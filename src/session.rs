//! Qwen chat session management.
//!
//! Reusing a `chat_id` across turns is fine. Qwen returns 429 ("chat in progress") only when
//! two messages are sent on the same chat before the prior response finishes — we serialize
//! per-chat with an in-flight mutex.

use anyhow::{bail, Context, Result};
use dashmap::DashMap;
use futures::lock::{Mutex, OwnedMutexGuard};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

fn session_ttl() -> Duration {
    let minutes = std::env::var("QWEN_PROXY_SESSION_TTL_MINUTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30u64);
    Duration::from_secs(minutes * 60)
}

fn max_sessions() -> usize {
    std::env::var("QWEN_PROXY_MAX_SESSIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100)
}

const MODEL_NAME: &str = "qwen3.7-max";
const QWEN_API_BASE: &str = "https://chat.qwen.ai/api/v2";

#[derive(Clone)]
struct SessionEntry {
    chat_id: String,
    parent_id: Arc<Mutex<Option<String>>>,
    created_at: Instant,
    in_flight: Arc<Mutex<()>>,
    last_used: Arc<Mutex<Instant>>,
}

/// Holds a per-chat in-flight lock until dropped (after the Qwen response completes).
pub struct AcquiredSession {
    pub chat_id: String,
    pub parent_id: Option<String>,
    pub parent_store: Arc<Mutex<Option<String>>>,
    _in_flight_guard: OwnedMutexGuard<()>,
    _last_used: Arc<Mutex<Instant>>,
}

impl AcquiredSession {
    pub async fn set_parent_id(&self, parent_id: String) {
        *self.parent_store.lock().await = Some(parent_id);
        *self._last_used.lock().await = Instant::now();
    }
}

pub struct SessionManager {
    sessions: DashMap<String, SessionEntry>,
}

impl SessionManager {
    pub fn new() -> Self {
        Self { sessions: DashMap::new() }
    }

    /// Get or create a Qwen chat for `client_key`, then wait until no other request is in-flight
    /// on that chat.
    pub async fn acquire(&self, client_key: &str, token: &str) -> Result<AcquiredSession> {
        self.cleanup_expired();

        if self.sessions.len() >= max_sessions() {
            self.evict_oldest();
        }

        let entry = match self.sessions.get(client_key) {
            Some(existing) if existing.created_at.elapsed() < session_ttl() => {
                debug!(client_key = %client_key, chat_id = %existing.chat_id, "Reusing existing session");
                existing.clone()
            }
            Some(_) => {
                warn!(client_key = %client_key, "Session expired, recreating");
                drop(self.sessions.remove(client_key));
                self.insert_new_entry(client_key, token).await?
            }
            None => {
                debug!(client_key = %client_key, "Creating new session");
                self.insert_new_entry(client_key, token).await?
            }
        };

        *entry.last_used.lock().await = Instant::now();

        let in_flight_guard = entry.in_flight.lock_owned().await;
        let parent_id = entry.parent_id.lock().await.clone();

        Ok(AcquiredSession {
            chat_id: entry.chat_id,
            parent_id,
            parent_store: entry.parent_id,
            _in_flight_guard: in_flight_guard,
            _last_used: entry.last_used,
        })
    }

    async fn insert_new_entry(&self, client_key: &str, token: &str) -> Result<SessionEntry> {
        let chat_id = self.create_chat(token).await?;
        info!(
            chat_id = %chat_id,
            client_key = %client_key,
            "Created Qwen chat (will reuse until TTL; concurrent sends on same chat are queued)"
        );
        let entry = SessionEntry {
            chat_id,
            parent_id: Arc::new(Mutex::new(None)),
            created_at: Instant::now(),
            in_flight: Arc::new(Mutex::new(())),
            last_used: Arc::new(Mutex::new(Instant::now())),
        };
        self.sessions.insert(client_key.to_string(), entry.clone());
        Ok(entry)
    }

    async fn create_chat(&self, token: &str) -> Result<String> {
        let token = token.to_string();
        smol::unblock(move || {
            let payload = serde_json::json!({
                "title": "Agent Chat",
                "models": [MODEL_NAME],
                "chat_mode": "normal",
                "chat_type": "t2t",
                "timestamp": chrono::Utc::now().timestamp_millis(),
            });

            let resp = ureq::post(&format!("{}/chats/new", QWEN_API_BASE))
                .set("accept", "application/json")
                .set("content-type", "application/json")
                .set("referer", "https://chat.qwen.ai/")
                .set("source", "web")
                .set("version", "0.8.0")
                .set("cookie", &format!("token={}", token))
                .send_json(&payload)
                .map_err(|e| anyhow::anyhow!("Failed to create Qwen chat: {}", e))?;

            if resp.status() == 401 {
                bail!("Qwen token expired or invalid");
            }

            let d: serde_json::Value = resp
                .into_json()
                .map_err(|e| anyhow::anyhow!("Failed to parse chat response: {}", e))?;
            d["data"]["id"]
                .as_str()
                .map(|s| s.to_string())
                .context("No chat ID in response")
        }).await
    }

    fn cleanup_expired(&self) {
        let cutoff = Instant::now() - session_ttl();
        let before = self.sessions.len();
        self.sessions.retain(|_, s| s.created_at > cutoff);
        let after = self.sessions.len();
        if before != after {
            debug!(evicted = before - after, "Expired sessions cleaned up");
        }
    }

    fn evict_oldest(&self) {
        // Evict by LAST USED (LRU), not creation time
        if let Some(oldest_key) = self
            .sessions
            .iter()
            .min_by_key(|e| {
                e.last_used.try_lock().map(|g| *g).unwrap_or(e.created_at)
            })
            .map(|e| e.key().clone())
        {
            warn!(key = %oldest_key, "Evicting oldest session (LRU)");
            self.sessions.remove(&oldest_key);
        }
    }
}
