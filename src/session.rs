//! Qwen chat session management.
//!
//! Fixes vs original:
//!  - Duplicate MODEL_NAME/QWEN_API_BASE removed; use crate::constants.
//!  - last_used stored as Arc<AtomicU64> — evict_oldest is always lock-free
//!    (original try_lock silently fell back to created_at, picking wrong victim).
//!  - cleanup_expired() gated to run at most once per 60 s (was on every request).
//!  - Redundant second last_used.lock().await in acquire() removed.
//!  - session_ttl() / max_sessions() cached with OnceLock — no env::var parse on hot path.

use crate::constants::QWEN_API_BASE;
use anyhow::{bail, Context, Result};
use dashmap::DashMap;
use futures::lock::{Mutex, OwnedMutexGuard};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Cached at first call — no env::var parse on the hot path.
fn session_ttl() -> Duration {
    static TTL: OnceLock<Duration> = OnceLock::new();
    *TTL.get_or_init(|| {
        let minutes = std::env::var("QWEN_PROXY_SESSION_TTL_MINUTES")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(30);
        Duration::from_secs(minutes * 60)
    })
}

/// Cached at first call — no env::var parse on the hot path.
fn max_sessions() -> usize {
    static MAX: OnceLock<usize> = OnceLock::new();
    *MAX.get_or_init(|| {
        std::env::var("QWEN_PROXY_MAX_SESSIONS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(100)
    })
}

#[derive(Clone)]
struct SessionEntry {
    chat_id: String,
    parent_id: Arc<Mutex<Option<String>>>,
    created_at: Instant,
    in_flight: Arc<Mutex<()>>,
    /// Lock-free last-used timestamp (Unix millis).
    last_used_ms: Arc<AtomicU64>,
}

/// Holds a per-chat in-flight lock until dropped (after the Qwen response completes).
pub struct AcquiredSession {
    pub chat_id: String,
    pub parent_id: Option<String>,
    pub parent_store: Arc<Mutex<Option<String>>>,
    _in_flight_guard: OwnedMutexGuard<()>,
    last_used_ms: Arc<AtomicU64>,
}

impl AcquiredSession {
    pub async fn set_parent_id(&self, parent_id: String) {
        *self.parent_store.lock().await = Some(parent_id);
        self.last_used_ms.store(now_millis(), Ordering::Relaxed);
    }
}

pub struct SessionManager {
    sessions: DashMap<String, SessionEntry>,
    /// Prevents cleanup running on every request hot path.
    last_cleanup_ms: AtomicU64,
}

impl SessionManager {
    pub fn new() -> Self {
        Self {
            sessions: DashMap::new(),
            last_cleanup_ms: AtomicU64::new(0),
        }
    }

    /// Get or create a Qwen chat for `client_key`, then wait until no other request is in-flight
    /// on that chat.
    pub async fn acquire(&self, client_key: &str, token: &str) -> Result<AcquiredSession> {
        let now = now_millis();
        let last = self.last_cleanup_ms.load(Ordering::Relaxed);
        if now.saturating_sub(last) > 60_000
            && self
                .last_cleanup_ms
                .compare_exchange(last, now, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
        {
            self.cleanup_expired();
        }

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

        entry.last_used_ms.store(now_millis(), Ordering::Relaxed);

        let in_flight_guard = entry.in_flight.lock_owned().await;
        let parent_id = entry.parent_id.lock().await.clone();

        Ok(AcquiredSession {
            chat_id: entry.chat_id,
            parent_id,
            parent_store: entry.parent_id,
            _in_flight_guard: in_flight_guard,
            last_used_ms: entry.last_used_ms,
        })
    }

    async fn insert_new_entry(&self, client_key: &str, token: &str) -> Result<SessionEntry> {
        let chat_id = self.create_chat(token).await?;
        info!(chat_id = %chat_id, client_key = %client_key,
            "Created Qwen chat (will reuse until TTL; concurrent sends are queued)");
        let entry = SessionEntry {
            chat_id,
            parent_id: Arc::new(Mutex::new(None)),
            created_at: Instant::now(),
            in_flight: Arc::new(Mutex::new(())),
            last_used_ms: Arc::new(AtomicU64::new(now_millis())),
        };
        self.sessions.insert(client_key.to_string(), entry.clone());
        Ok(entry)
    }

    async fn create_chat(&self, token: &str) -> Result<String> {
        let token = token.to_string();
        tokio::task::spawn_blocking(move || {
            let payload = serde_json::json!({
                "title": "Agent Chat",
                "models": [crate::constants::qwen_upstream_model(None)],
                "chat_mode": "normal",
                "chat_type": "t2t",
                "timestamp": chrono::Utc::now().timestamp_millis(),
            });
            let mut resp = ureq::agent().post(&format!("{}/chats/new", QWEN_API_BASE))
                .header("accept", "application/json")
                .header("content-type", "application/json")
                .header("referer", "https://chat.qwen.ai/")
                .header("source", "web")
                .header("version", "0.8.0")
                .header("cookie", &format!("token={}", token))
                .send(&serde_json::to_vec(&payload).context("serialize payload")?)
                .map_err(|e| anyhow::anyhow!("Failed to create Qwen chat: {}", e))?;
            if resp.status() == 401 {
                bail!("Qwen token expired or invalid");
            }
            let body = resp
                .body_mut()
                .read_to_string()
                .map_err(|e| anyhow::anyhow!("Failed to read chat response: {}", e))?;
            let d: serde_json::Value = serde_json::from_str(&body)
                .map_err(|e| anyhow::anyhow!("Failed to parse chat response: {}", e))?;
            d["data"]["id"]
                .as_str()
                .map(|s| s.to_string())
                .context("No chat ID in response")
        })
        .await
        .map_err(|e| anyhow::anyhow!("spawn_blocking join error: {}", e))?
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
        if let Some(key) = self
            .sessions
            .iter()
            .min_by_key(|e| e.last_used_ms.load(Ordering::Relaxed))
            .map(|e| e.key().clone())
        {
            warn!(key = %key, "Evicting oldest session (LRU)");
            self.sessions.remove(&key);
        }
    }
}
