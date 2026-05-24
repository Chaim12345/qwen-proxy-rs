mod qwen;
mod session;

use anyhow::{bail, Context, Result};
use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Json,
    },
    routing::{get, post},
    Router,
};
use futures::StreamExt;
use qwen::*;
use session::SessionManager;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, error, info};

use reqwest::header::{HeaderMap, HeaderValue};
use tower_http::cors::{Any, CorsLayer};

fn qwen_default_headers() -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(
        "user-agent",
        HeaderValue::from_static(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        ),
    );
    headers.insert("referer", HeaderValue::from_static("https://chat.qwen.ai/"));
    headers.insert("source", HeaderValue::from_static("web"));
    headers.insert("version", HeaderValue::from_static("0.8.0"));
    headers
}

fn qwen_http_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .default_headers(qwen_default_headers())
        .build()
        .context("Failed to build HTTP client")
}

const MODEL_NAME: &str = "qwen3.7-max";
const QWEN_API_BASE: &str = "https://chat.qwen.ai/api/v2";

struct AppState {
    sessions: SessionManager,
    http: reqwest::Client,
    token: String,
}

fn load_token() -> Result<String> {
    if let Ok(t) = std::env::var("QWEN_TOKEN") {
        if !t.is_empty() {
            info!("Loaded QWEN_TOKEN from environment");
            return Ok(t);
        }
    }

    if let Some(home) = dirs::home_dir() {
        let path = home.join(".qwen_session.json");
        if path.exists() {
            let content = std::fs::read_to_string(&path)
                .with_context(|| format!("Failed to read {}", path.display()))?;
            let sf: serde_json::Value =
                serde_json::from_str(&content).context("Failed to parse .qwen_session.json")?;
            if let Some(t) = sf["token"].as_str() {
                if !t.is_empty() {
                    info!("Loaded token from {}", path.display());
                    return Ok(t.to_string());
                }
            }
        }
    }

    bail!("No QWEN_TOKEN found. Set QWEN_TOKEN env var or create ~/.qwen_session.json");
}

fn model_info(id: &str) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "object": "model",
        "created": 1700000000,
        "owned_by": "qwen",
        "permission": [],
        "root": id,
        "parent": null
    })
}

fn openai_error(
    status: StatusCode,
    message: impl Into<String>,
    r#type: &str,
    param: Option<&str>,
    code: Option<&str>,
) -> (StatusCode, Json<serde_json::Value>) {
    let mut err = serde_json::json!({
        "message": message.into(),
        "type": r#type,
    });
    if let Some(p) = param {
        err["param"] = p.into();
    }
    if let Some(c) = code {
        err["code"] = c.into();
    }
    (status, Json(serde_json::json!({"error": err})))
}

fn bad_request(message: impl Into<String>) -> (StatusCode, Json<serde_json::Value>) {
    openai_error(
        StatusCode::BAD_REQUEST,
        message,
        "invalid_request_error",
        None,
        None,
    )
}

const MODELS_JSON: &str = r#"{"object":"list","data":[
{"id":"qwen3.7-max","object":"model","created":1700000000,"owned_by":"qwen","permission":[],"root":"qwen3.7-max","parent":null},
{"id":"qwen3.6-plus","object":"model","created":1700000000,"owned_by":"qwen","permission":[],"root":"qwen3.6-plus","parent":null},
{"id":"qwen3.6-max-preview","object":"model","created":1700000000,"owned_by":"qwen","permission":[],"root":"qwen3.6-max-preview","parent":null}
]}"#;

async fn models_handler() -> Json<serde_json::Value> {
    Json(serde_json::from_str(MODELS_JSON).unwrap())
}

async fn model_handler(
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    if model_id == MODEL_NAME
        || model_id == "qwen3.6-plus"
        || model_id == "qwen3.6-max-preview"
        || model_id == "gpt-4"
        || model_id == "gpt-4o"
        || model_id == "gpt-3.5-turbo"
    {
        let resolved_id =
            if model_id == "gpt-4" || model_id == "gpt-4o" || model_id == "gpt-3.5-turbo" {
                MODEL_NAME
            } else {
                &model_id
            };
        Ok(Json(model_info(resolved_id)))
    } else {
        Err(openai_error(
            StatusCode::NOT_FOUND,
            format!("Model '{}' not found", model_id),
            "invalid_request_error",
            Some("model"),
            Some("model_not_found"),
        ))
    }
}

async fn health_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok", "model": MODEL_NAME}))
}

async fn embeddings_handler(
    body: axum::body::Bytes,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let v: serde_json::Value =
        serde_json::from_slice(&body).map_err(|e| bad_request(format!("Invalid JSON: {}", e)))?;

    let input = v.get("input").and_then(|i| i.as_str()).unwrap_or("");
    let dims = v.get("dimensions").and_then(|d| d.as_u64()).unwrap_or(1536) as usize;
    let embedding: Vec<f64> = vec![0.0; dims];
    let tokens = std::cmp::max(1, input.len() / 4);

    let model = request_model(&v);

    Ok(Json(serde_json::json!({
        "object": "list",
        "data": [{
            "object": "embedding",
            "embedding": embedding,
            "index": 0
        }],
        "model": model,
        "usage": {
            "prompt_tokens": tokens,
            "total_tokens": tokens
        }
    })))
}

fn estimate_tokens(text: &str) -> usize {
    std::cmp::max(1, text.len() / 4)
}

fn request_model(v: &serde_json::Value) -> String {
    v.get("model")
        .and_then(|m| m.as_str())
        .filter(|m| !m.is_empty())
        .unwrap_or(MODEL_NAME)
        .to_string()
}

fn build_stream_chunk(
    id: &str,
    model: &str,
    created: i64,
    delta: serde_json::Value,
    finish_reason: Option<&str>,
) -> String {
    serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason
        }]
    })
    .to_string()
}

fn build_tool_call_stream_chunks(
    id: &str,
    model: &str,
    created: i64,
    tool_call_id: &str,
    name: &str,
    args_json: &str,
) -> Vec<String> {
    let mut chunks = Vec::new();
    chunks.push(build_stream_chunk(
        id,
        model,
        created,
        serde_json::json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "index": 0,
                "id": tool_call_id,
                "type": "function",
                "function": { "name": name, "arguments": "" }
            }]
        }),
        None,
    ));

    let args_bytes = args_json.as_bytes();
    let chunk_size = 8;
    for chunk_start in (0..args_bytes.len()).step_by(chunk_size) {
        let chunk_end = std::cmp::min(chunk_start + chunk_size, args_bytes.len());
        let arg_piece = String::from_utf8_lossy(&args_bytes[chunk_start..chunk_end]);
        chunks.push(build_stream_chunk(
            id,
            model,
            created,
            serde_json::json!({
                "tool_calls": [{
                    "index": 0,
                    "function": { "arguments": arg_piece.to_string() }
                }]
            }),
            None,
        ));
    }

    chunks.push(build_stream_chunk(
        id,
        model,
        created,
        serde_json::json!({}),
        Some("tool_calls"),
    ));
    chunks
}

fn append_sse_delta(full_text: &mut String, ch: &serde_json::Value) {
    if let Some(delta) = extract_qwen_sse_delta(ch) {
        full_text.push_str(&delta);
    }
}

/// Hash of the request `tools` array so identical prompts with different tool sets stay isolated.
fn tools_fingerprint(v: &serde_json::Value) -> u64 {
    let Some(tools) = v.get("tools").and_then(|t| t.as_array()) else {
        return 0;
    };
    if tools.is_empty() {
        return 0;
    }
    let serialized = serde_json::to_string(tools).unwrap_or_default();
    let mut hasher = DefaultHasher::new();
    serialized.hash(&mut hasher);
    hasher.finish()
}

fn session_tools_suffix(v: &serde_json::Value) -> String {
    let fp = tools_fingerprint(v);
    if fp == 0 {
        String::new()
    } else {
        format!(":t{:x}", fp)
    }
}

/// Stable key so multi-turn requests reuse the same Qwen chat_id.
fn client_session_key(v: &serde_json::Value) -> String {
    let tools_suffix = session_tools_suffix(v);
    if let Some(user) = v.get("user").and_then(|u| u.as_str()) {
        if !user.is_empty() {
            return format!("user:{}{}", user, tools_suffix);
        }
    }
    if let Some(meta) = v.get("metadata") {
        for key in ["session_id", "conversation_id", "chat_id"] {
            if let Some(id) = meta.get(key).and_then(|x| x.as_str()) {
                if !id.is_empty() {
                    return format!("meta:{}:{}{}", key, id, tools_suffix);
                }
            }
        }
    }
    let messages = v
        .get("messages")
        .or_else(|| v.get("input"))
        .and_then(|m| m.as_array());
    if let Some(msgs) = messages {
        for m in msgs {
            if m.get("role").and_then(|r| r.as_str()) == Some("user") {
                let content = m
                    .get("content")
                    .and_then(|c| c.as_str())
                    .unwrap_or("");
                if !content.is_empty() {
                    let mut hasher = DefaultHasher::new();
                    content.hash(&mut hasher);
                    return format!("conv:{:x}{}", hasher.finish(), tools_suffix);
                }
            }
        }
    }
    format!("ephemeral:{}", uuid::Uuid::new_v4())
}

async fn handler(
    State(st): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> Result<axum::response::Response, (StatusCode, Json<serde_json::Value>)> {
    let json_bytes = if body.len() >= 4 && body[0] == 0x28 && body[1] == 0xB5 {
        zstd::decode_all(&body[..])
            .map_err(|e| bad_request(format!("zstd decompression failed: {}", e)))?
    } else {
        body.to_vec()
    };

    let v: serde_json::Value = serde_json::from_slice(&json_bytes)
        .map_err(|e| bad_request(format!("Invalid JSON: {}", e)))?;

    let messages = if let Some(msgs) = v.get("messages").and_then(|m| m.as_array()) {
        msgs
    } else if let Some(input) = v.get("input").and_then(|m| m.as_array()) {
        input
    } else {
        return Err(bad_request("messages or input array is required"));
    };

    if messages.is_empty() {
        return Err(bad_request("messages array cannot be empty"));
    }

    let is_responses_api = v.get("input").is_some() && v.get("messages").is_none();
    let is_stream = v.get("stream").and_then(|x| x.as_bool()).unwrap_or(false);
    let msg = build_message(&v);
    let tools = parse_tools(&v);

    debug!(messages = messages.len(), stream = is_stream, "Processing request");

    let client_key = client_session_key(&v);
    let session = st.sessions.acquire(&client_key, &st.token).await.map_err(|e| {
        error!(error = %e, client_key = %client_key, "Failed to acquire session");
        let status = if e.to_string().contains("expired") {
            StatusCode::UNAUTHORIZED
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        };
        openai_error(status, e.to_string(), "server_error", None, None)
    })?;

    let session_id = session.chat_id.clone();
    let parent_id = session.parent_id.clone();
    let parent_store = session.parent_store.clone();

    let prompt = if tools.is_empty() {
        msg
    } else {
        format!(
            "{}\n\nAvailable functions:\n{}\n\nWhen you need to use a function, respond with: {{\"tool\":\"<name>\",\"args\":{{...}}}}.",
            msg,
            serde_json::to_string_pretty(&tools).unwrap_or_default()
        )
    };

    let payload = qwen_payload(&session_id, parent_id.as_deref(), &prompt);

    let resp = st
        .http
        .post(format!(
            "{}/chat/completions?chat_id={}",
            QWEN_API_BASE, session_id
        ))
        .header("accept", "text/event-stream")
        .header("content-type", "application/json")
        .header("referer", "https://chat.qwen.ai/")
        .header("source", "web")
        .header("version", "0.8.0")
        .header("cookie", format!("token={}", st.token))
        .json(&payload)
        .send()
        .await
        .map_err(|e| {
            error!(error = %e, "Failed to call Qwen API");
            openai_error(
                StatusCode::BAD_GATEWAY,
                format!("Qwen API error: {}", e),
                "server_error",
                None,
                None,
            )
        })?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body_text = resp.text().await.unwrap_or_default();
        error!(
            status = %status,
            body_preview = %&body_text[..body_text.len().min(500)],
            "Qwen chat/completions returned error"
        );
        if status.as_u16() == 429 || body_text.contains("in progress") {
            return Err(openai_error(
                StatusCode::TOO_MANY_REQUESTS,
                "Qwen chat is busy (another message in flight on this chat_id)",
                "rate_limit_exceeded",
                None,
                Some("chat_in_progress"),
            ));
        }
        return Err(openai_error(
            StatusCode::BAD_GATEWAY,
            format!("Qwen API returned {}", status),
            "server_error",
            None,
            None,
        ));
    }

    let completion_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();
    let prompt_tokens = estimate_tokens(&prompt);
    let model = request_model(&v);

    if is_stream {
        let stream = resp.bytes_stream();
        let completion_id_clone = completion_id.clone();
        let model_clone = model.clone();
        let tools_present = !tools.is_empty();

        let sse_stream = async_stream::stream! {
            let _session_guard = session;
            let mut buf = String::new();
            let mut full_text = String::new();
            let mut stream = std::pin::pin!(stream);

            while let Some(chunk_result) = stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        yield Ok::<_, std::convert::Infallible>(Event::default().data(build_stream_chunk(
                            &completion_id_clone, &model_clone, created,
                            serde_json::json!({"content": format!("[Stream error: {}]", e)}),
                            Some("stop")
                        )));
                        yield Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"));
                        return;
                    }
                };

                buf.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(nl) = buf.find('\n') {
                    let line = buf[..nl].trim().to_string();
                    buf = buf[nl + 1..].to_string();
                    if !line.starts_with("data: ") { continue; }
                    let d = line[6..].trim().to_string();
                    if d.is_empty() || d == "[DONE]" { continue; }
                    if let Ok(ch) = serde_json::from_str::<serde_json::Value>(&d) {
                        if let Some(pid) = extract_response_parent_id(&ch) {
                            *parent_store.lock().await = Some(pid);
                        }
                        append_sse_delta(&mut full_text, &ch);
                    }
                }
            }

            if full_text.is_empty() {
                if let Some(err) = parse_qwen_upstream_error(&buf) {
                    yield Ok::<_, std::convert::Infallible>(Event::default().data(build_stream_chunk(
                        &completion_id_clone, &model_clone, created,
                        serde_json::json!({"content": err}),
                        Some("stop")
                    )));
                    yield Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"));
                    return;
                }
            }

            let tc = detect_tool(&full_text, &tools);
            if let Some(tc) = tc {
                info!(tool = %tc.name, "Detected tool call (streaming)");
                let tool_call_id = format!("call_{}", uuid::Uuid::new_v4());
                let args = serde_json::to_string(&tc.args).unwrap_or_else(|_| "{}".to_string());
                for chunk in build_tool_call_stream_chunks(
                    &completion_id_clone, &model_clone, created, &tool_call_id, &tc.name, &args,
                ) {
                    yield Ok::<_, std::convert::Infallible>(Event::default().data(chunk));
                }
            } else {
                yield Ok::<_, std::convert::Infallible>(Event::default().data(build_stream_chunk(
                    &completion_id_clone, &model_clone, created,
                    serde_json::json!({"role": "assistant", "content": ""}),
                    None
                )));
                let visible = client_visible_content(&full_text, None, tools_present);
                let content_bytes = visible.as_bytes();
                let chunk_size = 16;
                for chunk_start in (0..content_bytes.len()).step_by(chunk_size) {
                    let chunk_end = std::cmp::min(chunk_start + chunk_size, content_bytes.len());
                    let piece = String::from_utf8_lossy(&content_bytes[chunk_start..chunk_end]);
                    yield Ok::<_, std::convert::Infallible>(Event::default().data(build_stream_chunk(
                        &completion_id_clone, &model_clone, created,
                        serde_json::json!({"content": piece.to_string()}),
                        None
                    )));
                }
                yield Ok::<_, std::convert::Infallible>(Event::default().data(build_stream_chunk(
                    &completion_id_clone, &model_clone, created,
                    serde_json::json!({}),
                    Some("stop")
                )));
            }

            yield Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"));
        };

        Ok(Sse::new(sse_stream)
            .keep_alive(
                KeepAlive::new()
                    .interval(Duration::from_secs(15))
                    .text("keep-alive-text"),
            )
            .into_response())
    } else {
        let mut buf = String::new();
        let mut full_text = String::new();
        let mut stream = resp.bytes_stream();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| {
                error!(error = %e, "Stream error");
                openai_error(
                    StatusCode::BAD_GATEWAY,
                    format!("Stream error: {}", e),
                    "server_error",
                    None,
                    None,
                )
            })?;

            buf.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(nl) = buf.find('\n') {
                let line = buf[..nl].trim().to_string();
                buf = buf[nl + 1..].to_string();
                if !line.starts_with("data: ") {
                    continue;
                }
                let d = line[6..].trim().to_string();
                if d.is_empty() || d == "[DONE]" {
                    continue;
                }
                if let Ok(ch) = serde_json::from_str::<serde_json::Value>(&d) {
                    if let Some(pid) = extract_response_parent_id(&ch) {
                        session.set_parent_id(pid).await;
                    }
                    append_sse_delta(&mut full_text, &ch);
                }
            }
        }

        let completion_tokens = estimate_tokens(&full_text);
        let total_tokens = prompt_tokens + completion_tokens;

        if full_text.is_empty() {
            if let Some(err) = parse_qwen_upstream_error(&buf) {
                return Err(openai_error(
                    StatusCode::TOO_MANY_REQUESTS,
                    err,
                    "rate_limit_exceeded",
                    None,
                    Some("rate_limit"),
                ));
            }
        }

        let tc = detect_tool(&full_text, &tools);

        if let Some(tc) = tc {
            info!(tool = %tc.name, "Detected tool call");
            let tool_call_id = format!("call_{}", uuid::Uuid::new_v4());
            let args = serde_json::to_string(&tc.args).unwrap_or_else(|_| "{}".to_string());

            if is_responses_api {
                Ok(Json(serde_json::json!({
                    "id": completion_id,
                    "object": "response",
                    "created_at": created,
                    "model": model,
                    "output": [{
                        "type": "function_call",
                        "id": tool_call_id,
                        "call_id": tool_call_id,
                        "name": tc.name,
                        "arguments": args
                    }],
                    "usage": {
                        "input_tokens": prompt_tokens,
                        "output_tokens": completion_tokens,
                        "total_tokens": total_tokens
                    }
                }))
                .into_response())
            } else {
                Ok(Json(serde_json::json!({
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": null,
                            "tool_calls": [{
                                "id": tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": args
                                }
                            }]
                        },
                        "finish_reason": "tool_calls"
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens
                    }
                }))
                .into_response())
            }
        } else {
            let visible = client_visible_content(&full_text, None, !tools.is_empty());
            info!(len = visible.len(), "Returning text response");

            if is_responses_api {
                Ok(Json(serde_json::json!({
                    "id": completion_id,
                    "object": "response",
                    "created_at": created,
                    "model": model,
                    "output": [{
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": visible}]
                    }],
                    "usage": {
                        "input_tokens": prompt_tokens,
                        "output_tokens": completion_tokens,
                        "total_tokens": total_tokens
                    }
                }))
                .into_response())
            } else {
                Ok(Json(serde_json::json!({
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": visible
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens
                    }
                }))
                .into_response())
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "qwen_proxy=info,tower_http=info".into()),
        )
        .init();

    let http = qwen_http_client()?;
    let token = load_token()?;

    let state = Arc::new(AppState {
        sessions: SessionManager::new(http.clone()),
        http,
        token,
    });

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/v1/models", get(models_handler))
        .route("/v1/models/:model", get(model_handler))
        .route("/v1/chat/completions", post(handler))
        .route("/v1/responses", post(handler))
        .route("/v1/embeddings", post(embeddings_handler))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .with_state(state);

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8765);
    let addr = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("Qwen OpenAI proxy listening on http://{}", addr);

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    info!("Shutting down");
}
