mod constants;
mod qwen;
mod session;
mod streaming;

use crate::constants::{MODEL_NAME, QWEN_API_BASE};
use anyhow::{bail, Context, Result};

use futures::StreamExt;
use http::{Method, StatusCode};
use http_body::Frame;
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::body::{Bytes, Incoming};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::TokioIo;
use qwen::*;
use session::SessionManager;
use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::Duration;
use tokio::net::TcpListener as TokioTcpListener;
use tracing::{debug, error, info, info_span, warn, Instrument};

struct AppState {
    sessions: SessionManager,
    token: String,
    http: reqwest::Client,
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

fn json_response<T: serde::Serialize>(status: StatusCode, body: &T) -> Response<Full<Bytes>> {
    let json = serde_json::to_string(body).unwrap_or_default();
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .header("content-length", json.len().to_string())
        .body(Full::new(Bytes::from(json)))
        .unwrap()
}

fn openai_error_response(
    status: StatusCode,
    message: impl Into<String>,
    r#type: &str,
    param: Option<&str>,
    code: Option<&str>,
) -> Response<Full<Bytes>> {
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
    json_response(status, &serde_json::json!({"error": err}))
}

fn bad_request(message: impl Into<String>) -> Response<Full<Bytes>> {
    openai_error_response(
        StatusCode::BAD_REQUEST,
        message,
        "invalid_request_error",
        None,
        None,
    )
}

fn internal_error(message: impl Into<String>) -> Response<Full<Bytes>> {
    openai_error_response(
        StatusCode::INTERNAL_SERVER_ERROR,
        message,
        "server_error",
        None,
        None,
    )
}

fn not_found_response() -> Response<Full<Bytes>> {
    json_response(
        StatusCode::NOT_FOUND,
        &serde_json::json!({"error": {"message": "Not found", "type": "not_found"}}),
    )
}

const MODELS_JSON: &str = r#"{"object":"list","data":[
{"id":"qwen3.7-max","object":"model","created":1700000000,"owned_by":"qwen","permission":[],"root":"qwen3.7-max","parent":null},
{"id":"qwen3.7-max-preview","object":"model","created":1700000000,"owned_by":"qwen","permission":[],"root":"qwen3.7-max-preview","parent":null},
{"id":"qwen3.6-plus","object":"model","created":1700000000,"owned_by":"qwen","permission":[],"root":"qwen3.6-plus","parent":null},
{"id":"qwen3.6-max-preview","object":"model","created":1700000000,"owned_by":"qwen","permission":[],"root":"qwen3.6-max-preview","parent":null}
]}"#;

/// perf(#12): build the response bytes exactly once; every request clones the Arc pointer,
/// not the payload.  Avoids a serde_json deserialise+serialise round-trip per call.
static MODELS_BYTES: LazyLock<Bytes> = LazyLock::new(|| Bytes::from_static(MODELS_JSON.as_bytes()));

fn models_handler() -> Response<Full<Bytes>> {
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .header("content-length", MODELS_BYTES.len().to_string())
        .body(Full::new(Bytes::clone(&MODELS_BYTES)))
        .unwrap()
}

fn model_handler(model_id: &str) -> Response<Full<Bytes>> {
    if model_id == MODEL_NAME
        || model_id == "qwen3.7-max-preview"
        || model_id == "qwen3.6-plus"
        || model_id == "qwen3.6-max-preview"
        || model_id == "gpt-4"
        || model_id == "gpt-4o"
        || model_id == "gpt-3.5-turbo"
    {
        let resolved_id = crate::constants::qwen_upstream_model(Some(model_id));
        json_response(StatusCode::OK, &model_info(&resolved_id))
    } else {
        openai_error_response(
            StatusCode::NOT_FOUND,
            format!("Model '{}' not found", model_id),
            "invalid_request_error",
            Some("model"),
            Some("model_not_found"),
        )
    }
}

fn health_handler() -> Response<Full<Bytes>> {
    json_response(
        StatusCode::OK,
        &serde_json::json!({"status": "ok", "model": MODEL_NAME}),
    )
}

fn estimate_tokens(text: &str) -> usize {
    std::cmp::max(1, text.len() / 4)
}

fn request_model(v: &serde_json::Value) -> String {
    crate::constants::qwen_upstream_model(
        v.get("model")
            .and_then(|m| m.as_str())
            .filter(|m| !m.is_empty()),
    )
}

/// OpenAI-compatible clients (Cursor, OpenCode, DeepSeek-compat) read `reasoning_content`
/// in the delta — not a custom `thinking` field.
fn build_reasoning_delta(text: &str, first: bool) -> serde_json::Value {
    let mut delta = serde_json::json!({
        "reasoning_content": text,
        "thinking": text,
    });
    if first {
        delta["role"] = serde_json::json!("assistant");
    }
    delta
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
    index: usize,
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
                "index": index,
                "id": tool_call_id,
                "type": "function",
                "function": { "name": name, "arguments": "" }
            }]
        }),
        None,
    ));

    // fix(#4): chunk by chars, not raw bytes, so multi-byte UTF-8 is never split.
    let chars: Vec<char> = args_json.chars().collect();
    let chunk_size = 8;
    for chunk_start in (0..chars.len()).step_by(chunk_size) {
        let chunk_end = std::cmp::min(chunk_start + chunk_size, chars.len());
        let arg_piece: String = chars[chunk_start..chunk_end].iter().collect();
        chunks.push(build_stream_chunk(
            id,
            model,
            created,
            serde_json::json!({
                "tool_calls": [{
                    "index": index,
                    "function": { "arguments": arg_piece }
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

fn append_sse_delta(acc: &mut AccumulatedText, ch: &serde_json::Value) {
    if let Some(delta) = extract_qwen_sse_delta(ch) {
        acc.append(&delta);
    }
}

/// Phase 3: small helper for best-effort detached feedback in stream error paths.
/// Client-visible error is emitted immediately; the send is spawned (no await, no added latency).
/// Reuses the send_qwen_chat_continuation (from 3.1) which handles logs, timeout, None-on-fail.
fn spawn_feedback_if_hallucinated(
    chat_id: String,
    parent_id: Option<String>,
    fb: String,
    token: String,
) {
    tokio::spawn(async move {
        let _ = send_qwen_chat_continuation(&chat_id, parent_id.as_deref(), &fb, &token).await;
    });
}

/// Small helper to DRY the 400 error shape for bad tool names (used by raw-body + non-stream validate err paths).
/// (The non-stream detect_qwen_tool_error path uses a similar shape but with upstream err_msg, so keeps local build.)
/// Keeps "available_tools" for client debugging (per Phase 2/4).
async fn handle_blocked_tools_nonstream(
    bad_names: &[String],
    tools: &[serde_json::Value],
    session_id: &str,
    latest_pid: Option<&str>,
    token: &str,
    session: &session::AcquiredSession,
    log_context: &str,
) -> Response<Full<Bytes>> {
    error!(
        hallucinated_tool_names = ?bad_names,
        client_tool_count = tools.len(),
        "{}: blocking hallucinated tool names", log_context
    );
    let fb = build_tool_hallucination_feedback(bad_names);
    if let Some(pid) = latest_pid {
        match send_qwen_chat_continuation(session_id, Some(pid), &fb, token).await {
            Ok(Some(new_pid)) => {
                session.set_parent_id(new_pid.clone()).await;
                info!(
                    chat_id = %session_id,
                    new_parent = %new_pid,
                    hallucinated = ?bad_names,
                    "Phase 3: feedback injected into Qwen chat — hallucination now in-context correction ({})",
                    log_context
                );
            }
            Ok(None) => {
                warn!(chat_id = %session_id, "Phase 3: feedback send completed with no new_pid ({}, best-effort)", log_context);
            }
            Err(e) => {
                warn!(error = %e, chat_id = %session_id, "Phase 3: feedback send failed ({}, non-fatal, client still gets clean 400)", log_context);
            }
        }
    } else {
        warn!(chat_id = %session_id, "Phase 3: no latest_pid captured; skipping feedback injection ({}, still returning 400 to client)", log_context);
    }
    let err_body = construct_tool_error_json(bad_names, tools);
    json_response(StatusCode::BAD_REQUEST, &err_body)
}

fn construct_tool_error_json(
    bad_names: &[String],
    tools: &[serde_json::Value],
) -> serde_json::Value {
    let mut err = serde_json::json!({
        "message": format!("Invalid tool call(s): {}. Only use exact names from the client's Available Tools list.", bad_names.join(", ")),
        "type": "invalid_request_error",
    });
    err["available_tools"] = serde_json::json!(tools);
    serde_json::json!({"error": err})
}

/// Phase 3 helpers (DRY): build the exact feedback strings used for in-context correction.
/// One for validate bad-names (used at all 4 gated emission sites + now detect_qwen non-stream),
/// one for upstream Qwen tool errors (detect_qwen paths). See plan §3.2/3.3/4.1.
fn build_tool_hallucination_feedback(bad_names: &[String]) -> String {
    format!(
        "TOOL RESULT: ERROR: You attempted to call tool(s) {:?} which are not in the Available Tools list. Only use exact names from the list the client provided. Retry with a valid tool or respond in plain text without a tool call.",
        bad_names
    )
}

fn extract_tool_name_from_error(err_msg: &str) -> String {
    if let Some(rest) = err_msg.strip_prefix("Tool ") {
        for suffix in &[" does not exist", " does not exists"] {
            if let Some(pos) = rest.find(suffix) {
                return rest[..pos].to_string();
            }
        }
    }
    err_msg
        .split(&[' ', '.'][..])
        .find(|w| !w.is_empty() && !["Tool", "does", "not", "exist", "exists"].contains(w))
        .unwrap_or("unknown_tool")
        .to_string()
}

fn build_synthetic_tool_ok_feedback_for_name(_tool_name: &str) -> String {
    "ok pretend it does exist and call it properly anyway".to_string()
}

/// Run validate_tool_calls; on synthetic-ok feedback, spawn best-effort Qwen continuation.
fn tool_gate_stream(
    raw: Vec<ToolCall>,
    tools: &[serde_json::Value],
    chat_id: &str,
    parent_id: Option<String>,
    token: &str,
) -> Result<Vec<ToolCall>, Vec<String>> {
    match validate_tool_calls(raw, tools) {
        ToolGateResult::Emit {
            emit,
            synthetic_feedback,
        } => {
            for fb in synthetic_feedback {
                spawn_feedback_if_hallucinated(
                    chat_id.to_string(),
                    parent_id.clone(),
                    fb,
                    token.to_string(),
                );
            }
            Ok(emit)
        }
        ToolGateResult::Blocked(bad) => Err(bad),
    }
}

/// Non-stream / raw-body: await synthetic-ok feedback injection when enabled.
async fn tool_gate_nonstream(
    raw: Vec<ToolCall>,
    tools: &[serde_json::Value],
    session_id: &str,
    latest_pid: Option<&str>,
    token: &str,
    session: &session::AcquiredSession,
) -> Result<Vec<ToolCall>, Vec<String>> {
    match validate_tool_calls(raw, tools) {
        ToolGateResult::Emit {
            emit,
            synthetic_feedback,
        } => {
            for fb in synthetic_feedback {
                if let Some(pid) = latest_pid {
                    match send_qwen_chat_continuation(session_id, Some(pid), &fb, token).await {
                        Ok(Some(new_pid)) => {
                            session.set_parent_id(new_pid.clone()).await;
                            info!(
                                chat_id = %session_id,
                                new_parent = %new_pid,
                                "TOOL_SYNTHETIC_OK: fake successful tool result injected into Qwen chat"
                            );
                        }
                        Ok(None) => {
                            warn!(chat_id = %session_id, "TOOL_SYNTHETIC_OK: no new parent_id returned");
                        }
                        Err(e) => {
                            warn!(error = %e, chat_id = %session_id, "TOOL_SYNTHETIC_OK: feedback send failed");
                        }
                    }
                } else {
                    warn!(chat_id = %session_id, "TOOL_SYNTHETIC_OK: no parent_id; skipping injection");
                }
            }
            Ok(emit)
        }
        ToolGateResult::Blocked(bad) => Err(bad),
    }
}

/// fix(#2): stable FNV-1a hasher — deterministic across process restarts,
/// zero extra dependencies, and faster than DefaultHasher for short strings.
fn fnv1a_hash(s: &str) -> u64 {
    const OFFSET: u64 = 14695981039346656037;
    const PRIME: u64 = 1099511628211;
    s.bytes()
        .fold(OFFSET, |h, b| (h ^ b as u64).wrapping_mul(PRIME))
}

fn tools_fingerprint(v: &serde_json::Value) -> u64 {
    let Some(tools) = v.get("tools").and_then(|t| t.as_array()) else {
        return 0;
    };
    if tools.is_empty() {
        return 0;
    }
    let serialized = serde_json::to_string(tools).unwrap_or_default();
    fnv1a_hash(&serialized)
}

fn session_tools_suffix(v: &serde_json::Value) -> String {
    let fp = tools_fingerprint(v);
    if fp == 0 {
        String::new()
    } else {
        format!(":t{:x}", fp)
    }
}

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
                let content: String = m
                    .get("content")
                    .and_then(|c| {
                        c.as_str().map(|s| s.to_string()).or_else(|| {
                            c.as_array().and_then(|arr| {
                                arr.iter().find_map(|part| {
                                    part.get("text")
                                        .and_then(|v| v.as_str())
                                        .or_else(|| part.get("input_text").and_then(|v| v.as_str()))
                                        .or_else(|| {
                                            part.get("output_text").and_then(|v| v.as_str())
                                        })
                                        .map(|s| s.to_string())
                                })
                            })
                        })
                    })
                    .unwrap_or_default();
                if !content.is_empty() {
                    // fix(#2): use fnv1a_hash instead of DefaultHasher
                    return format!("conv:{:x}{}", fnv1a_hash(&content), tools_suffix);
                }
            }
        }
    }
    format!("ephemeral:{}", uuid::Uuid::new_v4())
}

fn parse_qwen_sse_line(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed == "[DONE]" {
        return None;
    }
    trimmed
        .strip_prefix("data: ")
        .map(|rest| rest.trim().to_string())
}

fn qwen_request_headers(token: &str) -> Vec<(String, String)> {
    vec![
        ("accept".to_string(), "text/event-stream".to_string()),
        ("content-type".to_string(), "application/json".to_string()),
        ("referer".to_string(), "https://chat.qwen.ai/".to_string()),
        ("source".to_string(), "web".to_string()),
        ("version".to_string(), "0.8.0".to_string()),
        ("cookie".to_string(), format!("token={}", token)),
    ]
}

type BoxBody = http_body_util::combinators::UnsyncBoxBody<Bytes, std::io::Error>;

fn box_body<B>(body: B) -> BoxBody
where
    B: http_body::Body<Data = Bytes> + Send + 'static,
    B::Error: std::fmt::Display,
{
    body.map_err(|e| std::io::Error::other(format!("{}", e)))
        .boxed_unsync()
}

async fn handler(
    req: Request<Incoming>,
    st: Arc<AppState>,
) -> Result<Response<BoxBody>, Infallible> {
    let body_bytes = match req.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => {
            return Ok(bad_request(format!("Failed to read body: {}", e)).map(box_body));
        }
    };

    let json_bytes = body_bytes.to_vec();

    let v: serde_json::Value = match serde_json::from_slice(&json_bytes) {
        Ok(v) => v,
        Err(e) => {
            return Ok(bad_request(format!("Invalid JSON: {}", e)).map(box_body));
        }
    };

    let messages = if let Some(msgs) = v.get("messages").and_then(|m| m.as_array()) {
        msgs
    } else if let Some(input) = v.get("input").and_then(|m| m.as_array()) {
        input
    } else {
        return Ok(bad_request("messages or input array is required").map(box_body));
    };

    if messages.is_empty() {
        return Ok(bad_request("messages array cannot be empty").map(box_body));
    }

    let is_responses_api = v.get("input").is_some() && v.get("messages").is_none();
    let is_stream = v.get("stream").and_then(|x| x.as_bool()).unwrap_or(false);
    let response_format = v.get("response_format").cloned();
    let msg = build_message(&v);
    let tools = parse_tools(&v);

    debug!(
        messages = messages.len(),
        stream = is_stream,
        "Processing request"
    );

    let client_key = client_session_key(&v);
    let (session, session_id, parent_id, parent_store) = match st
        .sessions
        .acquire(&client_key, &st.token)
        .await
    {
        Ok(s) => {
            let sid = s.chat_id.clone();
            let pid = s.parent_id.clone();
            let ps = s.parent_store.clone();
            (s, sid, pid, ps)
        }
        Err(e) => {
            error!(error = %e, client_key = %client_key, "Failed to acquire session");
            let status = if e.to_string().contains("expired") {
                StatusCode::UNAUTHORIZED
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            return Ok(
                openai_error_response(status, e.to_string(), "server_error", None, None)
                    .map(box_body),
            );
        }
    };

    let prompt = if tools.is_empty() {
        msg
    } else {
        // Phase 1 Prompt Engineering Hardening (Robust Tool-Calling Translator; Phases 3-5 completed 2026-05 per docs/plans/finish-robust-tool-translator-20260526.md)
        // - Compact JSON (no pretty-print bloat)
        // - Reinforce codeblock + "exact name from the list" (removes the raw-JSON conflict with build_message)
        // - For response_format=json_* the build_message already emitted the correct "no markdown" rule;
        //   we still give the compact list + strict "use exact names" guidance.
        let compact_tools = serde_json::to_string(&tools).unwrap_or_default();
        format!(
            "{}\n\nAvailable Tools (exact names only — use these verbatim):\n{}\n\nWhen you need to call a tool, output ONLY a complete ```json fenced code block as the very last thing in your response, using an exact name from the list above. Never invent names. The block must contain nothing except the JSON.",
            msg,
            compact_tools
        )
    };

    let upstream_model = crate::constants::qwen_upstream_model(
        v.get("model")
            .and_then(|m| m.as_str())
            .filter(|m| !m.is_empty()),
    );
    let payload = qwen_payload(&session_id, parent_id.as_deref(), &prompt, &upstream_model);

    let qwen_url = format!("{}/chat/completions?chat_id={}", QWEN_API_BASE, session_id);
    let headers = qwen_request_headers(&st.token);

    let completion_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();
    let prompt_tokens = estimate_tokens(&prompt);
    let model = request_model(&v);

    // Fix #12: wrap in Arc to avoid cloning the full body Vec into the spawn closure.
    let body_arc: Arc<Vec<u8>> = Arc::new(serde_json::to_vec(&payload).unwrap_or_default());

    if is_stream {
        let rf = response_format.clone();
        let span = info_span!("stream", id = %completion_id, model = %model, tools = tools.len(), response_format = %rf.as_ref().map(|rf| format!("{:?}", rf.get("type"))).unwrap_or_else(|| "none".to_string()));
        async move {
        // Tokio mpsc replaces previous smol::channel for backpressure (8 line buffer)
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, String>>(64);

        let tx2 = tx.clone();
        let headers2 = headers.clone();
        let qwen_url2 = qwen_url.clone();
        let body2 = Arc::clone(&body_arc);
        let http = st.http.clone();

        // Spawn the upstream SSE reader — true reqwest streaming (incremental TTFB)
        tokio::spawn(async move {
            let mut sse = match streaming::post_sse(
                http,
                qwen_url2,
                headers2,
                (*body2).clone(),
            ).await {
                Ok(s) => s,
                Err(e) => {
                    let _ = tx2.send(Err(e)).await;
                    return;
                }
            };
            while let Some(item) = StreamExt::next(&mut sse).await {
                match item {
                    Ok(line_sse) => {
                        if tx2.send(Ok(line_sse.raw)).await.is_err() {
                            break;
                        }
                    }
                    Err(err_str) => {
                        let _ = tx2.send(Err(err_str)).await;
                        break;
                    }
                }
            }
            // Sender is dropped here → receiver will see None on next recv
        });

        let has_tools = !tools.is_empty();
        let is_responses_api = is_responses_api;
        // Phase 3: capture for detached feedback spawns in error arms below (no tuple bloat needed; closure captures persist across unfold steps)
        let chat_id_for_fb = session_id.clone();
        let token_for_fb = st.token.clone();
        let sse_stream = futures::stream::unfold(
            (rx, String::new(), AccumulatedText::new(), false, false, false, 0usize, 0usize, false, false, false, String::new()),
            move |(mut rx, buf, mut full_text, tool_emitted, mut content_emitted, done, mut prev_len, mut prev_thinking_len, mut thinking_role_sent, mut resp_created, mut output_started, mut item_id)| {
                let parent_store = parent_store.clone();
                let tools = tools.clone();
                let rf = rf.clone();
                let completion_id = completion_id.clone();
                let model = model.clone();
                // Phase 3: clone the fb context *per unfold step* (FnMut requires non-move for reusable closure; async move consumes the step-local clones)
                let chat_id_for_fb = chat_id_for_fb.clone();
                let token_for_fb = token_for_fb.clone();
                async move {
                    if done {
                        return None;
                    }
                    match rx.recv().await {
                        Some(Ok(line)) => {
                            // Track whether this chunk signals stream completion.
                            let mut stream_finished = false;
                            if let Some(data) = parse_qwen_sse_line(&line) {
                                if let Ok(ch) = serde_json::from_str::<serde_json::Value>(&data) {
                                    if let Some(pid) = extract_response_parent_id(&ch) {
                                        *parent_store.lock().await = Some(pid);
                                    }
                                    if let Some(delta) = extract_qwen_sse_delta(&ch) {
                                        stream_finished = delta.finished;
                                        full_text.append(&delta);
                                    }
                                }
                            }
                            let mut oai_chunks: Vec<String> = Vec::new();
                            if is_responses_api && !resp_created {
                                oai_chunks.push(serde_json::to_string(&serde_json::json!({
                                    "type": "response.created",
                                    "response": {
                                        "id": completion_id,
                                        "created_at": created,
                                        "model": model,
                                    }
                                })).unwrap_or_default());
                                resp_created = true;
                            }
                            let thinking = full_text.thinking();
                            if thinking.len() > prev_thinking_len {
                                let delta = &thinking[prev_thinking_len..];
                                let first_thinking = !thinking_role_sent;
                                oai_chunks.push(build_stream_chunk(
                                    &completion_id, &model, created,
                                    build_reasoning_delta(delta, first_thinking),
                                    None,
                                ));
                                thinking_role_sent = true;
                                prev_thinking_len = thinking.len();
                            }
                            // Fix #6: only run detect_tools on the finished delta to avoid
            // O(n²) scanning on every intermediate chunk.
                            // Phase 4: full_text accumulation (AccumulatedText) + terminal-only detect already provides
                            // complete payload for validate before any build_tool_call_stream_chunks or responses-api
                            // emission. No extra end-of-stream buffering required for name validation (tool JSON is atomic).
                            if !tool_emitted && !content_emitted && stream_finished {
                                let raw_tcs = detect_tools(full_text.full_answer(), &tools);
                                let pid_for_gate = parent_store.lock().await.clone();
                                let tcs = match tool_gate_stream(
                                    raw_tcs,
                                    &tools,
                                    &chat_id_for_fb,
                                    pid_for_gate,
                                    &token_for_fb,
                                ) {
                                    Ok(good) => good,
                                    Err(bad_names) => {
                                        error!(
                                            hallucinated_tool_names = ?bad_names,
                                            client_tool_count = tools.len(),
                                            "Phase 2 hard gate: blocking hallucinated tool names from mid-stream emission"
                                        );
                                        let err_msg = format!("Invalid tool call(s): {}. Only use exact names from the client's Available Tools list.", bad_names.join(", "));
                                        if is_responses_api {
                                            oai_chunks.push(serde_json::to_string(&serde_json::json!({
                                                "type": "error",
                                                "error": {"message": err_msg, "type": "tool_error"},
                                            })).unwrap_or_default());
                                        } else {
                                            oai_chunks.push(build_stream_chunk(
                                                &completion_id, &model, created,
                                                serde_json::json!({"content": format!("[Tool Error: {}]", err_msg)}),
                                                Some("stop"),
                                            ));
                                        }
                                        // Phase 3: spawn detached feedback (best-effort; client error sent immediately)
                                        let pid_for_fb = parent_store.lock().await.clone();
                                        let fb = build_tool_hallucination_feedback(&bad_names);
                                        spawn_feedback_if_hallucinated(chat_id_for_fb.clone(), pid_for_fb, fb, token_for_fb.clone());
                                        info!(chat_id = %chat_id_for_fb, hallucinated = ?bad_names, "Phase 3 feedback spawned async (mid-stream)");
                                        let out = oai_chunks.iter().map(|s| format!("data: {}\n\n", s)).collect::<String>();
                                        return Some((Ok::<_, std::io::Error>(Frame::data(Bytes::from(out))), (rx, buf, full_text, true, false, false, prev_len, prev_thinking_len, thinking_role_sent, resp_created, output_started, item_id)));
                                    }
                                };
                                if !tcs.is_empty() {
                                    info!(
                                        client_tool_count = tools.len(),
                                        emitted_tool_count = tcs.len(),
                                        tool_names = ?tcs.iter().map(|t| &t.name).collect::<Vec<_>>(),
                                        "Streaming: emitting validated tool calls to client (Phase 2 hard gate passed)"
                                    );
                                    for (i, tc) in tcs.iter().enumerate() {
                                        info!(tool = %tc.name, index = i, "Detected tool call");
                                        let tid = format!("call_{}", uuid::Uuid::new_v4());
                                        let args = serde_json::to_string(&tc.args).unwrap_or_else(|_| "{}".into());
                                        if is_responses_api {
                                            if !output_started {
                                                oai_chunks.push(serde_json::to_string(&serde_json::json!({
                                                    "type": "response.output_item.added",
                                                    "output_index": i,
                                                    "item": {
                                                        "id": tid,
                                                        "type": "function_call",
                                                        "call_id": tid.clone(),
                                                        "name": tc.name,
                                                        "arguments": "",
                                                    }
                                                })).unwrap_or_default());
                                                item_id = tid.clone();
                                                output_started = true;
                                            }
                                            oai_chunks.push(serde_json::to_string(&serde_json::json!({
                                                "type": "response.function_call_arguments.delta",
                                                "item_id": tid,
                                                "output_index": i,
                                                "delta": args,
                                            })).unwrap_or_default());
                                        } else {
                                            for s in build_tool_call_stream_chunks(&completion_id, &model, created, i, &tid, &tc.name, &args) {
                                                oai_chunks.push(s);
                                            }
                                        }
                                    }
                                    let out = oai_chunks.iter().map(|s| format!("data: {}\n\n", s)).collect::<String>();
                                    return Some((Ok::<_, std::io::Error>(Frame::data(Bytes::from(out))), (rx, buf, full_text, true, false, false, prev_len, prev_thinking_len, thinking_role_sent, resp_created, output_started, item_id)));
                                }
                                let answer = full_text.full_answer().to_string();
                                let visible = client_visible_content(&answer, None, has_tools);
                                if !visible.is_empty() {
                                    content_emitted = true;
                                }
                            }
                            if !tool_emitted && (!has_tools || content_emitted) {
                                let answer = full_text.full_answer().to_string();
                                let visible = client_visible_content(&answer, None, has_tools);
                                let visible = if rf.is_some() {
                                    strip_json_codeblock(&visible)
                                } else {
                                    visible
                                };
                                if visible.len() > prev_len {
                                    let delta = &visible[prev_len..];
                                    if is_responses_api {
                                        if !output_started {
                                            let new_id = format!("msg_{}", uuid::Uuid::new_v4());
                                            item_id = new_id.clone();
                                            oai_chunks.push(serde_json::to_string(&serde_json::json!({
                                                "type": "response.output_item.added",
                                                "output_index": 0,
                                                "item": {
                                                    "id": new_id,
                                                    "type": "message",
                                                    "role": "assistant",
                                                    "content": [{"type": "output_text", "text": "", "annotations": []}]
                                                }
                                            })).unwrap_or_default());
                                            output_started = true;
                                        }
                                        oai_chunks.push(serde_json::to_string(&serde_json::json!({
                                            "type": "response.output_text.delta",
                                            "item_id": item_id,
                                            "delta": delta,
                                        })).unwrap_or_default());
                                    } else {
                                        if prev_len == 0 {
                                            oai_chunks.push(build_stream_chunk(
                                                &completion_id, &model, created,
                                                serde_json::json!({"role": "assistant", "content": delta.to_string()}),
                                                None,
                                            ));
                                        } else {
                                            oai_chunks.push(build_stream_chunk(
                                                &completion_id, &model, created,
                                                serde_json::json!({"content": delta.to_string()}),
                                                None,
                                            ));
                                        }
                                    }
                                    prev_len = visible.len();
                                } else if visible.len() < prev_len {
                                    prev_len = visible.len();
                                }
                            }
                            if !oai_chunks.is_empty() {
                                let out = oai_chunks.iter().map(|s| format!("data: {}\n\n", s)).collect::<String>();
                                return Some((Ok::<_, std::io::Error>(Frame::data(Bytes::from(out))), (rx, buf, full_text, tool_emitted, content_emitted, false, prev_len, prev_thinking_len, thinking_role_sent, resp_created, output_started, item_id)));
                            }
                            Some((Ok::<_, std::io::Error>(Frame::data(Bytes::from(""))), (rx, buf, full_text, tool_emitted, content_emitted, false, prev_len, prev_thinking_len, thinking_role_sent, resp_created, output_started, item_id)))
                        }
                        Some(Err(err_str)) => {
                            let err_chunk = format!(
                                "data: {}\n\ndata: [DONE]\n\n",
                                build_stream_chunk(&completion_id, &model, created,
                                    serde_json::json!({"content": format!("[Error: {}]", err_str)}),
                                    Some("stop"),
                                )
                            );
                            Some((Ok::<_, std::io::Error>(Frame::data(Bytes::from(err_chunk))), (rx, buf, full_text, tool_emitted, content_emitted, true, prev_len, prev_thinking_len, thinking_role_sent, resp_created, output_started, item_id)))
                        }
                        None => {   // all senders dropped (normal end or error path)
                            let mut final_chunks = String::new();
                            if !tool_emitted && !content_emitted {
                                let answer = full_text.full_answer().to_string();
                                if has_tools {
                                    if let Some(err_msg) = detect_qwen_tool_error(&answer) {
                                        error!(error = %err_msg, "Qwen returned tool error in stream");
                                        let pid_for_recovery = parent_store.lock().await.clone();
                                        let tool_name = extract_tool_name_from_error(&err_msg);
                                        let ok_fb = build_synthetic_tool_ok_feedback_for_name(&tool_name);
                                        match send_qwen_continuation_and_get_response(
                                            &chat_id_for_fb,
                                            pid_for_recovery.as_deref(),
                                            &ok_fb,
                                            &token_for_fb,
                                        )
                                        .await
                                        {
                                            Ok(r) => {
                                                if let Some(new_pid) = r.new_parent_id {
                                                    *parent_store.lock().await = Some(new_pid);
                                                }
                                                let txt = r.response_text;
                                                if !txt.is_empty() {
                                                    info!(
                                                        chat_id = %chat_id_for_fb,
                                                        error = %err_msg,
                                                        tool_name = %tool_name,
                                                        "Stream tool error recovery: emitting synthetic response"
                                                    );
                                                    let recovery_tokens = estimate_tokens(&txt);
                                                    if is_responses_api {
                                                        let mid = format!("msg_{}", uuid::Uuid::new_v4());
                                                        final_chunks.push_str(&format!("data: {}\n\n", serde_json::to_string(&serde_json::json!({
                                                            "type": "response.output_item.added",
                                                            "output_index": 0,
                                                            "item": {
                                                                "id": mid,
                                                                "type": "message",
                                                                "role": "assistant",
                                                                "content": [{"type": "output_text", "text": "", "annotations": []}]
                                                            }
                                                        })).unwrap_or_default()));
                                                        final_chunks.push_str(&format!("data: {}\n\n", serde_json::to_string(&serde_json::json!({
                                                            "type": "response.output_text.delta",
                                                            "item_id": mid,
                                                            "delta": txt,
                                                        })).unwrap_or_default()));
                                                        final_chunks.push_str(&format!("data: {}\n\n", serde_json::to_string(&serde_json::json!({
                                                            "type": "response.output_item.done",
                                                            "output_index": 0,
                                                            "item": {
                                                                "id": mid,
                                                                "type": "message",
                                                                "role": "assistant",
                                                                "content": [{"type": "output_text", "text": txt, "annotations": []}]
                                                            }
                                                        })).unwrap_or_default()));
                                                        final_chunks.push_str(&format!("data: {}\n\n", serde_json::to_string(&serde_json::json!({
                                                            "type": "response.completed",
                                                            "response": {
                                                                "id": completion_id,
                                                                "object": "response",
                                                                "created_at": created,
                                                                "model": model,
                                                                "usage": {
                                                                    "input_tokens": prompt_tokens,
                                                                    "output_tokens": recovery_tokens,
                                                                    "total_tokens": prompt_tokens + recovery_tokens,
                                                                }
                                                            }
                                                        })).unwrap_or_default()));
                                                    } else {
                                                        final_chunks.push_str(&format!(
                                                            "data: {}\n\n",
                                                            build_stream_chunk(&completion_id, &model, created,
                                                                serde_json::json!({"role": "assistant", "content": txt}),
                                                                None,
                                                            )
                                                        ));
                                                        final_chunks.push_str(&format!(
                                                            "data: {}\n\n",
                                                            build_stream_chunk(&completion_id, &model, created,
                                                                serde_json::json!({}),
                                                                Some("stop"),
                                                            )
                                                        ));
                                                    }
                                                    final_chunks.push_str("data: [DONE]\n\n");
                                                    return Some((Ok::<_, std::io::Error>(Frame::data(Bytes::from(final_chunks))), (rx, buf, full_text, true, false, true, prev_len, prev_thinking_len, thinking_role_sent, true, true, String::new())));
                                                }
                                            }
                                            Err(e) => {
                                                warn!(error = %e, chat_id = %chat_id_for_fb, "Stream tool error recovery failed, falling back to error chunks");
                                            }
                                        }
                                        // Fallback: original error behavior
                                        if is_responses_api {
                                            final_chunks.push_str(&format!("data: {}\n\n", serde_json::to_string(&serde_json::json!({
                                                "type": "error",
                                                "error": {"message": err_msg, "type": "tool_error"},
                                            })).unwrap_or_default()));
                                        } else {
                                            final_chunks.push_str(&format!(
                                                "data: {}\n\n",
                                                build_stream_chunk(&completion_id, &model, created,
                                                    serde_json::json!({"content": format!("[Tool Error: {}]", err_msg)}),
                                                    Some("stop"),
                                                )
                                            ));
                                        }
                                        return Some((Ok::<_, std::io::Error>(Frame::data(Bytes::from(final_chunks))), (rx, buf, full_text, tool_emitted, content_emitted, true, prev_len, prev_thinking_len, thinking_role_sent, resp_created, output_started, item_id)));
                                    }
                                }
                                // Phase 4: full_text accumulation + terminal-only detect already provides complete payload
                                // for validate before emission. No extra buffering required (see mid-stream comment).
                                let raw_tcs = detect_tools(&answer, &tools);
                                let pid_for_gate = parent_store.lock().await.clone();
                                let tcs = match tool_gate_stream(
                                    raw_tcs,
                                    &tools,
                                    &chat_id_for_fb,
                                    pid_for_gate,
                                    &token_for_fb,
                                ) {
                                    Ok(good) => good,
                                    Err(bad_names) => {
                                        error!(
                                            hallucinated_tool_names = ?bad_names,
                                            client_tool_count = tools.len(),
                                            "Phase 2 hard gate: blocking hallucinated tool names from stream-end emission"
                                        );
                                        let err_msg = format!("Invalid tool call(s): {}. Only use exact names from the client's Available Tools list.", bad_names.join(", "));
                                        if is_responses_api {
                                            final_chunks.push_str(&format!("data: {}\n\n", serde_json::to_string(&serde_json::json!({
                                                "type": "error",
                                                "error": {"message": err_msg, "type": "tool_error"},
                                            })).unwrap_or_default()));
                                        } else {
                                            final_chunks.push_str(&format!(
                                                "data: {}\n\n",
                                                build_stream_chunk(&completion_id, &model, created,
                                                    serde_json::json!({"content": format!("[Tool Error: {}]", err_msg)}),
                                                    Some("stop"),
                                                )
                                            ));
                                        }
                                        // Phase 3: spawn detached feedback (stream-end validate err path)
                                        let pid_for_fb = parent_store.lock().await.clone();
                                        let fb = build_tool_hallucination_feedback(&bad_names);
                                        spawn_feedback_if_hallucinated(chat_id_for_fb.clone(), pid_for_fb, fb, token_for_fb.clone());
                                        info!(chat_id = %chat_id_for_fb, hallucinated = ?bad_names, "Phase 3 feedback spawned async (stream-end)");
                                        return Some((Ok::<_, std::io::Error>(Frame::data(Bytes::from(final_chunks))), (rx, buf, full_text, tool_emitted, content_emitted, true, prev_len, prev_thinking_len, thinking_role_sent, resp_created, output_started, item_id)));
                                    }
                                };
                                if !tcs.is_empty() {
                                    info!(
                                        client_tool_count = tools.len(),
                                        emitted_tool_count = tcs.len(),
                                        tool_names = ?tcs.iter().map(|t| &t.name).collect::<Vec<_>>(),
                                        "Stream end: emitting validated tool calls to client (Phase 2 hard gate passed)"
                                    );
                                    if is_responses_api {
                                        for (i, tc) in tcs.iter().enumerate() {
                                            info!(tool = %tc.name, index = i, "Detected tool call at stream end");
                                            let tid = format!("call_{}", uuid::Uuid::new_v4());
                                            let args = serde_json::to_string(&tc.args).unwrap_or_else(|_| "{}".into());
                                            if !output_started {
                                                final_chunks.push_str(&format!("data: {}\n\n", serde_json::to_string(&serde_json::json!({
                                                    "type": "response.output_item.added",
                                                    "output_index": i,
                                                    "item": {
                                                        "id": tid,
                                                        "type": "function_call",
                                                        "call_id": tid.clone(),
                                                        "name": tc.name,
                                                        "arguments": args,
                                                    }
                                                })).unwrap_or_default()));
                                                output_started = true;
                                            }
                                        }
                                    } else {
                                        for (i, tc) in tcs.iter().enumerate() {
                                            info!(tool = %tc.name, index = i, "Detected tool call at stream end");
                                            let tid = format!("call_{}", uuid::Uuid::new_v4());
                                            let args = serde_json::to_string(&tc.args).unwrap_or_else(|_| "{}".into());
                                            for s in build_tool_call_stream_chunks(&completion_id, &model, created, i, &tid, &tc.name, &args) {
                                                final_chunks.push_str(&format!("data: {}\n\n", s));
                                            }
                                        }
                                    }
                                } else {
                                    let visible = client_visible_content(&answer, None, has_tools);
                                    let visible = match process_structured_output(&visible, rf.as_ref()) {
                                        Ok(v) => v,
                                        Err(e) => {
                                            error!(error = %e, "Structured output validation failed at stream end");
                                            format!("[Structured Output Error: {}", e)
                                        }
                                    };
                                    if !visible.is_empty() {
                                        if is_responses_api {
                                            final_chunks.push_str(&format!("data: {}\n\n", serde_json::to_string(&serde_json::json!({
                                                "type": "response.output_item.added",
                                                "output_index": 0,
                                                "item": {
                                                    "id": format!("msg_{}", uuid::Uuid::new_v4()),
                                                    "type": "message",
                                                    "role": "assistant",
                                                    "content": [{"type": "output_text", "text": visible, "annotations": []}]
                                                }
                                            })).unwrap_or_default()));
                                        } else {
                                            final_chunks.push_str(&format!(
                                                "data: {}\n\n",
                                                build_stream_chunk(&completion_id, &model, created,
                                                    serde_json::json!({"role": "assistant", "content": visible}),
                                                    Some("stop"),
                                                )
                                            ));
                                        }
                                    }
                                }
                            }
                            if is_responses_api {
                                if output_started && !tool_emitted {
                                    final_chunks.push_str(&format!("data: {}\n\n", serde_json::to_string(&serde_json::json!({
                                        "type": "response.output_item.done",
                                        "output_index": 0,
                                        "item": {
                                            "id": item_id,
                                            "type": "message",
                                            "role": "assistant",
                                            "content": [{"type": "output_text", "text": client_visible_content(full_text.full_answer(), None, has_tools), "annotations": []}]
                                        }
                                    })).unwrap_or_default()));
                                }
                                let resp_completion_tokens = estimate_tokens(full_text.full_answer());
                                final_chunks.push_str(&format!("data: {}\n\n", serde_json::to_string(&serde_json::json!({
                                    "type": "response.completed",
                                    "response": {
                                        "id": completion_id,
                                        "object": "response",
                                        "created_at": created,
                                        "model": model,
                                        "usage": {
                                            "input_tokens": prompt_tokens,
                                            "output_tokens": resp_completion_tokens,
                                            "total_tokens": prompt_tokens + resp_completion_tokens,
                                        }
                                    }
                                })).unwrap_or_default()));
                            } else {
                                if tool_emitted {
                                    final_chunks.push_str(&format!(
                                        "data: {}\n\n",
                                        build_stream_chunk(&completion_id, &model, created,
                                            serde_json::json!({}),
                                            Some("tool_calls"),
                                        )
                                    ));
                                } else if content_emitted {
                                    final_chunks.push_str(&format!(
                                        "data: {}\n\n",
                                        build_stream_chunk(&completion_id, &model, created,
                                            serde_json::json!({}),
                                            Some("stop"),
                                        )
                                    ));
                                }
                            }
                            final_chunks.push_str("data: [DONE]\n\n");
                            Some((Ok::<_, std::io::Error>(Frame::data(Bytes::from(final_chunks))), (rx, buf, full_text, tool_emitted, content_emitted, true, prev_len, prev_thinking_len, thinking_role_sent, resp_created, output_started, item_id)))
                        }
                    }
                }
            },
        );

        let body = StreamBody::new(sse_stream);

        let response = Response::builder()
            .status(200)
            .header("content-type", "text/event-stream")
            .header("cache-control", "no-cache")
            .header("connection", "keep-alive")
            .header("access-control-allow-origin", "*")
            .header("access-control-allow-methods", "*")
            .header("access-control-allow-headers", "*")
            .body(box_body(body))
            .unwrap();

        Ok(response)
    }.instrument(span).await
    } else {
        // Use tokio::task::spawn_blocking for the blocking ureq call (now safe under Tokio runtime)
        let body_arc2 = Arc::clone(&body_arc);
        let qwen_url2 = qwen_url.clone();
        let headers2 = headers.clone();
        let http_res: Result<(u16, String), String> = tokio::task::spawn_blocking(move || {
            let mut req = ureq::post(&qwen_url2);
            for (k, v) in &headers2 {
                req = req.set(k.as_str(), v.as_str());
            }
            req = req.set("accept", "text/event-stream");
            match req.send_bytes(&body_arc2) {
                Ok(resp) => {
                    let status = resp.status();
                    let body_text = resp.into_string().unwrap_or_default();
                    Ok((status, body_text))
                }
                Err(e) => Err(format!("Qwen request failed: {}", e)),
            }
        })
        .await
        .map_err(|e| format!("spawn_blocking join error: {}", e))
        .unwrap_or_else(Err);

        match http_res {
            Ok((status, body_text)) => {
                if !(200..300).contains(&status) {
                    let preview: String = body_text.chars().take(500).collect();
                    error!(status = %status, body_preview = %preview, "Qwen chat/completions returned error");
                    if status == 429 || body_text.contains("in progress") {
                        return Ok(internal_error(
                            "Qwen chat is busy (another message in flight on this chat_id)",
                        )
                        .map(box_body));
                    }
                    return Ok(
                        internal_error(format!("Qwen API returned {}", status)).map(box_body)
                    );
                }
                // body_text now from blocking ureq; proceed with original line processing (includes set_parent_id awaits)
                let mut acc = AccumulatedText::new();
                for line in body_text.lines() {
                    if let Some(data) = parse_qwen_sse_line(line) {
                        if data == "[DONE]" {
                            continue;
                        }
                        if let Ok(ch) = serde_json::from_str::<serde_json::Value>(&data) {
                            if let Some(pid) = extract_response_parent_id(&ch) {
                                session.set_parent_id(pid).await;
                            }
                            append_sse_delta(&mut acc, &ch);
                        }
                    }
                }

                let latest_pid = parent_store.lock().await.clone(); // captured post-processing; all sets from the line loop above are visible
                let full_text = acc.full_answer().to_string();
                let completion_tokens = estimate_tokens(&full_text);
                let total_tokens = prompt_tokens + completion_tokens;

                if full_text.is_empty() {
                    if let Some(err) = parse_qwen_upstream_error(&body_text) {
                        return Ok(openai_error_response(
                            StatusCode::TOO_MANY_REQUESTS,
                            err,
                            "rate_limit_exceeded",
                            None,
                            Some("rate_limit"),
                        )
                        .map(box_body));
                    }
                }

                if full_text.is_empty() && !tools.is_empty() {
                    // Check for Qwen tool errors in the raw body even when accumulated answer is empty
                    // (e.g., all text was in a non-answer phase or the error is in the SSE metadata).
                    if let Some(err_msg) = detect_qwen_tool_error(&body_text) {
                        error!(error = %err_msg, "Qwen returned tool error in raw body (empty full_text)");
                        let mut err = serde_json::json!({
                            "message": err_msg,
                            "type": "invalid_request_error",
                        });
                        err["available_tools"] = serde_json::json!(tools);
                        return Ok(json_response(
                            StatusCode::BAD_REQUEST,
                            &serde_json::json!({"error": err}),
                        )
                        .map(box_body));
                    }
                    // Phase 4.1 / 3.4: close the raw-body bypass (was detect + direct emit with zero validate).
                    // Now uniform hard gate + feedback like the other 3 emission sites. Unknown names *never* leak.
                    let raw_tcs = detect_tools(&body_text, &tools);
                    let tcs = match tool_gate_nonstream(
                        raw_tcs,
                        &tools,
                        &session_id,
                        latest_pid.as_deref(),
                        &st.token,
                        &session,
                    )
                    .await
                    {
                        Ok(good) => good,
                        Err(bad_names) => {
                            return Ok(handle_blocked_tools_nonstream(
                                &bad_names,
                                &tools,
                                &session_id,
                                latest_pid.as_deref(),
                                &st.token,
                                &session,
                                "Phase 4.1 raw-body",
                            )
                            .await
                            .map(box_body));
                        }
                    };
                    if !tcs.is_empty() {
                        let tool_calls: Vec<serde_json::Value> = tcs
                            .iter()
                            .enumerate()
                            .map(|(i, tc)| {
                                info!(tool = %tc.name, index = i, "Detected tool call in raw body");
                                let tool_call_id = format!("call_{}", uuid::Uuid::new_v4());
                                let args = serde_json::to_string(&tc.args)
                                    .unwrap_or_else(|_| "{}".to_string());
                                serde_json::json!({
                                    "index": i,
                                    "id": tool_call_id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.name,
                                        "arguments": args
                                    }
                                })
                            })
                            .collect();
                        let resp_value = serde_json::json!({
                            "id": completion_id,
                            "object": "chat.completion",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": null,
                                    "tool_calls": tool_calls
                                },
                                "finish_reason": "tool_calls"
                            }],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": 0,
                                "total_tokens": prompt_tokens
                            }
                        });
                        info!(
                            count = tcs.len(),
                            "Returning tool calls from raw body (validated Phase 4.1)"
                        );
                        return Ok(json_response(StatusCode::OK, &resp_value).map(box_body));
                    }
                }

                if !full_text.is_empty() && !tools.is_empty() {
                    if let Some(err_msg) = detect_qwen_tool_error(&full_text) {
                        error!(error = %err_msg, "Qwen returned tool error in response text");
                        let tool_name = extract_tool_name_from_error(&err_msg);
                        let ok_fb = build_synthetic_tool_ok_feedback_for_name(&tool_name);
                        if let Some(pid) = &latest_pid {
                            match send_qwen_continuation_and_get_response(
                                &session_id,
                                Some(pid),
                                &ok_fb,
                                &st.token,
                            )
                            .await
                            {
                                Ok(result) => {
                                    if let Some(new_pid) = result.new_parent_id {
                                        session.set_parent_id(new_pid.clone()).await;
                                        info!(
                                            chat_id = %session_id,
                                            new_parent = %new_pid,
                                            error = %err_msg,
                                            tool_name = %tool_name,
                                            "Tool error recovery: synthetic OK injected, new response obtained"
                                        );
                                    } else {
                                        info!(
                                            chat_id = %session_id,
                                            error = %err_msg,
                                            "Tool error recovery: synthetic OK injected (no new_pid)"
                                        );
                                    }
                                    let new_text = result.response_text;
                                    if !new_text.is_empty() {
                                        let new_tokens = estimate_tokens(&new_text);
                                        let total = prompt_tokens + new_tokens;
                                        let resp_value = serde_json::json!({
                                            "id": completion_id,
                                            "object": "chat.completion",
                                            "created": created,
                                            "model": model,
                                            "choices": [{
                                                "index": 0,
                                                "message": {
                                                    "role": "assistant",
                                                    "content": new_text,
                                                },
                                                "finish_reason": "stop"
                                            }],
                                            "usage": {
                                                "prompt_tokens": prompt_tokens,
                                                "completion_tokens": new_tokens,
                                                "total_tokens": total
                                            }
                                        });
                                        return Ok(json_response(StatusCode::OK, &resp_value)
                                            .map(box_body));
                                    }
                                }
                                Err(e) => {
                                    warn!(error = %e, chat_id = %session_id, "Tool error recovery failed, falling back to error response");
                                }
                            }
                        } else {
                            warn!(chat_id = %session_id, "No latest_pid for tool error recovery; returning error");
                        }
                        let mut err = serde_json::json!({
                            "message": err_msg,
                            "type": "invalid_request_error",
                        });
                        err["available_tools"] = serde_json::json!(tools);
                        return Ok(json_response(
                            StatusCode::BAD_REQUEST,
                            &serde_json::json!({"error": err}),
                        )
                        .map(box_body));
                    }
                }

                let raw_tcs = detect_tools(&full_text, &tools);
                let tcs = match tool_gate_nonstream(
                    raw_tcs,
                    &tools,
                    &session_id,
                    latest_pid.as_deref(),
                    &st.token,
                    &session,
                )
                .await
                {
                    Ok(good) => good,
                    Err(bad_names) => {
                        return Ok(handle_blocked_tools_nonstream(
                            &bad_names,
                            &tools,
                            &session_id,
                            latest_pid.as_deref(),
                            &st.token,
                            &session,
                            "non-stream",
                        )
                        .await
                        .map(box_body));
                    }
                };

                if !tcs.is_empty() {
                    info!(
                        client_tool_count = tools.len(),
                        emitted_tool_count = tcs.len(),
                        tool_names = ?tcs.iter().map(|t| &t.name).collect::<Vec<_>>(),
                        "Non-stream final: emitting validated tool calls to client (Phase 2 hard gate passed)"
                    );
                }

                let resp_value = if !tcs.is_empty() {
                    let tool_calls: Vec<serde_json::Value> = tcs
                        .iter()
                        .enumerate()
                        .map(|(i, tc)| {
                            info!(tool = %tc.name, index = i, "Detected tool call");
                            let tool_call_id = format!("call_{}", uuid::Uuid::new_v4());
                            let args = serde_json::to_string(&tc.args)
                                .unwrap_or_else(|_| "{}".to_string());
                            serde_json::json!({
                                "index": i,
                                "id": tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": args
                                }
                            })
                        })
                        .collect();

                    if is_responses_api {
                        let output: Vec<serde_json::Value> = tcs.iter().map(|tc| {
                            serde_json::json!({
                                "type": "function_call",
                                "id": format!("call_{}", uuid::Uuid::new_v4()),
                                "call_id": format!("call_{}", uuid::Uuid::new_v4()),
                                "name": tc.name,
                                "arguments": serde_json::to_string(&tc.args).unwrap_or_else(|_| "{}".to_string())
                            })
                        }).collect();
                        serde_json::json!({
                            "id": completion_id,
                            "object": "response",
                            "created_at": created,
                            "model": model,
                            "output": output,
                            "usage": {
                                "input_tokens": prompt_tokens,
                                "output_tokens": completion_tokens,
                                "total_tokens": total_tokens
                            }
                        })
                    } else {
                        serde_json::json!({
                            "id": completion_id,
                            "object": "chat.completion",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": null,
                                    "tool_calls": tool_calls
                                },
                                "finish_reason": "tool_calls"
                            }],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": total_tokens
                            }
                        })
                    }
                } else {
                    let visible = client_visible_content(&full_text, None, !tools.is_empty());
                    let visible =
                        match process_structured_output(&visible, response_format.as_ref()) {
                            Ok(v) => v,
                            Err(e) => {
                                error!(error = %e, "Structured output processing failed");
                                return Ok(openai_error_response(
                                    StatusCode::UNPROCESSABLE_ENTITY,
                                    e,
                                    "invalid_response_error",
                                    None,
                                    None,
                                )
                                .map(box_body));
                            }
                        };
                    info!(len = visible.len(), "Returning text response");

                    if is_responses_api {
                        serde_json::json!({
                            "id": completion_id,
                            "object": "response",
                            "created_at": created,
                            "model": model,
                            "output": [{
                                "id": format!("msg_{}", uuid::Uuid::new_v4()),
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": visible, "annotations": []}]
                            }],
                            "usage": {
                                "input_tokens": prompt_tokens,
                                "output_tokens": completion_tokens,
                                "total_tokens": total_tokens
                            }
                        })
                    } else {
                        serde_json::json!({
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
                        })
                    }
                };

                Ok(json_response(StatusCode::OK, &resp_value).map(box_body))
            }
            Err(e) => {
                error!(error = %e, "Qwen API call failed (unblock)");
                Ok(internal_error(format!("Qwen API error: {}", e)).map(box_body))
            }
        }
    }
}

async fn embeddings_handler(req: Request<Incoming>) -> Result<Response<BoxBody>, Infallible> {
    let body_bytes = match req.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => {
            return Ok(bad_request(format!("Failed to read body: {}", e)).map(box_body));
        }
    };

    let v: serde_json::Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => {
            return Ok(bad_request(format!("Invalid JSON: {}", e)).map(box_body));
        }
    };

    let input = v.get("input").and_then(|i| i.as_str()).unwrap_or("");
    let dims = v.get("dimensions").and_then(|d| d.as_u64()).unwrap_or(1536) as usize;
    let embedding: Vec<f64> = vec![0.0; dims];
    let tokens = std::cmp::max(1, input.len() / 4);
    let model = request_model(&v);

    Ok(json_response(
        StatusCode::OK,
        &serde_json::json!({
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
        }),
    )
    .map(box_body))
}

async fn router(
    req: Request<Incoming>,
    st: Arc<AppState>,
) -> Result<Response<BoxBody>, Infallible> {
    let path = req.uri().path().to_string();
    let method = req.method().clone();

    let cors = |resp: Response<BoxBody>| {
        let (mut parts, body) = resp.into_parts();
        parts.headers.insert(
            "access-control-allow-origin",
            http::HeaderValue::from_static("*"),
        );
        parts.headers.insert(
            "access-control-allow-methods",
            http::HeaderValue::from_static("*"),
        );
        parts.headers.insert(
            "access-control-allow-headers",
            http::HeaderValue::from_static("*"),
        );
        Response::from_parts(parts, body)
    };

    if method == Method::OPTIONS {
        return Ok(cors(
            Response::builder()
                .status(204)
                .body(box_body(http_body_util::Empty::<Bytes>::new()))
                .unwrap(),
        ));
    }

    let resp = match (method.clone(), path.as_str()) {
        (Method::GET, "/health") => health_handler().map(box_body),
        (Method::GET, "/v1/models") => models_handler().map(box_body),
        (Method::GET, p) if p.starts_with("/v1/models/") => {
            let model_id = p.trim_start_matches("/v1/models/");
            model_handler(model_id).map(box_body)
        }
        (Method::POST, "/v1/chat/completions") | (Method::POST, "/v1/responses") => {
            handler(req, st).await.unwrap()
        }
        (Method::POST, "/v1/embeddings") => embeddings_handler(req).await.unwrap(),
        (Method::GET, "/") | (Method::GET, "") => json_response(
            StatusCode::OK,
            &serde_json::json!({"message": "Qwen OpenAI Proxy (smol+hyper)", "version": "0.1.0"}),
        )
        .map(box_body),
        _ => {
            if method == Method::POST {
                handler(req, st).await.unwrap()
            } else {
                not_found_response().map(box_body)
            }
        }
    };

    Ok(cors(resp))
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "qwen_proxy=info".into()),
        )
        .init();

    let token = load_token()?;

    let http = streaming::build_http_client()
        .map_err(|e| anyhow::anyhow!("Failed to build HTTP client: {}", e))?;
    let state = Arc::new(AppState {
        sessions: SessionManager::new(),
        token,
        http,
    });

    // fix(#5): read PORT once at startup rather than on every request.
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8765);
    let addr = format!("0.0.0.0:{}", port);

    info!(
        "Qwen OpenAI proxy (tokio+hyper) listening on http://{}",
        addr
    );

    let term = Arc::new(AtomicBool::new(false));
    if let Err(e) = signal_hook::flag::register(signal_hook::consts::SIGINT, Arc::clone(&term)) {
        error!(error = %e, "Failed to register SIGINT handler");
    }
    if let Err(e) = signal_hook::flag::register(signal_hook::consts::SIGTERM, Arc::clone(&term)) {
        error!(error = %e, "Failed to register SIGTERM handler");
    }

    // Tokio runtime (replaces previous smol::block_on)
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to build Tokio runtime");

    rt.block_on(async {
        let listener = TokioTcpListener::bind(&addr)
            .await
            .expect("Failed to bind address");

        loop {
            use futures::future::Either;
            let accept_fut = Box::pin(listener.accept());
            let check_fut = Box::pin(async {
                loop {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    if term.load(Ordering::Relaxed) {
                        break;
                    }
                }
            });
            match futures::future::select(accept_fut, check_fut).await {
                Either::Left((Ok((stream, peer)), _)) => {
                    let state = state.clone();
                    tokio::spawn(async move {
                        let stream = TokioIo::new(stream);
                        let service =
                            service_fn(move |req: Request<Incoming>| router(req, state.clone()));

                        if let Err(e) = http1::Builder::new()
                            .keep_alive(true)
                            .serve_connection(stream, service)
                            .await
                        {
                            debug!("Connection error from {}: {}", peer, e);
                        }
                    });
                    // No .detach() needed — Tokio tasks are fire-and-forget by default
                }
                Either::Left((Err(e), _)) => {
                    error!("Accept error: {}", e);
                    continue;
                }
                Either::Right(((), _)) => {
                    info!("Shutdown signal received, draining connections...");
                    break;
                }
            }
        }

        info!("Shutdown complete");
    });
    Ok(())
}
