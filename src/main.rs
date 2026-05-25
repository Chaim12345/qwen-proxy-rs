mod qwen;
mod session;

use anyhow::{bail, Context, Result};

use http::{Method, StatusCode};
use http_body::Frame;
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::body::{Bytes, Incoming};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response};
use qwen::*;
use session::SessionManager;
use smol_hyper;
use std::collections::hash_map::DefaultHasher;
use std::convert::Infallible;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use tracing::{debug, error, info};

const MODEL_NAME: &str = "qwen3.7-max";
const QWEN_API_BASE: &str = "https://chat.qwen.ai/api/v2";

struct AppState {
    sessions: SessionManager,
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

fn json_response<T: serde::Serialize>(status: StatusCode, body: &T) -> Response<Full<Bytes>> {
    let json = serde_json::to_string(body).unwrap_or_default();
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
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
{"id":"qwen3.6-plus","object":"model","created":1700000000,"owned_by":"qwen","permission":[],"root":"qwen3.6-plus","parent":null},
{"id":"qwen3.6-max-preview","object":"model","created":1700000000,"owned_by":"qwen","permission":[],"root":"qwen3.6-max-preview","parent":null}
]}"#;

fn models_handler() -> Response<Full<Bytes>> {
    json_response(
        StatusCode::OK,
        &serde_json::from_str::<serde_json::Value>(MODELS_JSON).unwrap(),
    )
}

fn model_handler(model_id: &str) -> Response<Full<Bytes>> {
    if model_id == MODEL_NAME
        || model_id == "qwen3.6-plus"
        || model_id == "qwen3.6-max-preview"
        || model_id == "gpt-4"
        || model_id == "gpt-4o"
        || model_id == "gpt-3.5-turbo"
    {
        let resolved_id = if model_id == "gpt-4" || model_id == "gpt-4o" || model_id == "gpt-3.5-turbo"
        {
            MODEL_NAME
        } else {
            model_id
        };
        json_response(StatusCode::OK, &model_info(resolved_id))
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
        id, model, created,
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

fn sse_line_to_openai(
    line: &str,
    completion_id: &str,
    model: &str,
    created: i64,
    _has_tools: bool,
    full_text: &mut AccumulatedText,
) -> Option<String> {
    let data = parse_qwen_sse_line(line)?;
    if data == "[DONE]" {
        return None;
    }
    let ch: serde_json::Value = serde_json::from_str(&data).ok()?;
    if extract_response_parent_id(&ch).is_some() {
        return None;
    }
    let delta = extract_qwen_sse_delta(&ch)?;
    full_text.append(&delta);
    if delta.phase == QwenPhase::ThinkingSummary || delta.phase == QwenPhase::Thinking {
        return None;
    }
    let text = delta.text;
    if text.is_empty() && !delta.finished {
        return None;
    }
    Some(build_stream_chunk(
        completion_id, model, created,
        serde_json::json!({"content": text}),
        if delta.finished { Some("stop") } else { None },
    ))
}

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

fn parse_qwen_sse_line(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed == "[DONE]" {
        return None;
    }
    if trimmed.starts_with("data: ") {
        Some(trimmed[6..].trim().to_string())
    } else {
        None
    }
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
    body.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{}", e)))
        .boxed_unsync()
}

async fn handler(
    req: Request<Incoming>,
    st: Arc<AppState>,
) -> Result<Response<BoxBody>, Infallible> {
    let body_bytes = match req.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => {
            return Ok(bad_request(format!("Failed to read body: {}", e)).map(|b| box_body(b)));
        }
    };

    let json_bytes = body_bytes.to_vec();

    let v: serde_json::Value = match serde_json::from_slice(&json_bytes) {
        Ok(v) => v,
        Err(e) => {
            return Ok(bad_request(format!("Invalid JSON: {}", e)).map(|b| box_body(b)));
        }
    };

    let messages = if let Some(msgs) = v.get("messages").and_then(|m| m.as_array()) {
        msgs
    } else if let Some(input) = v.get("input").and_then(|m| m.as_array()) {
        input
    } else {
        return Ok(bad_request("messages or input array is required").map(|b| box_body(b)));
    };

    if messages.is_empty() {
        return Ok(bad_request("messages array cannot be empty").map(|b| box_body(b)));
    }

    let is_responses_api = v.get("input").is_some() && v.get("messages").is_none();
    let is_stream = v.get("stream").and_then(|x| x.as_bool()).unwrap_or(false);
    let msg = build_message(&v);
    let tools = parse_tools(&v);

    debug!(messages = messages.len(), stream = is_stream, "Processing request");

    let client_key = client_session_key(&v);
    let (session, session_id, parent_id, parent_store) = match st.sessions.acquire(&client_key, &st.token).await {
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
            return Ok(openai_error_response(status, e.to_string(), "server_error", None, None).map(|b| box_body(b)));
        }
    };

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

    let qwen_url = format!("{}/chat/completions?chat_id={}", QWEN_API_BASE, session_id);
    let headers = qwen_request_headers(&st.token);

    let completion_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();
    let prompt_tokens = estimate_tokens(&prompt);
    let model = request_model(&v);

    let body_bytes: Vec<u8> = serde_json::to_vec(&payload).unwrap_or_default();

    if is_stream {
        let (tx, rx) = smol::channel::bounded::<Result<String, String>>(256);

        let tx2 = tx.clone();
        let headers2 = headers.clone();
        let qwen_url2 = qwen_url.clone();
        let body2 = body_bytes.clone();
        smol::spawn(async move {
            let result = smol::unblock(move || -> Result<(), String> {
                let mut resp = ureq::post(&qwen_url2);
                for (k, v) in &headers2 {
                    resp = resp.set(k, v);
                }
                let response = resp
                    .send_bytes(&body2)
                    .map_err(|e| format!("Qwen API error: {}", e))?;

                if !(200..300).contains(&response.status()) {
                    let status = response.status();
                    let body_text = response.into_string().unwrap_or_default();
                    let preview: String = body_text.chars().take(500).collect();
                    error!(status = %status, body_preview = %preview, "Qwen chat/completions returned error");
                    if status == 429 || body_text.contains("in progress") {
                        return Err("Qwen chat is busy (another message in flight on this chat_id)".to_string());
                    }
                    return Err(format!("Qwen API returned {}", status));
                }

                use std::io::Read;
                let mut reader = response.into_reader();
                let mut buf = vec![0u8; 4096];
                loop {
                    match reader.read(&mut buf) {
                        Ok(0) => break,
                        Ok(n) => {
                            if tx2.try_send(Ok(
                                String::from_utf8_lossy(&buf[..n]).to_string()
                            )).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            let _ = tx2.try_send(Err(format!("Stream read error: {}", e)));
                            break;
                        }
                    }
                }
                Ok(())
            }).await;
            if let Err(e) = result {
                let _ = tx.try_send(Err(e));
            }
            drop(tx);
        }).detach();

        let has_tools = !tools.is_empty();
        let sse_stream = futures::stream::unfold(
            (rx, String::new(), AccumulatedText::new(), false),
            move |(rx, mut buf, mut full_text, done)| {
                let parent_store = parent_store.clone();
                let tools = tools.clone();
                let completion_id = completion_id.clone();
                let model = model.clone();
                async move {
                    if done {
                        return None;
                    }
                    match rx.recv().await {
                        Ok(Ok(chunk)) => {
                            buf.push_str(&chunk);
                            let mut oai_chunks = Vec::new();
                            loop {
                                let nl = match buf.find('\n') {
                                    Some(p) => p,
                                    None => break,
                                };
                                let line = buf[..nl].to_string();
                                buf = buf[nl + 1..].to_string();
                                let data = match parse_qwen_sse_line(&line) {
                                    Some(d) => d,
                                    None => continue,
                                };
                                if data == "[DONE]" {
                                    continue;
                                }
                                if let Ok(ch) = serde_json::from_str::<serde_json::Value>(&data) {
                                    if let Some(pid) = extract_response_parent_id(&ch) {
                                        *parent_store.lock().await = Some(pid);
                                    }
                                    if let Some(delta) = extract_qwen_sse_delta(&ch) {
                                        full_text.append(&delta);
                                            if matches!(delta.phase, QwenPhase::Answer | QwenPhase::Other(_)) {
                                            if !delta.text.is_empty() {
                                                oai_chunks.push(build_stream_chunk(
                                                    &completion_id, &model, created,
                                                    serde_json::json!({"content": &delta.text}),
                                                    None,
                                                ));
                                            }
                                            if delta.finished {
                                                oai_chunks.push(build_stream_chunk(
                                                    &completion_id, &model, created,
                                                    serde_json::json!({}),
                                                    Some("stop"),
                                                ));
                                            }
                                        }
                                    }
                                }
                                let tc = detect_tool(full_text.full_answer(), &tools);
                                if tc.is_some() {
                                    break;
                                }
                            }
                            let tc = detect_tool(full_text.full_answer(), &tools);
                            if let Some(tc) = tc {
                                info!(tool = %tc.name, "Detected tool call");
                                let tid = format!("call_{}", uuid::Uuid::new_v4());
                                let args = serde_json::to_string(&tc.args).unwrap_or_else(|_| "{}".into());
                                for s in build_tool_call_stream_chunks(&completion_id, &model, created, &tid, &tc.name, &args) {
                                    oai_chunks.push(s);
                                }
                            }
                            if !oai_chunks.is_empty() {
                                let out = oai_chunks.iter().map(|s| format!("data: {}\n\n", s)).collect::<String>();
                                Some((Ok::<_, std::io::Error>(Frame::data(Bytes::from(out))), (rx, buf, full_text, false)))
                            } else {
                                Some((Ok::<_, std::io::Error>(Frame::data(Bytes::from(""))), (rx, buf, full_text, false)))
                            }
                        }
                        Ok(Err(err_str)) => {
                            let err_chunk = format!(
                                "data: {}\n\ndata: [DONE]\n\n",
                                build_stream_chunk(&completion_id, &model, created,
                                    serde_json::json!({"content": format!("[Error: {}]", err_str)}),
                                    Some("stop"),
                                )
                            );
                            Some((Ok::<_, std::io::Error>(Frame::data(Bytes::from(err_chunk))), (rx, buf, full_text, true)))
                        }
                        Err(_) => {
                            let mut final_chunks = String::new();
                            let answer = full_text.full_answer().to_string();
                            let already_emitted = !answer.is_empty();
                            let tc = detect_tool(&answer, &tools);
                            if let Some(tc) = tc {
                                info!(tool = %tc.name, "Detected tool call");
                                let tid = format!("call_{}", uuid::Uuid::new_v4());
                                let args = serde_json::to_string(&tc.args).unwrap_or_else(|_| "{}".into());
                                for s in build_tool_call_stream_chunks(&completion_id, &model, created, &tid, &tc.name, &args) {
                                    final_chunks.push_str(&format!("data: {}\n\n", s));
                                }
                            } else if !already_emitted {
                                let visible = client_visible_content(&answer, None, has_tools);
                                final_chunks.push_str(&format!(
                                    "data: {}\n\n",
                                    build_stream_chunk(&completion_id, &model, created,
                                        serde_json::json!({"role": "assistant", "content": visible}),
                                        Some("stop"),
                                    )
                                ));
                            }
                            final_chunks.push_str("data: [DONE]\n\n");
                            Some((Ok::<_, std::io::Error>(Frame::data(Bytes::from(final_chunks))), (rx, buf, full_text, true)))
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
    } else {
        match smol::unblock(move || -> Result<String> {
            let mut resp = ureq::post(&qwen_url);
            for (k, v) in &headers {
                resp = resp.set(k, v);
            }
            let response = resp
                .send_bytes(&body_bytes)
                .map_err(|e| anyhow::anyhow!("Qwen API error: {}", e))?;

            if !(200..300).contains(&response.status()) {
                let status = response.status();
                let body_text = response.into_string().unwrap_or_default();
                let preview: String = body_text.chars().take(500).collect();
                error!(status = %status, body_preview = %preview, "Qwen chat/completions returned error");
                if status == 429 || body_text.contains("in progress") {
                    bail!("Qwen chat is busy (another message in flight on this chat_id)");
                }
                bail!("Qwen API returned {}", status);
            }

            let body_text = response.into_string().map_err(|e| anyhow::anyhow!("Failed to read response: {}", e))?;
            Ok(body_text)
        }).await {
            Ok(body_text) => {
                let mut acc = AccumulatedText::new();
                for line in body_text.lines() {
                    if let Some(data) = parse_qwen_sse_line(line) {
                        if data == "[DONE]" { continue; }
                        if let Ok(ch) = serde_json::from_str::<serde_json::Value>(&data) {
                            if let Some(pid) = extract_response_parent_id(&ch) {
                                session.set_parent_id(pid).await;
                            }
                            append_sse_delta(&mut acc, &ch);
                        }
                    }
                }

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
                        ).map(|b| box_body(b)));
                    }
                }

                let tc = detect_tool(&full_text, &tools);

                let resp_value = if let Some(tc) = tc {
                    info!(tool = %tc.name, "Detected tool call");
                    let tool_call_id = format!("call_{}", uuid::Uuid::new_v4());
                    let args = serde_json::to_string(&tc.args).unwrap_or_else(|_| "{}".to_string());

                    if is_responses_api {
                        serde_json::json!({
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
                        })
                    }
                } else {
                    let visible = client_visible_content(&full_text, None, !tools.is_empty());
                    info!(len = visible.len(), "Returning text response");

                    if is_responses_api {
                        serde_json::json!({
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

                Ok(json_response(StatusCode::OK, &resp_value).map(|b| box_body(b)))
            }
            Err(e) => {
                error!(error = %e, "Qwen API call failed");
                Ok(internal_error(format!("Qwen API error: {}", e)).map(|b| box_body(b)))
            }
        }
    }
}

async fn embeddings_handler(
    req: Request<Incoming>,
) -> Result<Response<BoxBody>, Infallible> {
    let body_bytes = match req.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => {
            return Ok(bad_request(format!("Failed to read body: {}", e)).map(|b| box_body(b)));
        }
    };

    let v: serde_json::Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => {
            return Ok(bad_request(format!("Invalid JSON: {}", e)).map(|b| box_body(b)));
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
    ).map(|b| box_body(b)))
}

async fn router(
    req: Request<Incoming>,
    st: Arc<AppState>,
) -> Result<Response<BoxBody>, Infallible> {
    let path = req.uri().path().to_string();
    let method = req.method().clone();

    let cors = |resp: Response<BoxBody>| {
        let (mut parts, body) = resp.into_parts();
        parts.headers.insert("access-control-allow-origin", http::HeaderValue::from_static("*"));
        parts.headers.insert("access-control-allow-methods", http::HeaderValue::from_static("*"));
        parts.headers.insert("access-control-allow-headers", http::HeaderValue::from_static("*"));
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
        (Method::GET, "/health") => health_handler().map(|b| box_body(b)),
        (Method::GET, "/v1/models") => models_handler().map(|b| box_body(b)),
        (Method::GET, p) if p.starts_with("/v1/models/") => {
            let model_id = p.trim_start_matches("/v1/models/");
            model_handler(model_id).map(|b| box_body(b))
        }
        (Method::POST, "/v1/chat/completions") | (Method::POST, "/v1/responses") => {
            handler(req, st).await.unwrap()
        }
        (Method::POST, "/v1/embeddings") => {
            embeddings_handler(req).await.unwrap()
        }
        (Method::GET, "/") | (Method::GET, "") => {
            json_response(
                StatusCode::OK,
                &serde_json::json!({"message": "Qwen OpenAI Proxy (smol+hyper)", "version": "0.1.0"}),
            ).map(|b| box_body(b))
        }
        _ => {
            if method == Method::POST {
                handler(req, st).await.unwrap()
            } else {
                not_found_response().map(|b| box_body(b))
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

    let state = Arc::new(AppState {
        sessions: SessionManager::new(),
        token,
    });

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8765);
    let addr = format!("0.0.0.0:{}", port);

    info!("Qwen OpenAI proxy (smol+hyper) listening on http://{}", addr);

    smol::block_on(async {
        let listener = smol::net::TcpListener::bind(&addr)
            .await
            .expect("Failed to bind address");

        loop {
            let (stream, peer) = match listener.accept().await {
                Ok(conn) => conn,
                Err(e) => {
                    error!("Accept error: {}", e);
                    continue;
                }
            };

            let state = state.clone();

            smol::spawn(async move {
                let stream = smol_hyper::rt::FuturesIo::new(stream);
                let service = service_fn(move |req: Request<Incoming>| {
                    router(req, state.clone())
                });

                if let Err(e) = http1::Builder::new()
                    .keep_alive(true)
                    .serve_connection(stream, service)
                    .await
                {
                    debug!("Connection error from {}: {}", peer, e);
                }
            })
            .detach();
        }
    })
}
