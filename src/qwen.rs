use regex::Regex;
use serde_json::Value;
use std::sync::LazyLock;
use tracing::{debug, error, info, warn};

static MARKDOWN_CODE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"```(?:json)?\s*([\s\S]+?)```").unwrap());

/// Spinner / Braille patterns leaked into streamed Qwen output.
static SPINNER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[\u280B-\u283F]|⠋|⠙|⠹|⠸|⠼|⠴|⠦|⠧|⠇|⠏").unwrap()
});

static THINKING_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)Thinking\.{0,3}").unwrap());

/// Strip spinner/thinking artifacts before tool-call parsing.
pub fn normalize_tool_call_text(text: &str) -> String {
    let step1 = SPINNER_RE.replace_all(text, "");
    THINKING_RE.replace_all(&step1, "").into_owned()
}

/// Layer 4 (Phase 4.2): post-parsing tool name normalization for agent robustness.
/// Lowercases + strips common prefixes/suffixes that Cursor/Aider/etc sometimes invent
/// (e.g. "get_terminal_output", "bash_run", "Cursor_foo_tool" -> "terminal_output", "run", "foo").
/// Exact client-provided name is always preferred and emitted (preserves casing).
/// Conservative list only (no fuzzy); false-positive risk mitigated by exact-first + tests.
pub fn normalize_tool_name(name: &str) -> String {
    let mut n = name.trim().to_lowercase();
    for p in ["get_", "run_", "bash_", "execute_", "tool_", "cursor_", "api_"] {
        if let Some(s) = n.strip_prefix(p) {
            n = s.to_string();
            break;
        }
    }
    for s in ["_tool", "_cmd", "_function", "_op"] {
        if let Some(p) = n.strip_suffix(s) {
            n = p.to_string();
            break;
        }
    }
    // conservative cleanup for double-underscore artifacts like "api__call"
    n.trim_start_matches('_').trim_end_matches('_').to_string()
}

/// Shared matcher for accept + validate (Phase 4.2).
/// exact match first (preserve client casing for emission), then norm-match.
/// Returns Some(canonical_name_to_emit) or None (unknown/halluc).
/// When norm match used, caller can log; the returned name is the one from the allowed list.
fn is_tool_name_allowed(requested: &str, allowed: &[&str]) -> Option<String> {
    if allowed.is_empty() {
        return Some(requested.to_string());
    }
    // exact (case-sensitive, per client list)
    if allowed.contains(&requested) {
        return Some(requested.to_string());
    }
    let nreq = normalize_tool_name(requested);
    for &a in allowed {
        if normalize_tool_name(a) == nreq {
            info!(
                tool_requested = %requested,
                canonical_name = %a,
                normalized_match = true,
                "Layer 4 norm match: accepting prefixed/suffixed name, will emit canonical from client list"
            );
            return Some(a.to_string());
        }
    }
    None
}

fn sanitize_tool_args(args: &Value) -> Value {
    let Some(obj) = args.as_object() else { return args.clone(); };
    let mut out = serde_json::Map::new();
    for (key, value) in obj {
        let clean = key.trim();
        if !clean.is_empty() {
            out.insert(clean.to_string(), value.clone());
        }
    }
    Value::Object(out)
}

/// True when text still looks like a tool call after normalization (even if parse failed).
pub fn looks_like_tool_call_attempt(text: &str) -> bool {
    let cleaned = normalize_tool_call_text(text);
    cleaned.contains("```")
        || cleaned.contains("\"tool\":")
        || cleaned.contains("{\"tool\"")
}

/// Text safe to return as assistant `content` to OpenAI clients (never leak raw tool JSON).
pub fn client_visible_content(
    full_text: &str,
    tool_call: Option<&ToolCall>,
    tools_present: bool,
) -> String {
    if tool_call.is_some() {
        return String::new();
    }
    if tools_present && looks_like_tool_call_attempt(full_text) {
        return String::new();
    }
    full_text.to_string()
}

fn extract_string_value(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::String(s) => s.to_string(),
        Value::Array(arr) => arr
            .iter()
            .filter_map(|x| x.as_str())
            .collect::<Vec<_>>()
            .join(""),
        Value::Object(obj) => {
            if let Some(s) = obj.get("content").and_then(|c| c.as_str()) {
                return s.to_string();
            }
            if let Some(arr) = obj.get("content").and_then(|c| c.as_array()) {
                return arr
                    .iter()
                    .filter_map(|x| x.as_str())
                    .collect::<Vec<_>>()
                    .join("");
            }
            String::new()
        }
        _ => String::new(),
    }
}

/// Extract assistant-visible text from one Qwen SSE JSON chunk (mirrors qtalt proxies).
pub fn parse_qwen_upstream_error(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if !trimmed.starts_with('{') { return None; }
    let v: Value = serde_json::from_str(trimmed).ok()?;
    if v.get("success").and_then(|s| s.as_bool()) != Some(false) {
        return None;
    }
    let msg = v["data"]["details"]
        .as_str()
        .or_else(|| v["data"]["template"].as_str())
        .or_else(|| v["message"].as_str())
        .unwrap_or("Qwen API error");
    Some(msg.to_string())
}

/// Strip markdown code block fences from JSON text if present.
pub fn strip_json_codeblock(text: &str) -> String {
    let trimmed = text.trim();
    if let Some(caps) = MARKDOWN_CODE_RE.captures(trimmed) {
        if let Some(inner) = caps.get(1) {
            return inner.as_str().trim().to_string();
        }
    }
    trimmed.to_string()
}

/// Process response content for structured output (response_format).
pub fn process_structured_output(text: &str, rf: Option<&Value>) -> Result<String, String> {
    let rf = match rf { Some(r) => r, None => return Ok(text.to_string()) };
    let rf_type = rf.get("type").and_then(|t| t.as_str());
    if rf_type != Some("json_schema") && rf_type != Some("json_object") {
        return Ok(text.to_string());
    }
    let cleaned = strip_json_codeblock(text);
    if serde_json::from_str::<Value>(&cleaned).is_err() {
        return Err(format!(
            "Response is not valid JSON despite response_format. Got: {}",
            cleaned.chars().take(200).collect::<String>()
        ));
    }
    Ok(cleaned)
}

/// Detect tool-related error messages in Qwen response text.
pub fn detect_qwen_tool_error(text: &str) -> Option<String> {
    if text.is_empty() { return None; }
    if text.contains("Tool ") && (text.contains(" does not exist") || text.contains(" does not exists")) {
        let end = text.find('.').unwrap_or(text.len());
        return Some(text[..end].to_string());
    }
    if (text.contains("cannot use") || text.contains("can't use") || text.contains("unable to use")) && text.contains("tool") {
        let end = text.find('.').unwrap_or(text.len().min(200));
        return Some(text[..end].to_string());
    }
    if text.contains("tool not found") || text.contains("tool_not_found") {
        let end = text.find('.').unwrap_or(text.len().min(200));
        return Some(text[..end].to_string());
    }
    if text.len() < 100 && (text.contains("抱歉") || (text.contains("sorry") && text.contains("tool"))) {
        let end = text.find('.').unwrap_or(text.len().min(200));
        return Some(text[..end].to_string());
    }
    None
}

#[derive(Debug, Clone, PartialEq)]
pub enum QwenPhase {
    ThinkingSummary,
    Thinking,
    Search,
    Answer,
    Other(String),
}

pub struct QwenSseDelta {
    pub phase: QwenPhase,
    pub text: String,
    pub finished: bool,
}

pub struct AccumulatedText {
    pub thinking: String,
    pub answer: String,
}

impl AccumulatedText {
    pub fn new() -> Self {
        AccumulatedText { thinking: String::new(), answer: String::new() }
    }

    pub fn append(&mut self, delta: &QwenSseDelta) {
        match delta.phase {
            QwenPhase::ThinkingSummary | QwenPhase::Thinking => {
                if !delta.text.is_empty() {
                    if !self.thinking.is_empty() {
                        self.thinking.push('\n');
                    }
                    self.thinking.push_str(&delta.text);
                }
            }
            QwenPhase::Answer | QwenPhase::Other(_) => {
                self.answer.push_str(&delta.text);
            }
            QwenPhase::Search => {}
        }
    }

    pub fn full_answer(&self) -> &str { &self.answer }
    pub fn thinking(&self) -> &str { &self.thinking }
}

/// Extract delta text from a Qwen SSE JSON chunk.
/// CRITICAL FIX: properly extract thinking content from `extra.summary_thought`
/// and `extra.summary_title` fields, not just `delta.content`.
pub fn extract_qwen_sse_delta(ch: &Value) -> Option<QwenSseDelta> {
    let delta = ch
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|c| c.first())
        .and_then(|c| c.get("delta"));

    let phase_str = delta
        .as_ref()
        .and_then(|d| d.get("phase"))
        .and_then(|p| p.as_str())
        .or_else(|| ch.get("phase").and_then(|p| p.as_str()))
        .unwrap_or("");

    let phase = match phase_str {
        "thinking_summary" => QwenPhase::ThinkingSummary,
        "thinking" => QwenPhase::Thinking,
        "search" => QwenPhase::Search,
        "answer" => QwenPhase::Answer,
        other => if other.is_empty() { QwenPhase::Answer } else { QwenPhase::Other(other.to_string()) },
    };

    let finished = delta
        .as_ref()
        .and_then(|d| d.get("status"))
        .and_then(|s| s.as_str())
        .map(|s| s == "finished")
        .unwrap_or(false);

    let mut text = String::new();

    // ── THINKING PHASE: pull from extra.summary_thought / extra.summary_title ──
    if phase == QwenPhase::ThinkingSummary || phase == QwenPhase::Thinking {
        if let Some(extra) = delta.as_ref().and_then(|d| d.get("extra")) {
            if let Some(title) = extra.get("summary_title").and_then(|s| s.get("content")) {
                if let Some(arr) = title.as_array() {
                    for item in arr { if let Some(s) = item.as_str() { text.push_str(s); } }
                } else if let Some(s) = title.as_str() { text.push_str(s); }
            }
            if let Some(thought) = extra.get("summary_thought").and_then(|s| s.get("content")) {
                if let Some(arr) = thought.as_array() {
                    for item in arr {
                        if let Some(s) = item.as_str() {
                            if !text.is_empty() { text.push('\n'); }
                            text.push_str(s);
                        }
                    }
                } else if let Some(s) = thought.as_str() {
                    if !text.is_empty() { text.push('\n'); }
                    text.push_str(s);
                }
            }
        }
        // Fallback: some Qwen versions put thinking text directly in delta.content as array
        if text.is_empty() {
            text = extract_string_value(delta.and_then(|d| d.get("content")).unwrap_or(&Value::Null));
        }
    } else {
        // Normal answer phase
        text = extract_string_value(delta.and_then(|d| d.get("content")).unwrap_or(&Value::Null));
        if text.is_empty() {
            text = ch
                .get("response")
                .map(|r| extract_string_value(r.get("content").unwrap_or(&Value::Null)))
                .unwrap_or_default();
        }
        if text.is_empty() {
            text = extract_string_value(ch.get("content").unwrap_or(&Value::Null));
        }
    }

    // Always return a delta for thinking phases so the stream can accumulate them
    if phase == QwenPhase::ThinkingSummary || phase == QwenPhase::Thinking {
        return Some(QwenSseDelta { phase, text, finished });
    }

    if text.is_empty() && !finished {
        None
    } else {
        Some(QwenSseDelta { phase, text, finished })
    }
}

fn extract_json_blocks(text: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut search_from = 0usize;
    while let Some(rel) = text[search_from..].find('{') {
        let start = search_from + rel;
        if let Some(end) = find_json_object_end(text, start) {
            blocks.push(text[start..=end].to_string());
            search_from = end + 1;
        } else {
            search_from = start + 1;
        }
    }
    blocks
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub args: serde_json::Value,
}

pub fn qwen_payload(chat_id: &str, parent_id: Option<&str>, prompt: &str) -> serde_json::Value {
    serde_json::json!({
        "stream": true,
        "version": "2.1",
        "incremental_output": true,
        "chat_id": chat_id,
        "chat_mode": "normal",
        "model": "qwen3.7-max",
        "parent_id": parent_id,
        "messages": [{
            "fid": chat_id,
            "parentId": parent_id,
            "role": "user",
            "content": prompt,
            "user_action": "chat",
            "files": [],
            "timestamp": chrono::Utc::now().timestamp_millis(),
            "models": ["qwen3.7-max"],
            "chat_type": "t2t",
            "feature_config": {
                "thinking_enabled": true,
                "auto_search": true,
                "research_mode": "normal",
                "output_schema": "phase",
                "auto_thinking": true,
                "thinking_mode": "Auto",
                "thinking_format": "summary"
            },
            "extra": { "meta": { "subChatType": "t2t" } },
            "sub_chat_type": "t2t"
        }],
        "timestamp": chrono::Utc::now().timestamp_millis()
    })
}

/// Phase 3 Feedback/Recovery helper.
/// Sends a synthetic "TOOL RESULT: ERROR..." (or similar) user message as a continuation
/// in the *current* Qwen chat thread (using parent_id), then returns the new response_id
/// from Qwen's reply so we can set_parent_id and make the hallucination + correction
/// visible in-context for subsequent turns on the same client_session_key.
/// Best-effort: returns None on any failure (network, 429, parse, etc.); never panics the request path.
/// Uses stream:true payload for compatibility with qwen_payload; parses SSE body for extract_response_parent_id.
/// 30s timeout. Consistent ureq + spawn_blocking pattern (see create_chat).
pub async fn send_qwen_chat_continuation(
    chat_id: &str,
    parent_id: Option<&str>,
    feedback_content: &str,
    token: &str,
) -> ::anyhow::Result<Option<String>> {
    let chat_id = chat_id.to_string();
    let parent = parent_id.map(|s| s.to_string());
    let feedback = feedback_content.to_string();
    let token = token.to_string();

    tokio::task::spawn_blocking(move || -> ::anyhow::Result<Option<String>> {
        info!(
            chat_id = %chat_id,
            has_parent = parent.is_some(),
            feedback_len = feedback.len(),
            "Phase 3: sending feedback continuation to Qwen chat for in-context correction"
        );

        // Reuse the exact payload builder (stream:true, parent wiring, fid, feature_config etc.)
        let payload = qwen_payload(&chat_id, parent.as_deref(), &feedback);
        let url = format!("{}/chat/completions?chat_id={}", crate::constants::QWEN_API_BASE, chat_id);

        let req = ureq::post(&url)
            .timeout(std::time::Duration::from_secs(30))
            .set("accept", "text/event-stream")
            .set("content-type", "application/json")
            .set("referer", "https://chat.qwen.ai/")
            .set("source", "web")
            .set("version", "0.8.0")
            .set("cookie", &format!("token={}", token));

        let resp = match req.send_json(&payload) {
            Ok(r) => r,
            Err(e) => {
                warn!(error = %e, chat_id = %chat_id, "Phase 3 feedback POST failed (best-effort, no injection)");
                return Ok(None);
            }
        };

        if resp.status() == 401 {
            warn!(chat_id = %chat_id, "Phase 3 feedback: Qwen token expired during send");
            return Ok(None);
        }
        if !(200..300).contains(&resp.status()) {
            warn!(status = resp.status(), chat_id = %chat_id, "Phase 3 feedback: non-2xx from Qwen");
            return Ok(None);
        }

        let body_text = resp.into_string().unwrap_or_default();

        // Parse SSE (or fallback json) for the new response_id / parent from Qwen's reply to our feedback.
        let mut new_pid: Option<String> = None;
        for line in body_text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed == "[DONE]" { continue; }
            let data = if let Some(rest) = trimmed.strip_prefix("data: ") {
                rest.trim()
            } else {
                trimmed
            };
            if data == "[DONE]" { continue; }
            if let Ok(ch) = serde_json::from_str::<Value>(data) {
                if let Some(pid) = extract_response_parent_id(&ch) {
                    new_pid = Some(pid);
                    break;
                }
            }
        }
        if new_pid.is_none() {
            // Fallback: whole body as json?
            if let Ok(v) = serde_json::from_str::<Value>(&body_text) {
                new_pid = extract_response_parent_id(&v);
            }
        }

        info!(
            chat_id = %chat_id,
            new_parent = ?new_pid,
            "Phase 3: feedback continuation sent; new parent_id captured for subsequent turns on this chat"
        );
        Ok(new_pid)
    })
    .await
    .map_err(|e| ::anyhow::anyhow!("spawn_blocking join error for feedback: {}", e))?
}

fn message_content_to_string(content: &Value) -> String {
    if let Some(text) = content.as_str() { return text.to_string(); }
    if let Some(parts) = content.as_array() {
        return parts
            .iter()
            .filter_map(|part| {
                part.get("text").and_then(|v| v.as_str())
                    .or_else(|| part.get("output_text").and_then(|v| v.as_str()))
                    .or_else(|| part.get("input_text").and_then(|v| v.as_str()))
                    .or_else(|| part.get("output").and_then(|v| v.as_str()))
            })
            .collect::<Vec<_>>()
            .join("\n");
    }
    String::new()
}

pub fn build_message(v: &Value) -> String {
    let mut parts = vec![];
    let has_tools = v
        .get("tools")
        .and_then(|t| t.as_array())
        .map(|a| !a.is_empty())
        .unwrap_or(false);

    let input_messages = v
        .get("messages")
        .and_then(|m| m.as_array())
        .or_else(|| v.get("input").and_then(|m| m.as_array()));

    if let Some(msgs) = input_messages {
        for msg in msgs {
            let role = msg["role"].as_str().unwrap_or("user").to_uppercase();

            if let Some(tcs) = msg["tool_calls"].as_array() {
                for tc in tcs {
                    let name = tc["function"]["name"].as_str().unwrap_or("?");
                    let args = tc["function"]["arguments"].as_str().unwrap_or("{}");
                    parts.push(format!("ASSISTANT used tool: {}({})", name, args));
                }
            }

            if msg["type"].as_str() == Some("function_call") {
                let name = msg["name"].as_str().unwrap_or("?");
                let args = msg["arguments"].as_str().unwrap_or("{}");
                parts.push(format!("ASSISTANT used tool: {}({})", name, args));
            }

            if msg["type"].as_str() == Some("function_call_output") {
                let tool_result = message_content_to_string(&msg["output"]);
                if !tool_result.is_empty() {
                    parts.push(format!("TOOL RESULT: {}", tool_result));
                }
                continue;
            }

            let content = message_content_to_string(&msg["content"]);
            if !content.is_empty() {
                if role == "SYSTEM" {
                    parts.push(content);
                } else {
                    parts.push(format!("{}: {}", role, content));
                }
            }

            if msg["role"] == "tool" {
                let tool_result = message_content_to_string(&msg["content"]);
                parts.push(format!("TOOL RESULT: {}", tool_result));
                continue;
            }
        }
    }

    let mut result = parts.join("\n\n");

    if let Some(rf) = v.get("response_format") {
        match rf.get("type").and_then(|t| t.as_str()) {
            Some("json_schema") => {
                result.push_str("\n\nRespond with valid JSON only (no markdown) matching this schema:\n");
                if let Some(schema) = rf.get("json_schema").and_then(|s| s.get("schema")) {
                    result.push_str(&schema.to_string());
                }
            }
            Some("json_object") => {
                result.push_str("\n\nRespond with valid JSON only, no markdown fences.\n");
            }
            _ => {}
        }
    }

    if has_tools {
        // Phase 1 Prompt Engineering Hardening (Robust Tool-Calling Translator)
        // Loudest rule first: ONLY use exact names from the client's list. Never invent.
        result.push_str("\n\n## Tool Use Format — CRITICAL RULES\n");
        result.push_str("You may **only** call tools whose exact `name` appears in the Available Tools list provided by the client. Never invent, guess, or hallucinate tool names. If no tool is needed, respond in normal text without any JSON.\n\n");

        // Extract one real example tool name from the request for a concrete few-shot style example
        let example_tool = v.get("tools")
            .and_then(|t| t.as_array())
            .and_then(|arr| arr.first())
            .and_then(|t| t.get("function").and_then(|f| f.get("name")).or_else(|| t.get("name")))
            .and_then(|n| n.as_str())
            .unwrap_or("example_tool");

        result.push_str("When you need to call a tool, output **only** a single complete fenced code block with language `json` as the very last thing in your response (nothing after it):\n");
        result.push_str(&format!("```json\n{{\"tool\":\"{}\",\"args\":{{...}}}}\n```\n", example_tool));
        result.push_str("Always wrap the tool JSON in a markdown code block. Never output raw JSON outside a code block.\n");
        result.push_str("The code block must contain nothing except the JSON object and must be the absolute last output you produce.\n");
        result.push_str("Use the exact tool name spelling from the client's list — no prefixes, no suffixes, no made-up variations.\n");
    }

    result
}

pub fn parse_tools(v: &Value) -> Vec<Value> {
    v.get("tools")
        .and_then(|t| t.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|t| {
                    let function = t.get("function");
                    let name = function
                        .and_then(|f| f.get("name"))
                        .and_then(|v| v.as_str())
                        .or_else(|| t.get("name").and_then(|v| v.as_str()))?;
                    let desc = function
                        .and_then(|f| f.get("description"))
                        .and_then(|v| v.as_str())
                        .or_else(|| t.get("description").and_then(|v| v.as_str()))
                        .unwrap_or("");
                    let params = function
                        .and_then(|f| f.get("parameters"))
                        .or_else(|| t.get("parameters"))
                        .or_else(|| t.get("input_schema"))
                        .unwrap_or(&Value::Null);
                    Some(serde_json::json!({
                        "name": name,
                        "description": desc,
                        "parameters": params
                    }))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Parse a JSON object from text starting at `start` index.
fn find_json_object_end(text: &str, start: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut escape = false;

    for (i, &ch) in bytes.iter().enumerate().skip(start) {
        if escape { escape = false; continue; }
        if ch == b'\\' && in_string { escape = true; continue; }
        if ch == b'"' { in_string = !in_string; continue; }
        if !in_string {
            match ch {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 { return Some(i); }
                }
                _ => {}
            }
        }
    }
    None
}

fn try_parse_tool_json(json_str: &str) -> Option<ToolCall> {
    let normalized = normalize_tool_call_text(json_str);
    match serde_json::from_str::<Value>(&normalized) {
        Ok(parsed) => {
            let name = parsed.get("tool")?.as_str()?.trim();
            if name.is_empty() { return None; }
            let args = parsed.get("args")?;
            if !args.is_object() { return None; }
            debug!(tool = %name, "Successfully parsed tool call from JSON");
            Some(ToolCall { name: name.to_string(), args: sanitize_tool_args(args) })
        }
        Err(e) => {
            debug!(error = %e, json_preview = %&normalized[..normalized.len().min(100)],
                   "Failed to parse JSON as tool call");
            None
        }
    }
}

fn accept_tool_call(tc: ToolCall, tool_names: &[&str]) -> Option<ToolCall> {
    let client_tool_count = tool_names.len();
    if let Some(canonical) = is_tool_name_allowed(&tc.name, tool_names) {
        let mut accepted = tc;
        let used_norm = accepted.name != canonical;
        if used_norm {
            accepted.name = canonical;
        }
        info!(
            tool_requested = %accepted.name,
            tool_allowed = true,
            client_tool_count,
            normalized_match = used_norm,
            "Tool call accepted (Layer 4 norm integrated)"
        );
        Some(accepted)
    } else {
        warn!(
            tool_requested = %tc.name,
            tool_allowed = false,
            client_tool_count,
            hallucinated_tool_names = ?vec![&tc.name],
            "Tool call REJECTED - name not in client's allowed list (will not be emitted)"
        );
        None
    }
}

/// Phase 2 Hard Validation Gate (defense-in-depth; Phases 3-5 + Layer 4/5 completed 2026-05 per finish-robust... plan).
/// Returns the filtered good tool calls, or the list of hallucinated/unknown names.
/// This is the single choke point that guarantees unknown tool names are *never* emitted to the client.
pub fn validate_tool_calls(tcs: Vec<ToolCall>, allowed: &[Value]) -> Result<Vec<ToolCall>, Vec<String>> {
    let allowed_names: Vec<&str> = allowed
        .iter()
        .filter_map(|t| t["name"].as_str())
        .collect();
    let client_tool_count = allowed_names.len();

    let mut good = Vec::new();
    let mut bad = Vec::new();

    for tc in tcs {
        if let Some(canonical) = is_tool_name_allowed(&tc.name, &allowed_names) {
            let mut good_tc = tc;
            let used_norm = good_tc.name != canonical;
            if used_norm {
                good_tc.name = canonical;
            }
            info!(
                tool_requested = %good_tc.name,
                tool_allowed = true,
                client_tool_count,
                normalized_match = used_norm,
                "Tool validated and allowed for emission (Layer 4 norm)"
            );
            good.push(good_tc);
        } else {
            error!(
                tool_requested = %tc.name,
                tool_allowed = false,
                client_tool_count,
                hallucinated_tool_names = ?vec![&tc.name],
                "HALLUCINATED TOOL NAME — blocked from client emission (Phase 2 hard gate)"
            );
            bad.push(tc.name);
        }
    }

    if bad.is_empty() {
        Ok(good)
    } else if crate::constants::strict_tool_validation() {
        Err(bad)
    } else {
        info!(
            dropped_hallucinated_count = bad.len(),
            client_tool_count,
            "STRICT_TOOL_VALIDATION=false (burn-in): dropping hallucinated names, emitting only good ones or falling back to text. Monitor logs for hallucination rate."
        );
        Ok(good)
    }
}

/// Detect tool calls in text using multiple strategies.
/// Strategies are tried in order of cost; early returns avoid redundant work (#11).
/// Layer 4 (Phase 4.2): accept_tool_call (called by all strategies) now integrates
/// normalize_tool_name + is_tool_name_allowed so prefixed names (get_*, bash_* etc.) are
/// forgiven early and canonical name emitted. validate_tool_calls is the final choke.
pub fn detect_tools(text: &str, tool_defs: &[Value]) -> Vec<ToolCall> {
    let normalized = normalize_tool_call_text(text);
    let tool_names: Vec<&str> = tool_defs
        .iter()
        .filter_map(|t| t["name"].as_str())
        .collect();
    let client_tool_count = tool_names.len();

    debug!(
        text_len = normalized.len(),
        tool_names = ?tool_names,
        text_preview = %&normalized[..normalized.len().min(200)],
        client_tool_count,
        "Starting tool detection"
    );

    let mut found: Vec<ToolCall> = Vec::new();
    let mut used_codeblock_path = false;

    // Strategy 1: markdown code blocks (cheapest — one regex pass)
    // (Layer 4 norm happens inside accept_tool_call for all strategies)
    for cap in MARKDOWN_CODE_RE.captures_iter(&normalized) {
        let json_str = cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
        if let Some(tc) = try_parse_tool_json(json_str).and_then(|tc| accept_tool_call(tc, &tool_names)) {
            debug!(strategy = "markdown_code_block", tool = %tc.name, "Tool detected");
            used_codeblock_path = true;
            if !found.contains(&tc) { found.push(tc); }
        }
    }
    // Short-circuit: if Strategy 1 found results, skip costlier strategies (#11)
    if !found.is_empty() {
        info!(
            client_tool_count,
            detected_count = found.len(),
            used_codeblock_path,
            hallucinated_tool_names = ?Vec::<String>::new(),
            "Tool detection complete (markdown fast-path)"
        );
        return found;
    }

    // Strategy 2: scan for JSON objects that explicitly contain "tool" key
    let blocks = extract_json_blocks(&normalized);
    for json_str in blocks.iter().filter(|b| b.contains("\"tool\"")) {
        if let Some(tc) = try_parse_tool_json(json_str).and_then(|tc| accept_tool_call(tc, &tool_names)) {
            debug!(strategy = "json_object_scan", tool = %tc.name, "Tool detected");
            if !found.contains(&tc) { found.push(tc); }
        }
    }
    // Short-circuit: if Strategy 2 found results, skip Strategy 3 (#11)
    if !found.is_empty() {
        info!(
            client_tool_count,
            detected_count = found.len(),
            used_codeblock_path,
            hallucinated_tool_names = ?Vec::<String>::new(),
            "Tool detection complete (json scan)"
        );
        return found;
    }

    // Strategy 3: line-by-line scan
    for line in normalized.lines() {
        let trimmed = line.trim();
        if trimmed.contains("\"tool\"") && trimmed.contains("\"args\"") {
            if let Some(start) = trimmed.find('{') {
                if let Some(end) = find_json_object_end(trimmed, start) {
                    let json_str = &trimmed[start..=end];
                    if let Some(tc) = try_parse_tool_json(json_str).and_then(|tc| accept_tool_call(tc, &tool_names)) {
                        debug!(strategy = "line_scan", tool = %tc.name, "Tool detected");
                        if !found.contains(&tc) { found.push(tc); }
                    }
                }
            }
        }
    }

    let hallucinated: Vec<String> = vec![]; // Phase 0 baseline (strict filtering comes in Phase 2)
    if found.is_empty() {
        debug!(client_tool_count, "No tool call detected in text");
    } else {
        debug!(count = found.len(), client_tool_count, "Tool calls detected");
    }
    info!(
        client_tool_count,
        detected_count = found.len(),
        used_codeblock_path,
        hallucinated_tool_names = ?hallucinated,
        "Tool detection complete"
    );
    found
}

/// Response id from a Qwen SSE chunk — chain as `parent_id` on the next turn.
pub fn extract_response_parent_id(ch: &Value) -> Option<String> {
    ch.get("response")
        .and_then(|r| r.get("created"))
        .and_then(|c| c.get("response_id"))
        .and_then(|v| v.as_str())
        .or_else(|| ch.get("response_id").and_then(|v| v.as_str()))
        .map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_tool_simple() {
        let text = r#"{"tool":"write","args":{"path":"test.txt","content":"hello"}}"#;
        let tools = vec![serde_json::json!({"name":"write","description":"","parameters":{}})]; 
        let tcs = detect_tools(text, &tools);
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].name, "write");
    }

    #[test]
    fn test_detect_tool_in_markdown() {
        let text = "Here is the tool call:\n```json\n{\"tool\":\"bash\",\"args\":{\"command\":\"ls\"}}\n```\nDone.";
        let tools = vec![serde_json::json!({"name":"bash","description":"","parameters":{}})]; 
        let tcs = detect_tools(text, &tools);
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].name, "bash");
    }

    #[test]
    fn test_detect_tool_embedded_in_text() {
        let text = "I've created the file. {\"tool\":\"write\",\"args\":{\"path\":\"demo.py\",\"content\":\"print('hi')\"}} The file is ready.";
        let tools = vec![serde_json::json!({"name":"write","description":"","parameters":{}})]; 
        let tcs = detect_tools(text, &tools);
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].name, "write");
    }

    #[test]
    fn test_detect_tool_unknown_tool() {
        let text = r#"{"tool":"unknown","args":{}}"#;
        let tools: Vec<Value> = vec![];
        let tcs = detect_tools(text, &tools);
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].name, "unknown");
    }

    #[test]
    fn test_detect_tool_nested_braces_in_string() {
        let text = r#"{"tool":"bash","args":{"command":"echo {hello}"}}"#;
        let tools = vec![serde_json::json!({"name":"bash","description":"","parameters":{}})]; 
        let tcs = detect_tools(text, &tools);
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].name, "bash");
        assert_eq!(tcs[0].args["command"], "echo {hello}");
    }

    #[test]
    fn test_detect_tool_deeply_nested_json() {
        let text = r#"{"tool":"write","args":{"path":"test.json","content":"{\"key\": \"value\"}"}}"#;
        let tools = vec![serde_json::json!({"name":"write","description":"","parameters":{}})]; 
        let tcs = detect_tools(text, &tools);
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].name, "write");
    }

    #[test]
    fn test_detect_tool_escaped_quotes() {
        let text = r#"{"tool":"write","args":{"content":"He said \"hello\""}}"#;
        let tools = vec![serde_json::json!({"name":"write","description":"","parameters":{}})]; 
        let tcs = detect_tools(text, &tools);
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].name, "write");
        assert_eq!(tcs[0].args["content"], "He said \"hello\"");
    }

    #[test]
    fn test_detect_multiple_tools() {
        let text = "First I'll read the file:\n```json\n{\"tool\":\"read\",\"args\":{\"path\":\"test.txt\"}}\n```\nThen I'll write:\n```json\n{\"tool\":\"write\",\"args\":{\"path\":\"out.txt\",\"content\":\"done\"}}\n```\nFinished.";
        let tools = vec![
            serde_json::json!({"name":"read","description":"","parameters":{}}),
            serde_json::json!({"name":"write","description":"","parameters":{}}),
        ];
        let tcs = detect_tools(text, &tools);
        assert_eq!(tcs.len(), 2, "should detect both tool calls");
        assert_eq!(tcs[0].name, "read");
        assert_eq!(tcs[1].name, "write");
    }

    #[test]
    fn test_detect_multiple_tools_no_markdown() {
        let text = r#"{"tool":"read","args":{"path":"a.txt"}}{"tool":"write","args":{"path":"b.txt","content":"x"}}"#;
        let tools = vec![
            serde_json::json!({"name":"read","description":"","parameters":{}}),
            serde_json::json!({"name":"write","description":"","parameters":{}}),
        ];
        let tcs = detect_tools(text, &tools);
        assert_eq!(tcs.len(), 2, "should detect both bare JSON tool calls");
    }

    #[test]
    fn test_build_message() {
        let v = serde_json::json!({
            "messages": [
                {"role":"system","content":"You are helpful."},
                {"role":"user","content":"Hello"},
            ],
            "tools": [{"function":{"name":"bash","description":"run commands","parameters":{}}}]
        });
        let msg = build_message(&v);
        assert!(msg.contains("You are helpful"), "system prompt included");
        assert!(msg.contains("USER: Hello"), "user message included");
        assert!(msg.contains("Tool Use Format"), "tool instructions present");
        assert!(msg.contains("```json"), "code block instruction present");
        // Phase 1 hardening assertions
        assert!(msg.contains("CRITICAL RULES") || msg.contains("exact `name`"), "loud 'only exact names' rule present");
        assert!(msg.contains("exact tool name spelling from the client's list") || msg.contains("exact `name` appears"), "reinforced exact-name guidance");
    }

    #[test]
    fn test_find_json_object_end() {
        let text = r#"{"tool":"bash","args":{"command":"echo {hello}"}}"#;
        let end = find_json_object_end(text, 0).expect("should find end");
        assert_eq!(end, text.len() - 1);
    }

    #[test]
    fn test_detect_tool_spinner_in_args() {
        let text = r#"{"tool":"bash","args":{"⠧ Thinking...command":"echo TOOLCALL-LIVE"}}"#;
        let tools = vec![serde_json::json!({"name":"bash","description":"","parameters":{}})]; 
        let tcs = detect_tools(text, &tools);
        assert_eq!(tcs.len(), 1, "should detect tool with spinner in args");
        assert_eq!(tcs[0].name, "bash");
        assert_eq!(tcs[0].args["command"], "echo TOOLCALL-LIVE");
    }

    #[test]
    fn test_client_visible_content_hides_tool_json() {
        let raw = r#"{"tool":"bash","args":{"command":"ls"}}"#;
        assert_eq!(client_visible_content(raw, None, true), "");
        assert_eq!(client_visible_content("hello", None, true), "hello");
    }

    #[test]
    fn test_looks_like_tool_call_backtick() {
        assert!(looks_like_tool_call_attempt("```json\n{\"tool\":\"bash\"}"));
        assert!(looks_like_tool_call_attempt("```"));
        assert!(looks_like_tool_call_attempt("text ```json more"));
        assert!(!looks_like_tool_call_attempt("hello world"));
    }

    #[test]
    fn test_find_json_object_end_with_escaped() {
        let text = r#"{"tool":"write","args":{"content":"He said \"hi\""}}"#;
        let end = find_json_object_end(text, 0).expect("should find end");
        assert_eq!(end, text.len() - 1);
    }

    // ── NEW: thinking extraction tests ──

    #[test]
    fn test_extract_thinking_from_extra_summary_thought() {
        let ch = serde_json::json!({
            "choices": [{"delta": {
                "phase": "thinking",
                "extra": {
                    "summary_thought": {"content": "Let me think about this step by step."}
                }
            }}]
        });
        let delta = extract_qwen_sse_delta(&ch);
        assert!(delta.is_some(), "should extract thinking from extra.summary_thought");
        let d = delta.unwrap();
        assert_eq!(d.phase, QwenPhase::Thinking);
        assert!(d.text.contains("think about this"), "thinking text extracted: {}", d.text);
    }

    #[test]
    fn test_extract_thinking_from_extra_both_fields() {
        let ch = serde_json::json!({
            "choices": [{"delta": {
                "phase": "thinking_summary",
                "extra": {
                    "summary_title": {"content": "Analyzing request"},
                    "summary_thought": {"content": "Breaking down the problem."}
                }
            }}]
        });
        let delta = extract_qwen_sse_delta(&ch);
        assert!(delta.is_some());
        let d = delta.unwrap();
        assert!(d.text.contains("Analyzing"), "title in text");
        assert!(d.text.contains("Breaking down"), "thought in text");
    }

    #[test]
    fn test_extract_thinking_from_array_content() {
        let ch = serde_json::json!({
            "choices": [{"delta": {
                "phase": "thinking",
                "extra": {
                    "summary_thought": {"content": ["First thought.", "Second thought."]}
                }
            }}]
        });
        let delta = extract_qwen_sse_delta(&ch);
        assert!(delta.is_some());
        let d = delta.unwrap();
        assert!(d.text.contains("First thought."));
        assert!(d.text.contains("Second thought."));
    }

    #[test]
    fn test_extract_qwen_sse_delta_answer_phase() {
        let ch = serde_json::json!({
            "choices": [{"delta": {"phase": "answer", "content": ""}}],
            "response": {"content": "Hello"},
            "content": ""
        });
        let delta = extract_qwen_sse_delta(&ch);
        assert!(delta.is_some());
        assert_eq!(delta.unwrap().text, "Hello");
    }

    #[test]
    fn test_extract_qwen_sse_delta_skips_thinking_empty() {
        let ch = serde_json::json!({
            "choices": [{"delta": {"phase": "thinking", "content": ""}}]
        });
        let delta = extract_qwen_sse_delta(&ch);
        assert!(delta.is_some(), "empty thinking phase should return Some for accumulation");
        assert_eq!(delta.unwrap().text, "");
    }

    #[test]
    fn test_extract_qwen_sse_delta_chunk_content() {
        let ch = serde_json::json!({
            "choices": [{"delta": {"phase": "answer"}}],
            "content": "world"
        });
        let delta = extract_qwen_sse_delta(&ch);
        assert!(delta.is_some());
        assert_eq!(delta.unwrap().text, "world");
    }

    // ─────────────────────────────────────────────────────────────
    // NEW Phase 0 adversarial tests for Robust Tool-Calling Translator
    // (Layer 5 Observability & Testing requirements)
    // These document current permissive behavior and will be strengthened
    // after Phase 2 hard validation gate.
    // ─────────────────────────────────────────────────────────────

    #[test]
    fn test_adversarial_unknown_tool_name_with_empty_list_is_permissive_baseline() {
        // Current (pre-Phase 2) behavior: empty tool list ⇒ accept anything.
        // This is the root cause of "Tool X does not exist" leaks.
        let text = r#"```json
{"tool":"get_terminal_output","args":{"command":"ls -la"}}
```"#;
        let tools: Vec<Value> = vec![];
        let tcs = detect_tools(text, &tools);
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].name, "get_terminal_output"); // will be rejected after strict gate
    }

    #[test]
    fn test_adversarial_unknown_tool_name_is_rejected_when_list_nonempty() {
        // This already works today via accept_tool_call — important regression guard.
        let text = r#"```json
{"tool":"cursor_bash_run","args":{"cmd":"whoami"}}
```"#;
        let tools = vec![
            serde_json::json!({"name":"read_file","description":"","parameters":{}}),
            serde_json::json!({"name":"write_file","description":"","parameters":{}}),
        ];
        let tcs = detect_tools(text, &tools);
        assert!(tcs.is_empty(), "unknown tool must not be returned when client list is provided");
    }

    #[test]
    fn test_large_tool_list_20_plus_still_detects_correctly_via_codeblock() {
        let mut tools = vec![];
        for i in 0..25 {
            tools.push(serde_json::json!({
                "name": format!("tool_{}", i),
                "description": "synthetic",
                "parameters": {}
            }));
        }
        let text = r#"I need to use one.
```json
{"tool":"tool_17","args":{"x":42}}
```
Done."#;
        let tcs = detect_tools(&text, &tools);
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].name, "tool_17");
    }

    #[test]
    fn test_codeblock_vs_raw_json_prefers_fast_markdown_path() {
        // Reinforces that the markdown strategy (used_codeblock_path) is preferred.
        let text = "Some thinking...\n```json\n{\"tool\":\"bash\",\"args\":{\"cmd\":\"date\"}}\n```\n";
        let tools = vec![serde_json::json!({"name":"bash","description":"","parameters":{}})];
        let tcs = detect_tools(text, &tools);
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].name, "bash");
        // The logging emitted used_codeblock_path=true (visible with RUST_LOG=info)
    }

    #[test]
    fn test_response_format_json_schema_does_not_break_tool_detection() {
        // response_format + tools must coexist without crashing the prompt logic.
        let text = r#"```json
{"tool":"search","args":{"q":"rust"}}
```"#;
        let tools = vec![serde_json::json!({"name":"search","description":"","parameters":{}})];
        let tcs = detect_tools(text, &tools);
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].name, "search");
    }

    #[test]
    fn test_normalization_strips_spinner_and_thinking_before_detection() {
        let text = "⠋ Thinking...\n```json\n{\"tool\":\"ls\",\"args\":{}}\n```";
        let tools = vec![serde_json::json!({"name":"ls","description":"","parameters":{}})];
        let tcs = detect_tools(text, &tools);
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].name, "ls");
    }

    // Phase 3 skeleton (compile + format check for feedback message synthesis; real send tested in manual repro 5.5)
    #[test]
    fn test_normalize_feedback_message_format() {
        let bad_names = vec!["get_terminal_output".to_string(), "bash_run".to_string()];
        let fb = format!(
            "TOOL RESULT: ERROR: You attempted to call tool(s) {:?} which are not in the Available Tools list. Only use exact names from the list the client provided. Retry with a valid tool or respond in plain text without a tool call.",
            bad_names
        );
        assert!(fb.contains("TOOL RESULT: ERROR"));
        assert!(fb.contains("get_terminal_output"));
        assert!(fb.contains("bash_run"));
        // The send_qwen_chat_continuation helper (added 3.1) reuses qwen_payload + extract_response_parent_id and is async best-effort.
    }

    // Phase 4.2 Layer 4 norm tests (added per finish-plan 4.2)
    #[test]
    fn test_normalize_tool_name_strips_common_agent_prefixes_suffixes() {
        assert_eq!(normalize_tool_name("Get_Terminal_Output"), "terminal_output");
        assert_eq!(normalize_tool_name("bash_run"), "run");
        assert_eq!(normalize_tool_name("execute_command"), "command");
        assert_eq!(normalize_tool_name("cursor_foo_tool"), "foo");
        assert_eq!(normalize_tool_name("api__call"), "call");
        assert_eq!(normalize_tool_name("foo"), "foo");
        // chained prefix+suffix on synthetic input yields "bash" (sequential pass); real cases like "bash_run" stop at first hit
        assert_eq!(normalize_tool_name("  Run_Bash_Cmd  "), "bash");
        assert_eq!(normalize_tool_name("write_file"), "write_file"); // no strip for this
    }

    #[test]
    fn test_adversarial_validate_with_normalization_accepts_prefixed_and_emits_canonical() {
        // list has canonical short names; request has common agent prefixes
        let allowed = vec![
            serde_json::json!({"name":"foo","description":"","parameters":{}}),
            serde_json::json!({"name":"terminal","description":"","parameters":{}}),
        ];
        // build fake tcs as if detected with prefixed names
        let bad_prefixed = vec![
            ToolCall { name: "get_foo".into(), args: serde_json::json!({}) },
            ToolCall { name: "bash_terminal".into(), args: serde_json::json!({}) },
            ToolCall { name: "unknown_weird".into(), args: serde_json::json!({}) },
        ];
        let res = validate_tool_calls(bad_prefixed, &allowed);
        assert!(res.is_err(), "unknown must still be rejected even with norm");
        // but the goods: to test accept path + norm emit, use detect which goes thru accept
        let text = r#"```json
{"tool":"get_foo","args":{}}
{"tool":"bash_terminal","args":{}}
```"#;
        let tcs = detect_tools(text, &allowed);
        assert_eq!(tcs.len(), 2);
        let names: Vec<_> = tcs.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"foo"));
        assert!(names.contains(&"terminal")); // canonical emitted, not the prefixed
    }

    #[test]
    fn test_full_detect_validate_norm_roundtrip_blocks_true_unknowns() {
        let allowed = vec![serde_json::json!({"name":"ls","description":"","parameters":{}})];
        let text = r#"```json
{"tool":"bash_ls","args":{"cmd":"."}}
{"tool":"get_nonexistent_cursor_tool","args":{}}
```"#;
        let raw = detect_tools(text, &allowed);
        let validated = validate_tool_calls(raw, &allowed);
        // only the norm-matched ls should survive; the invented one rejected
        match validated {
            Ok(good) => {
                assert_eq!(good.len(), 1);
                assert_eq!(good[0].name, "ls");
            }
            Err(bads) => panic!("expected 1 good via norm, got err {:?}", bads),
        }
    }

    // Phase 5.2: explicit "0 unknown ever emitted" adversarial (even 20+ Cursor-like invented/prefixed)
    #[test]
    fn test_adversarial_validate_blocks_20_weird_invented_names_including_cursor_like() {
        let allowed = vec![
            serde_json::json!({"name":"read_file","description":"","parameters":{}}),
            serde_json::json!({"name":"write_file","description":"","parameters":{}}),
            serde_json::json!({"name":"list_dir","description":"","parameters":{}}),
            serde_json::json!({"name":"run_terminal","description":"","parameters":{}}),
            serde_json::json!({"name":"search","description":"","parameters":{}}),
        ];
        // 20+ weird invented + prefixed + case mixes + nonsense (simulating Cursor/Aider/agent toolsets)
        let mixed: Vec<ToolCall> = vec![
            ToolCall { name: "read_file".into(), args: serde_json::json!({}) }, // good
            ToolCall { name: "get_terminal_output".into(), args: serde_json::json!({}) },
            ToolCall { name: "bash_run".into(), args: serde_json::json!({}) },
            ToolCall { name: "execute_command".into(), args: serde_json::json!({}) },
            ToolCall { name: "cursor_get_output".into(), args: serde_json::json!({}) },
            ToolCall { name: "run_bash".into(), args: serde_json::json!({}) },
            ToolCall { name: "tool_foo".into(), args: serde_json::json!({}) },
            ToolCall { name: "api__call".into(), args: serde_json::json!({}) },
            ToolCall { name: "X7y9Z".into(), args: serde_json::json!({}) },
            ToolCall { name: "Get_File".into(), args: serde_json::json!({}) },
            ToolCall { name: "WRITE_FILE_TOOL".into(), args: serde_json::json!({}) },
            ToolCall { name: "listdir_cmd".into(), args: serde_json::json!({}) },
            ToolCall { name: "search_op".into(), args: serde_json::json!({}) },
            ToolCall { name: "cursor_write".into(), args: serde_json::json!({}) },
            ToolCall { name: "bash_list_dir".into(), args: serde_json::json!({}) },
            ToolCall { name: "run_foo_bar".into(), args: serde_json::json!({}) },
            ToolCall { name: "execute_ls".into(), args: serde_json::json!({}) },
            ToolCall { name: "tool_unknown_1".into(), args: serde_json::json!({}) },
            ToolCall { name: "weird__name__here".into(), args: serde_json::json!({}) },
            ToolCall { name: "API_SEARCH_V2".into(), args: serde_json::json!({}) },
            ToolCall { name: "run_terminal".into(), args: serde_json::json!({}) }, // good (norm or exact)
            ToolCall { name: "search".into(), args: serde_json::json!({}) }, // good
        ];
        let res = validate_tool_calls(mixed, &allowed);
        match res {
            Err(bads) => {
                // all bads must be unknowns (no good slipped to err)
                for b in &bads {
                    assert!(!["read_file","write_file","list_dir","run_terminal","search"].contains(&b.as_str()),
                        "good name leaked to bad list: {}", b);
                }
                assert!(!bads.is_empty(), "should have detected some hallucinations");
            }
            Ok(goods) => {
                // under default strict, if any bad in input list, should not reach Ok with only goods? Wait, since input had bads, but
                // actually because validate filters, but for this test we pass mixed including bads, expect Err under strict
                panic!("expected Err for unknowns in list, got Ok with {} goods", goods.len());
            }
        }
        // Also: direct good-only list -> Ok, 0 bads
        let only_goods = vec![
            ToolCall { name: "read_file".into(), args: serde_json::json!({}) },
            ToolCall { name: "bash_search".into(), args: serde_json::json!({}) }, // norm to search
        ];
        let ok_res = validate_tool_calls(only_goods, &allowed);
        assert!(ok_res.is_ok());
        let emitted: Vec<_> = ok_res.unwrap().iter().map(|t| t.name.clone()).collect();
        assert!(emitted.contains(&"read_file".to_string()));
        assert!(emitted.contains(&"search".to_string())); // canonical
    }
}
