use regex::Regex;
use serde_json::Value;
use std::sync::LazyLock;
use tracing::{debug, trace};

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
    if text.contains("cannot use") || text.contains("can't use") || text.contains("unable to use") {
        if text.contains("tool") {
            let end = text.find('.').unwrap_or(text.len().min(200));
            return Some(text[..end].to_string());
        }
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
        result.push_str("\n\n## Tool Use Format\n");
        result.push_str("When you need to call a tool, output a fenced code block with language `json`:\n");
        result.push_str("```json\n{\"tool\":\"tool_name\",\"args\":{...}}\n```\n");
        result.push_str("Always wrap the tool JSON in a markdown code block. Never output raw JSON outside a code block.\n");
        result.push_str("Never add text after the code block. The code block must be the last thing you output.\n");
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

    for i in start..bytes.len() {
        let ch = bytes[i];
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
    if tool_names.is_empty() || tool_names.contains(&tc.name.as_str()) {
        Some(tc)
    } else {
        None
    }
}

/// Detect tool calls in text using multiple strategies.
/// Strategies are tried in order of cost; early returns avoid redundant work (#11).
pub fn detect_tools(text: &str, tool_defs: &[Value]) -> Vec<ToolCall> {
    let normalized = normalize_tool_call_text(text);
    let tool_names: Vec<&str> = tool_defs
        .iter()
        .filter_map(|t| t["name"].as_str())
        .collect();

    debug!(
        text_len = normalized.len(),
        tool_names = ?tool_names,
        text_preview = %&normalized[..normalized.len().min(200)],
        "Starting tool detection"
    );

    let mut found: Vec<ToolCall> = Vec::new();
    let mut add_unique = |tc: ToolCall| {
        if !found.contains(&tc) { found.push(tc); }
    };

    // Strategy 1: markdown code blocks (cheapest — one regex pass)
    for cap in MARKDOWN_CODE_RE.captures_iter(&normalized) {
        let json_str = cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
        if let Some(tc) = try_parse_tool_json(json_str).and_then(|tc| accept_tool_call(tc, &tool_names)) {
            debug!(strategy = "markdown_code_block", tool = %tc.name, "Tool detected");
            add_unique(tc);
        }
    }
    // Short-circuit: if Strategy 1 found results, skip costlier strategies (#11)
    if !found.is_empty() { return found; }

    // Strategy 2: scan for JSON objects that explicitly contain "tool" key
    let blocks = extract_json_blocks(&normalized);
    for json_str in blocks.iter().filter(|b| b.contains("\"tool\"")) {
        if let Some(tc) = try_parse_tool_json(json_str).and_then(|tc| accept_tool_call(tc, &tool_names)) {
            debug!(strategy = "json_object_scan", tool = %tc.name, "Tool detected");
            add_unique(tc);
        }
    }
    // Short-circuit: if Strategy 2 found results, skip Strategy 3 (#11)
    if !found.is_empty() { return found; }

    // Strategy 3: line-by-line scan
    for line in normalized.lines() {
        let trimmed = line.trim();
        if trimmed.contains("\"tool\"") && trimmed.contains("\"args\"") {
            if let Some(start) = trimmed.find('{') {
                if let Some(end) = find_json_object_end(trimmed, start) {
                    let json_str = &trimmed[start..=end];
                    if let Some(tc) = try_parse_tool_json(json_str).and_then(|tc| accept_tool_call(tc, &tool_names)) {
                        debug!(strategy = "line_scan", tool = %tc.name, "Tool detected");
                        add_unique(tc);
                    }
                }
            }
        }
    }

    if found.is_empty() { debug!("No tool call detected in text"); }
    else { debug!(count = found.len(), "Tool calls detected"); }
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
}
