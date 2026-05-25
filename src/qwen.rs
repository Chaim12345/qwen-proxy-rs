use regex::Regex;
use serde_json::Value;
use std::sync::LazyLock;
use tracing::debug;

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
    let Some(obj) = args.as_object() else {
        return args.clone();
    };
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
        || cleaned.contains("\"tool_calls\":")
        || cleaned.contains("\"function\":")
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
    if !trimmed.starts_with('{') {
        return None;
    }
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
/// Handles ```json ... ``` and ``` ... ``` patterns.
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
/// Strips code blocks and validates JSON for json_object/json_schema.
pub fn process_structured_output(
    text: &str,
    rf: Option<&Value>,
) -> Result<String, String> {
    let rf = match rf {
        Some(r) => r,
        None => return Ok(text.to_string()),
    };
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
/// When the Qwen model can't use a tool, it returns text like
/// "Tool calculator does not exists" or "无法使用该工具".
pub fn detect_qwen_tool_error(text: &str) -> Option<String> {
    if text.is_empty() {
        return None;
    }

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

    pub fn full_answer(&self) -> &str {
        &self.answer
    }

    pub fn thinking(&self) -> &str {
        &self.thinking
    }
}

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

    if phase == QwenPhase::ThinkingSummary {
        if let Some(extra) = delta.as_ref().and_then(|d| d.get("extra")) {
            if let Some(title) = extra.get("summary_title").and_then(|s| s.get("content")) {
                if let Some(arr) = title.as_array() {
                    for item in arr {
                        if let Some(s) = item.as_str() {
                            text.push_str(s);
                        }
                    }
                } else if let Some(s) = title.as_str() {
                    text.push_str(s);
                }
            }
            if let Some(thought) = extra.get("summary_thought").and_then(|s| s.get("content")) {
                if let Some(arr) = thought.as_array() {
                    for item in arr {
                        if let Some(s) = item.as_str() {
                            if !text.is_empty() {
                                text.push('\n');
                            }
                            text.push_str(s);
                        }
                    }
                } else if let Some(s) = thought.as_str() {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(s);
                }
            }
        }
    } else {
        text = extract_string_value(
            delta
                .and_then(|d| d.get("content"))
                .unwrap_or(&Value::Null),
        );
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
    if let Some(text) = content.as_str() {
        return text.to_string();
    }

    if let Some(parts) = content.as_array() {
        return parts
            .iter()
            .filter_map(|part| {
                part.get("text")
                    .and_then(|v| v.as_str())
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

            // Handle tool_calls from assistant messages
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

/// Parse a JSON object from text starting at `start` index, properly handling
/// nested braces, strings with braces inside them, and escape characters.
/// Returns the end index (inclusive) of the JSON object, or None if parsing fails.
fn find_json_object_end(text: &str, start: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut escape = false;

    for i in start..bytes.len() {
        let ch = bytes[i];

        if escape {
            escape = false;
            continue;
        }

        if ch == b'\\' && in_string {
            escape = true;
            continue;
        }

        if ch == b'"' {
            in_string = !in_string;
            continue;
        }

        if !in_string {
            match ch {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i);
                    }
                }
                _ => {}
            }
        }
    }

    None
}

/// Try to parse a tool call from a JSON string.
/// Returns Some(ToolCall) if the JSON contains {"tool": "...", "args": {...}}
fn try_parse_tool_json(json_str: &str) -> Option<ToolCall> {
    let normalized = normalize_tool_call_text(json_str);
    match serde_json::from_str::<Value>(&normalized) {
        Ok(parsed) => {
            let name = parsed.get("tool")?.as_str()?.trim();
            if name.is_empty() {
                return None;
            }
            let args = parsed.get("args")?;
            if !args.is_object() {
                return None;
            }
            debug!(tool = %name, "Successfully parsed tool call from JSON");
            Some(ToolCall {
                name: name.to_string(),
                args: sanitize_tool_args(args),
            })
        }
        Err(e) => {
            debug!(error = %e, json_preview = %&normalized[..normalized.len().min(100)], "Failed to parse JSON as tool call");
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

/// Detect tool calls in text using multiple strategies:
/// 1. Markdown code blocks containing JSON
/// 2. Proper JSON parsing with string/escape handling
/// 3. Line-by-line scanning for tool-like patterns
pub fn detect_tool(text: &str, tool_defs: &[Value]) -> Option<ToolCall> {
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

    for cap in MARKDOWN_CODE_RE.captures_iter(&normalized) {
        let json_str = cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
        if let Some(tc) = try_parse_tool_json(json_str).and_then(|tc| accept_tool_call(tc, &tool_names)) {
            debug!(strategy = "markdown_code_block", tool = %tc.name, "Tool detected");
            return Some(tc);
        }
    }

    let blocks = extract_json_blocks(&normalized);
    for json_str in blocks.iter().filter(|b| b.contains("\"tool\"")) {
        if let Some(tc) = try_parse_tool_json(json_str).and_then(|tc| accept_tool_call(tc, &tool_names)) {
            debug!(strategy = "json_object_scan", tool = %tc.name, "Tool detected");
            return Some(tc);
        }
    }
    for json_str in &blocks {
        if let Some(tc) = try_parse_tool_json(json_str).and_then(|tc| accept_tool_call(tc, &tool_names)) {
            debug!(strategy = "json_object_scan_fallback", tool = %tc.name, "Tool detected");
            return Some(tc);
        }
    }

    for line in normalized.lines() {
        let trimmed = line.trim();
        if trimmed.contains("\"tool\"") && trimmed.contains("\"args\"") {
            if let Some(start) = trimmed.find('{') {
                if let Some(end) = find_json_object_end(trimmed, start) {
                    let json_str = &trimmed[start..=end];
                    if let Some(tc) = try_parse_tool_json(json_str).and_then(|tc| accept_tool_call(tc, &tool_names)) {
                        debug!(strategy = "line_scan", tool = %tc.name, "Tool detected");
                        return Some(tc);
                    }
                }
            }
        }
    }

    debug!("No tool call detected in text");
    None
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
        let tc = detect_tool(text, &tools).expect("should detect tool");
        assert_eq!(tc.name, "write");
    }

    #[test]
    fn test_detect_tool_in_markdown() {
        let text = "Here is the tool call:\n```json\n{\"tool\":\"bash\",\"args\":{\"command\":\"ls\"}}\n```\nDone.";
        let tools = vec![serde_json::json!({"name":"bash","description":"","parameters":{}})];
        let tc = detect_tool(text, &tools).expect("should detect tool in markdown");
        assert_eq!(tc.name, "bash");
    }

    #[test]
    fn test_detect_tool_embedded_in_text() {
        let text = "I've created the file. {\"tool\":\"write\",\"args\":{\"path\":\"demo.py\",\"content\":\"print('hi')\"}} The file is ready.";
        let tools = vec![serde_json::json!({"name":"write","description":"","parameters":{}})];
        let tc = detect_tool(text, &tools).expect("should detect embedded tool");
        assert_eq!(tc.name, "write");
    }

    #[test]
    fn test_detect_tool_unknown_tool() {
        let text = r#"{"tool":"unknown","args":{}}"#;
        let tools: Vec<Value> = vec![];
        let tc = detect_tool(text, &tools).expect("should detect unknown tool");
        assert_eq!(tc.name, "unknown");
    }

    #[test]
    fn test_detect_tool_nested_braces_in_string() {
        // This is the key test - args contain braces inside a string value
        let text = r#"{"tool":"bash","args":{"command":"echo {hello}"}}"#;
        let tools = vec![serde_json::json!({"name":"bash","description":"","parameters":{}})];
        let tc =
            detect_tool(text, &tools).expect("should detect tool with nested braces in string");
        assert_eq!(tc.name, "bash");
        assert_eq!(tc.args["command"], "echo {hello}");
    }

    #[test]
    fn test_detect_tool_deeply_nested_json() {
        let text =
            r#"{"tool":"write","args":{"path":"test.json","content":"{\"key\": \"value\"}"}}"#;
        let tools = vec![serde_json::json!({"name":"write","description":"","parameters":{}})];
        let tc = detect_tool(text, &tools).expect("should detect tool with deeply nested JSON");
        assert_eq!(tc.name, "write");
    }

    #[test]
    fn test_detect_tool_escaped_quotes() {
        let text = r#"{"tool":"write","args":{"content":"He said \"hello\""}}"#;
        let tools = vec![serde_json::json!({"name":"write","description":"","parameters":{}})];
        let tc = detect_tool(text, &tools).expect("should detect tool with escaped quotes");
        assert_eq!(tc.name, "write");
        assert_eq!(tc.args["content"], "He said \"hello\"");
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
        assert!(
            msg.contains("Tool Use Format"),
            "tool instructions present"
        );
        assert!(
            msg.contains("```json"),
            "code block instruction present"
        );
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
        let tc = detect_tool(text, &tools).expect("should detect tool with spinner in args");
        assert_eq!(tc.name, "bash");
        assert_eq!(tc.args["command"], "echo TOOLCALL-LIVE");
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
    fn test_extract_qwen_sse_delta_skips_thinking() {
        let ch = serde_json::json!({
            "choices": [{"delta": {"phase": "thinking", "content": "secret"}}],
            "response": {"content": "should not appear"}
        });
        assert!(extract_qwen_sse_delta(&ch).is_none());
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
