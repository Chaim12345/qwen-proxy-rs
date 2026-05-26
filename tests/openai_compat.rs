//! OpenAI API Compatibility Test Suite
//!
//! ## Running tests
//! - Unit tests (no server needed): `cargo test`
//! - Integration tests (needs running server): `cargo test -- --ignored`
//! - Set PROXY_URL env var if server is not on localhost:8765

use serde_json::{json, Value};

// ─── Helper functions ─────────────────────────────────

fn base_url() -> String {
    let url = std::env::var("PROXY_URL").unwrap_or_else(|_| "http://127.0.0.1:8765".to_string());
    url.trim_end_matches('/')
        .trim_end_matches("/v1")
        .to_string()
}

fn client() -> reqwest::Client {
    reqwest::Client::new()
}

fn assert_openai_error(resp: &Value) {
    assert!(
        resp.get("error").is_some(),
        "Response must have 'error' field"
    );
    let err = &resp["error"];
    assert!(
        err.get("message").is_some(),
        "Error must have 'message' field"
    );
    assert!(err["message"].is_string(), "Error message must be a string");
    assert!(err.get("type").is_some(), "Error must have 'type' field");
}

fn assert_chat_completion_shape(resp: &Value) {
    assert!(resp.get("id").is_some(), "Missing 'id'");
    assert!(
        resp["id"].as_str().unwrap_or("").starts_with("chatcmpl-"),
        "id must start with 'chatcmpl-'"
    );

    assert_eq!(
        resp["object"].as_str(),
        Some("chat.completion"),
        "object must be 'chat.completion'"
    );

    assert!(resp.get("created").is_some(), "Missing 'created'");
    assert!(resp["created"].is_number(), "created must be a number");

    assert!(resp.get("model").is_some(), "Missing 'model'");
    assert!(resp["model"].is_string(), "model must be a string");

    assert!(resp.get("choices").is_some(), "Missing 'choices'");
    let choices = resp["choices"]
        .as_array()
        .expect("choices must be an array");
    assert!(!choices.is_empty(), "choices must not be empty");

    for choice in choices {
        assert!(choice.get("index").is_some(), "Choice must have 'index'");
        assert!(
            choice.get("message").is_some(),
            "Choice must have 'message'"
        );
        assert!(
            choice.get("finish_reason").is_some(),
            "Choice must have 'finish_reason'"
        );

        let msg = &choice["message"];
        assert_eq!(
            msg["role"].as_str(),
            Some("assistant"),
            "Message role must be 'assistant'"
        );
    }

    assert!(resp.get("usage").is_some(), "Missing 'usage'");
    let usage = &resp["usage"];
    assert!(
        usage.get("prompt_tokens").is_some(),
        "Missing 'prompt_tokens'"
    );
    assert!(
        usage.get("completion_tokens").is_some(),
        "Missing 'completion_tokens'"
    );
    assert!(
        usage.get("total_tokens").is_some(),
        "Missing 'total_tokens'"
    );

    // Verify total_tokens = prompt_tokens + completion_tokens
    let pt = usage["prompt_tokens"].as_u64().unwrap_or(0);
    let ct = usage["completion_tokens"].as_u64().unwrap_or(0);
    let tt = usage["total_tokens"].as_u64().unwrap_or(0);
    assert_eq!(
        tt,
        pt + ct,
        "total_tokens must equal prompt_tokens + completion_tokens"
    );
}

// ─── Unit tests (no server needed) ────────────────────

#[test]
fn test_models_json_format() {
    // Verify MODELS_JSON is valid OpenAI format
    let models: Value = serde_json::from_str(r#"{"object":"list","data":[
{"id":"qwen3.6-plus","object":"model","created":1700000000,"owned_by":"qwen","permission":[],"root":"qwen3.6-plus","parent":null}
]}"#).unwrap();

    assert_eq!(models["object"].as_str(), Some("list"));
    let data = models["data"].as_array().unwrap();
    assert!(!data.is_empty());
    assert_eq!(data[0]["object"].as_str(), Some("model"));
    assert!(data[0]["id"].is_string());
}

#[test]
fn test_model_info_format() {
    let model: Value = serde_json::from_str(
        r#"{"id":"qwen3.6-plus","object":"model","created":1700000000,"owned_by":"qwen"}"#,
    )
    .unwrap();
    assert_eq!(model["id"].as_str(), Some("qwen3.6-plus"));
    assert_eq!(model["object"].as_str(), Some("model"));
}

#[test]
fn test_openai_error_format() {
    // Verify error format matches OpenAI spec
    let err = json!({
        "error": {
            "message": "test error",
            "type": "invalid_request_error",
            "param": "messages",
            "code": null
        }
    });
    assert_openai_error(&err);
}

#[test]
fn test_chat_completion_response_shape() {
    let resp = json!({
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "qwen3.6-plus",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    });
    assert_chat_completion_shape(&resp);
}

#[test]
fn test_chat_completion_tool_call_shape() {
    let resp = json!({
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "qwen3.6-plus",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{\"path\":\"/tmp\"}"}
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    });
    assert_chat_completion_shape(&resp);
}

#[test]
fn test_token_estimation() {
    // Simple token estimation: bytes / 4
    let text = "Hello world, this is a test message.";
    let tokens = std::cmp::max(1, text.len() / 4);
    assert!(tokens > 0);
}

// ─── Integration tests (need running server) ──────────

#[tokio::test]
#[ignore]
async fn test_health_endpoint() {
    let resp = client()
        .get(format!("{}/health", base_url()))
        .send()
        .await
        .expect("Failed to connect");
    assert!(
        resp.status().is_success(),
        "Health endpoint should return 200"
    );
    let body: Value = resp.json().await.expect("Failed to parse JSON");
    assert_eq!(body["status"].as_str(), Some("ok"));
}

#[tokio::test]
#[ignore]
async fn test_models_list() {
    let resp = client()
        .get(format!("{}/v1/models", base_url()))
        .send()
        .await
        .expect("Failed to connect");
    assert!(resp.status().is_success());
    let body: Value = resp.json().await.expect("Failed to parse JSON");

    assert!(
        body.get("data").is_some(),
        "Models response must have 'data' array"
    );
    let models = body["data"].as_array().expect("data must be an array");
    assert!(!models.is_empty(), "Models list must not be empty");

    for model in models {
        assert!(model.get("id").is_some(), "Each model must have an 'id'");
        assert_eq!(model["object"].as_str(), Some("model"));
    }
}

#[tokio::test]
#[ignore]
async fn test_models_get_specific() {
    let resp = client()
        .get(format!("{}/v1/models/qwen3.6-plus", base_url()))
        .send()
        .await
        .expect("Failed to connect");
    assert!(resp.status().is_success());
    let body: Value = resp.json().await.expect("Failed to parse JSON");
    assert!(body.get("id").is_some());
    assert_eq!(body["object"].as_str(), Some("model"));
}

#[tokio::test]
#[ignore]
async fn test_models_get_not_found() {
    let resp = client()
        .get(format!("{}/v1/models/nonexistent-model", base_url()))
        .send()
        .await
        .expect("Failed to connect");
    assert_eq!(resp.status().as_u16(), 404);
    let body: Value = resp.json().await.expect("Failed to parse JSON");
    assert_openai_error(&body);
}

#[tokio::test]
#[ignore]
async fn test_chat_completions_basic() {
    let resp = client()
        .post(format!("{}/v1/chat/completions", base_url()))
        .json(&json!({
            "model": "qwen3.6-plus",
            "messages": [{"role": "user", "content": "Say 'hello' in one word."}]
        }))
        .send()
        .await
        .expect("Failed to connect");

    assert!(
        resp.status().is_success(),
        "Expected 200, got {}",
        resp.status()
    );
    let body: Value = resp.json().await.expect("Failed to parse JSON");
    assert_chat_completion_shape(&body);
}

#[tokio::test]
#[ignore]
async fn test_chat_completions_system_message() {
    let resp = client()
        .post(format!("{}/v1/chat/completions", base_url()))
        .json(&json!({
            "model": "qwen3.6-plus",
            "messages": [
                {"role": "system", "content": "You only respond in French."},
                {"role": "user", "content": "Say hello"}
            ]
        }))
        .send()
        .await
        .expect("Failed to connect");

    assert!(resp.status().is_success());
    let body: Value = resp.json().await.expect("Failed to parse JSON");
    assert_chat_completion_shape(&body);
}

#[tokio::test]
#[ignore]
async fn test_chat_completions_multi_turn() {
    let resp = client()
        .post(format!("{}/v1/chat/completions", base_url()))
        .json(&json!({
            "model": "qwen3.6-plus",
            "messages": [
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Nice to meet you, Alice!"},
                {"role": "user", "content": "What is my name?"}
            ]
        }))
        .send()
        .await
        .expect("Failed to connect");

    assert!(resp.status().is_success());
    let body: Value = resp.json().await.expect("Failed to parse JSON");
    assert_chat_completion_shape(&body);
}

#[tokio::test]
#[ignore]
async fn test_chat_completions_temperature() {
    let resp = client()
        .post(format!("{}/v1/chat/completions", base_url()))
        .json(&json!({
            "model": "qwen3.6-plus",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.7
        }))
        .send()
        .await
        .expect("Failed to connect");
    assert!(
        resp.status().is_success(),
        "temperature param should be accepted"
    );
}

#[tokio::test]
#[ignore]
async fn test_chat_completions_max_tokens() {
    let resp = client()
        .post(format!("{}/v1/chat/completions", base_url()))
        .json(&json!({
            "model": "qwen3.6-plus",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 50
        }))
        .send()
        .await
        .expect("Failed to connect");
    assert!(
        resp.status().is_success(),
        "max_tokens param should be accepted"
    );
}

#[tokio::test]
#[ignore]
async fn test_chat_completions_top_p() {
    let resp = client()
        .post(format!("{}/v1/chat/completions", base_url()))
        .json(&json!({
            "model": "qwen3.6-plus",
            "messages": [{"role": "user", "content": "Hi"}],
            "top_p": 0.9
        }))
        .send()
        .await
        .expect("Failed to connect");
    assert!(resp.status().is_success(), "top_p param should be accepted");
}

#[tokio::test]
#[ignore]
async fn test_chat_completions_stop() {
    let resp = client()
        .post(format!("{}/v1/chat/completions", base_url()))
        .json(&json!({
            "model": "qwen3.6-plus",
            "messages": [{"role": "user", "content": "Count to 5"}],
            "stop": ["3"]
        }))
        .send()
        .await
        .expect("Failed to connect");
    assert!(resp.status().is_success(), "stop param should be accepted");
}

#[tokio::test]
#[ignore]
async fn test_chat_completions_presence_penalty() {
    let resp = client()
        .post(format!("{}/v1/chat/completions", base_url()))
        .json(&json!({
            "model": "qwen3.6-plus",
            "messages": [{"role": "user", "content": "Hi"}],
            "presence_penalty": 0.5
        }))
        .send()
        .await
        .expect("Failed to connect");
    assert!(
        resp.status().is_success(),
        "presence_penalty should be accepted"
    );
}

#[tokio::test]
#[ignore]
async fn test_chat_completions_frequency_penalty() {
    let resp = client()
        .post(format!("{}/v1/chat/completions", base_url()))
        .json(&json!({
            "model": "qwen3.6-plus",
            "messages": [{"role": "user", "content": "Hi"}],
            "frequency_penalty": 0.5
        }))
        .send()
        .await
        .expect("Failed to connect");
    assert!(
        resp.status().is_success(),
        "frequency_penalty should be accepted"
    );
}

#[tokio::test]
#[ignore]
async fn test_chat_completions_seed() {
    let resp = client()
        .post(format!("{}/v1/chat/completions", base_url()))
        .json(&json!({
            "model": "qwen3.6-plus",
            "messages": [{"role": "user", "content": "Hi"}],
            "seed": 42
        }))
        .send()
        .await
        .expect("Failed to connect");
    assert!(resp.status().is_success(), "seed param should be accepted");
}

#[tokio::test]
#[ignore]
async fn test_chat_completions_user() {
    let resp = client()
        .post(format!("{}/v1/chat/completions", base_url()))
        .json(&json!({
            "model": "qwen3.6-plus",
            "messages": [{"role": "user", "content": "Hi"}],
            "user": "test-user-123"
        }))
        .send()
        .await
        .expect("Failed to connect");
    assert!(resp.status().is_success(), "user param should be accepted");
}

#[tokio::test]
#[ignore]
async fn test_chat_completions_response_format() {
    let resp = client()
        .post(format!("{}/v1/chat/completions", base_url()))
        .json(&json!({
            "model": "qwen3.6-plus",
            "messages": [{"role": "user", "content": "Return JSON"}],
            "response_format": {"type": "json_object"}
        }))
        .send()
        .await
        .expect("Failed to connect");
    assert!(
        resp.status().is_success(),
        "response_format should be accepted"
    );
}

#[tokio::test]
#[ignore]
async fn test_chat_completions_all_params() {
    let resp = client()
        .post(format!("{}/v1/chat/completions", base_url()))
        .json(&json!({
            "model": "qwen3.6-plus",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.5,
            "max_tokens": 100,
            "top_p": 0.9,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.2,
            "seed": 12345,
            "user": "test",
            "stop": ["\n"]
        }))
        .send()
        .await
        .expect("Failed to connect");

    assert!(
        resp.status().is_success(),
        "All params together should be accepted"
    );
    let body: Value = resp.json().await.expect("Failed to parse JSON");
    assert_chat_completion_shape(&body);
}

#[tokio::test]
#[ignore]
async fn test_chat_completions_tools() {
    let resp = client()
        .post(format!("{}/v1/chat/completions", base_url()))
        .json(&json!({
            "model": "qwen3.6-plus",
            "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"}
                        },
                        "required": ["location"]
                    }
                }
            }]
        }))
        .send()
        .await
        .expect("Failed to connect");

    assert!(resp.status().is_success());
    let body: Value = resp.json().await.expect("Failed to parse JSON");
    assert_chat_completion_shape(&body);
}

#[tokio::test]
#[ignore]
async fn test_chat_completions_tool_result() {
    let resp = client()
        .post(format!("{}/v1/chat/completions", base_url()))
        .json(&json!({
            "model": "qwen3.6-plus",
            "messages": [
                {"role": "user", "content": "What is the weather in Tokyo?"},
                {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "call_abc123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"location\": \"Tokyo\"}"}
                }]},
                {"role": "tool", "tool_call_id": "call_abc123", "content": "22°C, sunny"}
            ]
        }))
        .send()
        .await
        .expect("Failed to connect");

    assert!(resp.status().is_success());
    let body: Value = resp.json().await.expect("Failed to parse JSON");
    assert_chat_completion_shape(&body);
}

#[tokio::test]
#[ignore]
async fn test_chat_completions_missing_messages() {
    let resp = client()
        .post(format!("{}/v1/chat/completions", base_url()))
        .json(&json!({"model": "qwen3.6-plus"}))
        .send()
        .await
        .expect("Failed to connect");

    assert_eq!(resp.status().as_u16(), 400);
    let body: Value = resp.json().await.expect("Failed to parse JSON");
    assert_openai_error(&body);
}

#[tokio::test]
#[ignore]
async fn test_chat_completions_empty_messages() {
    let resp = client()
        .post(format!("{}/v1/chat/completions", base_url()))
        .json(&json!({"model": "qwen3.6-plus", "messages": []}))
        .send()
        .await
        .expect("Failed to connect");

    assert_eq!(resp.status().as_u16(), 400);
    let body: Value = resp.json().await.expect("Failed to parse JSON");
    assert_openai_error(&body);
}

#[tokio::test]
#[ignore]
async fn test_chat_completions_invalid_json() {
    let resp = client()
        .post(format!("{}/v1/chat/completions", base_url()))
        .body("not json")
        .header("content-type", "application/json")
        .send()
        .await
        .expect("Failed to connect");

    assert_eq!(resp.status().as_u16(), 400);
    let body: Value = resp.json().await.expect("Failed to parse JSON");
    assert_openai_error(&body);
}

#[tokio::test]
#[ignore]
async fn test_response_content_type() {
    let resp = client()
        .post(format!("{}/v1/chat/completions", base_url()))
        .json(&json!({
            "model": "qwen3.6-plus",
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .send()
        .await
        .expect("Failed to connect");

    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("");
    assert!(
        ct.contains("application/json"),
        "Response should have application/json content type"
    );
}

#[tokio::test]
#[ignore]
async fn test_cors_headers() {
    let resp = client()
        .post(format!("{}/v1/chat/completions", base_url()))
        .json(&json!({
            "model": "qwen3.6-plus",
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .header("Origin", "http://example.com")
        .send()
        .await
        .expect("Failed to connect");

    let allow_origin = resp.headers().get("access-control-allow-origin");
    assert!(
        allow_origin.is_some(),
        "Should have CORS allow-origin header"
    );
}
