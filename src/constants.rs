//! Single source of truth for shared compile-time constants.
pub const MODEL_NAME: &str = "qwen3.7-max";
pub const QWEN_API_BASE: &str = "https://chat.qwen.ai/api/v2";

/// Maximum prompt characters accepted before truncation.
/// Qwen3.7-Max supports ~1M token context. Using ~3.5M chars as a conservative
/// estimate (≈ 1M tokens × 3.5 chars/token, with safety margin for
/// multi-language / special-token overhead). If the serialized prompt exceeds
/// this, old messages are dropped from the middle of the history while
/// preserving the system prompt and the most recent exchanges.
pub const MAX_PROMPT_CHARS: usize = 3_500_000;

use std::sync::OnceLock;

fn env_flag(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            let v = v.to_lowercase();
            v != "false" && v != "0" && v != "no"
        })
        .unwrap_or(default)
}

/// STRICT_TOOL_VALIDATION (default true): when true, unknown tool names cause hard 400
/// unless TOOL_PASS_THROUGH or TOOL_SYNTHETIC_OK is enabled.
pub fn strict_tool_validation() -> bool {
    static STRICT: OnceLock<bool> = OnceLock::new();
    *STRICT.get_or_init(|| env_flag("STRICT_TOOL_VALIDATION", true))
}

/// TOOL_PASS_THROUGH (default true): emit tool calls to the client even when the name
/// is not in the request's tools list — fixes agent clients where Qwen invents a name
/// the runtime actually has ("Tool X does not exist" from over-strict proxy blocking).
/// Set TOOL_PASS_THROUGH=false to restore strict blocking of unknown names.
pub fn tool_pass_through() -> bool {
    static PASS: OnceLock<bool> = OnceLock::new();
    *PASS.get_or_init(|| env_flag("TOOL_PASS_THROUGH", true))
}

/// TOOL_SYNTHETIC_OK (default false): for unknown tool names, inject a fake successful
/// TOOL RESULT back into the Qwen chat (pretend the tool ran) instead of erroring.
/// Combine with TOOL_PASS_THROUGH to also forward the call to the client.
pub fn tool_synthetic_ok() -> bool {
    static SYN: OnceLock<bool> = OnceLock::new();
    *SYN.get_or_init(|| env_flag("TOOL_SYNTHETIC_OK", false))
}

/// Upstream Qwen model for chat/completions API (not the OpenAI-facing echo string).
/// Override with QWEN_MODEL (e.g. qwen3.7-max-preview). Client aliases like qwen3.6-plus
/// are upgraded to 3.7-max unless QWEN_MODEL is set.
pub fn qwen_upstream_model(client_requested: Option<&str>) -> String {
    static ENV_MODEL: OnceLock<Option<String>> = OnceLock::new();
    if let Some(m) =
        ENV_MODEL.get_or_init(|| std::env::var("QWEN_MODEL").ok().filter(|s| !s.is_empty()))
    {
        return m.clone();
    }
    match client_requested {
        Some("qwen3.7-max-preview") | Some("qwen3.7-max") => client_requested.unwrap().to_string(),
        Some("qwen3.6-plus") | Some("qwen3.6-max-preview") => {
            tracing::info!(
                client_model = client_requested.unwrap(),
                upstream = MODEL_NAME,
                "Upgrading client qwen3.6* alias to upstream qwen3.7-max"
            );
            MODEL_NAME.to_string()
        }
        Some("gpt-4") | Some("gpt-4o") | Some("gpt-3.5-turbo") => MODEL_NAME.to_string(),
        Some(other) if other.starts_with("qwen3.7") => other.to_string(),
        Some(other) => {
            tracing::warn!(
                client_model = other,
                upstream = MODEL_NAME,
                "Unknown client model; using upstream default"
            );
            MODEL_NAME.to_string()
        }
        None => MODEL_NAME.to_string(),
    }
}
