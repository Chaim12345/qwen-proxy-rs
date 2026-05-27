# Fix Document — qwen-proxy-rs Codebase Review

Generated: 2026-05-27

## Completion Status

| Task | Status | Notes |
|------|--------|-------|
| F1 | ✅ DONE | StreamState struct, dead `buf` removed, all return sites converted |
| F2 | ✅ DONE | `tool_error_recovery` helper extracted, unified fallback format |
| F3 | ✅ DONE | Narrowed patterns, removed `抱歉`/`sorry`, 300-char guard, F10 tests added |
| F4 | ✅ DONE | `SseLine.request_id` and `QwenSseStream.request_id` removed |
| F5 | ✅ DONE | "smol+hyper" → "tokio+hyper" in root handler |
| F6 | ✅ DONE | Embeddings returns 501 Not Implemented |
| F7 | ✅ DONE | `.github/workflows/build.yml` deleted |
| F8 | ✅ DONE | `Arc<String>` for `chat_id_for_fb`/`token_for_fb`, deref in spawn calls |
| F9 | ✅ DONE | 15 unit tests for `client_session_key` branches in `#[cfg(test)]` mod |
| F10 | ✅ DONE | 9 unit tests for `detect_qwen_tool_error` (true/false/edge cases) |
| Context limit | ✅ DONE | `truncate_for_context_limit` drops oldest messages, keeps system + recent |
| clippy/fmt | ✅ DONE | `cargo fmt` applied, clippy validated on GA CI |

## Task List

### Context Limit — Handle clients sending history exceeding Qwen context (983K tokens)  [HIGH]
- **File**: `src/main.rs:632-652`, `src/qwen.rs:650-731`
- **Problem**: Clients like opencode send entire conversation history in a single request. Qwen3.7-Max rejects with `InternalError.Algo.InvalidParameter: Range of input length should be [1, 983616]`.
- **Fix**:
  1. Added `MAX_PROMPT_CHARS = 3_500_000` constant (≈ 900K tokens with safety margin)
  2. `truncate_for_context_limit(&mut Value, max_chars)` in `qwen.rs`: preserves all system messages at the start and the last 20 exchanges; removes oldest messages from the middle iteratively until the serialized JSON fits within `target_chars = max_chars * 0.9`
  3. Called in `main.rs` after message parsing but before `build_message`, with a `warn!` log when truncation occurs
  4. If truncation results in zero messages, returns 400 Bad Request
- **Risk**: Low — only activates for oversized requests; conservative 3.5M char limit gives ~900K token headroom

### F1 — Replace 12-tuple unfold state with named struct  [HIGH]
- **File**: `src/main.rs:719-721`
- **Problem**: 12-element tuple is unreadable, position-dependent, and `buf` (index 1) is dead — always `String::new()`, never read or written.
- **Fix**:
  1. Define `struct StreamState { rx, full_text, tool_emitted, content_emitted, done, prev_len, prev_thinking_len, thinking_role_sent, resp_created, output_started, item_id }`
  2. Remove dead `buf` field
  3. Replace all tuple destructuring with field access
  4. Replace all tuple reconstruction with `StreamState { .. }`
- **Risk**: Mechanical refactor; all 10+ return sites in the unfold must be updated.

### F2 — Unify tool-error recovery paths (stream vs non-stream)  [HIGH]
- **File**: `src/main.rs:937-1044` (stream), `main.rs:1393-1467` (non-stream)
- **Problem**:
  - Stream path recovers even when `pid == None` (uses `parent_store.lock()`); non-stream skips entirely.
  - Stream path uses `*parent_store.lock().await = Some(pid)`; non-stream uses `session.set_parent_id()`.
  - Fallback error format differs (stream: `[Tool Error: ...]`; non-stream: OpenAI JSON with `available_tools`).
- **Fix**:
  1. Extract a shared `async fn tool_error_recovery(chat_id, parent_id, token, session, parent_store) -> Result<ContinuationResponse, ()>` helper.
  2. Both paths call the same helper with consistent `parent_id` resolution.
  3. Fallback format: always emit OpenAI JSON with `available_tools` (both stream and non-stream).
  4. Stream path uses `session.set_parent_id()` via an async lock instead of direct mutation.

### F3 — Narrow `detect_qwen_tool_error` to reduce false positives  [HIGH]
- **File**: `src/qwen.rs:184-210`
- **Problem**: Substring matches on `"tool"`, `"cannot use"`, `"sorry"` are too broad. E.g. "I'm sorry, I cannot use informal language" triggers a false positive.
- **Fix**:
  1. Require the pattern to start with `"Tool "` (capital T + space) for the primary check.
  2. For the `"cannot use"` / `"can't use"` / `"unable to use"` patterns, require the exact phrase `"tool"` to appear within 30 chars.
  3. Remove the `抱歉` / `sorry` heuristic entirely (too fragile, and Qwen tool errors are in English).
  4. Add a max-length guard of 300 chars (tool error messages are short).
  5. Add unit tests for false-positive inputs.

### F4 — Remove dead `SseLine.request_id` allocation  [LOW]
- **File**: `src/streaming.rs:15-19`
- **Problem**: `request_id` field is `Arc<str>` allocated per SSE line but never read outside the struct.
- **Fix**: Remove `request_id` field from `SseLine`, remove the `#[allow(dead_code)]`, update `QwenSseStream` to not store `request_id`.

### F5 — Fix stale "smol+hyper" in root handler  [LOW]
- **File**: `src/main.rs:1727`
- **Problem**: Returns `"Qwen OpenAI Proxy (smol+hyper)"` but runtime is tokio+hyper.
- **Fix**: Change to `"Qwen OpenAI Proxy (tokio+hyper)"`.

### F6 — Fix embeddings handler (all-zeros)  [CRITICAL]
- **File**: `src/main.rs:1658`
- **Problem**: Returns `vec![0.0; dims]` — breaks any client using embeddings for similarity.
- **Fix**: Return 501 Not Implemented with a clear message. The proxy doesn't have an embedding model.
  ```rust
  return openai_error_response(
      StatusCode::NOT_IMPLEMENTED,
      "Embeddings are not supported by this proxy",
      "not_supported",
      None,
      None,
  );
  ```

### F7 — Remove duplicate CI workflow  [MEDIUM]
- **File**: `.github/workflows/build.yml`
- **Problem**: `build.yml` is a strict subset of `ci.yml`'s `build-release` job. Every push runs 2x redundant builds.
- **Fix**: Delete `.github/workflows/build.yml`.

### F8 — Eliminate per-step `clone()` of static strings in unfold  [LOW]
- **File**: `src/main.rs:725-729`
- **Problem**: `completion_id`, `model`, `chat_id_for_fb`, `token_for_fb` are cloned on every unfold iteration (every SSE line).
- **Fix**: Wrap them in `Arc<str>` or `Arc<String>` once, then `Arc::clone` (cheap pointer bump) inside the unfold.

### F9 — Add unit tests for `client_session_key`  [MEDIUM]
- **File**: `src/main.rs:484-533`
- **Problem**: Complex branching logic with zero test coverage.
- **Fix**: Add tests for: `user` field, `metadata.session_id`, first-user-message hash, tools suffix, ephemeral fallback.

### F10 — Add unit tests for `detect_qwen_tool_error`  [MEDIUM]
- **File**: `src/qwen.rs:184-210` (tests section)
- **Problem**: No tests for true positives, true negatives, or false-positive guard.
- **Fix**: Add tests covering: "Tool X does not exist", "Tool Y does not exists", false positives ("sorry, I cannot use..."), short messages, Chinese text.

## Execution Order

1. **Context limit** (new 2026-05-27) — handle clients sending > 983K tokens; truncate from middle
2. **F1** (struct refactor) — must go first since F2 and F8 touch the same code
3. **F3** (narrow detect) — independent, safe to do early
4. **F6** (embeddings 501) — one-line fix, do early
5. **F5** (stale string) — one-line fix
6. **F4** (dead SseLine field) — independent
7. **F2** (unify recovery) — depends on F1 being done
8. **F8** (avoid per-step clone) — depends on F1 being done
9. **F7** (delete duplicate CI) — independent
10. **F9** + **F10** (tests) — can be done in parallel at the end
11. `cargo clippy --all-targets -- -D warnings` + `cargo fmt --check` — final gate
