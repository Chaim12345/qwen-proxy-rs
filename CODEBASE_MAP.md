# Codebase Map & Architectural Analysis: Qwen-Proxy

An in-depth, structured documentation of the `qwen-proxy` architecture, modules, data flow, state management, and compatibility layers.

---

## 1. Architectural Overview

`qwen-proxy` is a high-performance Rust-based proxy server designed to bridge the gap between stateless **OpenAI-compatible Clients** (such as those using the OpenAI Agents SDK) and the stateful, web-session-based **Qwen Chat Web API (v2)**.

The primary architectural challenge solved by this proxy is translating a stateless request (which sends a complete, raw list of messages and expects a single, direct response) into Qwen’s stateful chat system. In the upstream Qwen API:
1. Every conversation requires a persistent `chat_id` created via `/chats/new`.
2. Each message turn is linked to its preceding turn via a stateful `parent_id` (representing the upstream response ID of the previous assistant turn).
3. Concurrent requests to the same `chat_id` fail with a `429 chat in progress` error.
4. Tool execution is not natively supported in the web-channel API, requiring prompt engineering and post-generation text scanning.

The proxy manages HTTP routing, session caching, locking, message reconstruction, regex-based response normalization, brace-matching JSON parsing, and Server-Sent Events (SSE) translation.

```
┌────────────────────────────────────────┐
│          OpenAI Agents SDK /           │
│        OpenAI-compatible Clients       │
└───────────────────┬────────────────────┘
                    │
                    │ HTTP (Chat Completions / Responses / Models / Embeddings)
                    ▼
┌────────────────────────────────────────┐
│           Axum Routing Engine          │
│            (src/main.rs)               │
└───────────────────┬────────────────────┘
                    │
                    │ Orchestration & Session Matching
                    ▼
┌────────────────────────────────────────┐
│         Session Manager (DashMap)      │◄───► [ Mutex / Session Locks ]
│           (src/session.rs)             │      (Prevents concurrent Qwen writes)
└───────────────────┬────────────────────┘
                    │
                    │ State & Context Setup
                    ▼
┌────────────────────────────────────────┐
│           Qwen Payload Builder         │
│             (src/qwen.rs)              │
└───────────────────┬────────────────────┘
                    │
                    │ Normalized Prompt + Parent Message IDs
                    ▼
┌────────────────────────────────────────┐
│           reqwest Client Pool          │
└───────────────────┬────────────────────┘
                    │
                    │ HTTP SSE (https://chat.qwen.ai/api/v2)
                    ▼
┌────────────────────────────────────────┐
│            Qwen Upstream API           │
└────────────────────────────────────────┘
```

---

## 2. Directory & File Structure

The project maintains a compact, clean Rust layout along with a TypeScript integration test suite:

```
├── Cargo.toml                  # Cargo dependencies and dev configurations
├── src/
│   ├── main.rs                 # Web server entry point, Axum routes, request handling
│   ├── session.rs              # Upstream Qwen chat session caching & concurrency control
│   └── qwen.rs                 # Text normalization, prompt construction, & tool call parsing
└── tests/
    └── agents_sdk_test.ts      # Bun integration tests validating Agents SDK scenarios
```

---

## 3. Detailed Component Deep-Dive

### A. Core HTTP Engine: `src/main.rs`
This file serves as the main entry point and request coordinator. It initializes tracing, parses environment configurations, and boots the Axum web server on port `8765`.

*   **Key Responsibilities**:
    *   **Environment Handling**: Pulls `QWEN_TOKEN` from the environment or searches the home directory for `~/.qwen_session.json` to extract the session token cookie.
    *   **Axum Server Setup**: Configures routes for `/health`, `/v1/models`, `/v1/models/:model`, `/v1/embeddings`, `/v1/chat/completions`, and `/v1/responses`.
    *   **Session Matching (`client_session_key`)**: Extracts unique identifiers (such as the `user` field, metadata properties like `session_id`, or a SHA-like hash of the first user message) to map stateless client requests onto the same stateful Qwen session.
    *   **Upstream Client Configuration**: Sets default headers mimicking modern browsers (`Mozilla/5.0...`), referral origins, and cookie-based authorization headers.
    *   **Request & Response Streaming Engine**: Feeds incoming payloads into the session management layers, intercepts upstream chunks, and dynamically formats downstream Server-Sent Events (SSE) or full-text JSON responses.

### B. Session & Concurrency Layer: `src/session.rs`
The session layer addresses Qwen's strict serialization requirements and manages chat creation lifetimes.

*   **Key Responsibilities**:
    *   **Session Caching**: Uses a thread-safe `DashMap<String, SessionEntry>` to cache session entries mapped to client keys.
    *   **Concurrency Serialization**: Employs an in-flight Mutex (`Arc<Mutex<()>>`). If multiple requests target the same client session concurrently, the Mutex forces sequential execution, preventing upstream `429 chat in progress` rejections.
    *   **Parent-Child Tracking (`parent_store`)**: Maintains a thread-safe reference (`Arc<Mutex<Option<String>>>`) storing the upstream `response_id` of the previous assistant message. Each subsequent turn utilizes this value as the `parent_id` in the next request's payload, preserving Qwen's history linkage.
    *   **Lifecycle Management**: Enforces a Session TTL (Time-To-Live) of 30 minutes and maintains a maximum session size of 100 via an LRU eviction strategy (`evict_oldest`).
    *   **Upstream Chat Provisioning (`create_chat`)**: Submits a POST request to Qwen's `/chats/new` to retrieve a fresh upstream `chat_id` when no session exists or a session has expired.

### C. Translation & Normalization Layer: `src/qwen.rs`
This module acts as the core compiler translating between the OpenAI-formatted schema structures and Qwen's plain-text prompt interfaces.

*   **Key Responsibilities**:
    *   **Prompt Construction (`build_message`)**: Iterates through the list of system, user, assistant, and tool messages to build a unified plain-text script. 
        *   Maps different roles into visual labels (e.g. `USER: ...`, `ASSISTANT: ...`).
        *   Injects system instructions.
        *   Injects a list of formatted available functions if tools are declared.
        *   Appends structured output commands (e.g., schemas for `json_schema` or formatting restrictions for `json_object`).
        *   Appends strict structural JSON instructions when tools are present, instructing Qwen to output tool calls on a single line matching `{"tool":"...","args":{...}}`.
    *   **Response Normalization (`normalize_tool_call_text`)**: Strip UI artifacts such as Braille spinner animation sequences (`SPINNER_RE`) or text prefixes like `"Thinking..."` that may contaminate the raw text output.
    *   **Tool Call Parsing & Detection (`detect_tool`)**: Implements multiple scanning strategies to locate and extract tool calls from Qwen's textual output:
        1.  *Markdown JSON Captures*: Scans for code blocks containing JSON payload templates using regex (`MARKDOWN_CODE_RE`).
        2.  *Brace-Counting Parser (`find_json_object_end`)*: Iterates through the text character-by-character, keeping track of nesting brackets, escaped quotes, and double-quoted strings. This safely isolates complete JSON sub-objects even when they contain nested braces.
        3.  *Line-by-Line Regex Scanning*: Scans individual lines for tool/argument configurations as a final fallback.
    *   **SSE Chunk Extraction (`extract_qwen_sse_delta`)**: Parses upstream Qwen event-stream chunks, discarding internal processing phases (such as `thinking_summary`, `thinking`, or web search) and passing only the raw user-visible response text.

---

## 4. End-to-End Data Flow

The following describes the exact sequence of operations for an incoming chat completion request:

```
[Client Request] 
      │
      ▼
1. Extract Stable Session Key (User / Meta / Hash)
      │
      ▼
2. DashMap Session Lookup (Cleanup & Evict if required)
      │
      ├──► [Session Missing/Expired] ──► Call Qwen /chats/new ──┐
      │                                                         │
      ▼                                                         ▼
3. Acquire Mutex Guard & Fetch Parent ID ◄──────────────────────┘
      │
      ▼
4. Prompt Engineering: Compile full history (build_message)
      │
      ▼
5. Send HTTP POST to Qwen /chat/completions (text/event-stream)
      │
      ▼
6. Stream Processing (SSE)
      │
      ├─► Extract and update Parent ID (parent_store)
      │
      ├─► Strip Thinking / Search / Spinner phase events
      │
      ├─► Accumulate stream delta content
      │
      ├─► Evaluate Tool Call Presence (detect_tool)
      │         │
      │         ├─► [Yes] ─► Generate artificial tool call chunks
      │         │
      │         └─► [No]  ─► Pass visible content downstream
      │
      ▼
7. Drop Mutex Guard (Release session for next concurrent turn)
```

---

## 5. Compatibility & Translation Mechanics

To deliver an authentic OpenAI API experience, `qwen-proxy` translates several incompatible API paradigms:

| OpenAI API Paradigm | Qwen Web API Constraint | Proxy Compatibility Bridge |
| :--- | :--- | :--- |
| **Stateless Turns** | Stateful Conversation Thread | Caches `chat_id` inside `DashMap` and associates stateless requests with a shared session. |
| **Structured Tools / Functions** | No native tool configuration | Prompts Qwen with tool definitions and instructs it to output a custom JSON scheme. Scans text patterns via regex & brace-counting parser to reconstruct OpenAI `tool_calls` payloads. |
| **Message History** | Expects upstream chat engine to hold context | Combines system, user, assistant and tool-call-output roles into a single unified prompt layout (`build_message`) fed to the Qwen Web UI on each turn. |
| **Parent/Child Linkage** | Requires sequential `parent_id` linking | Caches the upstream response ID in `parent_store` and submits it as `parent_id` on the subsequent request. |
| **Thinking Process** | Exposes thinking phases as raw tokens | Intercepts streaming events, skipping `thinking_summary` and `search` chunks to keep the downstream client response clean and responsive. |
| **Responses API** | Traditional chat completions only | Supports `/v1/responses` by detecting request parameters, rewriting the request, and mapping output models to `response` objects instead of standard `chat.completion` layouts. |

---

## 6. Integration Test Suite Analysis

The test suite in `tests/agents_sdk_test.ts` executes **11 comprehensive integration test cases** to prove compatibility with the OpenAI Agents SDK:

1.  **Agent-style single turn with system prompt**: Validates that system-role prompts are respected and the standard completion response ends with a `"stop"` finish reason.
2.  **Agent loop (tool call → result → response)**: Simulates a complete agent turn where the agent issues a `calculator` tool call, receives a dummy evaluation payload from the client, and uses it to construct a final mathematical response.
3.  **Agent with multiple tool choices**: Confirms the proxy's capability to select the appropriate function from multiple definitions (`read_file`, `write_file`, `list_files`).
4.  **Agent streaming with tool definitions**: Validates that event-stream chunks flow smoothly and carry valid JSON when tools are defined in the payload.
5.  **Agent handoff pattern (simulated)**: Verifies that the agent correctly triggers specialized routing tools (like `transfer_to_specialist`) during triage operations.
6.  **Agent input guardrails via system prompt**: Confirms the agent adheres to strict system prompt guardrails (such as returning a specific message when unrelated queries are received).
7.  **Agent output format enforcement**: Asserts that raw structured outputs (e.g. JSON templates requested via system directives) are returned cleanly.
8.  **Agent handles long conversation history**: Verifies context retention by testing multi-turn logic spanning 10 conversational turns and confirming the agent remembers the initial query.
9.  **Agent parallel tool calls**: Verifies robust handling of queries prompting multiple tools (e.g. requesting both stock prices and company details for a financial symbol).
10. **Responses API compatibility (`/v1/responses`)**: Validates that the proxy correctly processes the unique request-reply schemas demanded by advanced SDK endpoints.
11. **Agent tracing/metadata**: Asserts that downstream responses strictly mirror OpenAI metadata standards, ensuring fields like `id`, `object`, `created`, `model`, `choices`, and detailed token counts (`usage`) are returned correctly.

---

## 7. Cargo Dependencies & Technical Stack

The library stack chosen in `Cargo.toml` underscores the project's focus on safety, speed, and standard compliance:

*   **`axum`**: A fast, asynchronous HTTP routing engine built on top of `hyper` and `tower`.
*   **`tokio`**: The industry-standard async runtime supplying system task loops and file-system I/O.
*   **`reqwest`**: An asynchronous HTTP client featuring connection pooling for high-performance requests.
*   **`dashmap`**: A concurrently-accessible, shard-locked map providing thread-safe session caching without global locking bottlenecks.
*   **`serde` & `serde_json`**: Clean JSON serialization/deserialization used heavily for payload manipulation.
*   **`zstd`**: Integrates compression capabilities supporting high-throughput messaging.
*   **`eventsource-stream` & `async-stream`**: Provides streaming generators that compile upstream SSE payloads into dynamic, real-time downstream events.
*   **`tiktoken-rs`**: Leveraged as a foundation for reliable token counting.
*   **`regex`**: Fast pattern matching used to sanitize thinking/spinner frames and captured JSON structures.
