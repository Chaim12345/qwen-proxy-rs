# Qwen OpenAI API Proxy

A high-performance Rust proxy that exposes the Qwen AI API as an **OpenAI-compatible REST API** for chat, models, and a Responses API subset. Validated against the OpenAI Node SDK, Vercel AI SDK, OpenAI Agents SDK, Pi (`@earendil-works/pi-ai`), and raw HTTP clients.

## Features

- ✅ **OpenAI-compatible chat & models** — drop-in for clients that use `/v1/chat/completions` and `/v1/models`
- ✅ **SSE streaming** — emits `chat.completion.chunk` events and a `[DONE]` sentinel (upstream response is buffered before chunk emission)
- ✅ **Tool/Function Calling** — detects tool calls in Qwen responses and formats them as OpenAI tool calls
- ✅ **Models endpoint** — `/v1/models` and `/v1/models/:id` with GPT model aliases
- ✅ **Embeddings stub** — returns zero-vectors for API compatibility
- ✅ **Session pooling** — automatic Qwen chat session management with 30min TTL
- ✅ **CORS support** — works from browser-based clients
- ✅ **Zstd decompression** — handles compressed request bodies
- ✅ **Graceful shutdown** — handles Ctrl+C and SIGTERM

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/models` | GET | List available models |
| `/v1/models/:id` | GET | Get model info (accepts `gpt-4`, `gpt-4o`, `gpt-3.5-turbo` aliases) |
| `/v1/chat/completions` | POST | Chat completions (streaming + non-streaming) |
| `/v1/responses` | POST | Responses API (alias for chat/completions) |
| `/v1/embeddings` | POST | Embeddings (stub — returns zero vectors) |

## Quick Start

### 1. Build

```bash
cd qwen-proxy-rs
cargo build --release
```

### 2. Set Authentication

```bash
# Option A: Environment variable
export QWEN_TOKEN="your-qwen-token"

# Option B: Session file
echo '{"token": "your-qwen-token"}' > ~/.qwen_session.json
```

### 3. Run

```bash
# Default port 8765
cargo run --release

# Custom port
PORT=3000 cargo run --release
```

### 4. Use with OpenAI SDK

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://127.0.0.1:8765/v1",
  apiKey: "any-value", // Not used, but required by SDK
});

// Non-streaming
const response = await client.chat.completions.create({
  model: "qwen3.6-plus",
  messages: [{ role: "user", content: "Hello!" }],
});
console.log(response.choices[0].message.content);

// Streaming
const stream = await client.chat.completions.create({
  model: "qwen3.6-plus",
  messages: [{ role: "user", content: "Count to 5." }],
  stream: true,
});
for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content || "");
}
```

## Running Tests

Start the proxy first (`cargo run --release` from this directory). Install Node/Bun deps in the parent `qtalt/` directory.

### Full compatibility gate (recommended)

```bash
cd qwen-proxy-rs
chmod +x tests/run_compat.sh
./tests/run_compat.sh
```

Runs unit tests, HTTP contract tests, OpenAI SDK, AI SDK, Agents SDK, tool-call stress, live streaming, Pi/OpenCode smoke tests, and the HTTP contract checklist sequentially.

Options:

- `PROXY_URL=http://127.0.0.1:8765/v1 ./tests/run_compat.sh` — custom base URL
- `./tests/run_compat.sh --skip-perf` — skip live streaming performance suite

### Individual suites

```bash
# From qwen-proxy-rs/
cargo test
cargo test --test openai_compat -- --ignored

# From qtalt/ (needs node_modules)
bun run qwen-proxy-rs/tests/openai_sdk_compat_test.ts
node qwen-proxy-rs/tests/ai_sdk_test.mjs
bun run qwen-proxy-rs/tests/agents_sdk_test.ts
node qwen-proxy-rs/tests/toolcall_stress_test.mjs
bun run qwen-proxy-rs/tests/live_agents_streaming_test.ts
bun run qwen-proxy-rs/tests/pi_opencode_compat_test.ts
node qwen-proxy-rs/tests/contract_checklist_test.mjs
```

## Test Coverage

### Rust Tests (11 tests)
- Tool call detection (simple, markdown, embedded, unknown)
- Message building with tools
- Response shape validation
- Model info format
- Error format validation
- Token estimation

### OpenAI SDK Tests (30+ tests)
- Models endpoint (list, retrieve, aliases, 404)
- Basic chat completions
- System message following
- Multi-turn conversations
- SSE streaming (chunk format, role delta, finish_reason)
- Tool call detection and round-trip
- Error handling (400, invalid JSON, error format)
- Health endpoint
- CORS headers
- Content-Type headers
- Embeddings stub
- Edge cases (long input, unicode, empty content, multiple system messages)
- SDK parameters (max_tokens, stop, user, seed, frequency_penalty, presence_penalty, top_p, response_format)

### Agents SDK Tests (11 tests)
- Agent-style single turn
- Agent loop (tool call → result → response)
- Multiple tool choices
- Streaming with tool definitions
- Agent handoff pattern
- Input/output guardrails
- Long conversation history
- Parallel tool calls
- Responses API compatibility
- Tracing metadata
- Structured output with JSON schema

## Architecture

```
Client (OpenAI SDK) 
    ↓ HTTP POST /v1/chat/completions
Rust Proxy (Axum)
    ↓ SSE stream
Qwen API (chat.qwen.ai)
```

### Key Components

- **`main.rs`** — HTTP server, routing, request handling, SSE streaming
- **`qwen.rs`** — Qwen API payload building, message conversion, tool call detection
- **Session Pool** — Manages Qwen chat sessions with automatic cleanup

### Streaming Flow

1. Client sends `stream: true` in request
2. Proxy opens SSE connection to Qwen API
3. Each Qwen SSE chunk is converted to OpenAI `chat.completion.chunk` format
4. Chunks are streamed to client in real-time
5. Final chunk includes `finish_reason: "stop"` or `finish_reason: "tool_calls"`
6. `[DONE]` sentinel is sent to close the stream

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `QWEN_TOKEN` | — | Qwen authentication token |
| `PORT` | `8765` | Server listen port |
| `RUST_LOG` | `qwen_proxy=info` | Log level |

## Compatibility Matrix

### Supported (validated)

| Surface | Status | Notes |
|---------|--------|-------|
| `POST /v1/chat/completions` | ✅ | Stream + non-stream |
| Tool / function calling | ✅ | Prompt-simulated; parsed from model text |
| `GET /v1/models`, `GET /v1/models/:id` | ✅ | Includes `gpt-4`, `gpt-4o`, `gpt-3.5-turbo` aliases |
| `POST /v1/responses` | ✅ | Subset mapped to chat completions |
| `POST /v1/embeddings` | ⚠️ Stub | Correct JSON shape; zero vectors only |
| `GET /health` | ✅ | Health check |
| CORS | ✅ | Permissive for local dev |
| OpenAI Node SDK | ✅ | See `openai_sdk_compat_test.ts` |
| Vercel AI SDK | ✅ | See `ai_sdk_test.mjs` |
| OpenAI Agents SDK | ✅ | See `agents_sdk_test.ts` |
| Pi / OpenCode (`@earendil-works/pi-ai`) | ✅ | `openai-completions` + `openai-responses` |
| Raw HTTP / curl / fetch | ✅ | See `contract_checklist_test.mjs` |

### Not supported

| Surface | Status | Notes |
|---------|--------|-------|
| Assistants, Threads, Runs | ❌ | Not implemented |
| Files, Images, Audio | ❌ | Not implemented |
| Batches, Fine-tuning | ❌ | Not implemented |
| Real embedding vectors | ❌ | Stub returns zeros |
| Native OpenAI tool API | ❌ | Tools are prompt-engineered, not upstream-native |
| `@openai/codex-sdk` OAuth | ❌ | Codex cloud auth; use a custom OpenAI base URL instead |
| True token-by-token upstream streaming | ❌ | Proxy buffers upstream before emitting SSE chunks |

### Client quick reference

| Client | Status | Notes |
|--------|--------|-------|
| OpenAI Node / Python SDK | ✅ | Point `baseURL` at `http://127.0.0.1:8765/v1` |
| LangChain / LiteLLM | ✅ | OpenAI-compatible endpoint |
| Browser (fetch) | ✅ | CORS enabled |
| Pi TUI / OpenCode | ✅ | Custom model `baseUrl` to proxy `/v1` |

## License

MIT
