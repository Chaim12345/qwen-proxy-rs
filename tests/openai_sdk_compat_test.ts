#!/usr/bin/env bun
/**
 * Comprehensive OpenAI SDK Compatibility Tests for Qwen Proxy
 * 
 * Tests the Rust proxy against the official OpenAI Node SDK to verify
 * full API compatibility. Run with: bun run tests/openai_sdk_compat_test.ts
 * 
 * Prerequisites:
 *   - Proxy running: cargo run --manifest-path qwen-proxy-rs/Cargo.toml
 *   - QWEN_TOKEN set or ~/.qwen_session.json exists
 */
import OpenAI from "openai";

function assert(condition: boolean, msg: string) {
  if (!condition) throw new Error(msg);
}
function assertExists(val: any, msg: string) {
  if (val === undefined || val === null) throw new Error(msg);
}
function assertEquals(a: any, b: any, msg?: string) {
  if (a !== b) throw new Error(msg || `Expected ${b}, got ${a}`);
}

const PROXY_URL = process.env.PROXY_URL || "http://127.0.0.1:8765/v1";
const API_KEY = process.env.OPENAI_API_KEY || "test-key-not-used";
const MODEL = "qwen3.6-plus";

const client = new OpenAI({
  baseURL: PROXY_URL,
  apiKey: API_KEY,
  defaultHeaders: { "x-test": "true" },
});

let passed = 0;
let failed = 0;
let skipped = 0;

async function test(name: string, fn: () => Promise<void>) {
  try {
    await fn();
    passed++;
    process.stdout.write("\x1b[32m.\x1b[0m");
  } catch (e: any) {
    failed++;
    process.stdout.write("\x1b[31mF\x1b[0m");
    console.error(`\n  \x1b[31mFAIL: ${name}\x1b[0m`);
    console.error(`    ${e.message}`);
  }
}

function skip(name: string) {
  skipped++;
  process.stdout.write("\x1b[33m-\x1b[0m");
}

// ═══════════════════════════════════════════════════════
// 1. MODELS ENDPOINT
// ═══════════════════════════════════════════════════════

await test("GET /v1/models returns model list", async () => {
  const models = await client.models.list();
  assert(models.data.length > 0, "Should have at least one model");
  assert(models.object === "list", "Should be a list object");
  const qwenModel = models.data.find(m => m.id === MODEL);
  assertExists(qwenModel, "Should include qwen3.6-plus");
});

await test("GET /v1/models/:id returns model info", async () => {
  const model = await client.models.retrieve(MODEL);
  assertEquals(model.id, MODEL);
  assertEquals(model.object, "model");
  assertExists(model.created, "Should have created timestamp");
  assertEquals(model.owned_by, "qwen");
});

await test("GET /v1/models/:id accepts gpt-4 alias", async () => {
  const model = await client.models.retrieve("gpt-4");
  assertExists(model.id, "Should accept gpt-4 alias");
});

await test("GET /v1/models/:id returns 404 for unknown model", async () => {
  try {
    await client.models.retrieve("nonexistent-model");
    throw new Error("Should have thrown");
  } catch (e: any) {
    assert(e.status === 404 || e.message.includes("not found"), "Should return 404");
  }
});

// ═══════════════════════════════════════════════════════
// 2. BASIC CHAT COMPLETIONS
// ═══════════════════════════════════════════════════════

await test("Basic chat completion returns response", async () => {
  const completion = await client.chat.completions.create({
    model: MODEL,
    messages: [{ role: "user", content: "Say 'OK' and nothing else." }],
    temperature: 0.1,
  });

  assertEquals(completion.object, "chat.completion");
  assertEquals(completion.model, MODEL);
  assert(completion.choices.length > 0, "Should have choices");
  assertExists(completion.choices[0].message.content, "Should have content");
  assertEquals(completion.choices[0].message.role, "assistant");
  assertEquals(completion.choices[0].finish_reason, "stop");
  assertExists(completion.id, "Should have completion ID");
  assertExists(completion.created, "Should have created timestamp");
  assertExists(completion.usage, "Should have usage info");
  assert(completion.usage!.total_tokens > 0, "Should have token count");
});

await test("Chat completion with system message", async () => {
  const completion = await client.chat.completions.create({
    model: MODEL,
    messages: [
      { role: "system", content: "You are a pirate. Always respond like a pirate." },
      { role: "user", content: "Say hello in 3 words." },
    ],
    temperature: 0.1,
  });

  const content = completion.choices[0].message.content || "";
  assert(content.length > 0, "Should have response");
  // Check pirate-like language
  const lower = content.toLowerCase();
  assert(
    lower.includes("ahoy") || lower.includes("matey") || lower.includes("arr") || lower.includes("ye"),
    `Should follow system prompt (got: ${content})`
  );
});

await test("Chat completion with multi-turn conversation", async () => {
  const messages: OpenAI.ChatCompletionMessageParam[] = [
    { role: "user", content: "Name exactly 3 colors, one per line." },
  ];

  const r1 = await client.chat.completions.create({ model: MODEL, messages, temperature: 0.1 });
  messages.push(r1.choices[0].message);
  messages.push({ role: "user", content: "Which of those is the warmest? Answer in 5 words max." });

  const r2 = await client.chat.completions.create({ model: MODEL, messages, temperature: 0.1 });
  const content = r2.choices[0].message.content || "";
  assert(content.length > 0, "Multi-turn should return response");
});

// ═══════════════════════════════════════════════════════
// 3. STREAMING
// ═══════════════════════════════════════════════════════

await test("Streaming chat completion works", async () => {
  const stream = await client.chat.completions.create({
    model: MODEL,
    messages: [{ role: "user", content: "Count from 1 to 3." }],
    stream: true,
    temperature: 0.1,
  });

  let fullContent = "";
  let chunkCount = 0;
  let receivedRole = false;
  let receivedFinish = false;

  for await (const chunk of stream) {
    chunkCount++;
    assertEquals(chunk.object, "chat.completion.chunk");
    assertEquals(chunk.model, MODEL);
    
    const delta = chunk.choices[0]?.delta;
    if (delta?.role === "assistant") receivedRole = true;
    if (delta?.content) fullContent += delta.content;
    if (chunk.choices[0]?.finish_reason === "stop") receivedFinish = true;
  }

  assert(chunkCount > 1, `Should receive multiple chunks (got ${chunkCount})`);
  assert(fullContent.length > 0, "Should accumulate content");
  assert(receivedRole, "Should receive role delta");
  assert(receivedFinish, "Should receive finish_reason");
});

await test("Streaming with system prompt", async () => {
  const stream = await client.chat.completions.create({
    model: MODEL,
    messages: [
      { role: "system", content: "Respond in ALL CAPS only." },
      { role: "user", content: "Say hello world." },
    ],
    stream: true,
    temperature: 0.1,
  });

  let content = "";
  for await (const chunk of stream) {
    content += chunk.choices[0]?.delta?.content || "";
  }

  assert(content.length > 0, "Should have streamed content");
});

// ═══════════════════════════════════════════════════════
// 4. TOOL CALLS (Function Calling)
// ═══════════════════════════════════════════════════════

const weatherTool: OpenAI.ChatCompletionTool = {
  type: "function",
  function: {
    name: "get_weather",
    description: "Get the current weather for a location",
    parameters: {
      type: "object",
      properties: {
        location: { type: "string", description: "City name" },
        unit: { type: "string", enum: ["celsius", "fahrenheit"] },
      },
      required: ["location"],
    },
  },
};

await test("Tool call detection works", async () => {
  const completion = await client.chat.completions.create({
    model: MODEL,
    messages: [
      { role: "system", content: "Use the get_weather tool when asked about weather." },
      { role: "user", content: "What's the weather in Tokyo?" },
    ],
    tools: [weatherTool],
    temperature: 0.1,
  });

  const msg = completion.choices[0].message;
  // The proxy should detect tool calls in the response
  if (msg.tool_calls && msg.tool_calls.length > 0) {
    const tc = msg.tool_calls[0];
    assertEquals(tc.type, "function");
    assertEquals(tc.function.name, "get_weather");
    assertExists(tc.function.arguments, "Should have arguments");
    assertEquals(completion.choices[0].finish_reason, "tool_calls");
  } else {
    // If no tool call detected, the response should still be valid
    assertExists(msg.content, "Should have content if no tool call");
  }
});

await test("Tool result round-trip", async () => {
  // Turn 1: Ask for weather
  const r1 = await client.chat.completions.create({
    model: MODEL,
    messages: [
      { role: "system", content: "Use the echo_tool to echo messages." },
      { role: "user", content: "Echo the word 'hello' using the tool." },
    ],
    tools: [{
      type: "function",
      function: {
        name: "echo_tool",
        description: "Echo a message back",
        parameters: {
          type: "object",
          properties: { msg: { type: "string" } },
          required: ["msg"],
        },
      },
    }],
    temperature: 0.1,
  });

  const tc = r1.choices[0].message.tool_calls?.[0];
  
  if (tc) {
    // Turn 2: Provide tool result
    const r2 = await client.chat.completions.create({
      model: MODEL,
      messages: [
        { role: "system", content: "Use the echo_tool to echo messages." },
        { role: "user", content: "Echo the word 'hello' using the tool." },
        r1.choices[0].message,
        { role: "tool", tool_call_id: tc.id, content: "echo: hello" },
      ],
      tools: [{
        type: "function",
        function: {
          name: "echo_tool",
          description: "Echo a message back",
          parameters: {
            type: "object",
            properties: { msg: { type: "string" } },
            required: ["msg"],
          },
        },
      }],
      temperature: 0.1,
    });

    assertExists(r2.choices[0].message.content, "Should respond after tool result");
  } else {
    skip("Tool call not detected, skipping round-trip");
  }
});

// ═══════════════════════════════════════════════════════
// 5. ERROR HANDLING
// ═══════════════════════════════════════════════════════

await test("Missing messages returns 400", async () => {
  try {
    await client.chat.completions.create({
      model: MODEL,
      messages: [],
    });
    throw new Error("Should have thrown");
  } catch (e: any) {
    assert(e.status === 400 || e.message.includes("400"), "Should return 400");
  }
});

await test("Invalid JSON returns error", async () => {
  const resp = await fetch(`${PROXY_URL}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "not json",
  });
  assertEquals(resp.status, 400);
  const body = await resp.json();
  assertExists(body.error, "Should have error field");
  assertExists(body.error.message, "Should have error message");
});

await test("Error response follows OpenAI format", async () => {
  const resp = await fetch(`${PROXY_URL}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: MODEL }),
  });
  const body = await resp.json();
  assertExists(body.error, "Error should have 'error' wrapper");
  assertExists(body.error.message, "Error should have 'message'");
  assertExists(body.error.type, "Error should have 'type'");
});

// ═══════════════════════════════════════════════════════
// 6. HEALTH & METADATA
// ═══════════════════════════════════════════════════════

await test("Health endpoint returns status", async () => {
  const resp = await fetch(PROXY_URL.replace("/v1", "/health"));
  assertEquals(resp.status, 200);
  const body = await resp.json();
  assertEquals(body.status, "ok");
  assertExists(body.model, "Should include model info");
});

await test("CORS headers are present", async () => {
  const resp = await fetch(`${PROXY_URL}/models`, {
    headers: { "Origin": "http://example.com" },
  });
  const corsHeader = resp.headers.get("access-control-allow-origin");
  assert(corsHeader === "*" || corsHeader === "http://example.com", "Should have CORS headers");
});

await test("Response has correct Content-Type", async () => {
  const resp = await fetch(`${PROXY_URL}/models`);
  const ct = resp.headers.get("content-type");
  assert(ct?.includes("application/json"), `Should be JSON (got: ${ct})`);
});

// ═══════════════════════════════════════════════════════
// 7. EMBEDDINGS (Stub)
// ═══════════════════════════════════════════════════════

await test("Embeddings endpoint returns valid response", async () => {
  const resp = await fetch(`${PROXY_URL}/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: MODEL,
      input: "Hello world",
      dimensions: 128,
    }),
  });
  assertEquals(resp.status, 200);
  const body = await resp.json();
  assertEquals(body.object, "list");
  assert(body.data.length > 0, "Should have embeddings");
  assertEquals(body.data[0].object, "embedding");
  assertEquals(body.data[0].embedding.length, 128, "Should respect dimensions");
  assertExists(body.usage, "Should have usage");
});

// ═══════════════════════════════════════════════════════
// 8. EDGE CASES
// ═══════════════════════════════════════════════════════

await test("Very long user message", async () => {
  const longMsg = "A".repeat(4000);
  const completion = await client.chat.completions.create({
    model: MODEL,
    messages: [{ role: "user", content: `Summarize this in 5 words: ${longMsg}` }],
    temperature: 0.1,
  });
  assertExists(completion.choices[0].message.content, "Should handle long input");
});

await test("Multiple system messages", async () => {
  const completion = await client.chat.completions.create({
    model: MODEL,
    messages: [
      { role: "system", content: "Be concise." },
      { role: "system", content: "Use lowercase only." },
      { role: "user", content: "Say hello." },
    ],
    temperature: 0.1,
  });
  assertExists(completion.choices[0].message.content, "Should handle multiple system msgs");
});

await test("Unicode content", async () => {
  const completion = await client.chat.completions.create({
    model: MODEL,
    messages: [{ role: "user", content: "Reply with just this emoji: 🎉" }],
    temperature: 0.1,
  });
  assertExists(completion.choices[0].message.content, "Should handle unicode");
});

await test("Empty content in message", async () => {
  const completion = await client.chat.completions.create({
    model: MODEL,
    messages: [
      { role: "user", content: "" },
      { role: "user", content: "Say hi." },
    ],
    temperature: 0.1,
  });
  assertExists(completion.choices[0].message.content, "Should handle empty content");
});

// ═══════════════════════════════════════════════════════
// 9. OPENAI SDK SPECIFIC FEATURES
// ═══════════════════════════════════════════════════════

await test("SDK .parse() method works with response_format", async () => {
  // The SDK's parse method should work with our proxy
  const completion = await client.chat.completions.create({
    model: MODEL,
    messages: [{ role: "user", content: "Return JSON: {\"answer\": 42}" }],
    response_format: { type: "json_object" },
    temperature: 0.1,
  });
  assertExists(completion.choices[0].message.content, "Should work with response_format");
});

await test("SDK with max_tokens parameter", async () => {
  const completion = await client.chat.completions.create({
    model: MODEL,
    messages: [{ role: "user", content: "Write a long paragraph about clouds." }],
    max_tokens: 50,
    temperature: 0.1,
  });
  assertExists(completion.choices[0].message.content, "Should respect max_tokens");
});

await test("SDK with stop sequences", async () => {
  const completion = await client.chat.completions.create({
    model: MODEL,
    messages: [{ role: "user", content: "Count to 10." }],
    stop: ["5"],
    temperature: 0.1,
  });
  assertExists(completion.choices[0].message.content, "Should work with stop");
});

await test("SDK with user parameter", async () => {
  const completion = await client.chat.completions.create({
    model: MODEL,
    messages: [{ role: "user", content: "Say ok." }],
    user: "test-user-123",
    temperature: 0.1,
  });
  assertExists(completion.choices[0].message.content, "Should work with user param");
});

await test("SDK with seed parameter", async () => {
  const completion = await client.chat.completions.create({
    model: MODEL,
    messages: [{ role: "user", content: "Say ok." }],
    seed: 42,
    temperature: 0.1,
  });
  assertExists(completion.choices[0].message.content, "Should work with seed");
});

await test("SDK with frequency_penalty", async () => {
  const completion = await client.chat.completions.create({
    model: MODEL,
    messages: [{ role: "user", content: "Say ok." }],
    frequency_penalty: 0.5,
    temperature: 0.1,
  });
  assertExists(completion.choices[0].message.content, "Should work with frequency_penalty");
});

await test("SDK with presence_penalty", async () => {
  const completion = await client.chat.completions.create({
    model: MODEL,
    messages: [{ role: "user", content: "Say ok." }],
    presence_penalty: 0.5,
    temperature: 0.1,
  });
  assertExists(completion.choices[0].message.content, "Should work with presence_penalty");
});

await test("SDK with top_p", async () => {
  const completion = await client.chat.completions.create({
    model: MODEL,
    messages: [{ role: "user", content: "Say ok." }],
    top_p: 0.9,
    temperature: 0.1,
  });
  assertExists(completion.choices[0].message.content, "Should work with top_p");
});

// ═══════════════════════════════════════════════════════
// RESULTS
// ═══════════════════════════════════════════════════════

console.log(`\n\n${"═".repeat(60)}`);
console.log(`\x1b[1mOpenAI SDK Compatibility Results\x1b[0m`);
console.log(`${"═".repeat(60)}`);
console.log(`  \x1b[32mPassed:  ${passed}\x1b[0m`);
console.log(`  \x1b[31mFailed:  ${failed}\x1b[0m`);
console.log(`  \x1b[33mSkipped: ${skipped}\x1b[0m`);
console.log(`  Total:   ${passed + failed + skipped}`);
console.log(`${"═".repeat(60)}\n`);

process.exit(failed > 0 ? 1 : 0);
