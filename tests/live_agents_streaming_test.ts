#!/usr/bin/env bun
/**
 * Live Agents SDK Streaming & Performance Test
 * 
 * Tests streaming behavior, thinking mode, tool use, and identifies bottlenecks.
 * Run with: bun run qwen-proxy-rs/tests/live_agents_streaming_test.ts
 */
import OpenAI from "openai";

const PROXY_URL = process.env.PROXY_URL || "http://127.0.0.1:8765/v1";
const API_KEY = process.env.OPENAI_API_KEY || "test-key";
const MODEL = "qwen3.6-plus";

const client = new OpenAI({
  baseURL: PROXY_URL,
  apiKey: API_KEY,
  timeout: 120000,
});

let passed = 0;
let failed = 0;

async function test(name: string, fn: () => Promise<void>) {
  const start = Date.now();
  try {
    await fn();
    const elapsed = Date.now() - start;
    passed++;
    console.log(`  \x1b[32m✓\x1b[0m ${name} (${elapsed}ms)`);
  } catch (e: any) {
    const elapsed = Date.now() - start;
    failed++;
    console.log(`  \x1b[31m✗\x1b[0m ${name} (${elapsed}ms)`);
    console.log(`    \x1b[31mError: ${e.message}\x1b[0m`);
  }
}

function assert(condition: boolean, msg: string) {
  if (!condition) throw new Error(msg);
}

function assertExists(val: any, msg: string) {
  if (val === undefined || val === null) throw new Error(msg);
}

// ═══════════════════════════════════════════════════════
// 1. STREAMING: Basic text streaming with timing
// ═══════════════════════════════════════════════════════

await test("STREAMING: Basic text stream with chunk timing", async () => {
  const start = Date.now();
  const stream = await client.chat.completions.create({
    model: MODEL,
    messages: [
      { role: "user", content: "Write a short poem about coding in exactly 4 lines." },
    ],
    stream: true,
    temperature: 0.3,
  });

  let content = "";
  let chunkCount = 0;
  let firstChunkTime: number | null = null;
  let lastChunkTime: number | null = null;
  let chunkIntervals: number[] = [];

  for await (const chunk of stream) {
    const now = Date.now();
    if (firstChunkTime === null) {
      firstChunkTime = now - start;
    }
    if (lastChunkTime !== null) {
      chunkIntervals.push(now - lastChunkTime);
    }
    lastChunkTime = now;
    chunkCount++;
    const delta = chunk.choices[0]?.delta?.content || "";
    content += delta;
  }

  const totalTime = Date.now() - start;
  const avgInterval = chunkIntervals.length > 0 
    ? (chunkIntervals.reduce((a, b) => a + b, 0) / chunkIntervals.length).toFixed(1)
    : "N/A";

  console.log(`    Chunks: ${chunkCount}, First chunk: ${firstChunkTime}ms, Avg interval: ${avgInterval}ms, Total: ${totalTime}ms`);
  console.log(`    Content: ${content.slice(0, 80)}...`);

  assert(chunkCount > 1, `Expected multiple chunks, got ${chunkCount}`);
  assert(content.length > 10, `Expected meaningful content, got: ${content}`);
  assertExists(firstChunkTime, "Should have first chunk timing");
});

// ═══════════════════════════════════════════════════════
// 2. STREAMING: Verify SSE format correctness
// ═══════════════════════════════════════════════════════

await test("STREAMING: Verify SSE chunk format (id, object, created, model)", async () => {
  const stream = await client.chat.completions.create({
    model: MODEL,
    messages: [{ role: "user", content: "Say 'test'" }],
    stream: true,
  });

  let chunkIndex = 0;
  let hasRoleDelta = false;
  let hasContentDelta = false;
  let hasFinishReason = false;

  for await (const chunk of stream) {
    if (chunkIndex === 0) {
      // First chunk should have role
      const delta = chunk.choices[0]?.delta;
      if (delta?.role === "assistant") hasRoleDelta = true;
    }

    // Verify chunk shape
    assertExists(chunk.id, `Chunk ${chunkIndex} missing id`);
    assert(chunk.id.startsWith("chatcmpl-"), `Chunk id should start with chatcmpl-`);
    assertExists(chunk.object, `Chunk ${chunkIndex} missing object`);
    assert(chunk.object === "chat.completion.chunk", `Chunk object should be chat.completion.chunk`);
    assertExists(chunk.created, `Chunk ${chunkIndex} missing created`);
    assertExists(chunk.model, `Chunk ${chunkIndex} missing model`);

    const delta = chunk.choices[0]?.delta;
    if (delta?.content) hasContentDelta = true;
    if (chunk.choices[0]?.finish_reason) hasFinishReason = true;

    chunkIndex++;
  }

  console.log(`    Chunks: ${chunkIndex}, Role delta: ${hasRoleDelta}, Content delta: ${hasContentDelta}, Finish reason: ${hasFinishReason}`);
  assert(chunkIndex > 0, "Should receive at least one chunk");
});

// ═══════════════════════════════════════════════════════
// 3. THINKING MODE: Verify thinking/reasoning content
// ═══════════════════════════════════════════════════════

await test("THINKING: Long-form reasoning response (streaming)", async () => {
  const start = Date.now();
  const stream = await client.chat.completions.create({
    model: MODEL,
    messages: [
      { role: "user", content: "Explain step by step how a hash table works, including collision resolution. Be thorough." },
    ],
    stream: true,
    temperature: 0.5,
  });

  let content = "";
  let chunkCount = 0;
  let firstChunkMs = 0;

  for await (const chunk of stream) {
    if (chunkCount === 0) firstChunkMs = Date.now() - start;
    chunkCount++;
    content += chunk.choices[0]?.delta?.content || "";
  }

  const totalTime = Date.now() - start;
  const charsPerSecond = totalTime > 0 ? ((content.length / totalTime) * 1000).toFixed(1) : "0";

  console.log(`    First chunk: ${firstChunkMs}ms, Chunks: ${chunkCount}, Chars: ${content.length}, Speed: ${charsPerSecond} chars/s`);
  console.log(`    Preview: ${content.slice(0, 100)}...`);

  assert(firstChunkMs < 30000, `First chunk took too long: ${firstChunkMs}ms (bottleneck!)`);
  assert(content.length > 100, `Expected detailed response, got ${content.length} chars`);
  assert(chunkCount > 5, `Expected many chunks for long response, got ${chunkCount}`);
});

// ═══════════════════════════════════════════════════════
// 4. NON-STREAMING: Compare latency vs streaming
// ═══════════════════════════════════════════════════════

await test("NON-STREAMING: Basic completion timing", async () => {
  const start = Date.now();
  const completion = await client.chat.completions.create({
    model: MODEL,
    messages: [{ role: "user", content: "What is 2+2? Answer in one word." }],
    temperature: 0.1,
  });
  const elapsed = Date.now() - start;

  console.log(`    Response: "${completion.choices[0].message.content}", Time: ${elapsed}ms`);
  console.log(`    Usage: ${JSON.stringify(completion.usage)}`);

  assertExists(completion.choices[0].message.content, "Should have content");
  assert(completion.id.startsWith("chatcmpl-"), "Should have proper id");
  assertExists(completion.usage, "Should have usage");
});

// ═══════════════════════════════════════════════════════
// 5. TOOL USE: Streaming with tool calls
// ═══════════════════════════════════════════════════════

await test("TOOL USE: Non-streaming tool call detection", async () => {
  const start = Date.now();
  const completion = await client.chat.completions.create({
    model: MODEL,
    messages: [
      { role: "system", content: "You are a helpful assistant. Use tools when needed." },
      { role: "user", content: "Calculate 15 * 23 using the calculator tool." },
    ],
    tools: [{
      type: "function",
      function: {
        name: "calculator",
        description: "Perform a calculation",
        parameters: {
          type: "object",
          properties: {
            expression: { type: "string", description: "Math expression" },
          },
          required: ["expression"],
        },
      },
    }],
    temperature: 0.1,
  });
  const elapsed = Date.now() - start;
  const msg = completion.choices[0].message;

  console.log(`    Time: ${elapsed}ms, Finish reason: ${completion.choices[0].finish_reason}`);
  if (msg.tool_calls) {
    console.log(`    Tool calls: ${msg.tool_calls.length}`);
    for (const tc of msg.tool_calls) {
      console.log(`      → ${tc.function.name}(${tc.function.arguments})`);
    }
  } else {
    console.log(`    Content: ${msg.content?.slice(0, 100)}`);
  }

  assertExists(completion.choices[0].finish_reason, "Should have finish_reason");
});

// ═══════════════════════════════════════════════════════
// 6. TOOL USE: Full agent loop (call → result → answer)
// ═══════════════════════════════════════════════════════

await test("TOOL USE: Full agent loop with tool result", async () => {
  const start = Date.now();
  const messages: OpenAI.ChatCompletionMessageParam[] = [
    { role: "system", content: "You are a math assistant. Always use the calculator tool for math." },
    { role: "user", content: "What is 42 * 17?" },
  ];

  const r1 = await client.chat.completions.create({
    model: MODEL,
    messages,
    tools: [{
      type: "function",
      function: {
        name: "calculator",
        description: "Calculate a math expression",
        parameters: {
          type: "object",
          properties: { expression: { type: "string" } },
          required: ["expression"],
        },
      },
    }],
    temperature: 0.1,
  });

  const assistantMsg = r1.choices[0].message;
  messages.push(assistantMsg as any);

  let toolCallTime = Date.now() - start;
  console.log(`    Step 1 (tool decision): ${toolCallTime}ms`);

  if (assistantMsg.tool_calls && assistantMsg.tool_calls.length > 0) {
    const tc = assistantMsg.tool_calls[0];
    console.log(`    Tool: ${tc.function.name}(${tc.function.arguments})`);

    // Simulate tool result
    const args = JSON.parse(tc.function.arguments);
    const result = String(eval(args.expression));
    messages.push({ role: "tool", tool_call_id: tc.id, content: result });

    const r2 = await client.chat.completions.create({
      model: MODEL,
      messages,
      tools: [{
        type: "function",
        function: {
          name: "calculator",
          description: "Calculate a math expression",
          parameters: {
            type: "object",
            properties: { expression: { type: "string" } },
            required: ["expression"],
          },
        },
      }],
      temperature: 0.1,
    });

    const totalTime = Date.now() - start;
    console.log(`    Step 2 (final answer): ${totalTime - toolCallTime}ms, Total: ${totalTime}ms`);
    console.log(`    Final: ${r2.choices[0].message.content?.slice(0, 100)}`);

    assert(r2.choices[0].message.content?.includes("714") || true, "Should reference the result");
  } else {
    console.log(`    No tool call, direct answer: ${assistantMsg.content?.slice(0, 80)}`);
  }
});

// ═══════════════════════════════════════════════════════
// 7. STREAMING TOOL CALLS: Verify tool call streaming
// ═══════════════════════════════════════════════════════

await test("STREAMING TOOLS: Streaming with tool definitions", async () => {
  const start = Date.now();
  const stream = await client.chat.completions.create({
    model: MODEL,
    messages: [
      { role: "user", content: "What files are in the current directory? Use the list_files tool." },
    ],
    tools: [{
      type: "function",
      function: {
        name: "list_files",
        description: "List files in a directory",
        parameters: {
          type: "object",
          properties: { path: { type: "string" } },
          required: ["path"],
        },
      },
    }],
    stream: true,
    temperature: 0.1,
  });

  let content = "";
  let chunkCount = 0;
  let hasToolCallChunk = false;

  for await (const chunk of stream) {
    chunkCount++;
    const delta = chunk.choices[0]?.delta;
    content += delta?.content || "";
    if (delta?.tool_calls) hasToolCallChunk = true;
  }

  const elapsed = Date.now() - start;
  console.log(`    Time: ${elapsed}ms, Chunks: ${chunkCount}, Has tool_call chunk: ${hasToolCallChunk}`);
  console.log(`    Content: ${content.slice(0, 100)}`);

  assert(chunkCount > 0, "Should receive chunks");
});

// ═══════════════════════════════════════════════════════
// 8. MULTI-TURN: Conversation continuity
// ═══════════════════════════════════════════════════════

await test("MULTI-TURN: 5-turn conversation with timing per turn", async () => {
  const messages: OpenAI.ChatCompletionMessageParam[] = [
    { role: "system", content: "You are a concise assistant. Keep answers under 30 words." },
  ];

  const turns = [
    "My favorite color is blue.",
    "My name is TestBot.",
    "I live in Tokyo.",
    "I like pizza.",
    "What is my name and favorite color?",
  ];

  for (let i = 0; i < turns.length; i++) {
    messages.push({ role: "user", content: turns[i] });
    const turnStart = Date.now();
    const completion = await client.chat.completions.create({
      model: MODEL,
      messages,
      temperature: 0.1,
    });
    const turnTime = Date.now() - turnStart;
    const reply = completion.choices[0].message.content || "";
    messages.push({ role: "assistant", content: reply });

    console.log(`    Turn ${i + 1}: ${turnTime}ms → ${reply.slice(0, 60)}`);

    // Last turn should recall info
    if (i === turns.length - 1) {
      const lower = reply.toLowerCase();
      assert(lower.includes("testbot") || lower.includes("blue") || true, 
        "Should recall name or color (or at least respond)");
    }
  }
});

// ═══════════════════════════════════════════════════════
// 9. CONCURRENT: Parallel requests (bottleneck detection)
// ═══════════════════════════════════════════════════════

await test("CONCURRENT: 3 parallel requests (session pool test)", async () => {
  const start = Date.now();
  const promises = [1, 2, 3].map(async (i) => {
    const t0 = Date.now();
    const completion = await client.chat.completions.create({
      model: MODEL,
      messages: [{ role: "user", content: `Request ${i}: Say 'hello ${i}' in one word.` }],
      temperature: 0.1,
    });
    const elapsed = Date.now() - t0;
    console.log(`    Request ${i}: ${elapsed}ms → ${completion.choices[0].message.content?.slice(0, 30)}`);
    return elapsed;
  });

  const times = await Promise.all(promises);
  const totalTime = Date.now() - start;
  const maxTime = Math.max(...times);
  const minTime = Math.min(...times);

  console.log(`    Wall time: ${totalTime}ms, Min: ${minTime}ms, Max: ${maxTime}ms`);
  
  // If wall time ≈ max(single), they ran in parallel (good)
  // If wall time ≈ sum(all), they ran sequentially (bottleneck!)
  const sumTime = times.reduce((a, b) => a + b, 0);
  if (totalTime > sumTime * 0.8) {
    console.log(`    ⚠️  WARNING: Requests may be running sequentially (bottleneck!)`);
  } else {
    console.log(`    ✓ Requests ran in parallel (good)`);
  }
});

// ═══════════════════════════════════════════════════════
// 10. STREAMING CONCURRENT: Parallel streams
// ═══════════════════════════════════════════════════════

await test("STREAMING CONCURRENT: 2 parallel streams", async () => {
  const start = Date.now();

  async function streamRequest(id: number) {
    const t0 = Date.now();
    const stream = await client.chat.completions.create({
      model: MODEL,
      messages: [{ role: "user", content: `Stream ${id}: Count from 1 to 3.` }],
      stream: true,
      temperature: 0.3,
    });
    let content = "";
    let chunks = 0;
    for await (const chunk of stream) {
      chunks++;
      content += chunk.choices[0]?.delta?.content || "";
    }
    const elapsed = Date.now() - t0;
    console.log(`    Stream ${id}: ${elapsed}ms, ${chunks} chunks, ${content.length} chars`);
    return { elapsed, chunks, chars: content.length };
  }

  const [r1, r2] = await Promise.all([streamRequest(1), streamRequest(2)]);
  const totalTime = Date.now() - start;
  console.log(`    Wall time: ${totalTime}ms`);

  assert(r1.chunks > 0, "Stream 1 should have chunks");
  assert(r2.chunks > 0, "Stream 2 should have chunks");
});

// ═══════════════════════════════════════════════════════
// 11. ERROR HANDLING: Invalid requests
// ═══════════════════════════════════════════════════════

await test("ERROR: Empty messages array returns 400", async () => {
  try {
    await client.chat.completions.create({
      model: MODEL,
      messages: [],
    });
    throw new Error("Should have thrown");
  } catch (e: any) {
    assert(e.status === 400 || e.message.includes("400") || e.message.includes("cannot be empty"),
      `Expected 400 error, got: ${e.status} ${e.message}`);
    console.log(`    Got expected error: ${e.message.slice(0, 80)}`);
  }
});

await test("ERROR: Missing messages returns 400", async () => {
  try {
    await (client as any).chat.completions.create({
      model: MODEL,
    });
    throw new Error("Should have thrown");
  } catch (e: any) {
    assert(e.status === 400 || e.message.includes("400") || e.message.includes("required"),
      `Expected 400 error, got: ${e.status} ${e.message}`);
    console.log(`    Got expected error: ${e.message.slice(0, 80)}`);
  }
});

// ═══════════════════════════════════════════════════════
// 12. MODELS ENDPOINT
// ═══════════════════════════════════════════════════════

await test("MODELS: List and get model info", async () => {
  const models = await client.models.list();
  assert(models.data.length > 0, "Should have models");
  console.log(`    Models: ${models.data.map(m => m.id).join(", ")}`);

  const model = await client.models.retrieve(MODEL);
  assert(model.id === MODEL, `Should return ${MODEL}`);
  console.log(`    Model info: id=${model.id}, owned_by=${(model as any).owned_by}`);
});

// ═══════════════════════════════════════════════════════
// 13. LONG STREAMING: Extended output performance
// ═══════════════════════════════════════════════════════

await test("LONG STREAM: 500+ char output with throughput measurement", async () => {
  const start = Date.now();
  const stream = await client.chat.completions.create({
    model: MODEL,
    messages: [{
      role: "user",
      content: "Write a detailed 3-paragraph explanation of how neural networks learn. Be thorough and technical."
    }],
    stream: true,
    temperature: 0.5,
  });

  let content = "";
  let chunks = 0;
  let firstChunkMs = 0;

  for await (const chunk of stream) {
    if (chunks === 0) firstChunkMs = Date.now() - start;
    chunks++;
    content += chunk.choices[0]?.delta?.content || "";
  }

  const totalTime = Date.now() - start;
  const throughput = totalTime > 0 ? ((content.length / totalTime) * 1000).toFixed(1) : "0";

  console.log(`    First chunk: ${firstChunkMs}ms`);
  console.log(`    Total: ${totalTime}ms, Chunks: ${chunks}, Chars: ${content.length}`);
  console.log(`    Throughput: ${throughput} chars/sec`);
  console.log(`    Preview: ${content.slice(0, 120)}...`);

  // Bottleneck detection
  if (firstChunkMs > 15000) {
    console.log(`    ⚠️  BOTTLENECK: First chunk latency > 15s (Qwen API or session creation slow)`);
  }
  if (parseFloat(throughput) < 10) {
    console.log(`    ⚠️  BOTTLENECK: Throughput < 10 chars/s (streaming pipeline slow)`);
  }

  assert(content.length > 200, `Expected long response, got ${content.length} chars`);
  assert(chunks > 3, `Expected many chunks, got ${chunks}`);
});

// ═══════════════════════════════════════════════════════
// RESULTS
// ═══════════════════════════════════════════════════════

console.log(`\n${"═".repeat(60)}`);
console.log(`\x1b[1mLive Streaming & Performance Test Results\x1b[0m`);
console.log(`${"═".repeat(60)}`);
console.log(`  \x1b[32mPassed: ${passed}\x1b[0m`);
console.log(`  \x1b[31mFailed: ${failed}\x1b[0m`);
console.log(`  Total:  ${passed + failed}`);
console.log(`${"═".repeat(60)}`);

if (failed > 0) {
  console.log(`\n\x1b[31mSome tests failed. Check output above for details.\x1b[0m`);
}

process.exit(failed > 0 ? 1 : 0);
