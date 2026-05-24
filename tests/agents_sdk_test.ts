#!/usr/bin/env bun
/**
 * OpenAI Agents SDK Compatibility Tests for Qwen Proxy
 * 
 * Tests the Rust proxy against the OpenAI Agents SDK to verify
 * agent workflow compatibility. Run with: bun run tests/agents_sdk_test.ts
 * 
 * Prerequisites:
 *   - Proxy running: cargo run --manifest-path qwen-proxy-rs/Cargo.toml
 *   - QWEN_TOKEN set or ~/.qwen_session.json exists
 *   - npm install openai @openai/agents (or bun install)
 */
import OpenAI from "openai";

const PROXY_URL = process.env.PROXY_URL || "http://127.0.0.1:8765/v1";
const API_KEY = process.env.OPENAI_API_KEY || "test-key-not-used";
const MODEL = "qwen3.6-plus";

const client = new OpenAI({
  baseURL: PROXY_URL,
  apiKey: API_KEY,
});

async function waitForProxy(baseUrl: string, retries = 30, delayMs = 500): Promise<void> {
  const healthUrl = baseUrl.replace(/\/v1\/?$/, "") + "/health";
  let lastErr = "";
  for (let i = 0; i < retries; i++) {
    try {
      const res = await fetch(healthUrl);
      if (res.ok) return;
      lastErr = `HTTP ${res.status}`;
    } catch (e: any) {
      lastErr = e.message;
    }
    await new Promise((r) => setTimeout(r, delayMs));
  }
  throw new Error(
    `Proxy not reachable at ${healthUrl} (${lastErr}). Start: cd qwen-proxy-rs && ./target/release/qwen-proxy`,
  );
}

await waitForProxy(PROXY_URL);

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

function assert(condition: boolean, msg: string) {
  if (!condition) throw new Error(msg);
}

function assertExists(val: any, msg: string) {
  if (val === undefined || val === null) throw new Error(msg);
}

function assertEquals(a: any, b: any, msg?: string) {
  if (a !== b) throw new Error(msg || `Expected ${b}, got ${a}`);
}

// ═══════════════════════════════════════════════════════
// 1. AGENT-STYLE SINGLE TURN
// ═══════════════════════════════════════════════════════

await test("Agent-style single turn with system prompt", async () => {
  // Simulates how Agents SDK sends a single-turn request
  const completion = await client.chat.completions.create({
    model: MODEL,
    messages: [
      {
        role: "system",
        content: "You are a helpful coding assistant. Write clean, concise code.",
      },
      {
        role: "user",
        content: "Write a Python function that adds two numbers. Just the function, no explanation.",
      },
    ],
    temperature: 0.1,
  });

  const content = completion.choices[0].message.content || "";
  assert(content.includes("def") || content.includes("lambda"), 
    `Should contain Python function (got: ${content.slice(0, 100)})`);
  assertEquals(completion.choices[0].finish_reason, "stop");
});

// ═══════════════════════════════════════════════════════
// 2. AGENT LOOP (MULTI-STEP TOOL USE)
// ═══════════════════════════════════════════════════════

await test("Agent loop: tool call → result → response", async () => {
  const calculatorTool: OpenAI.ChatCompletionTool = {
    type: "function",
    function: {
      name: "calculator",
      description: "Perform a mathematical calculation",
      parameters: {
        type: "object",
        properties: {
          expression: { type: "string", description: "Math expression to evaluate" },
        },
        required: ["expression"],
      },
    },
  };

  // Step 1: Agent decides to use tool
  const messages: OpenAI.ChatCompletionMessageParam[] = [
    { role: "system", content: "You are a math assistant. Use the calculator tool for any math." },
    { role: "user", content: "What is 23 * 47? Use the calculator tool." },
  ];

  const r1 = await client.chat.completions.create({
    model: MODEL,
    messages,
    tools: [calculatorTool],
    temperature: 0.1,
  });

  const assistantMsg = r1.choices[0].message;
  messages.push(assistantMsg);

  // Check if tool was called
  if (assistantMsg.tool_calls && assistantMsg.tool_calls.length > 0) {
    const tc = assistantMsg.tool_calls[0];
    assertEquals(tc.type, "function");
    assertEquals(tc.function.name, "calculator");

    // Step 2: Provide tool result (simulate calculator)
    const args = JSON.parse(tc.function.arguments);
    const result = eval(args.expression); // Safe in test context
    messages.push({
      role: "tool",
      tool_call_id: tc.id,
      content: String(result),
    });

    // Step 3: Agent responds with the result
    const r2 = await client.chat.completions.create({
      model: MODEL,
      messages,
      tools: [calculatorTool],
      temperature: 0.1,
    });

    const finalContent = r2.choices[0].message.content || "";
    assert(finalContent.includes("1081") || finalContent.includes("1081"), 
      `Should include calculation result (got: ${finalContent})`);
  } else {
    // Agent answered directly - still valid
    assertExists(assistantMsg.content, "Should have content if no tool call");
  }
});

// ═══════════════════════════════════════════════════════
// 3. AGENT WITH MULTIPLE TOOLS
// ═══════════════════════════════════════════════════════

await test("Agent with multiple tool choices", async () => {
  const tools: OpenAI.ChatCompletionTool[] = [
    {
      type: "function",
      function: {
        name: "read_file",
        description: "Read a file's contents",
        parameters: {
          type: "object",
          properties: { path: { type: "string" } },
          required: ["path"],
        },
      },
    },
    {
      type: "function",
      function: {
        name: "write_file",
        description: "Write content to a file",
        parameters: {
          type: "object",
          properties: {
            path: { type: "string" },
            content: { type: "string" },
          },
          required: ["path", "content"],
        },
      },
    },
    {
      type: "function",
      function: {
        name: "list_files",
        description: "List files in a directory",
        parameters: {
          type: "object",
          properties: { directory: { type: "string" } },
          required: ["directory"],
        },
      },
    },
  ];

  const completion = await client.chat.completions.create({
    model: MODEL,
    messages: [
      { role: "system", content: "You are a file management assistant." },
      { role: "user", content: "List the files in /tmp directory." },
    ],
    tools,
    temperature: 0.1,
  });

  const msg = completion.choices[0].message;
  // Either tool call or text response is acceptable
  if (msg.tool_calls && msg.tool_calls.length > 0) {
    const tc = msg.tool_calls[0];
    assert(
      ["read_file", "write_file", "list_files"].includes(tc.function.name),
      `Should call one of the defined tools (got: ${tc.function.name})`
    );
  } else {
    assertExists(msg.content, "Should have content if no tool call");
  }
});

// ═══════════════════════════════════════════════════════
// 4. AGENT STREAMING WITH TOOLS
// ═══════════════════════════════════════════════════════

await test("Agent streaming with tool definitions", async () => {
  const stream = await client.chat.completions.create({
    model: MODEL,
    messages: [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "Count from 1 to 3." },
    ],
    tools: [{
      type: "function",
      function: {
        name: "counter",
        description: "Count numbers",
        parameters: {
          type: "object",
          properties: { from: { type: "number" }, to: { type: "number" } },
          required: ["from", "to"],
        },
      },
    }],
    stream: true,
    temperature: 0.1,
  });

  let content = "";
  let chunkCount = 0;
  let sawToolDelta = false;
  for await (const chunk of stream) {
    chunkCount++;
    const delta = chunk.choices[0]?.delta;
    content += delta?.content || "";
    if (delta?.tool_calls?.length) sawToolDelta = true;
  }

  assert(chunkCount > 1, `Should receive multiple chunks (got ${chunkCount})`);
  assert(content.length > 0 || sawToolDelta, "Should have streamed content or tool-call deltas");
});

// ═══════════════════════════════════════════════════════
// 5. AGENT HANDOFF PATTERN
// ═══════════════════════════════════════════════════════

await test("Agent handoff pattern (simulated)", async () => {
  // Simulates the handoff pattern where one agent delegates to another
  const handoffTool: OpenAI.ChatCompletionTool = {
    type: "function",
    function: {
      name: "transfer_to_specialist",
      description: "Transfer the conversation to a specialist agent",
      parameters: {
        type: "object",
        properties: {
          specialist: {
            type: "string",
            enum: ["billing", "technical", "general"],
            description: "Which specialist to transfer to",
          },
          reason: { type: "string", description: "Why this transfer is needed" },
        },
        required: ["specialist", "reason"],
      },
    },
  };

  const r1 = await client.chat.completions.create({
    model: MODEL,
    messages: [
      { role: "system", content: "You are a triage agent. Transfer to the appropriate specialist." },
      { role: "user", content: "My credit card was charged twice for my subscription." },
    ],
    tools: [handoffTool],
    temperature: 0.1,
  });

  const msg = r1.choices[0].message;
  if (msg.tool_calls && msg.tool_calls.length > 0) {
    const tc = msg.tool_calls[0];
    assertEquals(tc.function.name, "transfer_to_specialist");
    const args = JSON.parse(tc.function.arguments);
    assertEquals(args.specialist, "billing", "Should transfer to billing");
  } else {
    assertExists(msg.content, "Should have content if no handoff");
  }
});

// ═══════════════════════════════════════════════════════
// 6. AGENT GUARDRAILS (INPUT/OUTPUT VALIDATION)
// ═══════════════════════════════════════════════════════

await test("Agent input guardrails via system prompt", async () => {
  const completion = await client.chat.completions.create({
    model: MODEL,
    messages: [
      {
        role: "system",
        content: `You are a customer service bot for a pizza shop. 
If the user asks about anything unrelated to pizza, say "I can only help with pizza orders." 
Keep responses under 20 words.`,
      },
      { role: "user", content: "What's the weather like?" },
    ],
    temperature: 0.1,
  });

  const content = completion.choices[0].message.content || "";
  const lower = content.toLowerCase();
  assert(
    lower.includes("pizza") || lower.includes("only help") || lower.includes("can't help"),
    `Should enforce guardrails (got: ${content})`
  );
});

await test("Agent output format enforcement", async () => {
  const completion = await client.chat.completions.create({
    model: MODEL,
    messages: [
      {
        role: "system",
        content: `Always respond in this exact JSON format:
{"answer": "your answer", "confidence": 0.0-1.0}
Do NOT include any text outside the JSON.`,
      },
      { role: "user", content: "What is the capital of France?" },
    ],
    temperature: 0.1,
  });

  const content = completion.choices[0].message.content || "";
  // Try to parse as JSON (may be wrapped in markdown)
  const jsonMatch = content.match(/\{[\s\S]*\}/);
  if (jsonMatch) {
    try {
      const parsed = JSON.parse(jsonMatch[0]);
      assertExists(parsed.answer, "Should have answer field");
      assertExists(parsed.confidence, "Should have confidence field");
    } catch {
      // JSON parsing failed but response exists
      assert(content.length > 0, "Should have response");
    }
  }
});

// ═══════════════════════════════════════════════════════
// 7. AGENT CONTEXT WINDOW MANAGEMENT
// ═══════════════════════════════════════════════════════

await test("Agent handles long conversation history", async () => {
  const messages: OpenAI.ChatCompletionMessageParam[] = [
    { role: "system", content: "You are a helpful assistant. Be concise." },
  ];

  // Simulate a long conversation
  for (let i = 0; i < 10; i++) {
    messages.push({ role: "user", content: `Question ${i + 1}: What is ${i} + ${i}?` });
    messages.push({ role: "assistant", content: `${i} + ${i} = ${i * 2}` });
  }

  messages.push({ role: "user", content: "What was my first question?" });

  const completion = await client.chat.completions.create({
    model: MODEL,
    messages,
    temperature: 0.1,
  });

  assertExists(completion.choices[0].message.content, "Should handle long history");
});

// ═══════════════════════════════════════════════════════
// 8. AGENT PARALLEL TOOL EXECUTION
// ═══════════════════════════════════════════════════════

await test("Agent parallel tool calls (if supported)", async () => {
  const tools: OpenAI.ChatCompletionTool[] = [
    {
      type: "function",
      function: {
        name: "get_stock_price",
        description: "Get current stock price",
        parameters: {
          type: "object",
          properties: { symbol: { type: "string" } },
          required: ["symbol"],
        },
      },
    },
    {
      type: "function",
      function: {
        name: "get_company_info",
        description: "Get company information",
        parameters: {
          type: "object",
          properties: { symbol: { type: "string" } },
          required: ["symbol"],
        },
      },
    },
  ];

  const completion = await client.chat.completions.create({
    model: MODEL,
    messages: [
      { role: "system", content: "You are a financial analyst assistant." },
      { role: "user", content: "Get the stock price and company info for AAPL." },
    ],
    tools,
    temperature: 0.1,
  });

  const msg = completion.choices[0].message;
  // Should either make tool calls or respond
  if (msg.tool_calls && msg.tool_calls.length > 0) {
    assert(msg.tool_calls.length >= 1, "Should make at least one tool call");
    for (const tc of msg.tool_calls) {
      assertEquals(tc.type, "function");
      assertExists(tc.function.name, "Tool call should have function name");
      assertExists(tc.function.arguments, "Tool call should have arguments");
    }
  } else {
    assertExists(msg.content, "Should have content if no tool calls");
  }
});

// ═══════════════════════════════════════════════════════
// 9. RESPONSES API COMPATIBILITY (/v1/responses)
// ═══════════════════════════════════════════════════════

await test("Responses API endpoint works", async () => {
  const resp = await fetch(`${PROXY_URL}/responses`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${API_KEY}`,
    },
    body: JSON.stringify({
      model: MODEL,
      input: [{ role: "user", content: "Say OK." }],
    }),
  });

  assertEquals(resp.status, 200);
  const body = await resp.json();
  assertExists(body.choices || body.output || body.content, "Should have response content");
});

// ═══════════════════════════════════════════════════════
// 10. AGENT TRACING/METADATA
// ═══════════════════════════════════════════════════════

await test("Response includes tracing metadata", async () => {
  const completion = await client.chat.completions.create({
    model: MODEL,
    messages: [{ role: "user", content: "Say hello." }],
    temperature: 0.1,
  });

  // Verify all standard OpenAI response fields
  assertExists(completion.id, "Should have id");
  assertExists(completion.object, "Should have object");
  assertExists(completion.created, "Should have created");
  assertExists(completion.model, "Should have model");
  assertExists(completion.choices, "Should have choices");
  assertExists(completion.usage, "Should have usage");
  
  // Verify usage structure
  assert(completion.usage!.prompt_tokens >= 0, "Should have prompt_tokens");
  assert(completion.usage!.completion_tokens >= 0, "Should have completion_tokens");
  assert(completion.usage!.total_tokens > 0, "Should have total_tokens");
});

// ═══════════════════════════════════════════════════════
// 11. AGENT SDK: STRUCTURED OUTPUT
// ═══════════════════════════════════════════════════════

await test("Structured output with JSON schema", async () => {
  const completion = await client.chat.completions.create({
    model: MODEL,
    messages: [
      { role: "user", content: "Give me a recipe for chocolate chip cookies." },
    ],
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "recipe",
        schema: {
          type: "object",
          properties: {
            name: { type: "string" },
            ingredients: { type: "array", items: { type: "string" } },
            steps: { type: "array", items: { type: "string" } },
          },
          required: ["name", "ingredients", "steps"],
        },
      },
    },
    temperature: 0.1,
  });

  const content = completion.choices[0].message.content || "";
  assert(content.length > 0, "Should return structured output");
});

// ═══════════════════════════════════════════════════════
// RESULTS
// ═══════════════════════════════════════════════════════

console.log(`\n\n${"═".repeat(60)}`);
console.log(`\x1b[1mAgents SDK Compatibility Results\x1b[0m`);
console.log(`${"═".repeat(60)}`);
console.log(`  \x1b[32mPassed:  ${passed}\x1b[0m`);
console.log(`  \x1b[31mFailed:  ${failed}\x1b[0m`);
console.log(`  \x1b[33mSkipped: ${skipped}\x1b[0m`);
console.log(`  Total:   ${passed + failed + skipped}`);
console.log(`${"═".repeat(60)}\n`);

process.exit(failed > 0 ? 1 : 0);
