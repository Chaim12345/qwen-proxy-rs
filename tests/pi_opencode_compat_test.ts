#!/usr/bin/env bun
/**
 * Pi / OpenCode compatibility smoke tests via @earendil-works/pi-ai.
 *
 * Validates the proxy works the way Pi and OpenCode agents configure custom
 * OpenAI-compatible endpoints (openai-completions + openai-responses).
 *
 * Run from qtalt/: bun run qwen-proxy-rs/tests/pi_opencode_compat_test.ts
 */
import {
  Type,
  complete,
  stream,
  type Context,
  type Model,
  type Tool,
} from "@earendil-works/pi-ai";

const PROXY_BASE = (process.env.PROXY_URL || "http://127.0.0.1:8765/v1").replace(
  /\/+$/,
  "",
);
const API_KEY = process.env.OPENAI_API_KEY || "local";

let passed = 0;
let failed = 0;
let testCounter = 0;

function testUser(prefix: string) {
  testCounter += 1;
  return `pi-opencode-${prefix}-${testCounter}`;
}

async function waitForProxy(retries = 30, delayMs = 500): Promise<void> {
  const healthUrl = PROXY_BASE.replace(/\/v1$/, "") + "/health";
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
  throw new Error(`Proxy not reachable at ${healthUrl} (${lastErr})`);
}

function assert(condition: boolean, msg: string) {
  if (!condition) throw new Error(msg);
}

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

function completionsModel(
  id: string,
  overrides: Partial<Model<"openai-completions">> = {},
): Model<"openai-completions"> {
  return {
    id,
    name: `${id} (qwen-proxy)`,
    api: "openai-completions",
    provider: "qwen-proxy",
    baseUrl: PROXY_BASE,
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 128000,
    maxTokens: 8192,
    compat: {
      supportsStore: false,
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
    },
    ...overrides,
  };
}

function responsesModel(
  id: string,
  overrides: Partial<Model<"openai-responses">> = {},
): Model<"openai-responses"> {
  return {
    id,
    name: `${id} (qwen-proxy responses)`,
    api: "openai-responses",
    provider: "qwen-proxy",
    baseUrl: PROXY_BASE,
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 128000,
    maxTokens: 8192,
    ...overrides,
  };
}

const weatherTool: Tool = {
  name: "get_weather",
  description: "Get weather for a city",
  parameters: Type.Object({
    city: Type.String({ description: "City name" }),
  }),
};

await waitForProxy();

// ── Pi-style chat completions (non-stream) ─────────────────────────────

await test("pi openai-completions complete (non-stream)", async () => {
  const model = completionsModel("qwen3.6-plus");
  const context: Context = {
    systemPrompt: "Reply briefly.",
    messages: [{ role: "user", content: "Say hello in one word." }],
  };
  const msg = await complete(model, context, {
    apiKey: API_KEY,
    sessionId: testUser("complete"),
    maxTokens: 64,
    temperature: 0.1,
  });
  const text = msg.content
    .filter((b) => b.type === "text")
    .map((b) => (b.type === "text" ? b.text : ""))
    .join("");
  assert(text.length > 0, "Expected non-empty assistant text");
});

// ── Pi-style streaming ─────────────────────────────────────────────────

await test("pi openai-completions stream", async () => {
  const model = completionsModel("qwen3.6-plus");
  const context: Context = {
    messages: [{ role: "user", content: "Count from 1 to 3, one number per line." }],
  };
  const s = stream(model, context, {
    apiKey: API_KEY,
    sessionId: testUser("stream"),
    maxTokens: 128,
    temperature: 0.1,
  });

  let sawText = false;
  for await (const event of s) {
    if (event.type === "text_delta" && event.delta.length > 0) sawText = true;
    if (event.type === "error") throw new Error(String(event.error));
  }
  const final = await s.result();
  assert(final.role === "assistant", "Expected assistant role");
  assert(sawText || final.content.some((b) => b.type === "text"), "Expected streamed text");
});

// ── Responses API (OpenCode / newer Pi defaults) ───────────────────────

await test("pi openai-responses complete", async () => {
  const model = responsesModel("qwen3.6-plus");
  const context: Context = {
    messages: [{ role: "user", content: "Reply with exactly: pong" }],
  };
  const msg = await complete(model, context, {
    apiKey: API_KEY,
    sessionId: testUser("responses"),
    maxTokens: 32,
    temperature: 0,
  });
  const text = msg.content
    .filter((b) => b.type === "text")
    .map((b) => (b.type === "text" ? b.text : ""))
    .join("");
  assert(text.length > 0, "Expected responses API text output");
});

// ── Tool loop (Pi / OpenCode agent pattern) ────────────────────────────

await test("pi tool loop with toolResult message", async () => {
  const model = completionsModel("qwen3.6-plus");
  const context: Context = {
    systemPrompt: "Use get_weather when asked about weather.",
    messages: [{ role: "user", content: "What is the weather in Paris?" }],
    tools: [weatherTool],
  };

  const first = await complete(model, context, {
    apiKey: API_KEY,
    sessionId: testUser("tools"),
    maxTokens: 256,
    temperature: 0.1,
  });

  const toolCalls = first.content.filter((b) => b.type === "toolCall");
  assert(toolCalls.length > 0, "Expected at least one tool call");

  context.messages.push(first);
  for (const block of toolCalls) {
    if (block.type !== "toolCall") continue;
    context.messages.push({
      role: "toolResult",
      toolCallId: block.id,
      toolName: block.name,
      content: [{ type: "text", text: "Sunny, 22°C" }],
      isError: false,
      timestamp: Date.now(),
    });
  }

  const second = await complete(model, context, {
    apiKey: API_KEY,
    sessionId: testUser("tools-followup"),
    maxTokens: 256,
    temperature: 0.1,
  });

  const answer = second.content
    .filter((b) => b.type === "text")
    .map((b) => (b.type === "text" ? b.text : ""))
    .join("");
  assert(answer.length > 0, "Expected follow-up answer after tool result");
});

// ── GPT model aliases ──────────────────────────────────────────────────

await test("pi gpt-4o alias via openai-completions", async () => {
  const model = completionsModel("gpt-4o");
  const context: Context = {
    messages: [{ role: "user", content: "Say ok." }],
  };
  const msg = await complete(model, context, {
    apiKey: API_KEY,
    sessionId: testUser("gpt4o"),
    maxTokens: 16,
    temperature: 0,
  });
  assert(msg.content.length > 0, "Expected response via gpt-4o alias");
});

await test("pi gpt-4 alias via openai-completions", async () => {
  const model = completionsModel("gpt-4");
  const context: Context = {
    messages: [{ role: "user", content: "Say ok." }],
  };
  const msg = await complete(model, context, {
    apiKey: API_KEY,
    sessionId: testUser("gpt4"),
    maxTokens: 16,
    temperature: 0,
  });
  assert(msg.content.length > 0, "Expected response via gpt-4 alias");
});

// ── Results ────────────────────────────────────────────────────────────

console.log(`\n\n${"═".repeat(60)}`);
console.log(`\x1b[1mPi / OpenCode Compatibility Results\x1b[0m`);
console.log(`${"═".repeat(60)}`);
console.log(`  \x1b[32mPassed: ${passed}\x1b[0m`);
console.log(`  \x1b[31mFailed: ${failed}\x1b[0m`);
console.log(`  Total:  ${passed + failed}`);
console.log(`${"═".repeat(60)}\n`);

process.exit(failed > 0 ? 1 : 0);
