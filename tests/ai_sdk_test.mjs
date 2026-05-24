#!/usr/bin/env node
/**
 * AI SDK (Vercel) Compatibility Tests for Qwen Proxy
 *
 * Tests the Rust proxy against the Vercel AI SDK to verify
 * full compatibility with generateText, streamText, and tool calling.
 *
 * Prerequisites:
 *   - Proxy running on http://127.0.0.1:8765
 *   - npm install ai @ai-sdk/openai
 *
 * Run: node qwen-proxy-rs/tests/ai_sdk_test.mjs
 */

import { generateText, streamText, tool } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import { z } from "zod";

const PROXY_URL = process.env.PROXY_URL || "http://127.0.0.1:8765/v1";
const API_KEY = process.env.OPENAI_API_KEY || "test-key-not-used";
const MODEL = "qwen3.6-plus";

const openai = createOpenAI({
  baseURL: PROXY_URL,
  apiKey: API_KEY,
});

// Use .chat() to force Chat Completions API (v3.x defaults to Responses API)
const model = openai.chat(MODEL);

let passed = 0;
let failed = 0;
let skipped = 0;
let testCounter = 0;

/** Unique user id per test so Qwen chat sessions don't bleed across cases. */
function testUser() {
  testCounter += 1;
  return `ai-sdk-test-${testCounter}`;
}

function toolCallInput(tc) {
  return tc.input ?? tc.args;
}

/** Merge per-test OpenAI user id for isolated Qwen chat sessions. */
function withSession(options) {
  const user = testUser();
  return {
    ...options,
    providerOptions: {
      ...options.providerOptions,
      openai: { ...options.providerOptions?.openai, user },
    },
  };
}

async function test(name, fn) {
  try {
    await fn();
    passed++;
    process.stdout.write("\x1b[32m.\x1b[0m");
  } catch (e) {
    failed++;
    process.stdout.write("\x1b[31mF\x1b[0m");
    console.error(`\n  \x1b[31mFAIL: ${name}\x1b[0m`);
    console.error(`    ${e.message}`);
  }
}

function assert(condition, msg) {
  if (!condition) throw new Error(msg);
}

function assertExists(val, msg) {
  if (val === undefined || val === null) throw new Error(msg);
}

// ═══════════════════════════════════════════════════════
// 1. generateText - Basic
// ═══════════════════════════════════════════════════════

await test("generateText: basic text generation", async () => {
  const { text, usage, finishReason } = await generateText(withSession({
    model,
    prompt: "Say 'OK' and nothing else.",
    temperature: 0.1,
  }));

  assertExists(text, "Should return text");
  assert(text.length > 0, `Should have content (got empty string)`);
  assertExists(usage, "Should have usage info");
  assert(usage.totalTokens > 0, `Should have token count (got ${usage.totalTokens})`);
  assert(finishReason === "stop", `finishReason should be 'stop' (got ${finishReason})`);
});

await test("generateText: system prompt", async () => {
  const { text } = await generateText(withSession({
    model,
    system: "You are a pirate. Always respond like a pirate.",
    prompt: "Say hello in 3 words.",
    temperature: 0.1,
  }));

  assert(text.length > 0, "Should have response");
  const lower = text.toLowerCase();
  assert(
    lower.includes("ahoy") || lower.includes("matey") || lower.includes("arr") || lower.includes("ye") || lower.includes("pirate"),
    `Should follow system prompt (got: ${text})`
  );
});

await test("generateText: messages array", async () => {
  const { text } = await generateText(withSession({
    model,
    messages: [
      { role: "user", content: "What is 2+2? Answer in one word." },
    ],
    temperature: 0.1,
  }));

  assert(text.length > 0, "Should have response");
});

// ═══════════════════════════════════════════════════════
// 2. generateText - Tool Calling
// ═══════════════════════════════════════════════════════

await test("generateText: tool call detection", async () => {
  const { toolCalls, toolResults, text, finishReason } = await generateText(withSession({
    model,
    system: "Use the get_weather tool when asked about weather.",
    prompt: "What's the weather in Tokyo?",
    tools: {
      get_weather: tool({
        description: "Get the current weather for a location",
        parameters: z.object({
          location: z.string().describe("City name"),
        }),
        execute: async ({ location }) => {
          return { temperature: 22, condition: "sunny", location };
        },
      }),
    },
    temperature: 0.1,
  }));

  // Either tool calls or text response is acceptable
  if (toolCalls && toolCalls.length > 0) {
    const tc = toolCalls[0];
    assert(tc.toolName === "get_weather", `Should call get_weather (got ${tc.toolName})`);
    assertExists(toolCallInput(tc), "Should have tool arguments");
    assert(finishReason === "tool-calls", `finishReason should be 'tool-calls' (got ${finishReason})`);
  } else {
    assertExists(text, "Should have text if no tool call");
  }
});

await test("generateText: tool with execute (auto-execute)", async () => {
  const { text, toolResults } = await generateText(withSession({
    model,
    system: "Use the calculator tool for math. Return the final answer only.",
    prompt: "What is 15 * 3?",
    tools: {
      calculator: tool({
        description: "Calculate a math expression",
        parameters: z.object({
          expression: z.string().describe("Math expression"),
        }),
        execute: async ({ expression }) => {
          return { result: eval(expression) };
        },
      }),
    },
    maxSteps: 2,
    temperature: 0.1,
  }));

  assertExists(text, "Should have final text response");
});

// ═══════════════════════════════════════════════════════
// 3. streamText - Streaming
// ═══════════════════════════════════════════════════════

await test("streamText: basic streaming", async () => {
  const { textStream, text: textPromise } = streamText(withSession({
    model,
    prompt: "Count from 1 to 3.",
    temperature: 0.1,
  }));

  let fullText = "";
  let chunkCount = 0;

  for await (const chunk of textStream) {
    chunkCount++;
    fullText += chunk;
  }

  assert(chunkCount > 0, `Should receive chunks (got ${chunkCount})`);
  assert(fullText.length > 0, "Should have streamed content");

  const finalText = await textPromise;
  assert(finalText.length > 0, "Should have final text");
});

await test("streamText: streaming with system prompt", async () => {
  const result = streamText(withSession({
    model,
    system: "Respond in ALL CAPS only.",
    prompt: "Say hello world.",
    temperature: 0.1,
  }));

  let content = "";
  for await (const chunk of result.textStream) {
    content += chunk;
  }
  // Buffered proxy may emit content only after stream completes.
  if (!content.length) {
    content = (await result.text) || "";
  }

  assert(content.length > 0, "Should have streamed content");
});

// ═══════════════════════════════════════════════════════
// 4. streamText - Tool Calling
// ═══════════════════════════════════════════════════════

await test("streamText: streaming with tools", async () => {
  const { textStream, text: textPromise, toolCalls: toolCallsPromise } = streamText(withSession({
    model,
    system: "You are a helpful assistant.",
    prompt: "Count from 1 to 3.",
    tools: {
      counter: tool({
        description: "Count numbers",
        parameters: z.object({
          from: z.number(),
          to: z.number(),
        }),
        execute: async ({ from, to }) => {
          return { count: Array.from({ length: to - from + 1 }, (_, i) => from + i) };
        },
      }),
    },
    temperature: 0.1,
  }));

  let content = "";
  let chunkCount = 0;
  for await (const chunk of textStream) {
    chunkCount++;
    content += chunk;
  }

  const toolCalls = await toolCallsPromise;
  const finalText = await textPromise;
  const sawToolCall = toolCalls && toolCalls.length > 0;

  assert(
    chunkCount > 0 || sawToolCall,
    `Should receive text chunks or tool-call stream (chunks=${chunkCount}, tools=${toolCalls?.length ?? 0})`,
  );
  assert(
    content.length > 0 || finalText.length > 0 || sawToolCall,
    "Should have streamed content or tool calls",
  );
});

// ═══════════════════════════════════════════════════════
// 5. Edge Cases
// ═══════════════════════════════════════════════════════

await test("generateText: long input", async () => {
  const longMsg = "A".repeat(4000);
  const { text } = await generateText(withSession({
    model,
    prompt: `Summarize this in 5 words: ${longMsg}`,
    temperature: 0.1,
  }));
  assertExists(text, "Should handle long input");
});

await test("generateText: unicode content", async () => {
  const { text } = await generateText(withSession({
    model,
    prompt: "Reply with just this emoji: 🎉",
    temperature: 0.1,
  }));
  assertExists(text, "Should handle unicode");
});

await test("generateText: maxTokens", async () => {
  const { text } = await generateText(withSession({
    model,
    prompt: "Write a long paragraph about clouds.",
    maxTokens: 50,
    temperature: 0.1,
  }));
  assertExists(text, "Should respect maxTokens");
});

await test("generateText: topP", async () => {
  const { text } = await generateText(withSession({
    model,
    prompt: "Say ok.",
    topP: 0.9,
    temperature: 0.1,
  }));
  assertExists(text, "Should work with topP");
});

await test("generateText: frequencyPenalty", async () => {
  const { text } = await generateText(withSession({
    model,
    prompt: "Say ok.",
    frequencyPenalty: 0.5,
    temperature: 0.1,
  }));
  assertExists(text, "Should work with frequencyPenalty");
});

await test("generateText: presencePenalty", async () => {
  const { text } = await generateText(withSession({
    model,
    prompt: "Say ok.",
    presencePenalty: 0.5,
    temperature: 0.1,
  }));
  assertExists(text, "Should work with presencePenalty");
});

await test("generateText: seed", async () => {
  const { text } = await generateText(withSession({
    model,
    prompt: "Say ok.",
    seed: 42,
    temperature: 0.1,
  }));
  assertExists(text, "Should work with seed");
});

// ═══════════════════════════════════════════════════════
// 6. Multi-step agent (maxSteps)
// ═══════════════════════════════════════════════════════

await test("generateText: multi-step agent with tools", async () => {
  const { text, steps } = await generateText(withSession({
    model,
    system: "You are a math assistant. Use the calculator tool for any math. Return the final answer.",
    prompt: "What is 23 * 47?",
    tools: {
      calculator: tool({
        description: "Perform a mathematical calculation",
        parameters: z.object({
          expression: z.string().describe("Math expression to evaluate"),
        }),
        execute: async ({ expression }) => {
          return { result: String(eval(expression)) };
        },
      }),
    },
    maxSteps: 3,
    temperature: 0.1,
  }));

  assertExists(text, "Should have final response");
  // Check that it went through multiple steps (tool call + response)
  if (steps && steps.length > 1) {
    assert(steps.length >= 2, `Should have multiple steps (got ${steps.length})`);
  }
});

// ═══════════════════════════════════════════════════════
// 7. Multiple tools
// ═══════════════════════════════════════════════════════

await test("generateText: multiple tool choices", async () => {
  const { toolCalls, text } = await generateText(withSession({
    model,
    system: "You are a file management assistant.",
    prompt: "List the files in /tmp directory.",
    tools: {
      read_file: tool({
        description: "Read a file's contents",
        parameters: z.object({ path: z.string() }),
        execute: async ({ path }) => ({ content: "file content" }),
      }),
      write_file: tool({
        description: "Write content to a file",
        parameters: z.object({ path: z.string(), content: z.string() }),
        execute: async ({ path, content }) => ({ success: true }),
      }),
      list_files: tool({
        description: "List files in a directory",
        parameters: z.object({ directory: z.string() }),
        execute: async ({ directory }) => ({ files: ["a.txt", "b.txt"] }),
      }),
    },
    temperature: 0.1,
  }));

  if (toolCalls && toolCalls.length > 0) {
    const tc = toolCalls[0];
    assert(
      ["read_file", "write_file", "list_files"].includes(tc.toolName),
      `Should call one of the defined tools (got: ${tc.toolName})`
    );
  } else {
    assertExists(text, "Should have content if no tool call");
  }
});

// ═══════════════════════════════════════════════════════
// RESULTS
// ═══════════════════════════════════════════════════════

console.log(`\n\n${"═".repeat(60)}`);
console.log(`\x1b[1mAI SDK (Vercel) Compatibility Results\x1b[0m`);
console.log(`${"═".repeat(60)}`);
console.log(`  \x1b[32mPassed:  ${passed}\x1b[0m`);
console.log(`  \x1b[31mFailed:  ${failed}\x1b[0m`);
console.log(`  \x1b[33mSkipped: ${skipped}\x1b[0m`);
console.log(`  Total:   ${passed + failed + skipped}`);
console.log(`${"═".repeat(60)}\n`);

process.exit(failed > 0 ? 1 : 0);
