#!/usr/bin/env node
/**
 * Tool Call Stress Test for Qwen Proxy (Optimized)
 *
 * Focuses on reliability, speed, and edge cases for tool calling.
 * Tests concurrent requests, special characters, multi-step agents,
 * and various edge cases with realistic load patterns.
 *
 * Prerequisites:
 * - Proxy running on http://127.0.0.1:8765
 * - npm install ai @ai-sdk/openai zod
 *
 * Run: node qwen-proxy-rs/tests/toolcall_stress_test.mjs
 *
 * Phase 5.3 (Robust Tool Translator): for "0 unknown tool names ever emitted" guarantee,
 * after any generateText/streamText with tools, add client-side assert:
 *   const allowed = Object.keys(tools);
 *   const bad = (toolCalls||[]).filter(tc => !allowed.includes(tc.toolName));
 *   assert(bad.length===0, "unknown tool names leaked: "+bad.map(b=>b.toolName));
 *   or on error shape, assert no bad names in any event.
 * This .mjs is harness-dependent (live proxy + QWEN_TOKEN + `npm install ai @ai-sdk/openai zod` + net).
 * Run manually when available; do not make CI-flaky. See also ai_sdk_test.mjs + contract_checklist_test.mjs.
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

const model = openai.chat(MODEL);

let passed = 0;
let failed = 0;
let testCounter = 0;

function testUser() {
 testCounter += 1;
 return `stress-test-${testCounter}`;
}

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

// Timing helper
async function timed(fn) {
 const start = performance.now();
 const result = await fn();
 const elapsed = performance.now() - start;
 return { result, elapsed };
}

// Stats collector
class Stats {
 constructor(name) {
 this.name = name;
 this.times = [];
 this.successes = 0;
 this.failures = 0;
 }
 record(ms, ok) {
 this.times.push(ms);
 if (ok) this.successes++; else this.failures++;
 }
 p50() { return this.percentile(50); }
 p95() { return this.percentile(95); }
 avg() { return this.times.length > 0 ? this.times.reduce((a, b) => a + b, 0) / this.times.length : 0; }
 percentile(p) {
 if (this.times.length === 0) return 0;
 const sorted = [...this.times].sort((a, b) => a - b);
 const idx = Math.ceil((p / 100) * sorted.length) - 1;
 return sorted[Math.max(0, idx)];
 }
 summary() {
 return `${this.name}: ${this.successes} ok / ${this.failures} fail | avg ${this.avg().toFixed(0)}ms | p50 ${this.p50().toFixed(0)}ms | p95 ${this.p95().toFixed(0)}ms`;
 }
}

// ═══════════════════════════════════════════════════════
// 1. Concurrent Tool Call Load Test (5 parallel)
// ═══════════════════════════════════════════════════════

console.log("\n\n" + "═".repeat(60));
console.log("1. CONCURRENT TOOL CALL LOAD TEST (5 parallel)");
console.log("═".repeat(60));

const loadStats = new Stats("Concurrent Load (5 parallel)");
const loadPromises = [];

for (let i = 0; i < 5; i++) {
 loadPromises.push(
 timed(async () => {
 const { toolCalls, text } = await generateText(withSession({
 model,
 system: "Use the calculator tool. Return only the numeric answer.",
 prompt: `What is ${10 + i} * ${20 + i}?`,
 tools: {
 calculator: tool({
 description: "Calculate math",
 parameters: z.object({ expression: z.string() }),
 execute: async ({ expression }) => ({ result: String(eval(expression)) }),
 }),
 },
 maxSteps: 2,
 temperature: 0.1,
 }));
 return { toolCalls, text };
 }).then(({ result, elapsed }) => {
 loadStats.record(elapsed, true);
 process.stdout.write("\x1b[32m.\x1b[0m");
 passed++;
 }).catch((e) => {
 loadStats.record(0, false);
 process.stdout.write("\x1b[31mF\x1b[0m");
 console.error(`\n FAIL [load ${i}]: ${e.message}`);
 failed++;
 })
 );
}

await Promise.all(loadPromises);
console.log(`\n ${loadStats.summary()}`);

// ═══════════════════════════════════════════════════════
// 2. Rapid Sequential Multi-Step (5 iterations)
// ═══════════════════════════════════════════════════════

console.log("\n\n" + "═".repeat(60));
console.log("2. RAPID SEQUENTIAL MULTI-STEP (5 iterations)");
console.log("═".repeat(60));

const seqStats = new Stats("Sequential Multi-Step");

for (let i = 0; i < 5; i++) {
 try {
 const { result, elapsed } = await timed(async () => {
 const { text, steps } = await generateText(withSession({
 model,
 system: "You are a math assistant. Use calculator. Return only the final number.",
 prompt: `Calculate: ${i + 1} + ${i * 3} * ${i + 5}`,
 tools: {
 calculator: tool({
 description: "Calculate",
 parameters: z.object({ expression: z.string() }),
 execute: async ({ expression }) => ({ result: String(eval(expression)) }),
 }),
 },
 maxSteps: 3,
 temperature: 0.1,
 }));
 return { text, steps };
 });
 seqStats.record(elapsed, true);
 process.stdout.write("\x1b[32m.\x1b[0m");
 passed++;
 } catch (e) {
 seqStats.record(0, false);
 process.stdout.write("\x1b[31mF\x1b[0m");
 console.error(`\n FAIL [seq ${i}]: ${e.message}`);
 failed++;
 }
}
console.log(`\n ${seqStats.summary()}`);

// ═══════════════════════════════════════════════════════
// 3. Multiple Tools Selection (10 tools, 5 requests)
// ═══════════════════════════════════════════════════════

console.log("\n\n" + "═".repeat(60));
console.log("3. MULTIPLE TOOLS SELECTION (10 tools, 5 requests)");
console.log("═".repeat(60));

const multiToolStats = new Stats("Multi-Tool Selection");

const toolDefs = {
 add: tool({ description: "Add two numbers", parameters: z.object({ a: z.number(), b: z.number() }), execute: async ({ a, b }) => ({ result: a + b }) }),
 subtract: tool({ description: "Subtract b from a", parameters: z.object({ a: z.number(), b: z.number() }), execute: async ({ a, b }) => ({ result: a - b }) }),
 multiply: tool({ description: "Multiply two numbers", parameters: z.object({ a: z.number(), b: z.number() }), execute: async ({ a, b }) => ({ result: a * b }) }),
 divide: tool({ description: "Divide a by b", parameters: z.object({ a: z.number(), b: z.number() }), execute: async ({ a, b }) => ({ result: a / b }) }),
 power: tool({ description: "Raise a to power b", parameters: z.object({ a: z.number(), b: z.number() }), execute: async ({ a, b }) => ({ result: Math.pow(a, b) }) }),
 sqrt: tool({ description: "Square root", parameters: z.object({ n: z.number() }), execute: async ({ n }) => ({ result: Math.sqrt(n) }) }),
 abs: tool({ description: "Absolute value", parameters: z.object({ n: z.number() }), execute: async ({ n }) => ({ result: Math.abs(n) }) }),
 modulo: tool({ description: "Modulo", parameters: z.object({ a: z.number(), b: z.number() }), execute: async ({ a, b }) => ({ result: a % b }) }),
 max: tool({ description: "Maximum", parameters: z.object({ a: z.number(), b: z.number() }), execute: async ({ a, b }) => ({ result: Math.max(a, b) }) }),
 min: tool({ description: "Minimum", parameters: z.object({ a: z.number(), b: z.number() }), execute: async ({ a, b }) => ({ result: Math.min(a, b) }) }),
};

const multiToolPrompts = [
 "Add 5 and 3",
 "Multiply 7 by 8",
 "What is 2 to the power of 10?",
 "Square root of 144",
 "Which is bigger: 99 or 101?",
];

for (let i = 0; i < multiToolPrompts.length; i++) {
 try {
 const { result, elapsed } = await timed(async () => {
 const { toolCalls, text } = await generateText(withSession({
 model,
 system: "Use the most appropriate tool. Return only the final answer.",
 prompt: multiToolPrompts[i],
 tools: toolDefs,
 maxSteps: 3,
 temperature: 0.1,
 }));
 return { toolCalls, text };
 });
 multiToolStats.record(elapsed, true);
 process.stdout.write("\x1b[32m.\x1b[0m");
 passed++;
 } catch (e) {
 multiToolStats.record(0, false);
 process.stdout.write("\x1b[31mF\x1b[0m");
 console.error(`\n FAIL [multi-tool ${i}]: ${e.message}`);
 failed++;
 }
}
console.log(`\n ${multiToolStats.summary()}`);

// ═══════════════════════════════════════════════════════
// 4. Edge Case: Special Characters in Tool Arguments
// ═══════════════════════════════════════════════════════

console.log("\n\n" + "═".repeat(60));
console.log("4. EDGE CASE: SPECIAL CHARACTERS IN ARGUMENTS");
console.log("═".repeat(60));

const specialCharStats = new Stats("Special Chars");

const specialCases = [
 { name: "quotes", input: 'He said "hello" and \'goodbye\'' },
 { name: "backslashes", input: "path\\to\\file" },
 { name: "unicode", input: "日本語テスト 🎉 émojis" },
 { name: "html", input: "<div>Hello & goodbye</div>" },
 { name: "json_in_string", input: '{"key": "value"}' },
 { name: "sql_injection", input: "'; DROP TABLE users; --" },
 { name: "empty_string", input: "" },
];

for (const tc of specialCases) {
 try {
 const { result, elapsed } = await timed(async () => {
 const { toolCalls, text } = await generateText(withSession({
 model,
 system: "Use the echo tool to echo back the input exactly.",
 prompt: `Echo this: ${tc.input}`,
 tools: {
 echo: tool({
 description: "Echo input",
 parameters: z.object({ text: z.string() }),
 execute: async ({ text }) => ({ echoed: text }),
 }),
 },
 temperature: 0.1,
 }));
 return { toolCalls, text };
 });
 specialCharStats.record(elapsed, true);
 console.log(` \x1b[32m✓\x1b[0m ${tc.name}: ${elapsed.toFixed(0)}ms`);
 passed++;
 } catch (e) {
 specialCharStats.record(0, false);
 console.log(` \x1b[31m✗\x1b[0m ${tc.name}: ${e.message}`);
 failed++;
 }
}
console.log(` ${specialCharStats.summary()}`);

// ═══════════════════════════════════════════════════════
// 5. Edge Case: Complex Nested Arguments
// ═══════════════════════════════════════════════════════

console.log("\n\n" + "═".repeat(60));
console.log("5. EDGE CASE: COMPLEX NESTED ARGUMENTS");
console.log("═".repeat(60));

const nestedStats = new Stats("Nested Args");

try {
 const { result, elapsed } = await timed(async () => {
 const { toolCalls, text } = await generateText(withSession({
 model,
 system: "Use the complex_tool with nested parameters.",
 prompt: "Create config: users [{name: 'Alice', age: 30, tags: ['admin']}], settings {theme: 'dark'}",
 tools: {
 complex_tool: tool({
 description: "Handle complex nested data",
 parameters: z.object({
 users: z.array(z.object({ name: z.string(), age: z.number(), tags: z.array(z.string()) })),
 settings: z.object({ theme: z.string() }),
 }),
 execute: async (args) => ({ received: true, userCount: args.users?.length ?? 0 }),
 }),
 },
 temperature: 0.1,
 }));
 return { toolCalls, text };
 });
 nestedStats.record(elapsed, true);
 console.log(` \x1b[32m✓\x1b[0m Complex nested: ${elapsed.toFixed(0)}ms`);
 passed++;
} catch (e) {
 nestedStats.record(0, false);
 console.log(` \x1b[31m✗\x1b[0m Complex nested: ${e.message}`);
 failed++;
}
console.log(` ${nestedStats.summary()}`);

// ═══════════════════════════════════════════════════════
// 6. Streaming Tool Calls (3 concurrent)
// ═══════════════════════════════════════════════════════

console.log("\n\n" + "═".repeat(60));
console.log("6. STREAMING TOOL CALLS (3 concurrent)");
console.log("═".repeat(60));

const streamStats = new Stats("Streaming Tools");
const streamPromises = [];

for (let i = 0; i < 3; i++) {
 streamPromises.push(
 timed(async () => {
 const { textStream, toolCalls: tcPromise } = streamText(withSession({
 model,
 system: "Use the calculator tool.",
 prompt: `What is ${i + 10} * ${i + 20}?`,
 tools: {
 calculator: tool({
 description: "Calculate",
 parameters: z.object({ expression: z.string() }),
 execute: async ({ expression }) => ({ result: String(eval(expression)) }),
 }),
 },
 temperature: 0.1,
 }));

 let chunks = 0;
 for await (const chunk of textStream) {
 chunks++;
 }
 const toolCalls = await tcPromise;
 return { chunks, toolCalls };
 }).then(({ result, elapsed }) => {
 streamStats.record(elapsed, true);
 process.stdout.write("\x1b[32m.\x1b[0m");
 passed++;
 }).catch((e) => {
 streamStats.record(0, false);
 process.stdout.write("\x1b[31mF\x1b[0m");
 console.error(`\n FAIL [stream ${i}]: ${e.message}`);
 failed++;
 })
 );
}

await Promise.all(streamPromises);
console.log(`\n ${streamStats.summary()}`);

// ═══════════════════════════════════════════════════════
// 7. Reliability: 10 Rapid Fire Requests
// ═══════════════════════════════════════════════════════

console.log("\n\n" + "═".repeat(60));
console.log("7. RELIABILITY: 10 RAPID FIRE REQUESTS");
console.log("═".repeat(60));

const reliabilityStats = new Stats("Reliability (10 rapid)");

for (let i = 0; i < 10; i++) {
 try {
 const { result, elapsed } = await timed(async () => {
 const { toolCalls, text } = await generateText(withSession({
 model,
 system: "Use calc tool. Answer with number only.",
 prompt: `${i + 1} + ${i + 2}`,
 tools: {
 calc: tool({
 description: "Add",
 parameters: z.object({ a: z.number(), b: z.number() }),
 execute: async ({ a, b }) => ({ result: a + b }),
 }),
 },
 maxSteps: 2,
 temperature: 0.1,
 }));
 return { toolCalls, text };
 });
 reliabilityStats.record(elapsed, true);
 process.stdout.write("\x1b[32m.\x1b[0m");
 passed++;
 } catch (e) {
 reliabilityStats.record(0, false);
 process.stdout.write("\x1b[31mF\x1b[0m");
 failed++;
 }
}
console.log(`\n ${reliabilityStats.summary()}`);

// ═══════════════════════════════════════════════════════
// 8. Tool Name Edge Cases
// ═══════════════════════════════════════════════════════

console.log("\n\n" + "═".repeat(60));
console.log("8. EDGE CASE: UNUSUAL TOOL NAMES");
console.log("═".repeat(60));

const toolNameStats = new Stats("Tool Names");

const toolNameCases = [
 { name: "snake_case_tool", desc: "Snake case" },
 { name: "camelCaseTool", desc: "Camel case" },
 { name: "with-dashes", desc: "With dashes" },
 { name: "with_123_numbers", desc: "With numbers" },
 { name: "a", desc: "Single char" },
 { name: "very_long_tool_name_that_describes_exactly_what_it_does", desc: "Long name" },
];

for (const tc of toolNameCases) {
 try {
 const tools = {};
 tools[tc.name] = tool({
 description: tc.desc,
 parameters: z.object({ input: z.string() }),
 execute: async ({ input }) => ({ received: input }),
 });

 const { result, elapsed } = await timed(async () => {
 const { toolCalls, text } = await generateText(withSession({
 model,
 system: `Use the ${tc.name} tool.`,
 prompt: "Process: hello",
 tools,
 temperature: 0.1,
 }));
 return { toolCalls, text };
 });
 toolNameStats.record(elapsed, true);
 console.log(` \x1b[32m✓\x1b[0m ${tc.name}: ${elapsed.toFixed(0)}ms`);
 passed++;
 } catch (e) {
 toolNameStats.record(0, false);
 console.log(` \x1b[31m✗\x1b[0m ${tc.name}: ${e.message}`);
 failed++;
 }
}
console.log(` ${toolNameStats.summary()}`);

// ═══════════════════════════════════════════════════════
// 9. Burst Test: 10 Concurrent Requests
// ═══════════════════════════════════════════════════════

console.log("\n\n" + "═".repeat(60));
console.log("9. BURST TEST: 10 CONCURRENT TOOL CALLS");
console.log("═".repeat(60));

const burstStats = new Stats("Burst (10 concurrent)");
const burstPromises = [];

for (let i = 0; i < 10; i++) {
 burstPromises.push(
 timed(async () => {
 const { toolCalls, text } = await generateText(withSession({
 model,
 system: "Use the add tool.",
 prompt: `Add ${i} and ${i + 1}`,
 tools: {
 add: tool({
 description: "Add numbers",
 parameters: z.object({ a: z.number(), b: z.number() }),
 execute: async ({ a, b }) => ({ result: a + b }),
 }),
 },
 temperature: 0.1,
 }));
 return { toolCalls, text };
 }).then(({ result, elapsed }) => {
 burstStats.record(elapsed, true);
 return true;
 }).catch((e) => {
 burstStats.record(0, false);
 return false;
 })
 );
}

const burstResults = await Promise.all(burstPromises);
const burstOk = burstResults.filter(Boolean).length;
console.log(` ${burstOk}/10 succeeded`);
passed += burstStats.successes;
failed += burstStats.failures;
console.log(` ${burstStats.summary()}`);

// ═══════════════════════════════════════════════════════
// 10. Large Tool Arguments
// ═══════════════════════════════════════════════════════

console.log("\n\n" + "═".repeat(60));
console.log("10. EDGE CASE: LARGE TOOL ARGUMENTS");
console.log("═".repeat(60));

const largeArgStats = new Stats("Large Arguments");

for (let size of [500, 1000, 2000]) {
 try {
 const largeData = "x".repeat(size);
 const { result, elapsed } = await timed(async () => {
 const { toolCalls, text } = await generateText(withSession({
 model,
 system: "Use the process_data tool.",
 prompt: `Process this data: ${largeData}`,
 tools: {
 process_data: tool({
 description: "Process large data",
 parameters: z.object({ data: z.string().describe("The data to process") }),
 execute: async ({ data }) => ({ length: data.length, processed: true }),
 }),
 },
 temperature: 0.1,
 }));
 return { toolCalls, text };
 });
 largeArgStats.record(elapsed, true);
 console.log(` \x1b[32m✓\x1b[0m ${size} chars: ${elapsed.toFixed(0)}ms`);
 passed++;
 } catch (e) {
 largeArgStats.record(0, false);
 console.log(` \x1b[31m✗\x1b[0m ${size} chars: ${e.message}`);
 failed++;
 }
}
console.log(` ${largeArgStats.summary()}`);

// ═══════════════════════════════════════════════════════
// RESULTS
// ═══════════════════════════════════════════════════════

console.log(`\n\n${"═".repeat(60)}`);
console.log(`\x1b[1mTOOL CALL STRESS TEST RESULTS\x1b[0m`);
console.log(`${"═".repeat(60)}`);
console.log(` \x1b[32mPassed: ${passed}\x1b[0m`);
console.log(` \x1b[31mFailed: ${failed}\x1b[0m`);
console.log(` Total: ${passed + failed}`);
console.log(` Success Rate: ${((passed / (passed + failed)) * 100).toFixed(1)}%`);
console.log(`${"═".repeat(60)}`);
console.log(`\n\x1b[1mPerformance Summary:\x1b[0m`);
console.log(` ${loadStats.summary()}`);
console.log(` ${seqStats.summary()}`);
console.log(` ${multiToolStats.summary()}`);
console.log(` ${specialCharStats.summary()}`);
console.log(` ${streamStats.summary()}`);
console.log(` ${reliabilityStats.summary()}`);
console.log(` ${burstStats.summary()}`);
console.log(` ${largeArgStats.summary()}`);
console.log(`${"═".repeat(60)}\n`);

process.exit(failed > 0 ? 1 : 0);
