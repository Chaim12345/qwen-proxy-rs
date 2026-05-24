#!/usr/bin/env node
/**
 * HTTP contract checklist validator for qwen-proxy-rs.
 * Asserts OpenAI-shaped responses for raw fetch/curl clients.
 *
 * Run from qtalt/: node qwen-proxy-rs/tests/contract_checklist_test.mjs
 */
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const checklist = JSON.parse(
  readFileSync(join(__dirname, "contract_checklist.json"), "utf8"),
);

const BASE = (process.env.PROXY_URL || "http://127.0.0.1:8765/v1").replace(
  /\/v1\/?$/,
  "",
);

let passed = 0;
let failed = 0;

function assert(condition, msg) {
  if (!condition) throw new Error(msg);
}

function headerMap(res) {
  const out = {};
  res.headers.forEach((v, k) => {
    out[k.toLowerCase()] = v;
  });
  return out;
}

function checkJsonShape(data, rules, label) {
  if (rules.object) {
    assert(data.object === rules.object, `${label}: object must be '${rules.object}', got '${data.object}'`);
  }
  if (rules.idPrefix) {
    assert(
      typeof data.id === "string" && data.id.startsWith(rules.idPrefix),
      `${label}: id must start with '${rules.idPrefix}'`,
    );
  }
  for (const field of rules.requiredFields || []) {
    assert(data[field] !== undefined && data[field] !== null, `${label}: missing '${field}'`);
  }
  if (rules.arrayField) {
    const arr = data[rules.arrayField];
    assert(Array.isArray(arr) && arr.length > 0, `${label}: '${rules.arrayField}' must be non-empty array`);
    if (rules.arrayItemFields) {
      for (const item of arr) {
        for (const f of rules.arrayItemFields) {
          assert(item[f] !== undefined, `${label}: array item missing '${f}'`);
        }
      }
    }
  }
  if (rules.choicesFields) {
    const choice = data.choices?.[0];
    assert(choice, `${label}: choices[0] missing`);
    for (const f of rules.choicesFields) {
      assert(choice[f] !== undefined, `${label}: choice missing '${f}'`);
    }
  }
  if (rules.messageFields) {
    const msg = data.choices?.[0]?.message;
    assert(msg, `${label}: choices[0].message missing`);
    for (const f of rules.messageFields) {
      assert(msg[f] !== undefined, `${label}: message missing '${f}'`);
    }
  }
  if (rules.allowFinishReasons) {
    const fr = data.choices?.[0]?.finish_reason;
    assert(
      rules.allowFinishReasons.includes(fr),
      `${label}: finish_reason '${fr}' not in [${rules.allowFinishReasons.join(", ")}]`,
    );
  }
}

function checkErrorShape(data, label) {
  assert(data.error, `${label}: missing error object`);
  assert(typeof data.error.message === "string", `${label}: error.message must be string`);
  assert(typeof data.error.type === "string", `${label}: error.type must be string`);
}

async function parseSse(res) {
  const text = await res.text();
  const lines = text.split("\n");
  const dataLines = lines.filter((l) => l.startsWith("data: ")).map((l) => l.slice(6));
  assert(dataLines.length > 0, "SSE: no data lines");
  let sawChunk = false;
  let sawDone = false;
  for (const line of dataLines) {
    if (line.trim() === "[DONE]") {
      sawDone = true;
      continue;
    }
    const chunk = JSON.parse(line);
    if (chunk.object === "chat.completion.chunk") sawChunk = true;
  }
  return { sawChunk, sawDone };
}

async function runCase(item) {
  const url = `${BASE}${item.path}`;
  const headers = {
    "content-type": "application/json",
    ...(item.headers || {}),
  };

  const init = {
    method: item.method,
    headers,
  };
  if (item.body !== undefined) {
    init.body = JSON.stringify(item.body);
  }

  const res = await fetch(url, init);
  assert(
    res.status === item.expectStatus,
    `expected HTTP ${item.expectStatus}, got ${res.status}`,
  );

  if (item.expectContentTypeIncludes) {
    const ct = res.headers.get("content-type") || "";
    assert(
      ct.includes(item.expectContentTypeIncludes),
      `content-type must include '${item.expectContentTypeIncludes}', got '${ct}'`,
    );
  }

  if (item.expectResponseHeaders) {
    const h = headerMap(res);
    for (const [k, v] of Object.entries(item.expectResponseHeaders)) {
      assert(h[k.toLowerCase()] === v, `header ${k} expected '${v}', got '${h[k.toLowerCase()]}'`);
    }
  }

  if (item.expectSse) {
    const { sawChunk, sawDone } = await parseSse(res);
    assert(sawChunk, "SSE: expected chat.completion.chunk");
    if (item.expectSse.requireDone) assert(sawDone, "SSE: expected [DONE] sentinel");
    return;
  }

  const data = await res.json();
  if (item.expectError) {
    checkErrorShape(data, item.id);
    return;
  }
  if (item.expectJson) {
    checkJsonShape(data, item.expectJson, item.id);
  }
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

// Health gate
try {
  const h = await fetch(`${BASE}/health`);
  assert(h.ok, `health check failed: HTTP ${h.status}`);
} catch (e) {
  console.error(`\x1b[31mProxy not reachable at ${BASE}/health\x1b[0m`);
  console.error(`  ${e.message}`);
  process.exit(1);
}

for (const item of checklist.endpoints) {
  await test(`contract: ${item.id}`, () => runCase(item));
}

console.log(`\n\n${"═".repeat(60)}`);
console.log("\x1b[1mHTTP Contract Checklist Results\x1b[0m");
console.log(`${"═".repeat(60)}`);
console.log(`  \x1b[32mPassed: ${passed}\x1b[0m`);
console.log(`  \x1b[31mFailed: ${failed}\x1b[0m`);
console.log(`  Total:  ${passed + failed}`);
console.log(`${"═".repeat(60)}\n`);

process.exit(failed > 0 ? 1 : 0);
