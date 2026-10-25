import { strict as assert } from "node:assert";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "vitest";
import { createReporter } from "../bench/report.mjs";

test("save writes schema-v1 JSON with only reported cases", () => {
  const r = createReporter(["--json", "X"], { suite: "scene" });
  r.report("a", { median: 1, p95: 2, n: 7 }, "ms");
  const doc = r.document();
  assert.equal(doc.schema, 1);
  assert.equal(doc.cases.length, 1);
  assert.equal(doc.cases[0].unit, "ms");
});

test("unwritable --json path throws on save", () => {
  const r = createReporter(["--json", "/nonexistent-dir-xyz/out.json"], { suite: "scene" });
  assert.throws(() => r.save(), /ENOENT|EACCES/);
});

test("save writes the document to the --json path", () => {
  const out = join(mkdtempSync(join(tmpdir(), "bench-report-")), "out.json");
  const r = createReporter(["--json", out], { suite: "scene" });
  r.report("a", { median: 1, p95: 2, min: 0.5, max: 3, n: 7 }, "ms");
  r.log("a console-only line");
  r.save();
  const doc = JSON.parse(readFileSync(out, "utf8"));
  assert.equal(doc.schema, 1);
  assert.equal(doc.suite, "scene");
  assert.equal(doc.cases.length, 1);
  assert.deepEqual(doc.cases[0], { name: "a", unit: "ms", median: 1, p95: 2, min: 0.5, max: 3, n: 7 });
});

test("save without --json is a no-op", () => {
  const r = createReporter([], { suite: "scene" });
  r.report("a", { median: 1, p95: 2, n: 7 }, "ms");
  r.save();
});

test("report prints the exact bench stats line", () => {
  const lines = [];
  const orig = console.log;
  console.log = (line) => lines.push(line);
  try {
    createReporter([], { suite: "scene" }).report("a", { median: 1, p95: 2, min: 0.5, max: 3, n: 7 }, "ms");
  } finally {
    console.log = orig;
  }
  assert.deepEqual(lines, ["a: median 1.00 ms, p95 2.00 ms, min 0.50, max 3.00 (n=7)"]);
});

test("log passes the line through to the console", () => {
  const lines = [];
  const orig = console.log;
  console.log = (line) => lines.push(line);
  try {
    createReporter([], { suite: "scene" }).log("hello");
  } finally {
    console.log = orig;
  }
  assert.deepEqual(lines, ["hello"]);
});
