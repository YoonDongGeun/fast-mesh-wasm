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

test("mixed units are kept per case", () => {
  const r = createReporter(["--json", "X"], { suite: "store" });
  r.report("burst", { median: 10, p95: 11, n: 3 }, "ms");
  r.report("event", { median: 1.5, p95: 2.5, n: 200 }, "µs");
  const units = r.document().cases.map((c) => c.unit);
  assert.deepEqual(units, ["ms", "µs"]);
});

test("quick flag is recorded and console still prints", () => {
  const r = createReporter(["--json", "X", "--quick"], { suite: "store" });
  assert.equal(r.document().quick, true);
});

test("record stores a case without printing", () => {
  const lines = [];
  const orig = console.log;
  console.log = (line) => lines.push(line);
  let doc;
  try {
    const r = createReporter(["--json", "X"], { suite: "store" });
    r.record("row", { median: 1.5, p95: 2.5, n: 200 }, "µs");
    doc = r.document();
  } finally {
    console.log = orig;
  }
  assert.deepEqual(lines, []);
  assert.deepEqual(doc.cases, [{ name: "row", unit: "µs", median: 1.5, p95: 2.5, n: 200 }]);
});
