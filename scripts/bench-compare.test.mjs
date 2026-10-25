import { test } from "node:test";
import assert from "node:assert/strict";
import { compare } from "./bench-compare.mjs";

const doc = (cases) => ({ schema: 1, cases });

test("2% delta is noise, 10% faster is green, renames split", () => {
  const r = compare(
    doc([{ name: "a", unit: "ms", median: 100, p95: 100 }, { name: "old", unit: "ms", median: 50, p95: 50 }]),
    doc([{ name: "a", unit: "ms", median: 98, p95: 98 }, { name: "new", unit: "ms", median: 45, p95: 45 }]),
  );
  assert.equal(r.rows[0].status, "➖");
  assert.deepEqual(r.added, ["new"]);
  assert.deepEqual(r.removed, ["old"]);
});

test("10% faster is green, 10% slower is red", () => {
  const r = compare(
    doc([{ name: "g", unit: "ms", median: 100, p95: 100 }, { name: "b", unit: "ms", median: 100, p95: 100 }]),
    doc([{ name: "g", unit: "ms", median: 90, p95: 90 }, { name: "b", unit: "ms", median: 110, p95: 110 }]),
  );
  assert.deepEqual(r.rows.map((x) => x.status), ["🟢", "🔴"]);
});
