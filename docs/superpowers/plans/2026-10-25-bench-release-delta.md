# Bench deltas on the release PR — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Version Packages PRs get a sticky comment with per-case time/memory bench deltas vs the previous release.

**Architecture:** Bench runners gain a `--json` mode via a small shared reporter (console output unchanged); a dependency-free compare script diffs head vs the previous `fast-mesh-wasm@*` tag; a new workflow runs both sides back-to-back on one runner and posts one sticky comment.

**Tech Stack:** Node 24 ESM (no new deps), vitest (package tests), `node --test` (script test), GitHub Actions + github-script.

**Spec:** `docs/superpowers/specs/2026-10-25-bench-release-delta-design.md`

## Global Constraints

- Bench console output stays byte-identical without `--json` (existing log readers keep working).
- Scene bench JSON units are all `ms`; store bench uses `ms` for bursts/adds and `µs` for per-event rows.
- JSON schema field `schema` is exactly `1`.
- Noise threshold is exactly 5% (`|Δ| < 5%` → ➖), one named const.
- No new npm dependencies anywhere.
- Heap column appears only when sampled (never fabricated).

## Review Focus

- A case renamed between releases must show as added+removed, never silently dropped or mismatched — pinned in Task 4 tests.
- A missing baseline tag (shallow checkout, no tags) must fail with a clear error, not an empty comment — pinned in Task 5 rehearsal.
- An unwritable `--json` path must fail the bench loudly (non-zero exit), not skip silently — pinned in Task 1 tests.
- A same-code rerun must not flip 🟢/🔴 (threshold hysteresis is out of scope; the test pins ➖ for a 2% delta) — pinned in Task 4 tests.
- `--json` combined with `--quick`/`--worker`/filters must still emit valid JSON with only the cases that ran — pinned in Task 2 tests.

---

### Task 1: JSON reporter + scene bench adoption

**Files:**
- Create: `packages/fast-mesh-wasm/bench/report.mjs`
- Create: `packages/fast-mesh-wasm/test/bench-report.test.mjs`
- Modify: `packages/fast-mesh-wasm/bench/scene.bench.mjs` (all `report()` def + call sites, raw measurement `console.log`s)

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `createReporter(argv: string[], opts: {suite: string}) -> {report(name, stats, unit?), log(line), save(): void}` where `stats = {median, p95, min?, max?, n}`; `--json <path>` parsed from `argv`; `save()` writes the schema-v1 document (no-op without the flag).

- [ ] **Step 1: Write the failing test** in `packages/fast-mesh-wasm/test/bench-report.test.mjs`:

```js
import { strict as assert } from "node:assert";
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm --filter fast-mesh-wasm vitest run test/bench-report.test.mjs`
Expected: FAIL with "Cannot find module …/bench/report.mjs".

- [ ] **Step 3: Implement `createReporter` in `packages/fast-mesh-wasm/bench/report.mjs`**

Console passthrough: `report()`/`log()` print exactly what the bench printed
before; with `--json`, cases accumulate and `save()` writes the file (throw on
error). Expose `document()` for tests (returns what `save()` would write).

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm --filter fast-mesh-wasm vitest run test/bench-report.test.mjs`
Expected: PASS (2 tests).

- [ ] **Step 5: Adopt in `scene.bench.mjs`**

Route the `report()` definition (line ~136) and every measurement print
(`report()` calls plus raw `console.log` measurements such as the drag setup
lines) through the reporter; call `save()` at each `process.exit(0)` site and
at end of file. Console text unchanged.

- [ ] **Step 6: Verify adoption**

Run: `node --check packages/fast-mesh-wasm/bench/scene.bench.mjs` (PASS, no
output) and `pnpm --filter fast-mesh-wasm vitest run test/bench-report.test.mjs`
(PASS). Full `--json` smoke run happens in Task 5 (needs a release build).

- [ ] **Step 7: Commit**

```bash
git add packages/fast-mesh-wasm/bench/report.mjs packages/fast-mesh-wasm/test/bench-report.test.mjs packages/fast-mesh-wasm/bench/scene.bench.mjs
git commit -m "feat(bench): JSON reporter and scene bench adoption"
```

### Task 2: store bench adoption (mixed units)

**Files:**
- Modify: `packages/fast-mesh-wasm/bench/store.bench.mjs` (`row()` + burst/adds logs)
- Modify: `packages/fast-mesh-wasm/test/bench-report.test.mjs` (add unit-mixing + filter tests)

**Interfaces:**
- Consumes: `createReporter` from Task 1 (same signature).
- Produces: store bench `--json` output with `ms` bursts and `µs` rows.

- [ ] **Step 1: Write the failing tests** (append to `bench-report.test.mjs`):

```js
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pnpm --filter fast-mesh-wasm vitest run test/bench-report.test.mjs`
Expected: FAIL (`quick` key missing; anything asserting new behavior).

- [ ] **Step 3: Implement**

Add `quick: argv.includes("--quick")` to the document in `report.mjs`; adopt the
reporter in `store.bench.mjs` (`row()` gains a unit arg: `µs` for per-event
rows, `ms` for the add burst and extra-adds lines). Console text unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pnpm --filter fast-mesh-wasm vitest run test/bench-report.test.mjs`
Expected: PASS (4 tests).

- [ ] **Step 5: Verify adoption**

Run: `node --check packages/fast-mesh-wasm/bench/store.bench.mjs` (PASS, no output).

- [ ] **Step 6: Commit**

```bash
git add packages/fast-mesh-wasm/bench/report.mjs packages/fast-mesh-wasm/bench/store.bench.mjs packages/fast-mesh-wasm/test/bench-report.test.mjs
git commit -m "feat(bench): store bench JSON adoption with mixed units"
```

### Task 3: heap sampling (memory column)

**Files:**
- Modify: `packages/fast-mesh-wasm/bench/report.mjs` (sampler)
- Modify: `packages/fast-mesh-wasm/test/bench-report.test.mjs`

**Interfaces:**
- Consumes: `createReporter` from Task 1.
- Produces: cases gain optional `heapMB` (number, one decimal) sampled after
  `globalThis.gc?.()` at each `report()` call; omitted when sampling is
  impossible (never zero-filled).

- [ ] **Step 1: Write the failing test**:

```js
test("report samples heapMB after gc when available", () => {
  const r = createReporter(["--json", "X"], { suite: "scene" });
  r.report("a", { median: 1, p95: 2, n: 7 }, "ms");
  const c = r.document().cases[0];
  assert.ok(typeof c.heapMB === "number" && c.heapMB > 0);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm --filter fast-mesh-wasm vitest run test/bench-report.test.mjs`
Expected: FAIL (`heapMB` undefined).

- [ ] **Step 3: Implement**

In `report()`: `globalThis.gc?.()` then
`heapMB: process.memoryUsage().heapUsed / 1048576` rounded to one decimal.
(Under vitest without `--expose-gc`, `gc` is absent and sampling still runs —
document that CI passes `NODE_OPTIONS=--expose-gc` for stable numbers.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `NODE_OPTIONS=--expose-gc pnpm --filter fast-mesh-wasm vitest run test/bench-report.test.mjs`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/fast-mesh-wasm/bench/report.mjs packages/fast-mesh-wasm/test/bench-report.test.mjs
git commit -m "feat(bench): heap sampling per case"
```

### Task 4: compare script

**Files:**
- Create: `scripts/bench-compare.mjs`
- Create: `scripts/bench-compare.test.mjs`

**Interfaces:**
- Consumes: schema-v1 JSON documents from Tasks 1–3.
- Produces: `compare(baseline, current) -> {rows, added[], removed[]}`,
  `renderMarkdown(result, {baseRef, headRef}) -> string`; CLI
  `node scripts/bench-compare.mjs <base.json> <head.json> [--out body.md]`
  (markdown to stdout without `--out`). Row: `{name, unit, base, head,
  deltaPct, p95DeltaPct, heapDeltaMB?, status}` with `status ∈ {🟢,🔴,➖}`
  at the 5% threshold (`NOISE_PCT = 5` const). Match cases by exact `name`;
  unit mismatch → treat as removed+added.

- [ ] **Step 1: Write the failing test** in `scripts/bench-compare.test.mjs`
  (`node:test` + `node:assert/strict`):

```js
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node --test scripts/bench-compare.test.mjs`
Expected: FAIL (cannot find module `./bench-compare.mjs`).

- [ ] **Step 3: Implement `compare`/`renderMarkdown`/CLI in `scripts/bench-compare.mjs`**

Delta = `(head - base) / base * 100` on medians (same for p95); heap delta in
MB when both sides sampled. Markdown: one table per suite document pair plus
added/removed lines; header names both refs.

- [ ] **Step 4: Run test to verify it passes**

Run: `node --test scripts/bench-compare.test.mjs`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/bench-compare.mjs scripts/bench-compare.test.mjs
git commit -m "feat(bench): head-vs-release compare script"
```

### Task 5: release-bench workflow + docs

**Files:**
- Create: `.github/workflows/bench-release.yml`
- Modify: `packages/fast-mesh-wasm/bench/scene.bench.mjs` header comment (document `--json`; 3 lines)
- Modify: `packages/fast-mesh-wasm/bench/store.bench.mjs` header comment (document `--json`; 3 lines)

**Interfaces:**
- Consumes: `--json` mode (Tasks 1–3), `scripts/bench-compare.mjs` (Task 4).
- Produces: sticky PR comment with the delta table on every `changeset-release/*` PR update.

Workflow (pin exactly): `on: pull_request: types: [opened, synchronize]`,
`if: startsWith(github.head_ref, 'changeset-release/')`, one job on
`ubuntu-latest`, permissions `contents: read` + `pull-requests: write`.
Steps: checkout PR (`fetch-tags: true`, so the baseline tag exists) →
resolve baseline `git describe --tags --abbrev=0 origin/main` (fail loudly
with "no release tag found" when absent) → second checkout of the tag to
`baseline/` → setup (node 24, pnpm, rust + nightly + wasm-bindgen, same pins
as `release.yml`) → `pnpm install` + `pnpm build` in both trees →
`NODE_OPTIONS=--expose-gc` bench runs with `--json` (default suite, both
files) → `node --test scripts/bench-compare.test.mjs` as a gate →
compare both suites → sticky comment via `actions/github-script` (find bot
comment by marker `<!-- bench-delta -->`, update or create).

- [ ] **Step 1: Rehearse every job command locally** (proves the chain before any YAML runs)

Run, in order: two release builds (current tree + a `git worktree` of the
latest tag), both bench files with `--json` in each, the compare CLI on both
pairs. Expected: two valid JSON pairs, one markdown table each, exit 0.

- [ ] **Step 2: Write `.github/workflows/bench-release.yml`** per the interface above.

- [ ] **Step 3: Validate registration**

Push the branch, then run: `gh workflow view bench-release --json path -q .path`. Expected: prints `.github/workflows/bench-release.yml` (proves GitHub parsed and registered it).

- [ ] **Step 4: Live smoke test (needs explicit user go-ahead at execution time)**

Push the branch, open a draft PR from `changeset-release/smoke-test`, confirm
the workflow runs and the sticky comment appears with real numbers, then close
the PR unmerged and delete the branch.

- [ ] **Step 5: Document the flags** (3-line header comments in both bench files).

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/bench-release.yml packages/fast-mesh-wasm/bench/scene.bench.mjs packages/fast-mesh-wasm/bench/store.bench.mjs
git commit -m "feat(ci): bench deltas on the release PR"
```
