# Region remesh ops — Implementation Plan (editor ops 2 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One-shot region remeshing (split long / collapse short / flip to Delaunay) to a target edge length, callable through `scene.repair()`, one undo entry.

**Architecture:** New `remesh` module in `fast-mesh-ops`: pure computation over `&Mesh` (rounds of split → collapse → flip, midpoint only, no projection) committed through one `Transaction`, wired through a new `RepairOp` variant + `decode_repair` + TS `repair()` types, tested in Rust and in `edit.test.ts`.

**Tech Stack:** Rust (engine), wasm-bindgen bindings, TypeScript vitest.

**Spec:** `docs/superpowers/specs/2026-10-25-mesh-editor-ops-design.md` (Decided section; remesh uses `Region{center, radius}` — binary membership, no falloff — by deliberate deviation recorded below; join is Plan 3, not this plan).

## Global Constraints

- `cargo fmt --all --check` clean and `cargo clippy -p fast-mesh-ops --all-targets -- -D warnings` clean.
- No History changes; one undo entry per `repair()` call as today.
- Region membership is brute-force `|p - center| < radius` (strict interior, as Plan 1).
- `radius <= 0` or any non-finite region number → error (`OpsError::InvalidInput`, surfaced as `INVALID_ARGUMENT`).
- `targetLength` NaN or `<= 0` → auto: the global mean live-edge length of the mesh (never the region's own mean — a sparse region must still refine).
- `SPLIT_FACTOR = 4.0 / 3.0` (split edges longer than `σ * SPLIT_FACTOR`), `COLLAPSE_FACTOR = 4.0 / 5.0` (collapse shorter than `σ * COLLAPSE_FACTOR`), `REMESH_MAX_ROUNDS = 10`; all named consts.
- New vertices at edge midpoints (no surface projection); existing vertex positions are never modified, only added/removed (bit-identical survivors).
- Boundary edges: splits allowed (midpoint subdivision preserves the loop polyline); collapse and flip forbidden on boundary edges.
- TS option names verbatim: `{ op: "remesh", center, radius, targetLength }` (`targetLength: number | null`; null → auto); report `{ splits, collapses, flips, rounds }`.
- New `RepairOp` f64 tag is the next free number after the existing ones (read `decode_repair` first, noting the Plan 1 tags).

## Review Focus

- Empty region must return all-zero counts, not an error — pinned in Task 1 tests.
- Target larger than every edge must split nothing (`splits == 0`) while still collapse/flip checking — pinned in Task 1 tests.
- Boundary loops must survive: every original boundary vertex id is still a live vertex afterwards — pinned in Task 2 tests.
- Degenerate input triangles must be skipped, never crash, and their count must not grow — pinned in Task 2 tests.
- NaN `targetLength` must behave exactly like auto (same counts as an explicit global-mean run) — pinned in Task 3 tests.

---

### Task 1: split long edges to target

**Files:**
- Create: `crates/fast-mesh-ops/src/remesh.rs` (with `#[cfg(test)] mod tests` colocated)
- Modify: `crates/fast-mesh-ops/src/lib.rs` (add `pub mod remesh;` + re-exports following the `deform` pattern)

**Interfaces:**
- Consumes: `Mesh::positions/triangles`, `VertexFans`/`EdgeMap` (topology.rs), `Transaction::{add_vertex_with_parents, add_triangles, remove_triangles}` (verify exact signatures in transaction.rs; `add_vertex_with_parents` takes the edge endpoints for subdivision), `OpsError::InvalidInput`.
- Produces: `pub struct Region { pub center: [f32; 3], pub radius: f32 }`, `pub struct RemeshOptions { pub region: Region, pub target_length: Option<f32>, pub max_rounds: u32 }` + `RemeshOptionsBuilder` (`center/radius/target_length/max_rounds/build`, following `*OptionsBuilder`), `pub struct RemeshReport { pub splits: usize, pub collapses: usize, pub flips: usize, pub rounds: usize }`, `pub fn remesh(tx: &mut Transaction, options: &RemeshOptions) -> Result<RemeshReport>` (same `Result` alias as `fill_holes`). Task 1 implements the split pass only (collapses/flips stay 0): rounds of midpoint splits of region edges longer than `σ * SPLIT_FACTOR` until a round splits nothing or `max_rounds` (default `REMESH_MAX_ROUNDS`); degenerate triangles never split; one terminal commit through `tx`.

- [ ] **Step 1: Write the failing tests** in `remesh.rs` (`#[cfg(test)]`, mesh builders from the crate's `test_util`):

```rust
#[test]
fn split_refines_sparse_quad_to_target() {
    // Two triangles forming a 4x1 quad (long edges 4.0), region covering
    // all, target 1.0. Assert: splits > 0, every region edge <= 4/3 + 1e-4,
    // original vertex positions bit-identical, triangle count increased.
}

#[test]
fn split_empty_region_and_coarse_target() {
    // Far-away region → all counts 0, mesh untouched; target 100.0 on the
    // quad → splits == 0.
}
```

Plus: `radius: 0.0` / NaN center → `InvalidInput`; boundary edge split keeps every original boundary vertex live.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p fast-mesh-ops remesh`
Expected: FAIL with "cannot find" (`remesh` module / `RemeshOptions` / `remesh` not defined).

- [ ] **Step 3: Implement** `remesh.rs` (Region/options+builder/report/`remesh` with the split pass; target resolution: `Some(t)` if finite and > 0 else global mean; midpoint splits via `add_vertex_with_parents` + triangle rewrite).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p fast-mesh-ops remesh`
Expected: PASS (all Task 1 tests).

- [ ] **Step 5: Run lint gate**

Run: `cargo fmt --all --check && cargo clippy -p fast-mesh-ops --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add crates/fast-mesh-ops/src/remesh.rs crates/fast-mesh-ops/src/lib.rs
git commit -m "feat(ops): region remesh splits to target length"
```

### Task 2: collapse short edges + Delaunay flips

**Files:**
- Modify: `crates/fast-mesh-ops/src/remesh.rs` (append collapse + flip passes + tests)
- Test: same file's `#[cfg(test)] mod tests`

**Interfaces:**
- Consumes: Task 1's `Region`/`RemeshOptions`/`RemeshReport`/`remesh`; same tx primitives plus position reads for fold checks.
- Produces: full `remesh` = per round split → collapse → flip until fixpoint or `max_rounds`. Collapse: region edges shorter than `σ * COLLAPSE_FACTOR` collapse to midpoint; skip boundary edges, skip when a resulting triangle duplicates an existing one or is degenerate. Flip: interior region edges whose opposite angles sum past π flip when the new diagonal is new and neither new triangle folds against the old pair (same criterion family as `refine.rs` lines 68-71 — read it, do not refactor it).

- [ ] **Step 1: Write the failing tests**:

```rust
#[test]
fn collapse_densifies_tiny_cluster() {
    // Small triangles (edges ~0.1), region covering all, target 1.0.
    // Assert: collapses > 0, triangle count decreased, every remaining
    // region edge >= 0.8 - 1e-4, no degenerate triangles, survivors bit-identical.
}

#[test]
fn flip_fixes_needle_diagonal() {
    // Kite quad P1(0,0,0) P2(1,0,0) P3(0.5,0.42,0) P4(0.5,-0.42,0),
    // tris (P1,P2,P3),(P1,P4,P2) (both CCW from +Z); region covering
    // all; target 0.78. Opposite angles ~100° + ~100° > 180° so the
    // P1P2 diagonal flips; edges (1.0 longest, ~0.653 shortest) sit
    // strictly inside the split (> 1.04) and collapse (< 0.624) bands.
    // Assert: splits == 0, collapses == 0, flips == 1, edge (P3,P4)
    // exists afterwards, edge (P1,P2) gone.
}
```

Plus: boundary loop survival (all original boundary vertex ids live, via `boundary_loops` before/after); degenerate input skipped (no error, degenerate count not grown); `max_rounds: 0` → all counts 0, untouched.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p fast-mesh-ops remesh`
Expected: FAIL (collapse/flip behavior missing — e.g. `flips == 1` fails with 0; Task 1 tests still pass).

- [ ] **Step 3: Implement** the collapse + flip passes and the round loop in `remesh.rs`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p fast-mesh-ops remesh`
Expected: PASS (Task 1 + Task 2 tests).

- [ ] **Step 5: Run lint gate**

Run: `cargo fmt --all --check && cargo clippy -p fast-mesh-ops --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add crates/fast-mesh-ops/src/remesh.rs
git commit -m "feat(ops): remesh collapse and Delaunay flips"
```

### Task 3: RepairOp + TS wiring and integration tests

**Files:**
- Modify: `crates/fast-mesh-wasm/src/edit.rs` (`RepairOp::Remesh` + `decode_repair` tag + `run_repair` arm calling `ops::remesh::remesh`, following the Inflate/Smooth precedent)
- Modify: `packages/fast-mesh-wasm/src/scene.ts` (`RepairOp` union + option/report types, following `InflateReport`)
- Modify: `packages/fast-mesh-wasm/src/node.ts`, `packages/fast-mesh-wasm/src/web.ts` (re-export `RemeshReport` wherever `FillReport` is re-exported — learned from Plan 1)
- Test: `packages/fast-mesh-wasm/test/edit.test.ts` (append `remesh` cases mirroring the inflate/smooth cases)

**Interfaces:**
- Consumes: Tasks 1–2 `remesh`; existing decode/tag/report-encoder precedent.
- Produces: `s.repair(m, { op: "remesh", center, radius, targetLength })` returning `{ splits, collapses, flips, rounds }`; `targetLength: null` → auto; invalid region → `INVALID_ARGUMENT`; one undo entry per call (existing behavior — test pins it).

- [ ] **Step 1: Write the failing tests** in `edit.test.ts` (mirror the inflate case's mesh setup): sparse-region remesh returns `splits > 0` with edge lengths bounded; dense-region returns `collapses > 0`; `{ radius: -1 }` → `INVALID_ARGUMENT`; NaN `targetLength` behaves exactly like auto (same counts as an explicit `targetLength` equal to the mesh mean — compute the mean in-test from the same mesh); undo restores the triangle count (one entry).

- [ ] **Step 2: Run tests to verify they fail**

Run (after one `pnpm --filter fast-mesh-wasm test` build if `dist/` is absent): `pnpm --filter fast-mesh-wasm vitest run test/edit.test.ts -t "remesh"`
Expected: FAIL (unknown op / decode error — not a pass).

- [ ] **Step 3: Implement** the Rust variant + decode + run arm + TS types + entry re-exports. Report encoding mirrors the inflate report path exactly.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pnpm --filter fast-mesh-wasm vitest run test/edit.test.ts`
Expected: PASS (whole file, old + new).

- [ ] **Step 5: Run full gates**

Run: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` (root `lint`, clean) and `pnpm --filter fast-mesh-wasm test` (full package gate incl. wasm rebuild, PASS).

- [ ] **Step 6: Commit**

```bash
git add crates/fast-mesh-wasm/src/edit.rs packages/fast-mesh-wasm/src/scene.ts packages/fast-mesh-wasm/src/node.ts packages/fast-mesh-wasm/src/web.ts packages/fast-mesh-wasm/test/edit.test.ts
git commit -m "feat(js): remesh repair op"
```
