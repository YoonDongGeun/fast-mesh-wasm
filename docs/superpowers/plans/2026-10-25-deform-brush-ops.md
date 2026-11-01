# Brush deformation ops — Implementation Plan (editor ops 1 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One-shot inflate + Taubin-smooth brush ops callable through `scene.repair()`, each a single undo entry.

**Architecture:** New `deform` module in `fast-mesh-ops` (pure Transaction ops reusing `VertexFans` adjacency and `compute_vertex_normals`), wired through new `RepairOp` variants + `decode_repair` + TS `repair()` types, tested in Rust and in `edit.test.ts`.

**Tech Stack:** Rust (engine), wasm-bindgen bindings, TypeScript vitest.

**Spec:** `docs/superpowers/specs/2026-10-25-mesh-editor-ops-design.md` (Decided section binds field names, constants, and error semantics; remesh/join are Plans 2–3, not this plan).

## Global Constraints

- `cargo fmt --all --check` clean and `cargo clippy -p fast-mesh-ops --all-targets -- -D warnings` clean.
- No History changes; one undo entry per `repair()` call as today.
- `radius <= 0` or any non-finite brush number → error (`OpsError::InvalidInput`, surfaced as `INVALID_ARGUMENT`); negative strength (deflate) is allowed.
- Region membership is brute-force `|p - center| < radius` (strict interior; weight is exactly 0 at t=1 either way, so rim vertices never move — no BVH in this plan).
- Falloff exact: `Linear: (1-t)`, `Smooth: s = 1-t; s*s*(3-2*s)`, `t = d/radius`.
- `TAUBIN_LAMBDA = 0.33`, `TAUBIN_MU = -0.34`; one cycle = one λ pass then one μ pass, each reading a pre-pass snapshot.
- TS option names verbatim: `{ op: "inflate", center, radius, falloff: "linear"|"smooth", strength }`, `{ op: "smooth", center, radius, falloff, strength, cycles }`.
- New `RepairOp` f64 tags are the next free numbers after the existing ones (read `decode_repair` first).

## Review Focus

- Zero/negative radius and NaN strength must error, never silently no-op — pinned in Task 1 tests.
- Empty mesh (or empty region) must return `moved: 0`, not an error — pinned in Task 1 tests.
- Non-manifold fans must not crash neighbor lookup (union of fan vertices, build errors propagate) — pinned in Task 2 tests.
- Cycles = 0 must be a clean no-op returning `moved: 0` — pinned in Task 2 tests.
- Deflate (negative strength) must move vertices opposite the normal — pinned in Task 1 tests.

---

### Task 1: inflate op

**Files:**
- Create: `crates/fast-mesh-ops/src/deform.rs` (with `#[cfg(test)] mod tests` colocated)
- Modify: `crates/fast-mesh-ops/src/lib.rs` (add `pub mod deform;` + re-exports following the file's existing pattern)

**Interfaces:**
- Consumes: `Transaction::set_positions`, `Mesh::positions/triangles`, `compute_vertex_normals`, `OpsError::InvalidInput`, `VertexFans` (not needed for inflate; listed for orientation).
- Produces: `pub enum Falloff { Linear, Smooth }`, `pub struct Brush { pub center: [f32; 3], pub radius: f32, pub falloff: Falloff, pub strength: f32 }`, `pub struct InflateOptions { pub brush: Brush }` + `InflateOptionsBuilder` (`center/radius/falloff/strength/build`, following `FillOptionsBuilder`), `pub struct InflateReport { pub moved: usize, pub max_displacement: f32 }`, `pub fn inflate(tx: &mut Transaction, options: &InflateOptions) -> Result<InflateReport>` (same `Result` alias as `fill_holes`). Semantics: validate first (radius/non-finite → `InvalidInput`); region by brute-force distance; displacement `strength * falloff * normal`; single `set_positions`; `moved` counts vertices with displacement > 0.

- [ ] **Step 1: Write the failing tests** in `deform.rs` (`#[cfg(test)]`, mesh builders from the crate's `test_util`):

```rust
#[test]
fn falloff_exact_values() {
    assert!((falloff_value(Falloff::Linear, 0.5) - 0.5).abs() < 1e-6);
    assert!((falloff_value(Falloff::Smooth, 0.0) - 1.0).abs() < 1e-6);
    assert!((falloff_value(Falloff::Smooth, 1.0) - 0.0).abs() < 1e-6);
}

#[test]
fn inflate_moves_center_along_normal() {
    // 3x3 flat grid in z=0, spacing 1, centered at origin; brush center
    // origin, radius 10, strength 2, Linear. Expected: center z ≈ +2 (or
    // -2 following compute_vertex_normals' sign — see note), |corner| < |center|.
}
```

Note on the sign: do NOT hardcode +Z. Compute `n = compute_vertex_normals` for the center vertex in the test and assert `center == old + 2.0 * n` within `1e-4` (this pins inflate's wiring — falloff × strength along the normal — without pinning the normals convention). Also assert monotonic falloff (`|corner_disp| < |center_disp|`), `moved == 9`, `max_displacement ≈ 2.0`, deflate (`strength: -2.0` → `center == old - 2.0 * n`), empty region (`moved == 0`), and `radius: 0.0` / NaN strength → `InvalidInput` (one test each, exact names of your choice).

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p fast-mesh-ops deform`
Expected: FAIL with "cannot find" (`deform` module / `inflate` / `falloff_value` not defined).

- [ ] **Step 3: Implement `deform.rs`** (Brush/Falloff/options+builder/`inflate`/`InflateReport` per Interfaces; `falloff_value(f: Falloff, t: f32) -> f32` helper, `pub(crate)`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p fast-mesh-ops deform`
Expected: PASS (all Task 1 tests).

- [ ] **Step 5: Run lint gate**

Run: `cargo fmt --all --check && cargo clippy -p fast-mesh-ops --all-targets -- -D warnings`
Expected: clean, no output / no warnings.

- [ ] **Step 6: Commit**

```bash
git add crates/fast-mesh-ops/src/deform.rs crates/fast-mesh-ops/src/lib.rs
git commit -m "feat(ops): brush inflate op"
```

### Task 2: Taubin smooth op

**Files:**
- Modify: `crates/fast-mesh-ops/src/deform.rs` (append smooth + tests)
- Test: same file's `#[cfg(test)] mod tests`

**Interfaces:**
- Consumes: Task 1's `Brush`/`Falloff`/`falloff_value`; `VertexFans::build/fan` for neighbor triangles (neighbor vertices = deduped fan-triangle vertices excluding self); pre-pass snapshots.
- Produces: `pub struct SmoothOptions { pub brush: Brush, pub cycles: u32 }` + `SmoothOptionsBuilder` (`center/radius/falloff/strength/cycles/build`), `pub const TAUBIN_LAMBDA: f32 = 0.33`, `pub const TAUBIN_MU: f32 = -0.34`, `pub struct SmoothReport { pub moved: usize, pub max_displacement: f32 }`, `pub fn smooth(tx: &mut Transaction, options: &SmoothOptions) -> Result<SmoothReport>` (same alias). Semantics: validate as Task 1; each cycle = λ pass then μ pass, `p += w * k * (centroid(neighbors) - p)` with `w = strength * falloff`; vertices outside the region never move but still serve as neighbor sources; `cycles: 0` returns `moved: 0` without touching the mesh.

- [ ] **Step 1: Write the failing tests**:

```rust
#[test]
fn taubin_pyramid_exact_after_one_cycle() {
    // Open pyramid: base A(0,0,0) B(1,0,0) C(1,1,0) D(0,1,0), apex
    // E(0.5,0.5,1); 4 side triangles; brush center E, radius 0.1 (only E
    // in region), strength 1, Linear, cycles 1. E's neighbors stay fixed
    // (outside region), centroid (0.5,0.5,0): λ → z=0.67, μ → z=0.8978.
    // Assert E.z ≈ 0.8978 (abs < 1e-4), A/B/C/D bit-identical, moved == 1.
}
```

(If `VertexFans::build` rejects the open fan, close the base with 2 triangles — E's fan and the expected value are unchanged since base triangles don't touch E; note it in the test comment.) Plus: `cycles: 0` → `moved == 0` and mesh untouched; empty region → `moved == 0`; invalid brush → `InvalidInput`; one non-manifold fan test (two triangles sharing only vertex E with disjoint other vertices — smooth must not error and E must move toward their centroid: `dist(E_new, centroid) < dist(E_old, centroid)`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p fast-mesh-ops deform`
Expected: FAIL (`smooth`/`SmoothOptions`/consts not defined; Task 1 tests still pass).

- [ ] **Step 3: Implement** smooth + options/builder/consts/report in `deform.rs`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p fast-mesh-ops deform`
Expected: PASS (Task 1 + Task 2 tests).

- [ ] **Step 5: Run lint gate**

Run: `cargo fmt --all --check && cargo clippy -p fast-mesh-ops --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add crates/fast-mesh-ops/src/deform.rs
git commit -m "feat(ops): Taubin brush smooth op"
```

### Task 3: RepairOp + TS wiring and integration tests

**Files:**
- Modify: `crates/fast-mesh-wasm/src/edit.rs` (`RepairOp::{Inflate, Smooth}` variants + `decode_repair` tags + `run_repair` arms calling `ops::deform::{inflate, smooth}` and encoding reports, all following the `Fill` precedent at lines ~334-340, ~386-510)
- Modify: `packages/fast-mesh-wasm/src/scene.ts` (`RepairOp` union + option/report types at ~line 479, following the `fillHoles` precedent)
- Test: `packages/fast-mesh-wasm/test/edit.test.ts` (append `inflate`/`smooth` cases mirroring the `orient`/`fillHoles` cases at ~line 300)

**Interfaces:**
- Consumes: Tasks 1–2 ops; existing `decode_repair` f64 tag scheme; existing report encoders; TS `repair()` overloads.
- Produces: `s.repair(m, { op: "inflate", center, radius, falloff, strength })` and `s.repair(m, { op: "smooth", center, radius, falloff, strength, cycles })` returning `{ moved, maxDisplacement }` (camelCase keys per the existing report convention — verify against the `fillHoles` report keys in the same file), invalid args → `INVALID_ARGUMENT` (mirror lines ~358-359).

- [ ] **Step 1: Write the failing tests** in `edit.test.ts` (mirror an existing `orient` case's mesh setup): inflate moves the brushed vertex along its normal (`moved > 0`, displacement direction matches a `smooth`-normals expectation loosely — assert displacement magnitude > 0 and unbrushed vertices unchanged); smooth reduces a spike (`dist(apex, base-plane)` shrinks, `moved > 0`); `{ op: "inflate", radius: -1, ... }` and NaN strength → `codeOf(...) === "INVALID_ARGUMENT"`; undo restores positions (`s.undo()` then positions equal pre-op snapshot — one entry per call).

- [ ] **Step 2: Run tests to verify they fail**

Run (after one `pnpm --filter fast-mesh-wasm test` build if `dist/` is absent): `pnpm --filter fast-mesh-wasm vitest run test/edit.test.ts -t "inflate"`
Expected: FAIL (unknown op / decode error — not a pass).

- [ ] **Step 3: Implement** the Rust variants + decode + run arms + TS types. Report encoding mirrors the `fillHoles` report path exactly (same encoder style).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pnpm --filter fast-mesh-wasm vitest run test/edit.test.ts`
Expected: PASS (whole file, old + new).

- [ ] **Step 5: Run full gates**

Run: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` (root `lint`, clean) and `pnpm --filter fast-mesh-wasm test` (full package gate incl. wasm rebuild, PASS).

- [ ] **Step 6: Commit**

```bash
git add crates/fast-mesh-wasm/src/edit.rs packages/fast-mesh-wasm/src/scene.ts packages/fast-mesh-wasm/test/edit.test.ts
git commit -m "feat(js): inflate/smooth repair ops"
```
