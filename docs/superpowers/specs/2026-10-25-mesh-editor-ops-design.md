# Mesh editor ops — short design

**Date:** 2026-10-25. **Status:** draft (my calls; user opted out of questions).

## Intent and why

Make the mesh engine editor-like: brush deformation (inflate/smooth),
density remeshing under deformation (split/collapse to target length),
and mesh joining (bridge two meshes). Delivered as one-shot ops through
the existing `Scene::edit` + undo pipeline, shaped brush-first so an
interactive stroke loop can call them later.

## Goals

- Deform (Plan 1): inflate along smooth normals + Taubin smooth, masked
  by a brush (center/radius/falloff/strength), one undo entry per call.
- Remesh (Plan 2): region-targeted split/collapse/flip to local target
  length, generalizing `refine.rs` beyond hole patches.
- Join (Plan 3): loop-to-loop bridge between two meshes
  (resample + zip + weld) via existing `boundary_loops`.

## Non-goals (draft)

- Real-time sculpt loop, stroke coalescing UI, viewport brushes.
- BVH-accelerated brush lookup (brute-force distance filter first).
- Remesh surface projection (midpoint splits/collapses; drift bounded).
- Boolean union (robustness tarpit; bridging covers "similar shapes").
- TS-side topology ops (engine owns topology).

## Anti-goals

- Breaking undo granularity (one History entry per op call, as today).
- Perf cliffs on large meshes without measurement (bench exists; use it).
- Changing existing console/API output or compat.

## Constraints

- Ops live in `fast-mesh-ops`, `Transaction`-based, builder options
  following the `*OptionsBuilder` convention, `OpsError` results.
- Exposed via `RepairOp` variants + `decode_repair` + `scene.repair()`.
- Rust unit tests colocated; TS integration in `edit.test.ts`.
- `cargo fmt` clean, `clippy -- -D warnings` (root `lint`).

## Approaches

1. **Engine ops + thin bindings (chosen).** Topology stays in one
   place, wasm perf, undo free via `Scene::edit`.
2. TS-side ops. Rejected: duplicates topology, slow, no undo.
3. Real-time stroke loop now. Rejected for the draft: ops are
   brush-shaped so a later loop can call them per stroke.

## Decided

- Three plans in order: deform → remesh → join.
- Shared brush shape for deform: `{center, radius, falloff, strength}`; falloff
  `Linear(1-t)` / `Smooth(smoothstep(1-t))`, `t = d/radius`. Remesh uses
  `Region{center, radius}` instead (binary membership; falloff/strength
  are meaningless for topology ops).
- Smooth is Taubin (`TAUBIN_LAMBDA = 0.33`, `TAUBIN_MU = -0.34`),
  masked; neighbors read from a pre-pass snapshot.
- Invalid brush (`radius <= 0`, non-finite numbers) is an error
  (`INVALID_ARGUMENT`); negative strength (deflate) is allowed.
- Remesh: split/collapse/flip to per-vertex target length, no projection.
- Join: boundary-loop bridging only; boolean union stays a non-goal.

## Left to the builder

- Per-plan algorithms and exact report fields (in each plan).
- Whether `VertexFans` needs a vertex-neighbor view (read it first).
