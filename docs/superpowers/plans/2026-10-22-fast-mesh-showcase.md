# Showcase app and contact heatmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a `contactHeatmap` colouring preset and `SceneStore.fetchColoringValues` to the library, and build `examples/showcase`. The showcase is a React + R3F tool where users load meshes, move them, repair them, and inspect the collision between two meshes as a depth heatmap.

**Spec:** `docs/superpowers/specs/2026-10-22-fast-mesh-showcase-design.md`. §6 overrides earlier sections.

**Workspace:** C:/Users/YoonCarrot/orca/projects/fast-mesh-wasm-showcase, on branch `feat/showcase`. This is a separate clone; never touch the main checkout.

## Global Constraints
- **Library:**
  - Changes are additive only: `contactHeatmap` in `./store`, and `SceneStore.fetchColoringValues(mesh)`.
  - Add a minor changeset.
  - `./store` still must not import React, three or the wasm glue.
- **Exact colouring:** no approximation.
  - `exactDepth: true` sets `maxDistance: Infinity`.
  - `exactDepth: false` (the library default) sets `maxDistance: max(depth, gap)`.
  - The showcase defaults to `true`, with a "Fast mode" toggle.
- **Never freeze the UI:**
  - Parsing happens in the example's parser Worker, never on the main thread.
  - Long operations show busy states.
  - Readouts throttle to about 4 Hz during drags and refresh on drag end.
- **Undo:** every repair goes through `store.repair` and is undoable. Analyze is read-only.
- **No new gizmo dependency:** use three's `TransformControls` from `three/addons`, the same way `examples/react` does.
- **Gate** (each task ends green):
  - `cargo fmt --check` / `clippy -D warnings` / `cargo test --workspace`
  - `pnpm build`, `pnpm --filter fast-mesh-wasm test`, `pnpm check:pkg`
  - the webgpu, react and showcase example builds
  - `node scripts/browser-check.mjs --threads 4`
  - from Task 3 on, also `node scripts/showcase-check.mjs`
  - a final `pnpm build`

## Review Focus
1. **Open target mesh.** A heatmap against an open mesh still colours, and the panel shows the open/inconsistent warning. Owned by Task 3 (`showcase-check`: the C-mesh collision step).
2. **Tolerance change.** Changing the tolerance never rebuilds the field; the field-add count stays the same. Owned by Task 1 (`tolerance_change_reuses_field`).
3. **Removing a collision mesh.** Removing A or B while it is coloured resets the picker, raises no errors and stops the readouts. Owned by Task 3.
4. **Large file.** A 1M-triangle STL upload keeps the main thread responsive: the longest main-thread block stays under 50 ms during load. Owned by Task 2 (`showcase-check --big`, measured with a `PerformanceObserver` long-task probe).
5. **Repair while coloured.** Repairing a mesh that is coloured keeps the colouring, which updates, and undo restores both. Owned by Task 3.

---

### Task 1: Library `contactHeatmap` and `fetchColoringValues`

**Files:**
- Create: `packages/fast-mesh-wasm/src/store/heatmap.ts`.
- Modify: `packages/fast-mesh-wasm/src/store/index.ts` (export `contactHeatmap`, `ContactHeatmapOptions`).
- Modify: `packages/fast-mesh-wasm/src/store/store.ts` and `host.ts` (`fetchColoringValues`).
- Modify: README (a store section on the heatmap preset).
- Create: `.changeset/<name>.md` (minor).
- Test: `packages/fast-mesh-wasm/test/store-heatmap.test.ts`.

**Interfaces (produces):**
```ts
interface ContactHeatmapOptions { depth: number; gap: number; tolerance?: number; exactDepth?: boolean; sign?: SignMode;
  colors?: { deep?: Rgba; shallow?: Rgba; contact?: Rgba; near?: Rgba; far?: Rgba } }
function contactHeatmap(o: ContactHeatmapOptions): Omit<DistanceColoring, "distanceTo">;
SceneStore.fetchColoringValues(mesh: MeshHandle): Promise<Float32Array | null>;
```
- **Defaults** (spec §2.2): `#7f0000`, `#ff6b6b`, `#ffffff`, `#4caf50`, `#1e3a8a`.
- **Range** is `[-depth, gap]`. `below` is `clamp` and `above` is `base`.
- **Stops:**
  - deep at t=0;
  - shallow just below the band;
  - contact over the band: `[-tolerance, +tolerance]` mapped into t, collapsing to a single stop at tolerance 0;
  - near just above the band;
  - far at t=1.
  - Stops must be strictly increasing.
- **Validation** (spec §6.5): throw `INVALID_ARGUMENT` unless `depth > 0`, `gap > 0` and `0 ≤ tolerance < min(depth, gap)`.

- [ ] **Step 1: Failing tests**
  - `heatmap_defaults`: range, maxDistance for both `exactDepth` values, stop count and order, and the colours at t of `-depth`, 0 and `gap`.
  - `heatmap_band`: with tolerance 0.1, the colour at ±0.05 is white.
  - `heatmap_validation`: each invalid case throws.
  - `tolerance_change_reuses_field` (RF2): on both hosts, changing only the tolerance through `setDisplay` keeps the field-add count unchanged.
  - `fetch_coloring_values`: on both hosts, it returns values equal to a fresh field, or `null` for an uncoloured mesh.
  - `exact_depth_colours_deep_interior`: with `exactDepth: true`, a vertex deeper than `depth` gets the deep colour.
- [ ] **Step 2: Run** `pnpm --filter fast-mesh-wasm exec vitest run test/store-heatmap.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the gate.**
- [ ] **Step 5: Commit** `feat(store): contactHeatmap preset and fetchColoringValues`.

### Task 2: `examples/showcase` app shell — load, list, move, undo

**Files:**
- Create `examples/showcase/` (package `fast-mesh-wasm-example-showcase`) containing:
  - `index.html`
  - `vite.config.ts`, with the COOP/COEP middleware plugin copied from `examples/react`
  - `tsconfig.json`
  - `package.json`
  - `src/main.tsx`, `src/App.tsx`
  - `src/scene/demo.ts` (demo meshes A, B and C per spec §6.8, plus `?big`)
  - `src/load/parser.worker.ts` and `src/load/{stl,obj,ply}.ts`
  - `src/panels/MeshList.tsx`, `src/panels/StatusBar.tsx`
  - `src/viewport/Viewport.tsx` (R3F `Canvas`, `<FastMesh onBind>` for each mesh, click to select, `TransformControls` gizmo with T/R modes, focus camera on a mesh)
- Create: `examples/showcase/test/parsers.test.ts` (vitest; small fixtures for STL binary and ASCII, OBJ with polygons, and PLY binary and ASCII).
- Create: `scripts/showcase-check.mjs`. It drives headless Chrome through `scripts/chrome-cdp.mjs` against `vite preview`. The checks are filled in this task and in Task 3. This task covers: loading the demo, uploading a fixture through the file input, dragging with the gizmo, undo, and no console errors. With `--big` it also uploads a generated 1M-triangle binary STL and checks the long-task probe stays under 50 ms (RF4).
- Modify: `.github/workflows/ci.yml` (build the showcase; run `showcase-check`).

**Behaviour:**
- The engine is a `WorkerScene` store, with `?local` for the synchronous `Scene`.
- The status bar shows the overall status, the thread count, whether the page is cross-origin isolated, the error message, and a Restart button.
- The mesh list shows each mesh's name, triangle count, visibility, an opacity slider, selection, remove and focus.

- [ ] **Step 1: Write the parser tests**, then run them. Expected: FAIL.
- [ ] **Step 2: Implement the parsers.** STL merges bit-identical positions; OBJ fan-triangulates polygons; PLY reads binary little-endian and ASCII. Run the tests. Expected: PASS.
- [ ] **Step 3: Implement the app shell.**
- [ ] **Step 4: Write `showcase-check.mjs`** with the Task 2 checks and `--big`. Run it. Expected: PASS.
- [ ] **Step 5: Run the gate.**
- [ ] **Step 6: Commit** `feat(examples): showcase app shell`.

### Task 3: Repair and collision panels, checks and docs

**Files:**
- Create: `examples/showcase/src/panels/RepairPanel.tsx`
  - **Analyze:** call `analyze` on the app's scene and show holes with their sizes, orientation, components, non-manifold elements and degenerates.
  - **Buttons:** weld, remove degenerate, orient, remove small components (threshold), fill holes (max size, fair toggle).
  - **Each repair:** show its report, then re-analyze. Show a busy state while it runs.
- Create: `examples/showcase/src/panels/CollisionPanel.tsx`
  - **Choices:** A/B pickers, and sliders for depth, gap and tolerance (clamped per spec §6.5).
  - **Fast mode** toggle.
  - **Open-mesh warning** taken from `fieldStatus`.
  - **Readouts:**
    - max penetration through `fetchColoringValues` ("≥ depth" in fast mode);
    - min gap through `meshDistance`;
    - `intersects`;
    - contact count through `findContacts({ margin: tolerance })`.

    Readouts throttle to about 4 Hz and refresh on drag end.
  - **Colouring:** both meshes, each against the other, via `contactHeatmap`.
- Modify: `scripts/showcase-check.mjs`. Add:
  - heatmap on A/B with readouts updating while dragging;
  - a C-mesh collision step that shows the open warning (RF1);
  - removing B while coloured (RF3);
  - fill holes on C, with fewer holes after and undo restoring them (RF5 on a coloured mesh);
  - a `?local` pass.
- Modify: `examples/showcase/README.md` and the root README (a short "Showcase" pointer).

- [ ] **Step 1: Extend `showcase-check.mjs`** with the Task 3 checks. Run it. Expected: FAIL.
- [ ] **Step 2: Implement the panels.**
- [ ] **Step 3: Run `showcase-check`** in Chrome and the `?local` pass. Expected: PASS. Also run Firefox and WebKit once through Playwright, and record the results.
- [ ] **Step 4: Run the gate.**
- [ ] **Step 5: Commit** `feat(examples): showcase repair and collision panels`, then `docs(showcase): …`.
