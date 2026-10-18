# fast-mesh three.js adapter and WebGPU example (Phase 8b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two deliverables:
- `fast-mesh-wasm/three`, an official three.js adapter: geometry sync, matrix sync, opacity, TransformControls drags, and section clipping and lines.
- `examples/webgpu`, a raw WebGPU app reproducing the owner's scene: 2–5 meshes, drag to move or rotate, distance colouring on a key, opacity, undo and redo, a section toggle.

**Architecture:**
- The adapter is a new tsdown entry in the existing package. `three` is an optional peer dependency.
- It keeps JS-owned mirror arrays per attribute (spec §11.2). On every `scene.commitFrame()` it copies only the dirty ranges from fresh views and marks update ranges. When a buffer's capacity changes it recreates that attribute.
- The WebGPU example is a private Vite workspace app that writes dirty ranges from views straight into GPU buffers.

**Tech Stack:** TypeScript, tsdown, vitest (Node, no WebGL), three `^0.186` as peer and dev dependency, Vite 8, WebGPU (WGSL).

**Spec:** `docs/superpowers/specs/2026-10-18-fast-mesh-js-api-design.md`, §6, §7, §11.2 and §12 (implementation notes from 8a). Read §12: capacity changes report every buffer `"all"`; in debug mode views are Proxies, so call `.subarray()` before handing them to GPU or three APIs; coalesce `drag.move` to one call per animation frame.

## Global Constraints
- **Packaging.**
  - `three` is an optional peer dependency (`peerDependencies` plus `peerDependenciesMeta.three.optional = true`, range `>=0.160`) and a devDependency for tests.
  - The core entries (`.`) must never import `three`.
  - `publint --strict` and `attw --profile esm-only` stay green with the new `./three` export.
- **No duplicated glue.** The `./three` chunk imports the shared core and does not inline the wasm glue (tsdown `neverBundle`).
- **Change stream.**
  - The adapter consumes changes only through `scene.onChanges` / `commitFrame`. It never calls `takeChanges`.
  - Apps call `scene.commitFrame()` once per animation frame.
- **Display settings.** Opacity, colours on/off and visibility never touch history.
- **Mirrors.** Mirrors are re-sized only on `capacityChanged` or when colours or normals turn on or off. Otherwise each frame copies only dirty ranges.
- **Views.** Read views fresh inside each frame handler. Call `.subarray()` before handing them to `Float32Array.set`, `writeBuffer` or three attributes, so debug-mode Proxies work.
- **WebGPU example.**
  - No 3D library. If `navigator.gpu` is missing, show a message.
  - It builds in CI with `vite build`; there is no runtime test.
- **Commits.**
  - Conventional messages ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (or the implementing model).
  - Each message goes in a NEW temp file, committed with `git commit -F`.
  - Files are UTF-8 with LF.
- **Every task ends green on:**
  - `cargo fmt --all -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
  - `pnpm build`
  - `pnpm --filter fast-mesh-wasm test`
  - `pnpm check:pkg`
  - from Task 3 on, also `pnpm --filter fast-mesh-wasm-example-webgpu build`
  - Finish with `pnpm build` so `dist/` is a release build.
- **Changeset:** a minor bump in the last task.

## Review Focus
1. **Capacity growth while bound.** Add vertices through `scene.edit` to a bound mesh. The attributes are recreated with the new length, all data is uploaded, `geometry.setDrawRange` / index count is correct, and nothing reads past the old array. Test: Task 1 `capacity_growth_recreates_attributes`.
2. **Colouring toggled on and off repeatedly.** Each toggle adds or removes the colour attribute and flips `material.vertexColors`. Opacity is kept. History is untouched. Test: Task 1 `coloring_toggle_adds_and_removes_attribute`.
3. **Undo during a TransformControls drag.** Ctrl+Z while dragging ends the drag first. The object then follows the restored placement through the change stream. Test: Task 2 `undo_mid_transform_drag`.
4. **Disposing a bound mesh, then removing the mesh (and the reverse order).** No listener leaks (the onChanges unsubscribe is called), no errors, and the geometry is disposed. Test: Task 1 `dispose_and_remove_any_order`.
5. **Debug-mode scene.** With `new Scene({ debug: true })` the adapter still works, because it uses `.subarray()` before `set`. Test: Task 1 `works_with_debug_scene`.

---

### Task 1: `fast-mesh-wasm/three`: `bindMesh` (geometry, matrix, opacity, colours, dispose)

**Files:**
- Create: `packages/fast-mesh-wasm/src/three/index.ts` and `packages/fast-mesh-wasm/src/three/mesh.ts`.
- Modify:
  - `packages/fast-mesh-wasm/package.json`: `exports["./three"]` with types and default, the peer dependency plus meta, and the devDependency `three` and `@types/three` at matching versions.
  - `packages/fast-mesh-wasm/tsdown.config.ts`: new entry `three: "src/three/index.ts"`; keep the wasm glue external.
- Test: `packages/fast-mesh-wasm/test/three-mesh.test.ts`.

**Interfaces (produces):**
```ts
import type { Scene, MeshHandle } from "fast-mesh-wasm";
import * as THREE from "three";
export interface BindMeshOptions { material?: THREE.MeshStandardMaterial | THREE.MeshPhongMaterial | THREE.MeshLambertMaterial; color?: THREE.ColorRepresentation; opacity?: number }
export interface BoundMesh {
  readonly object: THREE.Mesh;          // geometry: position, normal (if on), color (if colouring on), index
  readonly mesh: MeshHandle;
  setOpacity(alpha: number): void;      // 0..1; <1 → transparent=true, depthWrite=false; display only
  dispose(): void;                      // unsubscribes, disposes geometry (and material if created by bindMesh)
}
export function bindMesh(scene: Scene, mesh: MeshHandle, options?: BindMeshOptions): BoundMesh;
```
- **Construction.** Read the views. Create mirrors (`Float32Array` for position and normal, `Uint8Array` for colour with `normalized: true`, `Uint32Array` for index), set the attributes, then set `object.matrix` from `scene.matrix4(mesh)` (Float64 to the Matrix4 elements) with `matrixAutoUpdate = false`.
- **Per frame (onChanges listener), for records of this mesh:**
  - `removed`: detach. Leave the object in place and stop updating it. Document this.
  - `capacityChanged` or `"all"`: recreate the attribute and its mirror at the new length, copy everything, call `geometry.computeBoundingSphere()` over live data (see the bounds note below), and set `needsUpdate`.
  - Ranges: copy each range (elements × itemSize) from the fresh view's `subarray` into the mirror, call `attribute.addUpdateRange(start*itemSize, count*itemSize)`, and set `needsUpdate = true`.
  - `placementChanged`: update `object.matrix`, then `object.matrixWorldNeedsUpdate = true`.
  - Colours turning on or off: `colors()` becomes non-null or null, with `"all"`. Add or remove the colour attribute and set `material.vertexColors` plus `material.needsUpdate`.
- **Bounds.** Dead slots keep stale positions. Compute the bounding sphere from live triangles' vertices only (via the index mirror), or fall back to `scene` world bounds if that is cheaper; builder's choice, documented. Recompute it only on `"all"` or capacity changes and on placement-independent geometry edits, never on placement-only frames.

- [ ] **Step 1: Write failing tests** (Node, three without a renderer):
  - `bind_creates_attributes_matching_views`
  - `ranges_only_update_dirty_parts`: after a `setPositions` edit of 3 vertices, `position.updateRanges` covers only those elements (with the merge gap), and the mirror equals the views.
  - `placement_updates_matrix_only`: a drag gives a matrix change and no position update ranges.
  - `capacity_growth_recreates_attributes` (Review Focus 1)
  - `coloring_toggle_adds_and_removes_attribute` (Review Focus 2)
  - `opacity_sets_transparent_and_depthwrite`: `setOpacity(0.5)` gives transparent and no depth write; `setOpacity(1)` restores both. `canUndo` is unchanged.
  - `dispose_and_remove_any_order` (Review Focus 4)
  - `works_with_debug_scene` (Review Focus 5)
  - `core_entry_does_not_import_three`: after build, grep `dist/web.js` and `dist/node.js` for `from "three"` and find nothing.
- [ ] **Step 2: Run** `pnpm build && pnpm --filter fast-mesh-wasm test`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full gate. Expected: PASS.
- [ ] **Step 5: Commit** `feat(three): bindMesh keeps a BufferGeometry in sync with a scene mesh`.

### Task 2: TransformControls drags and `bindSection`

**Files:**
- Create: `packages/fast-mesh-wasm/src/three/controls.ts` and `packages/fast-mesh-wasm/src/three/section.ts`.
- Modify: `packages/fast-mesh-wasm/src/three/index.ts` (exports).
- Test: `packages/fast-mesh-wasm/test/three-controls.test.ts` and `packages/fast-mesh-wasm/test/three-section.test.ts`.

**Interfaces (produces):**
```ts
export interface TransformControlsLike extends THREE.EventDispatcher<any> { object?: THREE.Object3D; dragging: boolean }
export function attachTransformControls(controls: TransformControlsLike, scene: Scene, bound: BoundMesh): () => void; // returns detach
export interface BindSectionOptions { lineColor?: THREE.ColorRepresentation; fillColor?: THREE.ColorRepresentation; clip?: boolean /* default true */ }
export interface BoundSection { readonly group: THREE.Group; /* contours LineSegments + fill Mesh per selected mesh */ readonly plane: THREE.Plane; dispose(): void }
export function bindSection(scene: Scene, section: SectionHandle, bound: BoundMesh[], options?: BindSectionOptions): BoundSection;
```

**`attachTransformControls`:**
- On `dragging-changed` to true, open `scene.drag(mesh)` with the object's current placement as the start.
- On `objectChange` while dragging, read the object's position and quaternion and call `drag.move({ rotation: quat, translation: pos − rotation·origin })`.
  - The object's matrix represents `placement · translate(origin)`, so convert back. `scene.placement(mesh)` and `matrix4` give the origin relation; document the math.
  - Coalesce to one move per animation frame: store the latest pose and flush it with `requestAnimationFrame`, or on `dragging-changed` to false.
  - In Node tests, flush is synchronous through an injectable scheduler option.
- On `dragging-changed` to false, `drag.end()`.
- `controls.object` matrix updates come from the change stream. The adapter must not fight TransformControls while the user is dragging, so skip writing the matrix for the dragged object until the drag ends, then apply the final placement.

**`bindSection`:**
- Adds `plane` to `material.clippingPlanes` of each bound mesh's material and documents that the renderer needs `localClippingEnabled = true`.
- Builds `LineSegments` from the contours (consecutive point pairs; closed contours wrap) and a fill `Mesh` from the fill.
- Rebuilds when `sectionsChanged` lists the section for any of the meshes.
- On `setSectionPlane`, the next frame updates `plane` (normal and constant) from `scene`.
- `dispose()` removes the clipping planes and disposes the geometry.

- [ ] **Step 1: Write failing tests:**
  - `drag_maps_to_one_undo_step`: a mocked controls object (EventDispatcher with `dragging` and `object`) emits dragging-changed true, several objectChange events and dragging-changed false. The result is one undo step, and the placement equals the object pose.
  - `moves_are_coalesced_per_frame`: ten objectChange events within one frame give a single `drag.move` call. Count it with a spy on the scene.
  - `undo_mid_transform_drag` (Review Focus 3)
  - `rotated_mesh_with_origin_round_trips`: object pose to placement to `matrix4` reproduces the object matrix within 1e-9.
  - `section_clipping_and_lines`: the clipping plane is added to the materials, the line positions equal the `scene.section` contours, and after `setSectionPlane` + `commitFrame` the lines and plane update. `dispose()` removes the planes.
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full gate. Expected: PASS.
- [ ] **Step 5: Commit** `feat(three): TransformControls drags and section clipping`.

### Task 3: `examples/webgpu`

**Files:**
- Create the Vite workspace app `examples/webgpu/` with package name `fast-mesh-wasm-example-webgpu`, private, depending on `fast-mesh-wasm: workspace:*` and `vite`. Contents:
  - `index.html` and `src/main.ts` (setup, input, keys)
  - `src/gpu.ts` (device, pipeline, per-mesh buffers, writing dirty ranges)
  - `src/shader.wgsl` (lit shader: per-vertex colour or base colour, opacity uniform, alpha blending)
  - `src/meshes.ts` (generators: sphere, torus, wavy plate; plus drag-and-drop of a binary STL, kept tiny)
  - `README.md`
- Modify:
  - `pnpm-workspace.yaml` if needed, so the example is a workspace member like `examples/vite`.
  - `.github/workflows/ci.yml`: add `pnpm --filter fast-mesh-wasm-example-webgpu build` next to the existing vite example build.

**Behaviour (spec §7):**
- **Scene:** 3 generated meshes (sphere, torus, wavy plate) placed apart.
- **Per frame:**
  1. `scene.commitFrame()`.
  2. For each change: on `capacityChanged` or `"all"`, recreate that mesh's GPU buffers. Otherwise `queue.writeBuffer` each dirty range from `views().*().subarray(...)`. The byte offset is `start · itemSize · 4`; colours use `start · 4`.
  3. Update the per-object uniform from `scene.matrix4(mesh)`.
  4. Draw opaque meshes, then transparent ones sorted back to front by world centre.
- **Input:**
  - Click picks with `scene.raycast`, using a ray from camera unprojection.
  - Drag moves the picked mesh in a camera-facing plane. Shift+drag rotates it (arcball).
  - Moves are coalesced per animation frame via `scene.drag`.
- **Keys:**
  - Ctrl+Z / Ctrl+Y and Ctrl+Shift+Z: undo / redo.
  - C: toggle distance colouring between the picked mesh and its nearest other mesh (`addDistanceField` + `setColoring`, then `setColoring(null)`), with `maxDistance` = 10% of the scene diagonal and the turbo map.
  - 1–9: set the picked mesh's opacity to 0.1–0.9; 0 sets 1.0.
  - S: toggle a section plane through the scene centre, drawing contour lines from `scene.section` as a line-list pipeline.
- **WebGPU unavailable:** a visible message, and no exception thrown.
- **README:** the renderer-neutral pattern in about 20 lines (views, dirty ranges to `writeBuffer`, recreate on capacity, apply `matrix4`), how to run it, and a manual checklist.

- [ ] **Step 1: Scaffold** the app. Write `vite build` into CI. Run `pnpm --filter fast-mesh-wasm-example-webgpu build`. Expected: FAIL until the code exists, then PASS.
- [ ] **Step 2: Implement** the pieces in the order `meshes.ts`, `gpu.ts`, `shader.wgsl`, `main.ts`.
- [ ] **Step 3: Type-check and build.** `tsc --noEmit -p examples/webgpu` (add a tsconfig) and `vite build` both pass. Expected: PASS.
- [ ] **Step 4: Run** the full gate. Expected: PASS.
- [ ] **Step 5: Commit** `feat(examples): raw WebGPU example of the owner's scene`.

### Task 4: Adapter benchmark, docs, changeset

**Files:**
- Modify:
  - `packages/fast-mesh-wasm/bench/scene.bench.mjs`: add an adapter case. Run a 64-pose drag of a 1M mesh against another with colouring on, with `bindMesh` attached, and measure per-frame time for `commitFrame` plus the adapter's range application (Node, no GPU upload).
  - `docs/benchmarks.md`: "three.js adapter" rows.
  - Root `README.md`: a three.js section (`bindMesh`, `attachTransformControls`, `bindSection`, `localClippingEnabled`, `commitFrame` in the render loop) and a pointer to `examples/webgpu`.
  - The spec: add §13 "8b implementation notes" (rulings and numbers).
- Create: `.changeset/<name>.md`, a minor bump for `fast-mesh-wasm` describing `fast-mesh-wasm/three`.

- [ ] **Step 1: Add** the bench case and run it on a release build. Record real numbers.
- [ ] **Step 2: Write** the docs and the changeset.
- [ ] **Step 3: Run** the full gate. Expected: green.
- [ ] **Step 4: Commit** `bench(three): adapter frame cost`, `docs(three): document the three.js adapter and WebGPU example`, `chore: changeset for fast-mesh-wasm/three`.
