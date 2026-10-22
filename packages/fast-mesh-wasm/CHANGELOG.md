# fast-mesh-wasm

## 0.3.0

### Minor Changes

- [#41](https://github.com/YoonDongGeun/fast-mesh-wasm/pull/41) [`569fc9d`](https://github.com/YoonDongGeun/fast-mesh-wasm/commit/569fc9d46638fe81cade36ba6566ab31b7e7c553) Thanks [@YoonDongGeun](https://github.com/YoonDongGeun)! - Add the `Scene` API: many meshes in one WebAssembly scene with placements, a BVH per mesh, raycast / closest-point / mesh-distance / contact queries, signed distance fields with colour maps, planar sections, edit op lists, mesh analysis and hole repair, and undo/redo with drags as one step. Render buffers (positions, normals, RGBA8 colours, indices) are read through zero-copy typed-array views, and `commitFrame()` reports the dirty ranges to upload. Errors are `FastMeshError`s with a `code`. The wasm binary grows from 19.5 KB to 733 KB (292 KB gzip).

- [#49](https://github.com/YoonDongGeun/fast-mesh-wasm/pull/49) [`be8ecbb`](https://github.com/YoonDongGeun/fast-mesh-wasm/commit/be8ecbbb646d66ffaab04c6c8b4d31784df952de) Thanks [@YoonDongGeun](https://github.com/YoonDongGeun)! - Add `fast-mesh-wasm/store`, `fast-mesh-wasm/react` and `fast-mesh-wasm/r3f`.
  
  `fast-mesh-wasm/store`: `createSceneStore(scene)` wraps a `Scene` or a `WorkerScene` in a `SceneStore` with frozen, structurally shared snapshots (`getSnapshot` / `subscribe`): per mesh its applied `display`, the `desired` display and a `status` (`"idle"`, `"computing"`, `"ready"` or `"error"`), plus sections, history, selection and an overall status. Besides the scene's actions it adds declarative ones: `setDisplay(mesh, { coloring, opacity, visible })`, where `coloring` colours a mesh by its distance to another through a distance field the store owns (one per coloured mesh, reused when the colouring is toggled back on), `setSection(section, { plane, meshes, fill })` and `select(mesh)`. Declarative changes apply at the next `store.commitFrame()`, the last value per frame winning, so rapid toggling never queues work; a mesh or section reads `"computing"` until its change is shown, including while a placement, edit or undo it depends on is pending. `opacity`, `visible` and the selection live only in the store and never enter history. The store has no dependencies (no React, three.js or wasm glue).
  
  `fast-mesh-wasm/react` (optional peer `react >=18`): `SceneProvider`, `useScene`, `useStoreSelector`, `useMeshDisplay`, `useSection`, `useHistory`, `useSceneStatus` and `useSelection`, built on `useSyncExternalStore` with memoised selectors, so components re-render only when their slice changes and render an empty snapshot on the server. Bind controlled inputs to `state.desired`; the top-level fields are what is shown.
  
  `fast-mesh-wasm/r3f` (optional peer `@react-three/fiber >=9`, tested with 9.x; v8 is not supported): `<FastMesh mesh />` draws a scene mesh through `bindMesh`, follows its opacity and visibility, and disposes on unmount or removal; `useFrameDriver(store)` calls `store.commitFrame()` once per rendered frame however many `<FastMesh>` are mounted.
  
  A new example, `examples/react`, runs the store on a Worker scene with colouring toggles, opacity, visibility, a section, undo/redo, Restart and TransformControls dragging.

- [#44](https://github.com/YoonDongGeun/fast-mesh-wasm/pull/44) [`9d89ed3`](https://github.com/YoonDongGeun/fast-mesh-wasm/commit/9d89ed3744839ae212150a0d20a5a566ce0e3c78) Thanks [@YoonDongGeun](https://github.com/YoonDongGeun)! - Add `fast-mesh-wasm/three`, a three.js adapter for the `Scene` API. `bindMesh(scene, mesh, options?)` returns a `THREE.Mesh` whose geometry follows the scene through `commitFrame()` (dirty ranges only, colour attribute while colouring is on, `setOpacity`, `dispose`); `attachTransformControls(controls, scene, bound)` turns a TransformControls drag into one scene drag and one undo step; `bindSection(scene, section, bound, options?)` draws a section's contours and fill and clips the bound meshes at its plane (set `renderer.localClippingEnabled = true`). `three` (`>=0.160`) is an optional peer dependency that only this entry imports, and TypeScript users also need `@types/three`. A caller's material is used as given and must not be shared between bound meshes.
  
  Core additions: `DragSession.active` tells whether the drag is still open, `Scene.sectionPlane(section)` returns a section's current plane, and `Scene.removeMesh` now ends an active drag on that mesh (as `undo()` does) instead of leaving it open.
  
  There is also a raw WebGPU example in `examples/webgpu` that uses no 3D library.

- [#47](https://github.com/YoonDongGeun/fast-mesh-wasm/pull/47) [`bddc257`](https://github.com/YoonDongGeun/fast-mesh-wasm/commit/bddc2570c88b17138ef37112f83c47d9e1bb29aa) Thanks [@YoonDongGeun](https://github.com/YoonDongGeun)! - Add `fast-mesh-wasm/worker` (browsers): `createWorkerScene(options?)` runs the `Scene` engine in a Web Worker and resolves to a `WorkerScene` with the same methods. Handles are returned at once; state changes are fire-and-forget and report progress through `status(key)` / `onStatus`; queries (`raycast`, `closestPoint`, `meshDistance`, `intersects`, `findContacts`, `analyze`, `fetchFieldValues`, `fieldStatus`, `sectionGap`) and `edit` / `repair` return promises. Call `commitFrame()` once per animation frame: it sends the frame's requests as one batch and delivers the change records of the results received, whose dirty ranges were transferred into main-thread mirrors, so `views`, `matrix4`, `placement`, `section` and `sectionPlane` stay synchronous and the three.js adapter works unchanged. Requests run in order; superseded colouring, section and drag-move requests are dropped before they start, and results older than what is shown are ignored. Drags are optimistic (the placement follows the pointer at once; `end` and `cancel` are synchronous). After a trap every call fails with `INSTANCE_POISONED` until `restart()`, which starts a new Worker and replays the meshes, placements, fields, colourings and sections (history is cleared). The Worker script is loaded with `new Worker(new URL("./worker-entry.js", import.meta.url), { type: "module" })`; the `workerUrl` and `createWorker` options override it. The Worker scene is always eager and has no `beginStep` / `endStep` / `cancelStep` or `takeChanges`.
  
  The three.js adapter (`bindMesh`, `bindSection`, `attachTransformControls`) now accepts any `SceneLike`, a new exported interface (`views`, `matrix4`, `placement`, `section`, `sectionPlane`, `onChanges`, `commitFrame`, `drag`) that both `Scene` and `WorkerScene` implement. This is a type-only change: code passing a `Scene` is unaffected.

### Patch Changes

- [#51](https://github.com/YoonDongGeun/fast-mesh-wasm/pull/51) [`0d7ab20`](https://github.com/YoonDongGeun/fast-mesh-wasm/commit/0d7ab20cb7990264b71930f29516625ad2a4ef88) Thanks [@YoonDongGeun](https://github.com/YoonDongGeun)! - Faster wide-range distance colouring while dragging: when most of a mesh is within a field's `maxDistance`, the per-pose update now recomputes every vertex instead of first listing every nearby triangle pair, which makes a 1M-triangle drag with `maxDistance` 8 about 29 times faster per pose. Values and colours are unchanged, bit for bit. In the Worker scene (`fast-mesh-wasm/worker`), the field recompute after a drag move now runs in small steps, and a recompute whose pose is already stale (a newer move, drag cancel, `setPlacement`, edit, repair, compact, undo/redo, removal of either mesh or removal of the field is queued) is dropped, so colours follow the latest pose without a backlog; during a sustained drag a recompute is kept and shown once 200 ms have passed since the last colours, so they still update while the pointer moves. Each pose's colours are still applied all at once. In Node `worker_threads` the engine now yields with `setImmediate`, so requests sent during a long recompute arrive between its steps. No API change.

## 0.2.0

### Minor Changes

- [#9](https://github.com/YoonDongGeun/fast-mesh-wasm/pull/9) [`d7fed6d`](https://github.com/YoonDongGeun/fast-mesh-wasm/commit/d7fed6da89c1010cd3cc4bf9e14c02ac74ec4783) Thanks [@YoonDongGeun](https://github.com/YoonDongGeun)! - Easier JavaScript API (breaking):
  
  - `computeBounds` now returns a plain object `{ min, max, center, size }` with `[x, y, z]` arrays. The `Bounds` class, `free()` and `using` are gone.
  - Inputs accept `Uint16Array` indices (three.js), `Float64Array` and plain `number[]`; `Float32Array`/`Uint32Array` are still used without copying.
  - Calling a function before `await init()` in the browser throws a clear error instead of an internal one.
  - Fixed: the Node.js entry no longer gets dropped by bundlers that honor `sideEffects` (webpack server builds, e.g. Next.js), which made every call crash.
  - `InitInput` is exported, and the Node.js `init()` accepts the same argument as the browser one.

## 0.1.0

### Minor Changes

- [`bf94aa5`](https://github.com/YoonDongGeun/fast-mesh-wasm/commit/bf94aa5088886d69886f9e02b535e162cffe9065) Thanks [@YoonDongGeun](https://github.com/YoonDongGeun)! - Initial release: `computeVertexNormals` and `computeBounds`, with ESM entry points for browsers/bundlers and Node.js/Bun.
