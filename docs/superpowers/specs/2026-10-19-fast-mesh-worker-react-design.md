# fast-mesh Phase 9: Worker, UI synchronisation and React

- **Status:** draft for review
- **Parent spec:** `2026-10-14-fast-mesh-engine-design.md` §11.1, §12 and §13.2. Where they differ, this document wins.
- **Builds on:** Phase 8a's synchronous `Scene` API (`fast-mesh-wasm`) and 8b's three.js adapter. Multithreading stage 2 (a wasm threads build) is a separate spec that comes after this one. It reuses this phase's Worker and shared-memory plumbing.
- **Labels:** *(owner)* marks what the owner said. *(my call)* marks choices made without asking.

## 1. Intent

In a React app the owner wants to:
- toggle features such as distance colouring and sections, trigger undo and redo, adjust per-mesh opacity, and show computation status *(owner)*;
- have the UI reflect the current state: toggle buttons that look pressed when a feature is on, undo and redo buttons that enable and disable, sliders that show the current value *(owner)*.

Heavy work must never freeze the page. Fast repeated clicks must settle on the final state without doing the intermediate work (parent spec §12).

The work comes in two parts:
- **9a: Worker and sync.** `fast-mesh-wasm/worker` runs the same engine in a Web Worker behind an async API. It has an ordered queue, coalescing, revisions and cooperative cancellation, and moves data through shared memory or transferred dirty ranges.
- **9b: React.** `fast-mesh-wasm/react` provides hooks over either the synchronous `Scene` or the Worker scene, built on `useSyncExternalStore`.

Each part gets its own plan.

## 2. Goals, non-goals, anti-goals

### Goals
1. **One state model, two hosts.** A `SceneStore` exposes the observable state (§5.1) and its actions. It runs over either:
   - `LocalHost`: the synchronous `Scene`, on the main thread;
   - `WorkerHost`: the Worker.

   React and other UIs depend only on the store.
2. **Declarative display state** *(owner: toggles reflect state)*. `store.setDisplay(mesh, { coloring, opacity, visible })` and `store.setSection(...)` describe a desired state. These calls are:
   - idempotent;
   - safe under StrictMode double invocation;
   - coalesced while not yet started.

   The store also reports the applied state and its status.
3. **History state** *(owner)*. `canUndo` and `canRedo` are observable. `undo()` and `redo()` are actions. Display state never enters history (8a).
4. **Status** *(owner)*. Each declared piece of state reports `{ status: "idle" | "computing" | "ready" | "error", revision, error? }`.
5. **Worker data paths.**
   - On cross-origin-isolated pages, render views are `SharedArrayBuffer` views over the Worker's wasm memory: no copies.
   - Otherwise each frame transfers only the dirty ranges, as `ArrayBuffer`s, into main-thread mirrors.
6. **No main-thread stalls.** In Worker mode, adding a 1M-triangle mesh, repairing it or building fields never blocks the main thread for more than the cost of applying the dirty ranges.
7. **React hooks.**
   - `useScene()` / `<SceneProvider>`
   - `useMeshDisplay(mesh)` returns `[state, setState]`
   - `useHistory()`
   - `useSection(id)`
   - `useSceneStatus()`
   - `useSelection()`

   All are subscribed through `useSyncExternalStore`, so components never tear.
8. **React Three Fiber.** Works through the 8b adapter. A small `<FastMesh mesh={...} />` component binds a scene mesh into an R3F scene *(my call: thin, built on `bindMesh`)*.

### Non-goals (this phase)
- The wasm threads build (multithreading stage 2, its own spec).
- Server-side rendering beyond "the hooks don't crash during SSR": no data on the server.
- Persisting state (save/load) and collaborative sync.
- Redux, Zustand or other store integrations. The store is plain and subscribable, so apps can wrap it.

### Anti-goals
- **No torn UI.** A button must never show "on" while the mesh shows colours off, beyond the documented pending state (`status: "computing"`).
- **No stale results.** A result older than state the UI has already shown must never be applied.
- **No queue blow-up.** Rapid toggling must never queue unbounded work.
- **Sync stays.** Worker mode must not degrade the synchronous API. Apps that stay on the main thread keep 8a behaviour unchanged.

## 3. Constraints
- **Package.** Same package `fast-mesh-wasm`, with new exports `./worker` and `./react`.
  - `react` is an optional peer, `>=18`.
  - The core entries must not import React. The worker entry must not import three or React.
  - publint and attw stay green.
- **Worker bundling.** The Worker script is shipped by the package as `new Worker(new URL("./worker-entry.js", import.meta.url), { type: "module" })`. That pattern works in Vite, webpack 5 and modern bundlers. The wasm loads inside the Worker through the existing web `init()`. A user-supplied Worker URL or factory option covers unusual bundlers *(my call)*.
- **Messages.**
  - Structured clone plus transferable `ArrayBuffer`s.
  - SAB views only when `crossOriginIsolated` is true.
  - No `eval`. No `Atomics.wait` on the main thread.
- **Wasm memory.** On isolated pages the Worker builds its wasm with a shared memory. This needs a build flag (`+atomics,+bulk-memory` and shared memory).
  - If that conflicts with the single-threaded release build, ship a second artifact, `fast_mesh_wasm_shared_bg.wasm`, chosen at runtime.
  - Multithreading stage 2 will use the same artifact.
  - *(My call: decide in 9a's first task, by trying the toolchain. If a shared-memory build requires nightly, 9a falls back to transfers only and SAB moves to stage 2.)*
- **Cancellation is cooperative.** It happens only between chunks of work, or by discarding stale results. Long single Rust calls cannot be interrupted in this phase.
- **Errors.** `FastMeshError` codes cross the Worker boundary unchanged. A poisoned Worker reports `INSTANCE_POISONED` and can be recreated with `store.restart()`, which rebuilds the scene from the last snapshot *(my call: snapshot = meshes plus placements plus display state; history is not restored)*.
- **Tests.**
  - vitest in Node using `worker_threads` behind a small adapter, so the Worker protocol is exercised.
  - React hooks tested with `@testing-library/react` and `react-dom` in jsdom.
  - R3F: a smoke test with `@react-three/test-renderer`, if it runs headless; otherwise a type-check plus example build.

## 4. Worker protocol (9a)

### 4.1 Requests and ordering
- Every state-changing request carries a monotonically increasing `revision` assigned on the main thread, plus a `key` naming the state it targets:
  - `display:<meshId>`
  - `section:<id>`
  - `placement:<meshId>` for drags
  - `edit:<meshId>`
  - and so on.
- The Worker processes requests strictly in arrival order (parent §12.2).
- **Coalescing (§12.3).** Before starting a request, the Worker drops queued requests that will be superseded:
  - an earlier, not-yet-started request with the same `key`;
  - for keys whose semantics are "last wins": display, section planes and drag moves.

  Edits, undo and redo are never coalesced.
- **Drag moves.** At most one move per frame is sent, as in 8a. The Worker coalesces any backlog anyway.

### 4.2 Results and revisions
- After each processed batch, the Worker posts `{ revision, changes, state }`:
  - `changes` are the dirty ranges per mesh (the 8a change records), with data attached for transfer mode.
  - `state` is the observable state delta (§5.1).
- The main thread applies results in order. Any result whose `revision` is below the highest revision already applied for that key is ignored (§12.5).

### 4.3 Cooperative cancellation
- Long operations run in chunks inside the Worker, with a yield between chunks (a message-loop turn). Before each chunk, the Worker checks whether a newer request with the same key is queued; if so, it stops (§12.4). These operations are:
  - full field recomputation;
  - repair pipelines;
  - section recompute over many meshes.
- Chunking uses Rust APIs that already work in steps: field updates per field, repair per op, sections per entry. No new Rust interruptibility is required *(my call)*.

### 4.4 Data paths
- **SAB mode:** the main-thread store exposes the same `views(mesh)` getters as 8a, backed by the Worker's shared wasm memory. The fresh-view rule and the debug guard still apply. Memory growth replaces the SAB, and a new buffer is posted.
- **Transfer mode:** the main thread keeps mirrors per mesh and buffer. Change records arrive with the bytes of their dirty ranges, which are applied to the mirrors. `capacityChanged` resizes the mirrors. `views(mesh)` returns the mirrors.
- **Adapters:** the 8b three.js adapter and the WebGPU pattern work unchanged on the store's `views` plus `commitFrame`-equivalent change delivery.

### 4.5 API shape
- `createWorkerScene(options?)` returns `Promise<WorkerScene>`.
- `WorkerScene` mirrors the 8a `Scene` methods, but returns promises for anything that returns data.
- Fire-and-forget state changes return `void` and are tracked through status.
- `views(mesh)` and `onChanges` / `commitFrame` stay synchronous, reading the main-thread side.
- Queries (`raycast`, `meshDistance` and the rest) are async (parent §12, known limitation). For latency-critical picking, the docs recommend a main-thread `Scene` holding a light picking copy.

## 5. Store and React (9b)

### 5.1 Observable state
`SceneStore` holds an immutable snapshot, replaced on every applied result:

```ts
interface SceneSnapshot {
  revision: number;
  meshes: ReadonlyMap<MeshHandle, MeshState>;
  history: { canUndo: boolean; canRedo: boolean };
  sections: ReadonlyMap<SectionHandle, SectionState>;
  selection: MeshHandle | null;
  status: "idle" | "computing" | "error";
  error?: FastMeshError;
}
interface MeshState {
  display: { coloring: ColorRule | null; opacity: number; visible: boolean }; // applied state
  desired: { coloring: ColorRule | null; opacity: number; visible: boolean }; // last requested
  status: "idle" | "computing" | "ready" | "error"; revision: number; error?: FastMeshError;
}
```

- `opacity` and `visible` are display-only. The store keeps them and adapters read them. The engine never sees them.
- `selection` is UI state kept in the store for convenience. It is never sent to the Worker.

### 5.2 Hooks
- `<SceneProvider store>` / `useScene()` gives the store.
- `useMeshDisplay(mesh)` returns `[{ ...display, desired, status }, setDisplay]`. The toggle-button pattern is `pressed = state.desired.coloring !== null`, with `busy = state.status === "computing"`.
- `useHistory()` returns `{ canUndo, canRedo, undo, redo }`.
- `useSection(section)` returns `[{ plane, meshes, fill, status }, setSection]`.
- `useSceneStatus()` returns the overall status and error.
- `useSelection()` returns `[mesh | null, select]`.
- All of these use `useSyncExternalStore(store.subscribe, selector)` with selector memoisation, so components re-render only when their slice changes.

### 5.3 R3F
`<FastMesh store mesh material? />` binds with `bindMesh` on mount and disposes on unmount. Opacity and visibility follow `MeshState.display`. It calls `commitFrame` in `useFrame` once per frame for the whole scene, through a shared frame driver.

## 6. Testing
- **Protocol:** ordering, coalescing (rapid on/off/on resolves to on with one computation), stale-result rejection, cancellation of a superseded long field build, error codes across the boundary, poison and restart.
- **Data paths:** SAB and transfer modes each keep a mirror byte-identical to the Worker's engine across a randomised op sequence. This is the same check as 8a's final review, run through the Worker.
- **Main-thread budget:** in Worker mode, `addMesh` of 1M triangles keeps the main thread responsive. Measure max main-thread block time with a timer probe; target under 16 ms per message.
- **Store:** snapshot immutability; selectors re-render only on slice change; StrictMode double-invoke is idempotent.
- **React:**
  - toggle button pressed and busy states follow status;
  - undo/redo buttons enable and disable;
  - a slider bound to opacity;
  - external changes (a keyboard undo) update every subscribed component;
  - no tearing under concurrent rendering (a test with `startTransition`).
- **Examples:**
  - `examples/react`, a Vite app: the owner's scene with React UI (toggle buttons, opacity sliders, undo/redo buttons with enable state, status panel) on R3F, using the Worker scene.
  - It builds in CI.

## 7. Benchmarks
- Worker round-trip latency for a display toggle.
- Per-frame cost of the transfer path vs the SAB path for a 1M drag with colouring.
- Main-thread block time during a 1M `addMesh` in Worker mode.
- Results go into `docs/benchmarks.md`.

## 8. Left to the builder
- Message encoding details.
- Chunk sizes for cooperative cancellation.
- Selector memoisation helpers.
- Test harness for `worker_threads`.
- Splitting into plan tasks: 9a and 9b plans.

## 9. Decisions after the builder check (all *my call*; they override earlier sections)

1. **9a ships transfer mode only.** SAB views over Worker memory need one of three things: a frame handshake that freezes mutation, double-buffered render buffers, or accepting torn frames. They also need a shared-memory wasm build. Both belong with multithreading stage 2, which needs the same shared build. So 9a always transfers dirty ranges into main-thread mirrors. §2 goal 5 and §4.4 SAB mode move to stage 2.
2. **Handles are allocated synchronously on the main thread.**
   - `addMesh`, `addSection` and the store's field creation return a handle immediately, with a client id.
   - The Worker keeps a client-id → engine-id map, and messages name client ids.
   - Coalescing keys use client ids.
   - Handles survive `restart()`: the map is rebuilt.
   - Errors from a failed add appear on the handle's status (`error`), and later calls on it give `UNKNOWN_MESH`.
3. **Adapter data reaches the main thread with every result.**
   - Each result carries, per changed mesh, its placement and `matrix4`. Per changed section entry it carries the section result (contours and fill), as transferred arrays.
   - The main-thread proxy keeps these mirrors, so `matrix4`, `placement`, `section` and `sectionPlane` stay synchronous.
   - `fieldValues` is async in Worker mode (`fetchFieldValues`).
   - The 8b adapters are retyped from the concrete `Scene` class to a structural `SceneLike` interface (`views`, `matrix4`, `placement`, `section`, `sectionPlane`, `onChanges`, `commitFrame`, `drag`). Both `Scene` and the Worker proxy implement it.
   - This is type-only, so existing users are unaffected; it is noted in the changeset.
4. **Drags are optimistic.**
   - In Worker mode, `drag(mesh).move(p)` updates the main-thread placement mirror and `matrix4` immediately, so the object follows the pointer without lag. It then sends a coalesced move to the Worker.
   - Colours and sections catch up when results arrive, shown as `status: "computing"` on the dependent state. This visible lag is documented and is not tearing.
   - `end()` and `cancel()` are synchronous on the main side and enqueue the Worker step.
   - `cancel()` restores the start placement and records nothing, as in 8a's nested semantics.
5. **Distance colouring is declarative, and the store owns the fields.**
   - `setDisplay(mesh, { coloring: { distanceTo: MeshHandle, maxDistance, map, range, below?, above?, center? } | null })`.
   - The store creates the field on first need, keeps it while colouring is off (engine §12.6: "hiding keeps values"), and reuses it when colouring comes back on with the same target and `maxDistance`. It replaces the field only when those change.
   - Toggling therefore costs a recolour, not a field build, after the first time.
   - The low-level `ColorRule` / `addDistanceField` path stays on `Scene` / `WorkerScene` for advanced use.
6. **`restart()` snapshot.**
   - Built on the main thread from what it already has: the transfer-mode mirrors (live triangles, local f32 positions), each mesh's origin and placement (kept by the proxy), and the store's declarative state (display, sections, selection).
   - Rebuilding re-adds each mesh from its live data, compacted (slot ids change, handles do not), then re-applies placements, display (fields are re-created) and sections.
   - History is not restored.
   - No extra copy of the inputs is kept.
7. **API scope.**
   - **`WorkerScene` / main proxy:** `addMesh`, `removeMesh`, `setPlacement`, `placement`, `matrix4`, `drag`, `edit`, `compact`, `analyze`, `repair`, `undo`, `redo`, `canUndo`/`canRedo` (as state), sections (add, set, remove, `section`, `sectionPlane`), async queries (`raycast`, `closestPoint`, `meshDistance`, `intersects`, `findContacts`), the low-level fields API, `views`, `onChanges`, `commitFrame`, `restart`, `terminate`.
   - **Not in 9a:** user-opened steps (`beginStep`/`endStep`/`cancelStep`), Lazy/Manual modes (the Worker scene is always Eager), and `takeChanges` (`onChanges` replaces it).
   - **`SceneStore` actions:** `addMesh`, `removeMesh`, `setPlacement`, `drag`, `edit`, `repair`, `undo`, `redo`, `setDisplay`, `setSection`, `addSection`, `removeSection`, `select`, `restart`.
8. **Node.** `worker_threads` is a test harness only. `fast-mesh-wasm/worker` targets browsers and has no Node export condition in 9a.
9. **Package layout.** `<FastMesh>` (R3F) goes in a separate subpath, `fast-mesh-wasm/r3f`, so `./react` does not depend on three or R3F. Optional peers: `react >=18`, `@react-three/fiber >=8`.

## 10. 9a implementation notes

What 9a shipped, the rulings made while building it, the open points 9b inherits, and the measured costs. Like §9, this section wins over earlier sections where they differ.

### 10.1 Rulings
1. **`setPlacement` is never coalesced.** Each call is an undo step. Only drag moves coalesce, so pointer-rate updates must use `drag()`. If an app calls `setPlacement` at pointer rate, the work queues (documented).
2. **Section `set` keys follow the aspects present.** The default key is `section:<id>:plane`, `:meshes` or `:fill` (combined, such as `section:<id>:plane+fill`), so only sets of the same aspects coalesce. Coalescing also checks the target id.
3. **A poisoned batch applies nothing.** Every request in it reports `INSTANCE_POISONED`, none of its ranges reach the mirrors, and clients do not advance applied revisions from it. After `restart()` its unacknowledged intent is re-sent or surfaced as errors.
4. **Promise calls follow the frame loop.** Once `commitFrame()` has been seen, promise-returning calls queue for the next `commitFrame()`, with a fallback timer (100 ms, `FALLBACK_FLUSH_MS`) that flushes if no frame comes. Without a frame loop they flush on a microtask. Last-wins requests and queries are never reordered. Cost: a query can gain up to one frame of latency.
5. **`idOf`, `fieldStatus` and `sectionGap` stay public** (the 9b store needs them). The test-only `debugDump` is not in the public `.d.ts`.
6. **Chunking stays at request, field-build and repair-operation granularity.** There is no new Rust step in 9a (§4.3). Finer Rust-side chunking is a follow-up if benchmarks show long blocks; §10.3 has the numbers. If one recompute is huge, it blocks the Worker queue (the page stays responsive and the status stays `computing`).
7. **Restart after poisoning replays engine-acknowledged placements.** Post-poison calls already failed with `INSTANCE_POISONED` and are not replayed. Requests that cannot be replayed fail with `INSTANCE_POISONED`.
8. **Placements, colourings, fields and sections are declarative state.** A healthy restart replays the newest requested value; a poisoned one replays the engine-acknowledged value. In-flight non-declarative requests (edits, repair, undo/redo) are lost and fail. Pending declarative requests settle with the replay instead of failing.
9. **After any successful restart, every replayed or settled key reflects the replay**: `ready`, or `error` only if the replay itself failed for that key. Post-poison failures were already reported through their rejected calls, so placement keys of a poisoned restart settle `ready` too.

### 10.2 Open points for 9b
- **Status covers only the request's own key (overrides §9.4).** In 9a a key's status is `"computing"` only while a request under that key is pending. A drag, `setPlacement` or edit does not mark the dependent `display:<id>` or `section:<id>:*` keys, although colours and sections catch up only when its result arrives. The 9b store derives the dependent "catching up" status (for example from the placement or edit key of the meshes a colouring or section depends on).
- **Removed state is forgotten once its removal settles.** When the result of a removal has been applied (or at a restart), the client drops the removed mesh's, field's or section's state and the statuses of its keys (`status()` then reports `"idle"` for them); the handles keep giving `UNKNOWN_MESH`, `UNKNOWN_FIELD` or `UNKNOWN_SECTION`. The store should read a removal's outcome from its own action, not from a later `status()` of the removed key. After a poisoned result, field and section removal keys go `"idle"` at once (their removal counts as done), while mesh removal keys stay at `error` until a restart, because a mesh is pruned only when its removal record is delivered.
- **Deferred field-build errors.** A field `add` sent under a display key (`key: "display:<mesh>"`) is abandoned when the colouring it feeds is already superseded, and built when first needed. Its errors (a taken name, an unknown mesh) then surface on the request that first needs the field, not on the add. This is unchanged in 9a; nothing in 9a sends display-keyed field adds, so it only matters once the store does.
- **The store must send display-keyed field adds.** `WorkerScene.addDistanceField` keeps the `field:<id>` key, which is never cancelled. To get cancellation of superseded field builds, the 9b store, which owns the fields (§9.5), should send its field adds with `key: "display:<mesh>"`. This may need a small internal option on `WorkerScene`.
- **Undo snap-back (cosmetic).** `undo()` or `redo()` right after a drag drops the poses held for ended drags, so the engine's last acknowledged pose shows until the undo's result arrives. A mesh can briefly jump to that intermediate pose (for a frame or two) before it lands on the undone pose. The store could hold the pre-drag pose instead.
- **Retry-pose edge case.** When a restart's replay is posted and then fails, the next `restart()` replays the same poses (`#retryPoses`). Those can win over a `setPlacement` sent after the failed replay was posted.
- Smaller deferred points: the error code of an unpostable replay is mixed (`INVALID_ARGUMENT` for the batch, `INSTANCE_POISONED` for the scene); a removed mesh whose add failed in the Worker leaves a small state entry; non-`FastMeshError` exceptions cross as `INVALID_ARGUMENT` with `"<name>: <message>"`.

### 10.3 Measured costs
Measured on 2026-10-19 in the Node `worker_threads` harness (`pnpm --filter fast-mesh-wasm bench -- --worker --10m`), i5-13600K, Windows 11, Node 24.13.0, release wasm without the `parallel` feature (single-threaded, as published). Full tables and method: `docs/benchmarks.md`, "Worker scene". Medians of run 1 of two; run 2 agreed within about 10% except where noted there.

| | 1M triangles | 10M triangles |
| --- | --- | --- |
| Longest main-thread block during `addMesh` (target §6: < 16 ms per message) | 9.3 ms | 63 ms |
| of which the synchronous `addMesh()` call (input copies + box-centre scan) | 9.3 ms | 63 ms |
| Longest message handler during `addMesh` | 0.32 ms | 0.11 ms |
| Colouring toggle round trip, on / off (contact) | 1.0 / 0.06 ms | 9.9 / 0.2 ms |
| 64-pose drag with colouring: main-thread cost per frame (contact) | 0.03 ms | 0.12 ms |
| of which applying the transferred ranges (result handler) | 0.01 ms | 0.06 ms |
| 64-pose drag: round trip move to colours applied (contact) | 1.8 ms | 31 ms |
| Worker block of one `addMesh` request | 1.1 s | 12.7 s |

- **§6 target.** No message blocks the main thread for more than 0.32 ms. The 16 ms budget is met at 1M; at 10M the synchronous `addMesh()` call takes 63 ms (copies 33 ms, box-centre scan 34 ms; passing `origin` skips the scan). A transfer-taking `addMesh` variant, or a scan inside the Worker, would be needed to meet it at 10M.
- **Worst single-request Worker blocks (ruling 6).** On 1M meshes: a one-vertex edit of A takes 94 ms when a field is measured *to* A (B to A), and 348 ms with four fields (two of them `maxDistance` 8); one field build takes 106 to 218 ms; a wide-range drag move takes 0.4 s (up to 0.74 s); a section update over three 1M meshes takes under 1.3 ms. **The edit and wide-move cases exceed ~100 ms**, so finer Rust-side chunking (per-field recompute steps, or incremental sign data after small edits) is worth a follow-up issue; sections need none.
- **Many small requests.** 1000 `setPlacement` calls in one frame run in 2.3 ms in the Worker (longest block 3.2 ms, within the 4 ms yield budget) and 3.9 ms round trip, against 2.0 ms on the synchronous `Scene`.
- **Bundling (T3 concern 4).** The default `new URL("./worker-entry.js", import.meta.url)` survives Vite 8: `examples/webgpu` built with `vite build` emits `worker-entry-*.js` and the `.wasm`, and the page with `?worker` ran in headless Chrome (SwiftShader WebGPU) under both `vite preview` and `vite dev`, with picking, colouring, a drag, a section and undo working. In the monorepo the package is a workspace link, so Vite's dependency pre-bundling of an npm-installed copy was not exercised.

## 11. 9b implementation notes

What 9b shipped (`fast-mesh-wasm/store`, `/react` and `/r3f`, and `examples/react`), the decisions the plan made, the rulings made while building it, and what is left open. Like §9 and §10, this section wins over earlier sections where they differ.

### 11.1 Decisions made in the plan
1. **A new subpath `fast-mesh-wasm/store` holds `SceneStore`.** `./react` and `./r3f` import it, and non-React apps can use the store without React installed. The store entry imports only types from `scene.ts`, plus the wasm-free Worker client, so it never pulls the wasm glue into a bundle. The host is detected structurally (`"status" in scene` means a Worker scene), not with `instanceof Scene`.
2. **Declarative changes are flushed per frame, as in 9a ruling 4.** `setDisplay` and `setSection` update `desired` at once and reach the scene at the next `store.commitFrame()`, or on a microtask without a frame loop. Only the last value per mesh or section within a frame is applied, on both hosts, so the synchronous `Scene` gets the same coalescing.
3. **The store owns at most one distance field per coloured mesh.** It is keyed by `(distanceTo, maxDistance, sign)`, kept while colouring is off, reused when colouring comes back with the same key, and replaced (the old one removed) only when the key changes. Field names are `fast-mesh-store:<n>`.
4. **In Worker mode the store's field adds use the display key** (`key: "display:<meshId>"`, §10.2), through the internal `addFieldKeyed(scene, desc, key)` in `src/worker/client.ts`. Like `debugDump`, it is not exported from `./worker` and not in the public `.d.ts`.
5. **Dependent status (§10.2).** A mesh's status is `"computing"` while any of these is: its own `display:<id>` key, the `field:<id>` key of its store field, `placement:<id>` or `edit:<id>` of the mesh or of its `distanceTo` target, or `history`. A section follows the same rule over its `section:<id>…` keys, the `placement:`/`edit:` keys of its meshes, and `history`. `"error"` wins over `"computing"`, which wins over `"ready"` and `"idle"`. Rulings 1 and 3 below narrow which keys can give `"error"`.
6. **The local host is synchronous.** Every status is `"ready"` (or `"error"` when the call threw) right after the flush that applied it. `restart()` on a local store rejects with `INVALID_ARGUMENT` ("restart needs a Worker scene").
7. **`opacity`, `visible` and `selection` live only in the store.** Opacity is a number in [0, 1], default 1; visible defaults to true. Removing the selected mesh clears the selection.
8. **The undo snap-back (§10.2) is not addressed in 9b.** It is cosmetic, documented, and left as it is.

### 11.2 Rulings made during 9b
1. **Dependent keys only add `"computing"`.** The keys a state depends on (`placement:`/`edit:` of the meshes involved, and `history`) can make it `"computing"`, never `"error"`. `"error"` comes only from the state's own keys (`display:`/`field:` for a mesh, `section:<id>*` for a section). A failed edit is reported by its own rejected promise, not by colouring. Cost if wrong: a colouring that is stale after a failed edit shows `"ready"`.
2. **The microtask flush (no frame loop) does not call `scene.commitFrame()`.** Without a frame loop nothing renders, and calling it would steal changes from an app that commits the scene itself. Cost if wrong: none observed.
3. **`mesh:<id>` is an own key of a mesh** (§9.2: a failed add shows on the handle's status). Cost if wrong: a failed add puts the mesh's display status in `"error"`.
4. **`./store` may ship the wasm-free `WorkerScene` client in a shared chunk.** The chunk sits in `dist/` beside `worker-entry.js`, so `new URL("./worker-entry.js", import.meta.url)` still resolves. Cost if wrong: a few KB for Scene-only apps. Checked in Task 5: `examples/react` built with Vite 8 emits `worker-entry-*.js`, the page runs on the Worker store in headless Chrome, and CI asserts the file.
5. **Re-setting an identical rule that the engine rejected is not re-sent.** The error stays until the rule changes (idempotent declarative semantics). Cost if wrong: the user must change the rule to retry.
6. **Controlled inputs bind to `desired`.** The §5.2 shape is kept: the top-level display (and section) fields are the applied values, and `desired` is what was last requested. Both are exposed, and the README says which to bind. Cost if wrong: none.
7. **A pending declarative change reads `"computing"` on both hosts.** A mesh or section whose `desired` differs from what was last sent (a change waiting for the flush) reports `"computing"`, as the no-torn-UI constraint requires. Cost if wrong: a one-frame busy flicker on every toggle.
8. **Frame dedupe is a leader per store.** The first mounted `useFrameDriver` for a store commits, and leadership passes on when it unmounts. A frame stamp is not observable under `@react-three/test-renderer` or in R3F's frame order. Cost if wrong: one store driving two `<Canvas>` roots stops committing if the leader's canvas pauses (documented limitation).
9. **(Task 5) The `@react-three/fiber` peer is narrowed to `>=9`.** Only 9.x is installed and tested; v8 (the React 18 line) is not. `react` stays `>=18` for `./react`, which uses only React 18 APIs (`useSyncExternalStore`, context, memo hooks), but the hook tests run on React 19.
10. **(Final review) `<FastMesh>` exposes its binding through `onBind`.** `onBind(bound)` is called with the `BoundMesh` after `bindMesh`, and with `null` in the cleanup; it is held in a ref, so an inline callback never rebinds. `./three` exports `type MeshBinding = Pick<BoundMesh, "mesh" | "object">`, which `bindSection` (`readonly MeshBinding[]`) and `attachTransformControls` take. `bindSection` clips the materials it sees at bind time, so an app rebinds the section when `onBind` delivers a new binding (the README says so). `examples/react` uses `onBind` for its section and gizmo. Supersedes the Task 5 ruling that kept the binding private. Cost if wrong: none; the extra prop is optional.
11. **(Final review) The r3f driver invalidates the canvas on store and scene changes.** Per canvas (keyed by its `invalidate`), one `store.subscribe` and one `store.scene.onChanges` subscription (the latter only for non-empty change lists, since the scene's listeners run on every `commitFrame()`) call `invalidate()`, so `frameloop="demand"` renders each change. Cost if wrong: one extra frame after each change in `"always"` mode (none: R3F renders anyway).
12. **(Final review) The store returns to microtask flushing when its last driver unmounts.** `useFrameDriver`'s cleanup calls the internal `setFrameLoop(store, false)` (set from `SceneStore`'s static block, like `listenerCount`, and absent from every `.d.ts`), which also schedules the flush of changes waiting for a frame. Cost if wrong: none; a later `commitFrame()` switches frame flushing back on.
13. **(Final review) `<FastMesh>` applies opacity and visibility from a store listener.** The listener runs inside `store.commitFrame()` (before R3F renders the frame), so they show in the same frame as a colouring flushed with them, with one apply at bind time. This removes the one-frame lag of the earlier layout effect.
14. **(Final review) `store.dispose()` removes the store's owned fields** (`fast-mesh-store:<n>`, and so the colourings by them). Errors from a scene that cannot take the removal (a poisoned Worker scene, a freed `Scene`) are swallowed. On a Worker scene the removals are queued and go with the scene's next `commitFrame()`.

### 11.3 Open points and deferred minors
- **Snapshot refresh is O(meshes + sections) per event.** The store rescans every mesh and section on each scene or status event, so a burst of N adds costs O(N²). This is fine for tens of meshes; revisit it for scenes with many meshes.
- **Opacity and visibility lag colouring by one frame.** Resolved by ruling 13.
- **`<FastMesh>` with and without a `store` prop** has a different element tree (the prop wraps it in a `SceneProvider`), so adding or removing the prop remounts it and rebinds the mesh. Documented in the README.
- **`<FastMesh>`'s `color` prop** is normalised through `new THREE.Color(...).getHex()` on every render: an allocation per render, and colours quantised to 8 bits per channel.
- **No way to reach `<FastMesh>`'s `BoundMesh`.** Resolved by ruling 10 (`onBind`, `MeshBinding`).
- **Scene edits made straight through `store.scene`** reach the scene's change listeners only at the next `commitFrame()`, so under `frameloop="demand"` they render when something else invalidates the canvas. Edits made through the store invalidate it at once (ruling 11).
- **Undo snap-back** (§10.2, plan decision 8): unchanged.
- **Meshes removed straight through the scene** (not through the store) stay in the store's snapshot. Store actions throw `INVALID_ARGUMENT` after `dispose()`; the last snapshot stays readable.
- **Cosmetic or low-risk:** store field numbers (`fast-mesh-store:<n>`) skip after a failed add, and when a later tracked call throws synchronously the store may drop the display of an older in-flight request.
- **Bundling.** As in §10.3, the monorepo uses a workspace link, so Vite's dependency pre-bundling of an npm-installed copy of `./store` (with the Worker client in a shared chunk) was not exercised.
- **`THREE.Clock` deprecation warning.** `@react-three/fiber` 9.8 on three 0.186 logs "THREE.Clock: This module has been deprecated" once at start-up. It comes from R3F, not from this package.
