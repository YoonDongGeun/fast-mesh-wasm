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
