# fast-mesh Worker scene (Phase 9a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `fast-mesh-wasm/worker` runs the engine in a Web Worker behind an async API. It provides:
- ordered requests with coalescing, revisions and cooperative cancellation;
- dirty ranges transferred into main-thread mirrors;
- synchronous handles, `views`, `matrix4`, `section` and `onChanges` on the main side;
- optimistic drags and `restart()`.

The 8b adapters become structurally typed (`SceneLike`), so they work with both the sync `Scene` and the Worker scene.

**Architecture:**
- `src/worker/protocol.ts` defines message types.
- `src/worker/engine.ts` is the Worker-side core:
  - holds a sync `Scene`;
  - maps client ids to engine handles;
  - runs a keyed queue with coalescing and cancellation;
  - packs results.
  It is pure and talks through a `Port` interface, so it runs in a browser Worker or in Node `worker_threads` for tests.
- `src/worker/entry.ts` is the browser Worker script.
- `src/worker/client.ts` is the main-side `WorkerScene` proxy (mirrors, handles, optimistic drags).
- `src/worker/index.ts` exports `createWorkerScene`.

**Tech Stack:** TypeScript, tsdown, vitest in Node with `worker_threads` as the Worker harness, wasm-bindgen output from 8a.

**Spec:** `docs/superpowers/specs/2026-10-19-fast-mesh-worker-react-design.md`. §9 overrides earlier sections; in particular, transfer mode only, client-side handles and optimistic drags. 9b (store, React, R3F) gets its own plan.

## Global Constraints
- **Package exports.** New export `./worker`, browser target, with no Node export condition.
  - The Worker script ships as a separate built file loaded via `new Worker(new URL("./worker-entry.js", import.meta.url), { type: "module" })`.
  - A `workerUrl` / `createWorker` option overrides it.
  - The core entries must not import worker code. The worker entry must not import three or React.
  - `publint --strict` and `attw --profile esm-only` must stay green.
- **Messages.** Structured clone plus transferable `ArrayBuffer`s only. No SharedArrayBuffer and no Atomics in 9a.
- **Ordering.**
  - The Worker processes requests in arrival order.
  - Coalescing drops not-yet-started requests that a later request with the same key supersedes, for "last wins" keys only:
    - display and colouring per mesh
    - section plane, meshes and fill per section
    - drag moves per mesh
  - Edits, repair, undo, redo, add and remove are never coalesced.
- **Revisions.**
  - Every state-changing request gets a monotonically increasing revision on the main side.
  - Results carry the highest processed revision.
  - State and status deltas older than what the main side has already applied for a key are ignored.
  - Dirty-range bytes are never dropped; they are always applied, in order.
- **Handles.** Handles are created synchronously on the main side with client ids. Messages name client ids. The Worker maps client ids to engine handles. Handles survive `restart()`.
- **Mirrors.**
  - Transfer mode: results carry the bytes of every dirty range, per buffer.
  - `capacityChanged` and `"all"` carry whole buffers.
  - Placements and `matrix4` are sent for every mesh whose placement changed.
  - Section results (contours and fill) are sent for every changed section entry.
- **Drags (optimistic).**
  - `drag(mesh).move(p)` updates the main placement and `matrix4` mirror immediately and emits a placement change to local `onChanges` listeners on the next `commitFrame`.
  - It sends at most one coalesced move per frame.
  - `end()` and `cancel()` are synchronous on the main side.
  - `cancel()` restores the start placement and records nothing.
- **Errors.**
  - `FastMeshError` codes cross the boundary unchanged.
  - A Worker trap gives `INSTANCE_POISONED` on every pending and later call until `restart()`.
- **Scope.**
  - No user-opened steps, no Lazy/Manual modes (the Worker scene is always Eager), no `takeChanges`.
  - Queries are async.
  - `fieldValues` becomes `fetchFieldValues`, which is async.
- **Commits.**
  - Conventional messages ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (or the implementing model).
  - Each message goes in a NEW temp file and is committed with `git commit -F`.
  - Files are UTF-8 with LF.
  - No git worktrees.
  - Do not touch `../fast-mesh-wasm-parallel`.
- **Gate.** Every task ends green on all of the following, finishing with `pnpm build`:
  - `cargo fmt --all -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
  - `pnpm build`
  - `pnpm --filter fast-mesh-wasm test`
  - `pnpm check:pkg`
  - `pnpm --filter fast-mesh-wasm-example-webgpu build`
- **Changeset.** Minor bump, in the last task.

## Review Focus
1. **Rapid toggle storm.** 50 alternating colouring on/off requests sent within one frame while a long field build runs. The result is exactly one build plus at most one recolour, and the final applied state equals the last request. Test: Task 3 `toggle_storm_coalesces`.
2. **Edit then display in the same frame.** An edit (never coalesced) followed by display changes keeps its order. The mirror equals the engine after the batch. Test: Task 2 `mixed_batch_keeps_edit_order`.
3. **Worker trap mid-batch.** Every pending promise rejects with `INSTANCE_POISONED`. Then `restart()` rebuilds the meshes, placements and sections from mirrors, and the handles still work. Test: Task 4 `poison_then_restart_rebuilds`.
4. **Handle used before its add is processed.** `setPlacement` on a just-created handle queues behind the add and applies. If the add failed, the handle reports error and later calls give `UNKNOWN_MESH`. Test: Task 2 `handle_before_add_processed`.
5. **Drag while results are in flight.** The pose stays where the user put it (it is not snapped back by an older placement result). After `end()` the final placement matches the Worker's. Test: Task 3 `optimistic_drag_not_snapped_back`.

---

### Task 1: `SceneLike` and adapter retyping (type-only)

**Files:**
- Create: `packages/fast-mesh-wasm/src/scene-like.ts`.
- Modify:
  - `packages/fast-mesh-wasm/src/three/mesh.ts`, `controls.ts` and `section.ts`: accept `SceneLike` instead of `Scene`.
  - `packages/fast-mesh-wasm/src/scene.ts`: `Scene implements SceneLike`.
  - Exports in `web.ts`, `node.ts` and `three/index.ts`, as types.
- Test: `packages/fast-mesh-wasm/test/scene-like.test.ts`.

**Interfaces (produces):**
```ts
export interface SceneLike {
  views(mesh: MeshHandle): MeshViews;
  matrix4(mesh: MeshHandle): Float64Array;
  placement(mesh: MeshHandle): { rotation: Mat3; translation: Vec3 };
  section(s: SectionHandle, mesh: MeshHandle): SectionResult;
  sectionPlane(s: SectionHandle): { origin: Vec3; normal: Vec3 };
  onChanges(listener: (c: MeshChange[]) => void): () => void;
  commitFrame(): MeshChange[];
  drag(mesh: MeshHandle): DragSession;
}
```
- `MeshHandle`, `SectionHandle` and `DragSession` become structural or opaque types shared by both hosts. Keep the 8a runtime classes. Where the adapters do `instanceof` checks or need nominal brands, replace them with a `SceneLike`-safe check.

- [ ] **Step 1: Write failing tests.**
  - `adapters_accept_scene_like`: a hand-written fake `SceneLike` (in-memory views and a change emitter) drives `bindMesh` and `bindSection` in Node.
  - A type test: `Scene` is assignable to `SceneLike`.
- [ ] **Step 2: Run** `pnpm build && pnpm --filter fast-mesh-wasm test`. Expected: FAIL.
- [ ] **Step 3: Implement.** No behaviour change: all 8a and 8b tests stay green.
- [ ] **Step 4: Run** the full gate. Expected: PASS.
- [ ] **Step 5: Commit** `refactor(js): type the three.js adapter against SceneLike`.

### Task 2: Protocol and Worker-side engine (queue, id map, results)

**Files:**
- Create:
  - `packages/fast-mesh-wasm/src/worker/protocol.ts`
  - `packages/fast-mesh-wasm/src/worker/engine.ts`
  - `packages/fast-mesh-wasm/test/harness/node-worker.mjs` (a `worker_threads` entry that loads the wasm via the node entry, then runs `engine.ts` on `parentPort`)
  - `packages/fast-mesh-wasm/test/worker-engine.test.ts`

**Interfaces (produces):**
```ts
// protocol.ts
type ClientId = number;
type Request =
  | { kind: "addMesh"; rev: number; id: ClientId; positions: ArrayBuffer /* f64 or f32, tagged */; f64: boolean; indices: ArrayBuffer; options: AddMeshOptions }
  | { kind: "removeMesh" | "compact"; rev: number; id: ClientId }
  | { kind: "setPlacement"; rev: number; id: ClientId; rotation: number[]; translation: number[]; key: string }
  | { kind: "drag"; rev: number; id: ClientId; phase: "begin" | "move" | "end" | "cancel"; rotation?: number[]; translation?: number[]; key: string }
  | { kind: "edit" | "repair"; rev: number; id: ClientId; payload: unknown }
  | { kind: "undo" | "redo"; rev: number }
  | { kind: "coloring"; rev: number; id: ClientId; rule: ColorRuleWire | null; key: string }
  | { kind: "field"; rev: number; field: ClientId; op: "add" | "remove"; desc?: FieldDescWire }
  | { kind: "section"; rev: number; section: ClientId; op: "add" | "set" | "remove"; desc?: SectionDescWire; key?: string }
  | { kind: "query"; rev: number; qid: number; name: string; args: unknown }
  | { kind: "fetchFieldValues"; rev: number; qid: number; field: ClientId };
type Result = {
  rev: number;                            // highest processed revision
  changes: MeshChangeWire[];              // 8a change records with client ids + dirty-range bytes (transferred)
  placements: { id: ClientId; rotation: number[]; translation: number[]; matrix4: number[] }[];
  sections: { section: ClientId; mesh: ClientId; result: SectionResultWire }[];
  history: { canUndo: boolean; canRedo: boolean };
  status: { key: string; rev: number; status: "ready" | "error"; error?: { code: string; message: string } }[];
  replies: { qid: number; ok: boolean; value?: unknown; error?: { code: string; message: string } }[];
};
// engine.ts
export interface Port { post(msg: Result, transfer: Transferable[]): void; onMessage(cb: (batch: Request[]) => void): void }
export function runEngine(port: Port): void; // assumes wasm initialised
```

**Queue (per spec §4.1).**
- Requests arrive in batches, one per main-side frame flush. The queue appends them.
- Before starting each request, drop any queued, not-started request with the same "last wins" `key` that a later queued request supersedes.
- Process in order. After the batch, call `scene.commitFrame()`, then build one `Result`:
  - each dirty range is copied into a transferred `ArrayBuffer` per buffer;
  - `"all"` or a capacity change sends the whole buffer.
- **Errors.** A request that throws records a status error for its key and the queue continues. A trap (RuntimeError) posts a poisoned result, and the engine stops.

- [ ] **Step 1: Write failing tests** (Node harness):
  - `add_place_round_trip`: a mirror rebuilt from results equals the engine views.
  - `mixed_batch_keeps_edit_order` (Review Focus 2)
  - `handle_before_add_processed` (Review Focus 4)
  - `coalesces_last_wins_keys_only`
  - `section_and_placement_mirrors_delivered`
  - `errors_cross_boundary_with_codes`
  - `trap_posts_poisoned`, using the test-panic build
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full gate. Expected: PASS.
- [ ] **Step 5: Commit** `feat(worker): protocol and worker-side engine`.

### Task 3: Main-side `WorkerScene` (mirrors, handles, optimistic drags, async queries)

**Files:**
- Create: `packages/fast-mesh-wasm/src/worker/client.ts`, `packages/fast-mesh-wasm/src/worker/index.ts`, `packages/fast-mesh-wasm/src/worker/entry.ts` (the browser Worker script: `init()`, then `runEngine(selfPort)`).
- Modify:
  - `packages/fast-mesh-wasm/tsdown.config.ts`: add `worker` and `worker-entry` entries. The worker-entry chunk is a standalone ES module whose wasm glue resolves next to it.
  - `packages/fast-mesh-wasm/package.json`: `exports["./worker"]`.
- Test: `packages/fast-mesh-wasm/test/worker-client.test.ts`, using the Node harness through a `createWorker` option.

**Interfaces (produces):**
```ts
export interface WorkerSceneOptions { workerUrl?: string | URL; createWorker?: () => WorkerLike; debug?: boolean }
export interface WorkerLike { postMessage(msg: unknown, transfer?: Transferable[]): void; addEventListener(type: "message" | "error", cb: (e: any) => void): void; terminate(): void }
export function createWorkerScene(options?: WorkerSceneOptions): Promise<WorkerScene>;
export class WorkerScene implements SceneLike {
  addMesh(positions: Positions64, indices: Indices, options?: { origin?: Vec3; normals?: boolean }): MeshHandle;  // sync handle
  removeMesh(mesh): void; setPlacement(mesh, p): void; placement(mesh); matrix4(mesh);
  drag(mesh): DragSession; edit(mesh, ops): Promise<{ added: Uint32Array[] }>; compact(mesh): void;
  analyze(mesh, o?): Promise<Analysis>; repair(mesh, op): Promise<RepairReport | RepairReport[]>;
  undo(): void; redo(): void; readonly canUndo: boolean; readonly canRedo: boolean;
  addDistanceField(d): FieldHandle; removeField(f): void; setColoring(mesh, rule | null): void; fetchFieldValues(f): Promise<Float32Array>;
  addSection(d): SectionHandle; setSectionPlane(s, p): void; setSectionMeshes(s, m): void; setSectionFill(s, f): void; removeSection(s): void;
  section(s, mesh): SectionResult; sectionPlane(s);
  raycast(...): Promise<...>; closestPoint(...): Promise<...>; meshDistance(...): Promise<...>; intersects(...): Promise<boolean>; findContacts(...): Promise<...>;
  views(mesh): MeshViews; onChanges(l): () => void; commitFrame(): MeshChange[];
  status(key: string): { status: "idle" | "computing" | "ready" | "error"; revision: number; error?: FastMeshError };
  onStatus(l: () => void): () => void;  // fires when any status/history changes (for the 9b store)
  restart(): Promise<void>; terminate(): void;
}
```

**Main-side mirrors.**
- Per mesh, keep a `Float32Array` for positions and normals, a `Uint8Array` for colours and a `Uint32Array` for indices. Apply transferred ranges in result order.
- `views()` returns the mirrors, following the same fresh-view rule; the debug guard applies to mirror reallocations.
- `commitFrame()` flushes the batch of requests queued this frame to the Worker. It then returns, and delivers to listeners, the change records from results received since the last call, with their ranges already applied to the mirrors, plus optimistic placement changes.

**Optimistic drags.**
- `move` sets a local pose. That pose overrides placement results for the mesh until the drag's `end` or `cancel` result is processed.

**Status.**
- Requests put their key into "computing".
- The result for that key's revision sets "ready" or "error", unless a newer request for the key is still pending.

- [ ] **Step 1: Write failing tests:**
  - `views_mirror_engine_after_edits`: a randomised sequence keeps the mirror byte-identical to the Worker engine. Fetch the engine views through a test-only `debugDump` reply.
  - `toggle_storm_coalesces` (Review Focus 1)
  - `optimistic_drag_not_snapped_back` (Review Focus 5)
  - `async_queries_resolve_and_reject_with_codes`
  - `three_adapter_works_on_worker_scene`: `bindMesh` on a `WorkerScene` stays in sync. Use three in Node.
  - `status_computing_then_ready`
  - `package_worker_export_resolves`: after the build, `dist/worker.js` and `dist/worker-entry.js` exist, the core `dist/web.js` does not import them, and `check:pkg` passes.
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full gate. Expected: PASS.
- [ ] **Step 5: Commit** `feat(worker): WorkerScene with mirrors, optimistic drags and async queries`.

### Task 4: Cooperative cancellation, stale rejection and `restart()`

**Files:**
- Modify: `packages/fast-mesh-wasm/src/worker/engine.ts` and `client.ts`.
- Test: `packages/fast-mesh-wasm/test/worker-lifecycle.test.ts`.

**Behaviour (spec §4.3, §9.6).**
- **Chunking.** Long operations are chunked, yielding to the Worker's message loop between chunks with `setTimeout(0)` or a `MessageChannel` tick:
  - `updateFields` over many due fields, one field per chunk;
  - repair pipelines, one op per chunk;
  - section updates over many entries.
- **Cancellation.** Before each chunk, if a newer request with the same "last wins" key has arrived, abandon the rest. Status stays "computing" until the newer request completes.
- **Restart.** `restart()` terminates the Worker, starts a new one, and replays the scene:
  - `addMesh` from mirrors (live triangles only, compacted; local positions plus each mesh's origin);
  - placements;
  - fields, colourings and sections as last requested.

  The client-id map is rebuilt, so handles keep working. History is cleared.

- [ ] **Step 1: Write failing tests:**
  - `superseded_long_field_build_is_cancelled`: five due fields; a newer colouring request arrives after chunk 1, and fewer than five field computations run (counted through a test hook).
  - `stale_status_ignored`
  - `poison_then_restart_rebuilds` (Review Focus 3)
  - `restart_preserves_handles_and_display`
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full gate. Expected: PASS.
- [ ] **Step 5: Commit** `feat(worker): cooperative cancellation and restart`.

### Task 5: Benchmarks, docs, changeset, example wiring

**Files:**
- Modify: `packages/fast-mesh-wasm/bench/scene.bench.mjs`. Add a Worker case run in the Node harness, measuring:
  - round-trip latency of a colouring toggle;
  - per-frame cost of applying transferred ranges for a 1M drag with colouring (main side);
  - the longest main-thread block during a 1M `addMesh`, using a timer probe; the target is under 16 ms per message.
- Modify: `docs/benchmarks.md` ("Worker scene"), the root `README.md` (Worker section: `createWorkerScene`, the `commitFrame` loop, optimistic drags, async queries, `restart`, the bundler note) and the spec (add §10 "9a implementation notes": rulings and numbers).
- Create: `.changeset/<name>.md`, a minor bump describing `fast-mesh-wasm/worker` and the `SceneLike` typing.
- Optional: in `examples/webgpu`, add a `?worker` query flag that switches to `createWorkerScene`, to prove the pattern. Build it in CI.

- [ ] **Step 1: Add** the bench cases and run them on a release build. Record real numbers.
- [ ] **Step 2: Write** the docs and the changeset, plus the optional example flag.
- [ ] **Step 3: Run** the full gate. Expected: green.
- [ ] **Step 4: Commit** `bench(worker): worker scene costs`, `docs(worker): document the Worker scene`, `chore: changeset for fast-mesh-wasm/worker`.
