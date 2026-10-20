# Phase 9b: SceneStore, React and R3F Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a renderer-neutral, subscribable `SceneStore` with declarative display, section, history, selection and status state, running over either the synchronous `Scene` or the `WorkerScene`. Add React hooks on top of it, a thin React Three Fiber `<FastMesh>`, and an `examples/react` app.

**Architecture:**
- `src/store/` holds the store. It wraps a scene through a small `Host` interface with two implementations, one for `Scene` and one for `WorkerScene`.
- The store keeps an immutable `SceneSnapshot`. It replaces the snapshot, sharing unchanged parts, only when something visible changed.
- The store owns the distance fields behind declarative colouring.
- In Worker mode it derives the "catching up" status of dependent state from the request keys of the meshes that state depends on.
- `src/react/` provides hooks over `useSyncExternalStore`. `src/r3f/` provides `<FastMesh>` over the 8b `bindMesh`, with one `commitFrame` per rendered frame per store.

**Tech Stack:**
- TypeScript, built with tsdown and tested with vitest.
- React ≥18 as an optional peer.
- @react-three/fiber ≥8 as an optional peer.
- Tests use @testing-library/react with jsdom, and @react-three/test-renderer.
- The example uses Vite.

**Spec:** `docs/superpowers/specs/2026-10-19-fast-mesh-worker-react-design.md`. Sections §5, §9 and §10 apply; §9 and §10 override earlier sections, and §10.2 lists the open points this phase inherits.

## Decisions made here (my call; recorded in spec §11 by Task 5)

1. **A new subpath `fast-mesh-wasm/store` holds `SceneStore`.** `./react` and `./r3f` import it. Non-React apps use the store without React being installed.
   - The store entry imports only types from `scene.ts`, plus the wasm-free worker client, so it never pulls the wasm glue into the bundle.
   - Hosts are detected structurally (`"status" in scene` means Worker), not with `instanceof Scene`.
2. **Declarative changes are flushed per frame, as in 9a ruling 4.**
   - `setDisplay` and `setSection` update `desired` immediately.
   - They reach the scene at the next `store.commitFrame()`. Without a frame loop, they reach it on a microtask.
   - Only the last value per mesh or section within a frame is applied. This holds on both hosts, so the synchronous `Scene` gets the same coalescing.
3. **The store owns at most one distance field per coloured mesh.**
   - The field is keyed by `(distanceTo, maxDistance, sign)`.
   - It is kept while colouring is off and reused when colouring comes back on with the same key. It is replaced (old one removed) only when the key changes.
   - Field names are `fast-mesh-store:<n>`.
4. **In Worker mode, the store's field adds use the display key** (`key: "display:<meshId>"`, §10.2) through an internal module function `addFieldKeyed(scene, desc, key)` in `src/worker/client.ts`. It follows the `debugDump` pattern: it is not exported from `./worker` and not in the public `.d.ts`.
5. **Dependent status (§10.2).** A mesh's `display` status is `"computing"` while any of these is `"computing"`:
   - its own `display:<id>` key;
   - the `field:<id>` key of its store field;
   - `placement:<id>` or `edit:<id>` of the coloured mesh or of its `distanceTo` target;
   - `history`.

   A section's status follows the same rule over its `section:<id>…` keys and the `placement:` / `edit:` keys of its meshes, plus `history`. `"error"` wins over `"computing"`, which wins over `"ready"` and `"idle"`.
6. **The local host is synchronous.**
   - Every status is `"ready"` (or `"error"` when the call threw) right after the flush that applied it.
   - `restart()` on a local store rejects with `INVALID_ARGUMENT` ("restart needs a Worker scene").
7. **`opacity`, `visible` and `selection` live only in the store.** Opacity is a number in [0, 1], default 1. Visible defaults to true. Removing a selected mesh clears the selection.
8. **The undo snap-back (§10.2) is not addressed in 9b.** It is cosmetic, documented and left as it is.

## Global Constraints

- **Package.** Same package `fast-mesh-wasm`, with new exports `./store`, `./react` and `./r3f` (browser and neutral targets).
  - `react` is an optional peer at `>=18`, and `@react-three/fiber` is an optional peer at `>=8`.
  - The core entries (`.`, `./three`, `./worker`) must not import React. `./store` must not import React, three or the wasm glue. `./react` must not import three or R3F.
  - publint and attw stay green.
- **No torn UI.** A button must never show "on" while the mesh shows colours off, beyond the documented pending state (`status: "computing"`).
- **No stale results.** A result older than state the UI has already shown must never be applied.
- **No queue blow-up.** Rapid toggling must never queue unbounded work.
- **Sync stays.** Apps that stay on the main thread keep 8a behaviour unchanged.
- **Display state never enters history (8a).**
- **SSR.** The hooks must not crash during SSR. No data is produced on the server (`getServerSnapshot` returns an empty snapshot).
- **Tests.**
  - Store tests run in vitest, on Node with `Scene` from `src/node.js`, and with `WorkerScene` through the `worker_threads` harness (`test/harness/node-worker.mjs`, `createWorker` option).
  - Hooks are tested with @testing-library/react and react-dom in jsdom.
  - R3F gets a smoke test with @react-three/test-renderer if it runs headless. Otherwise it gets a type-check plus the example build.
- **Gate.** Every task ends green on all of these, in this order:
  - `cargo fmt --all -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
  - `pnpm build`
  - `pnpm --filter fast-mesh-wasm test`
  - `pnpm check:pkg`
  - `pnpm --filter fast-mesh-wasm-example-webgpu build`
  - from Task 5, `pnpm --filter fast-mesh-wasm-example-react build`
  - finally `pnpm build` again, so that `dist` is a release build.

## Review Focus

1. **A colouring target is removed.** `removeMesh(B)` while A is coloured by distance to B must not throw. A's `display.coloring` becomes `null`, `desired` keeps the rule, and status is `"error"` with `UNKNOWN_MESH`. Owned by Task 2: `colouring_target_removed`.
2. **StrictMode double effects.** Calling `setDisplay` twice with equal (not identical) rules in one render, as StrictMode does, builds exactly one field and sends one colouring. Owned by Task 2: `equal_rules_build_one_field`.
3. **Hooks for a removed mesh.** A component still mounted for a handle whose mesh was just removed renders a removed state (`status: "idle"`, `display` at defaults) and does not throw. Selection clears. Owned by Tasks 1 and 3: `removed_mesh_state` and `hook_survives_removed_mesh`.
4. **`<FastMesh>` unmounted after its mesh was removed.** Disposal runs once, raises no error, and leaves no frame-driver subscription. Owned by Task 4: `fastmesh_unmount_after_remove`.
5. **Many `<FastMesh>` instances.** N mounted `<FastMesh>` call `store.commitFrame()` exactly once per rendered frame. Owned by Task 4: `one_commit_per_frame`.

---

### Task 1: `SceneStore` core and hosts (`fast-mesh-wasm/store`)

**Files:**
- Create: `packages/fast-mesh-wasm/src/store/index.ts` (the public entry), `store.ts` (the `SceneStore` class), `host.ts` (`Host`, `localHost`, `workerHost`) and `snapshot.ts` (types plus structural-sharing helpers).
- Modify: `packages/fast-mesh-wasm/tsdown.config.ts` (entry `store`) and `package.json` (`exports["./store"]`).
- Test: `packages/fast-mesh-wasm/test/store-core.test.ts`.

**Interfaces:**
- **Produces, from `fast-mesh-wasm/store`:**
  ```ts
  type Status = "idle" | "computing" | "ready" | "error";
  interface DisplayState { coloring: DistanceColoring | null; opacity: number; visible: boolean }
  interface DistanceColoring { distanceTo: MeshHandle; maxDistance: number; sign?: SignMode;
    map: Colormap; range: readonly [number, number]; below?: OutOfRange; above?: OutOfRange;
    center?: number; base?: Rgba; nan?: Rgba }
  interface MeshState { display: DisplayState; desired: DisplayState; status: Status; revision: number; error?: FastMeshError }
  interface SectionState { plane: SectionPlane; meshes: readonly MeshHandle[]; fill: boolean;
    desired: { plane: SectionPlane; meshes: readonly MeshHandle[]; fill: boolean }; status: Status; revision: number; error?: FastMeshError }
  interface SceneSnapshot { revision: number; meshes: ReadonlyMap<MeshHandle, MeshState>;
    history: { canUndo: boolean; canRedo: boolean }; sections: ReadonlyMap<SectionHandle, SectionState>;
    selection: MeshHandle | null; status: "idle" | "computing" | "error"; error?: FastMeshError }
  function createSceneStore(scene: Scene | WorkerScene): SceneStore;
  class SceneStore {
    readonly scene: SceneLike;           // for the adapters
    readonly mode: "local" | "worker";
    getSnapshot(): SceneSnapshot; subscribe(listener: () => void): () => void;
    commitFrame(): MeshChange[];          // flushes declarative work, then scene.commitFrame()
    addMesh(positions, indices, options?): MeshHandle; removeMesh(mesh): void;
    setPlacement(mesh, placement): void; drag(mesh): DragSession;
    edit(mesh, ops): Promise<EditResult>; repair(mesh, op): Promise<RepairReport | RepairReport[]>;
    undo(): void; redo(): void;
    setDisplay(mesh, patch: Partial<DisplayState>): void;
    addSection(desc: SectionDesc): SectionHandle; setSection(s, patch: Partial<{ plane; meshes; fill }>): void; removeSection(s): void;
    select(mesh: MeshHandle | null): void;
    restart(): Promise<void>; dispose(): void;
  }
  const EMPTY_SNAPSHOT: SceneSnapshot;   // for SSR
  const REMOVED_MESH_STATE: MeshState;   // frozen; status "idle", default display
  function meshState(snap: SceneSnapshot, mesh: MeshHandle): MeshState; // REMOVED_MESH_STATE when absent
  ```
- **`Host` (internal, `host.ts`):** a thin layer that makes both scenes look alike to the store:
  - synchronous `canUndo()` / `canRedo()`;
  - `edit` and `repair` always returning promises;
  - `status(key): KeyStatus`, which is always `ready` on the local host;
  - `onStatus(cb)`, which is a no-op subscription on the local host;
  - `idOf(handle)`, which on the local host is a WeakMap counter;
  - field add/remove, `setColoring`, the section calls, and `restart`.
- In this task `setDisplay` handles `opacity` and `visible` only. Colouring and sections, beyond storing `desired`, arrive in Task 2.

- [ ] **Step 1: Write the failing tests** in `test/store-core.test.ts`. Run each test on both hosts, using `describe.each([["local", makeLocal], ["worker", makeWorker]])`; `makeWorker` uses the harness through `createWorker`.
  - `snapshot_is_immutable_and_shared`: after `setDisplay(a, { opacity: 0.5 })` and a flush, `getSnapshot() !== before`, `snap.meshes.get(b) === before.meshes.get(b)`, and `Object.isFrozen(snap)` and `Object.isFrozen(snap.meshes.get(a)!.display)` are both true.
  - `no_change_no_new_snapshot`: `setDisplay(a, { opacity: 1 })` when the opacity is already 1 leaves `getSnapshot()` identical and notifies no listener.
  - `history_follows_actions`: after `setPlacement(a, …)` and a flush, `history.canUndo === true`. After `undo()` and a flush, `canRedo === true`. Display changes never change `history`.
  - `selection_and_removed_mesh`: `select(a)`, then `removeMesh(a)`, then a flush gives `selection === null` and `meshes.has(a) === false`. A `setDisplay(a, …)` after that throws `FastMeshError` with code `UNKNOWN_MESH`.
  - `removed_mesh_state` (Review Focus 3): `getSnapshot().meshes.get(removed)` is `undefined`, and the store exports `meshState(snap, mesh)`, which returns the frozen `REMOVED_MESH_STATE` (`status: "idle"`, defaults) for a missing handle.
  - `flush_on_microtask_without_frames`: `setDisplay(a, { visible: false })` followed by `await Promise.resolve()` makes `display.visible` false. With a frame loop (after one `commitFrame()`), it applies only at the next `commitFrame()`.
  - `subscribe_unsubscribe`: the listener count goes back to 0, and `dispose()` removes the scene subscriptions (`onChanges`, `onStatus`).
  - `package_store_export`: `dist/store.js` and its transitive imports contain no `react`, `three` or `fast_mesh_wasm.js`. Follow the pattern of `package_worker_export_resolves` in `worker-client.test.ts`.
- [ ] **Step 2: Run** `pnpm --filter fast-mesh-wasm exec vitest run test/store-core.test.ts`. Expected: FAIL (cannot resolve `../src/store/index.js`).
- [ ] **Step 3: Implement** `host.ts`, `snapshot.ts`, `store.ts`, `index.ts`, the tsdown entry `store: "src/store/index.ts"` and `exports["./store"]` with `types` and `default`.
  - The snapshot is rebuilt by a single `#refresh()`, called after each flush, on the scene's `onStatus` and `onChanges`, and after actions.
  - `#refresh()` compares each mesh or section state with the previous one field by field. It reuses the old frozen object when they are equal, and bumps `revision` only when something changed.
- [ ] **Step 4: Run** the test file, then the full gate. Expected: PASS.
- [ ] **Step 5: Commit** `feat(store): SceneStore over Scene and WorkerScene`.

### Task 2: Declarative colouring, sections, dependent status and restart

**Files:**
- Modify: `packages/fast-mesh-wasm/src/store/store.ts` and `host.ts`, and `packages/fast-mesh-wasm/src/worker/client.ts` (internal `addFieldKeyed`).
- Test: `packages/fast-mesh-wasm/test/store-display.test.ts`.

**Interfaces:**
- **Consumes:** Task 1's `SceneStore`, `Host` and snapshot types.
- **Produces:**
  - `setDisplay(mesh, { coloring })`, with field ownership as in decision 3.
  - `addSection`, `setSection` and `removeSection` with `SectionState`.
  - Overall `snapshot.status` and `snapshot.error`.
  - `restart()`.
  - `export function addFieldKeyed(scene: WorkerScene, desc: DistanceFieldDesc, key: string): FieldHandle` in `src/worker/client.ts`. It is wired like `debugDump` and not re-exported from `src/worker/index.ts`.

- [ ] **Step 1: Write the failing tests** in `test/store-display.test.ts`, on both hosts unless a test says otherwise:
  - `toggle_storm_one_build`: `on`, `off`, `on` with the same rule within one frame, then `commitFrame()` and settle. Exactly one field add reaches the host. In Worker mode, count `"field"` requests in the harness `sent` log; locally, spy on `Scene.prototype.addDistanceField`. One colouring is applied, and `display.coloring` deep-equals the rule.
  - `toggle_reuses_field`: on, settle, off, settle, on, settle. There is still one field add in total, and `fetchFieldValues` / `fieldValues` is never needed.
  - `field_replaced_on_key_change`: changing `maxDistance` removes the old field (one `field` `remove`) and adds one new field.
  - `equal_rules_build_one_field` (Review Focus 2): two `setDisplay` calls with structurally equal, non-identical rules give one field and one colouring.
  - `colouring_target_removed` (Review Focus 1): colour A by B, then `removeMesh(B)` and settle. A's `display.coloring === null`, `desired.coloring` is kept, `status === "error"` and `error.code === "UNKNOWN_MESH"`. No uncaught error.
  - `dependent_status_during_drag` (Worker only): colour A by B. During `drag(B).move(...)` with the result held, `meshes.get(A).status === "computing"`. After the result is released, it becomes `"ready"`.
  - `superseded_field_build_is_cancelled` (Worker only): the store's field add carries `key: "display:<idOf(A)>"` (asserted on the harness `sent` log). With five meshes coloured and every colouring superseded before its build, fewer than five builds run. Use the engine's `onChunk` hook as `worker-lifecycle.test.ts` does.
  - `section_declarative`:
    1. `addSection({ plane, meshes: [a, b], fill: true })`, then a flush. The `SectionState` matches.
    2. `setSection(s, { plane: p2 })` twice in one frame sends one plane set.
    3. `removeSection(s)` removes it from the snapshot.
  - `overall_status`: `snapshot.status === "computing"` while any store-issued key is computing (Worker), and `"error"` with `error` set after a failed action. It goes back to `"idle"` once that key is ready again.
  - `restart_keeps_declarative_state` (Worker only): colour A, add a section, poison through the test-panic hook (as in `worker-lifecycle.test.ts`), then `await store.restart()`. Every `MeshState` and `SectionState` returns to `"ready"`, `display.coloring` deep-equals `desired.coloring`, and `history` is `{ canUndo: false, canRedo: false }`.
  - `local_restart_rejects`: the code is `INVALID_ARGUMENT`.
- [ ] **Step 2: Run** `pnpm --filter fast-mesh-wasm exec vitest run test/store-display.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement** the following:
  - Colouring flush per mesh:
    1. Resolve the owned field; reuse it or replace it.
    2. Map `DistanceColoring` to `ColorRule` (`field` plus the colour fields).
    3. Call `host.setColoring`.
  - The `display` value in the snapshot reflects what was last applied successfully. On the Worker host that is the colouring whose `display:` key is ready.
  - Dependent status per decision 5.
  - Sections flush only the aspects that changed.
  - `restart()` calls `host.restart()`, then `#refresh()`. WorkerScene already replays its fields, colourings and sections (9a ruling 8), so the store re-sends nothing.
- [ ] **Step 4: Run** the test file, then the full gate. Expected: PASS.
- [ ] **Step 5: Commit** `feat(store): declarative colouring, sections and status`.

### Task 3: React hooks (`fast-mesh-wasm/react`)

**Files:**
- Create: `packages/fast-mesh-wasm/src/react/index.ts`.
- Modify:
  - `tsdown.config.ts`: entry `react`, with `react` in `neverBundle`.
  - `package.json`: `exports["./react"]`; `peerDependencies` `react: ">=18"` (optional); devDependencies `react`, `react-dom`, `@types/react`, `@types/react-dom`, `@testing-library/react`, `jsdom`.
- Test: `packages/fast-mesh-wasm/test/react.test.tsx`, with `// @vitest-environment jsdom`.

**Interfaces:**
- **Consumes:** Task 1/2 `SceneStore`, `SceneSnapshot`, `meshState` and `EMPTY_SNAPSHOT`.
- **Produces:**
  ```ts
  function SceneProvider(props: { store: SceneStore; children?: ReactNode }): JSX.Element;
  function useScene(): SceneStore;                       // throws INVALID_ARGUMENT outside a provider
  function useStoreSelector<T>(selector: (s: SceneSnapshot) => T, isEqual?: (a: T, b: T) => boolean): T;
  function useMeshDisplay(mesh: MeshHandle): [MeshState & DisplayState, (patch: Partial<DisplayState>) => void];
  function useHistory(): { canUndo: boolean; canRedo: boolean; undo(): void; redo(): void };
  function useSection(section: SectionHandle): [SectionState, (patch: Partial<{ plane: SectionPlane; meshes: readonly MeshHandle[]; fill: boolean }>) => void];
  function useSceneStatus(): { status: "idle" | "computing" | "error"; error?: FastMeshError };
  function useSelection(): [MeshHandle | null, (mesh: MeshHandle | null) => void];
  ```
  - Every hook is built on `useSyncExternalStore(store.subscribe, get, () => EMPTY_SNAPSHOT-derived)`.
  - Every returned setter is stable across renders, through `useCallback` on `[store, handle]`.

- [ ] **Step 1: Write the failing tests** in `test/react.test.tsx`, using a local store over `Scene` from `src/node.js`:
  - `toggle_button_pressed_and_busy`: a `<Toggle mesh>` renders `aria-pressed={state.desired.coloring !== null}`. Clicking it updates `aria-pressed` before the flush. After `act(() => store.commitFrame())` the button shows `display.coloring` set.
  - `undo_redo_enable`: buttons are disabled at first. After `setPlacement` and a flush, Undo is enabled. After clicking Undo and a flush, Redo is enabled.
  - `opacity_slider`: an `<input type="range">` bound to `opacity` reflects `0.3` after `setDisplay(a, { opacity: 0.3 })` from outside React.
  - `external_change_updates_all`: two components subscribed to `useHistory()` both re-render after `store.undo()` is called outside React (a keyboard handler).
  - `rerender_only_on_slice_change`: a render counter on `useMeshDisplay(b)` does not increase when only `a` changes.
  - `no_tearing_with_transition`: an update inside `startTransition` while two components read the same mesh. Every render sees the same `revision`; record the revisions per commit.
  - `hook_survives_removed_mesh` (Review Focus 3): a mounted `useMeshDisplay(a)` component, after `removeMesh(a)` and a flush, renders `status "idle"` without throwing.
  - `ssr_does_not_crash`: `renderToString(<SceneProvider store={store}><Toggle mesh={a} /></SceneProvider>)` from `react-dom/server` returns a string, using `EMPTY_SNAPSHOT`.
  - `strict_mode_idempotent`: under `<StrictMode>`, an effect calling `setDisplay(a, { coloring: rule })` results in one field add (spy).
  - `package_react_export`: `dist/react.js` imports `react` and `./store`-chunk code, but never `three`, `@react-three/fiber` or the wasm glue.
- [ ] **Step 2: Run** `pnpm --filter fast-mesh-wasm exec vitest run test/react.test.tsx`. Expected: FAIL.
- [ ] **Step 3: Implement** `src/react/index.ts` and the packaging. Add a `vitest` config change only if `.tsx` tests need one; prefer the per-file `// @vitest-environment jsdom` comment.
- [ ] **Step 4: Run** the test file, then the full gate. Expected: PASS.
- [ ] **Step 5: Commit** `feat(react): hooks over SceneStore`.

### Task 4: React Three Fiber `<FastMesh>` (`fast-mesh-wasm/r3f`)

**Files:**
- Create: `packages/fast-mesh-wasm/src/r3f/index.ts`.
- Modify:
  - `tsdown.config.ts`: entry `r3f`, with `@react-three/fiber` and `react` in `neverBundle`.
  - `package.json`: `exports["./r3f"]`; optional peer `@react-three/fiber: ">=8"`; devDependencies `@react-three/fiber`, `@react-three/test-renderer`.
- Test: `packages/fast-mesh-wasm/test/r3f.test.tsx`.

**Interfaces:**
- **Consumes:** `useScene()` and the hooks from Task 3, and `bindMesh(scene: SceneLike, mesh, { material?, opacity? }): BoundMesh` from `fast-mesh-wasm/three`.
- **Produces:**
  ```ts
  interface FastMeshProps { mesh: MeshHandle; store?: SceneStore; material?: BindMeshOptions["material"]; color?: THREE.ColorRepresentation;
    onPointerDown?: (e: ThreeEvent<PointerEvent>) => void; /* other R3F pointer events passed through */ }
  function FastMesh(props: FastMeshProps): JSX.Element;   // <primitive object={bound.object} />
  function useFrameDriver(store?: SceneStore): void;       // exported; FastMesh calls it
  ```
- **Behaviour:**
  - `<FastMesh>` calls `bindMesh` on mount and `dispose()` on unmount. `opacity` follows `display.opacity` through `bound.setOpacity`, and `object.visible` follows `display.visible`.
  - The frame driver calls `store.commitFrame()` at most once per rendered frame per store, however many `<FastMesh>` instances are mounted. Use `useFrame` with a per-store `WeakMap` record of the last frame stamp; the stamp is R3F's `state.clock.elapsedTime`, or a frame counter read from `useThree`'s `gl.info.render.frame`.

- [ ] **Step 1: Write the failing tests** in `test/r3f.test.tsx`, using @react-three/test-renderer with a local store:
  - `fastmesh_binds_and_follows_display`: after `create(<SceneProvider store><FastMesh mesh={a} /></SceneProvider>)`, the scene graph has one mesh. `store.setDisplay(a, { opacity: 0.4, visible: false })`, a flush and `advanceFrames(1)` leave `material.opacity === 0.4`, `material.transparent === true` and `object.visible === false`.
  - `one_commit_per_frame` (Review Focus 5): with three `<FastMesh>` mounted, a spy on `store.commitFrame` and `advanceFrames(5)`, the spy is called 5 times.
  - `fastmesh_unmount_after_remove` (Review Focus 4): `removeMesh(a)`, a flush, then unmounting raises no error, `dispose` runs once (spy on the `BoundMesh`), and further `advanceFrames` calls do not call `commitFrame` for an unmounted-only tree.
  - `package_r3f_export`: `dist/r3f.js` imports `@react-three/fiber`, `three` and `react`, and does not inline them.

  If @react-three/test-renderer cannot run headless in this environment, record that and replace these tests with a type-check of a `.tsx` fixture plus unit tests of `useFrameDriver`'s per-store dedupe against a fake `useFrame`. Report this in DONE_WITH_CONCERNS.
- [ ] **Step 2: Run** `pnpm --filter fast-mesh-wasm exec vitest run test/r3f.test.tsx`. Expected: FAIL.
- [ ] **Step 3: Implement** `src/r3f/index.ts` and the packaging.
- [ ] **Step 4: Run** the test file, then the full gate. Expected: PASS.
- [ ] **Step 5: Commit** `feat(r3f): FastMesh component and frame driver`.

### Task 5: `examples/react`, CI, docs and changeset

**Files:**
- Create `examples/react/` (Vite + React + R3F, package name `fast-mesh-wasm-example-react`). It contains `index.html`, `src/main.tsx`, `src/App.tsx`, `src/Panel.tsx`, `vite.config.ts`, `tsconfig.json`, `package.json` (scripts `dev`, `build` = `tsc --noEmit -p . && vite build`, `preview`) and `README.md`.
- Modify:
  - `.github/workflows/ci.yml`: build the example and assert that `dist/assets/worker-entry-*.js` is emitted, as for webgpu.
  - Root `README.md`: a "React" section covering the store, hooks and `<FastMesh>`.
  - `docs/superpowers/specs/2026-10-19-fast-mesh-worker-react-design.md`: add §11, "9b implementation notes". It lists the decisions from this plan, the rulings recorded during execution, and the open points.
  - `.changeset/<name>.md`: a minor bump describing `./store`, `./react` and `./r3f`.

**Example behaviour:**
- **Scene:** a `WorkerScene` store with two generated meshes (for example a sphere and a displaced plane), plus STL drop to replace them, reusing the webgpu example's STL loader.
- **Per-mesh panel:**
  - a colouring toggle (distance to the other mesh) with `aria-pressed` and a busy indicator;
  - an opacity slider;
  - a visibility checkbox.
- **Other panels:**
  - **Section:** a toggle and a plane-offset slider.
  - **History:** Undo and Redo buttons with their enabled state, plus Ctrl+Z / Ctrl+Shift+Z.
  - **Status:** shows `useSceneStatus()`, and Restart when the status is `error`.
- **Dragging:** with three's `TransformControls` through `attachTransformControls`.
- **`?local` flag:** switches to the synchronous `Scene` store.

- [ ] **Step 1: Build the example.** Run `pnpm --filter fast-mesh-wasm-example-react build`. Expected: PASS, with `worker-entry-*.js` in `dist/assets`.
- [ ] **Step 2: Check it in a browser.** Run `vite preview` and open the page in headless Chrome. Toggling colouring, the slider, undo/redo and the section must work, with no console errors. Record the outcome in the report. If no WebGL is available headless, record that instead.
- [ ] **Step 3: Write the docs.** Update the README, add spec §11 and add the changeset.
- [ ] **Step 4: Run** the full gate, including the react example build. Expected: green.
- [ ] **Step 5: Commit** `feat(examples): React example`, `docs(react): document the store, hooks and FastMesh`, and `chore: changeset for store, react and r3f`.
