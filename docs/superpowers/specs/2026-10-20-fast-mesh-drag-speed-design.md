# fast-mesh: fast wide-range distance colouring during drags (parallel stage 2)

- **Status:** draft for review
- **Parent specs:**
  - `2026-10-18-fast-mesh-parallel-design.md`. Its §1 defines stage 2 as the wasm threads build.
  - `2026-10-19-fast-mesh-worker-react-design.md` (§9.1 SAB, §10.3 measured costs, §11).
  - Follow-up issue #48: Rust-side chunking.
- **Labels:** *(owner)* marks what the owner said. *(my call)* marks choices made without asking.

## 1. Intent

When the owner drags a mesh while it is coloured by distance to another mesh, the colours lag behind *(owner)*. This happens when the colouring range (`maxDistance`) is wide *(owner)*. With a contact-sized range the update is fast. Measured in 9a §10.3, at 1M triangles, one pose costs:

| Range | Cost per pose |
| --- | --- |
| contact | ~2 ms |
| wide (`maxDistance` 8) | ~0.4 s, up to 0.74 s |

The wide case is slow because nearly every vertex of the coloured mesh is in range, so every pose recomputes almost all of them.

What the owner accepts *(owner)*:
- Colours exact on every frame are welcome.
- A small delay is also fine, by throttling or debouncing, as long as performance improves.

The rule this design takes from that *(my call, from the owner's answer)*: the colours shown are always the complete, exact result for one pose. That pose may be a few frames behind the pointer, and intermediate poses may be skipped. Partial or progressive colouring, where nearby vertices update first and far ones later, is not used, because it would show colours from two different poses at once.

The work has two parts, in this order *(owner agreed)*:
- **A. Make one recompute cheaper and interruptible.** This is Rust engine work and helps every user, whether the page has security headers or not.
- **B. Wasm threads build.** This is stage 2 of the parallel plan. It runs the per-vertex work on all cores, on pages that enable cross-origin isolation. The owner will make sure the pages enable it, and the library should guide users to do the same *(owner)*.

There is one spec and two plans and PRs: A first, then B.

## 2. Goals, non-goals, anti-goals

### Goals
1. **Warm start (A).**
   - **Remember the closest triangle.** A distance field remembers, per vertex, the triangle of the target that was closest at the last update.
   - **Seed the next query.** The next update searches from that triangle. Its distance, measured at the new pose, is a valid upper bound, so the BVH search starts with a small radius and prunes most of the tree.
   - **Exactness.** Results stay bit-identical to a cold search: the same closest feature, distance and sign.
2. **Interruptible recompute (A, closes #48).**
   - **Budgeted steps.** A field recompute can run in resumable steps over its candidate vertices: a cursor plus a time or vertex budget.
   - **Cancellation in the Worker.** Between steps, the Worker engine checks for a newer request with the same last-wins key, for example a newer drag move of the same mesh. If one is waiting, it abandons the stale recompute and starts on the newest pose.
   - **No partial results.** A partial recompute is never published. Values and colours change only when a pose completes (see Intent).
3. **Latest pose wins, with a bounded delay (A).**
   - While a drag is in progress, the Worker processes only the newest pose. A recompute already in progress for an older pose is abandoned at the next step boundary.
   - On drag end, the final pose is computed to completion.
   - *Target:* the time from the last pointer move to exact colours is at most one full recompute plus one step.
4. **Threads (B).**
   - **Second artifact.** A second wasm artifact is built with atomics and shared memory, and with the existing `parallel` feature (rayon), using `wasm-bindgen-rayon`. Both artifacts ship in the npm package.
   - **Selection.** The Worker loads the threaded artifact when `crossOriginIsolated` is true. Otherwise it loads the current single-threaded one.
   - **Identical results.** Results are bit-identical between the two (parallel stage 1 already guarantees this natively).
5. **Header guidance (B).** The owner will enable the headers, and the library must make that easy *(owner)*:
   - **Docs.** The README gives COOP/COEP setup for Vite (dev and preview), webpack-dev-server and a generic server (nginx/Express headers).
   - **Examples.** `examples/react` and `examples/webgpu` turn the headers on in their Vite config, so the examples run threaded out of the box.
   - **Dev warning.** In development builds, when the Worker scene starts without cross-origin isolation, it warns once in the console. The warning says the scene is single-threaded and links to the README section.
6. **Benchmarks.** Results go into `docs/benchmarks.md`.
   - **Scenarios:** the 9a wide drag at 1M and 10M (median time per pose and time from the last move to colours applied), plus the contact drag to confirm no regression.
   - **Native:** sequential versus warm start.
   - **Worker harness:** single-threaded versus threaded. Threaded numbers are measured in a browser if the Node harness cannot run the threaded artifact (left to the builder).

### Non-goals
- **SAB render views** (9a §9.1, deferred): sharing the Worker's wasm memory with the main thread for zero-copy render buffers. *(My call: dropped.)* The measured cost of applying transferred ranges is 0.01–0.06 ms per frame at 1M–10M (9a §10.3). Sharing would also need a frame handshake or double buffering to avoid torn frames. The shared-memory build in part B is only for the Worker's own threads.
- **Threads on the main thread.** The synchronous `Scene` stays on the single-threaded artifact. Rayon blocks with `Atomics.wait`, which is not allowed on a browser main thread *(my call)*.
- **Progressive or partial colouring** (see Intent).
- **Approximate colouring:** lower resolution while dragging, or skipping vertices whose colour "would barely change". Anything inexact is out.
- **Hosting workarounds** such as a service worker that injects COOP/COEP (coi-serviceworker). The docs mention that this exists, but the package does not ship it *(my call)*.
- **Parallel normals and sort helpers** from the stage 1 leftovers. They are unrelated to this lag. Follow-up.

### Anti-goals
- **Mixed-pose colours.** Showing colours from more than one pose at once.
- **Unbounded or growing lag.** A drag must never fall further and further behind. Abandoning stale poses keeps the lag at about one recompute.
- **Results depending on build or warm-start state.** The threaded and single-threaded builds, and warm and cold starts, must give bit-identical values.
- **Breaking non-isolated pages.** Pages without the headers keep working exactly as today, only without threads.
- **A slower contact case.** The fast contact case must not get slower.

## 3. Constraints

- **Toolchains.** Stable Rust (MSRV 1.88) for everything except the threaded artifact. That artifact needs nightly plus `-Z build-std` with `+atomics,+bulk-memory,+mutable-globals`.
  - Pin the nightly version in a file or CI env, so builds are reproducible.
  - CI builds both artifacts.
  - If `wasm-bindgen-rayon` needs a different `wasm-bindgen` than the pinned `=0.2.129`, align both, and say so in the plan.
- **Package.** Same package `fast-mesh-wasm`.
  - The threaded artifact and its glue sit next to the current ones in `dist/wasm/`.
  - Only the Worker entry chooses between them. Core entries and the main-thread `Scene` never load the threaded artifact.
  - publint and attw stay green.
  - The npm size grows by roughly one more wasm file, which is acceptable.
- **Thread pool.** The Worker scene starts the rayon pool once, at start-up, through `initThreadPool`. The default size is `navigator.hardwareConcurrency`, capped and overridable by a `threads` option on `createWorkerScene`; `threads: 1` forces the single-threaded artifact.
  - The pool's workers are nested inside the scene Worker. Check that every target browser supports nested module workers, and record the outcome.
- **Exactness.**
  - Warm start must not change any value bit. A test compares warm and cold over randomised drags.
  - Threaded and single-threaded must agree bit for bit. A test compares them in the Node harness or a browser.
- **Rust API.** Interruptible recompute is a new stepping API in `fast-mesh-scene`. Its shape is left to the builder; something like `begin_update` / `step(budget)` / `finish` / `abandon`.
  - The existing `update_fields` stays and keeps its behaviour.
  - It runs on stable and in both artifacts.
- **Worker protocol.** Abandoning work reuses 9a's last-wins keys and cooperative cancellation (9a §4.3, ruling 6). The step budget replaces the request-level granularity for field recomputes only.
- **9a statuses.** While a drag is in progress, `display:<id>` statuses stay correct: a mesh being recomputed or abandoned reads `"computing"` until its newest pose completes.

## 4. What is decided

1. Exact colours for one pose at a time. Delay is acceptable; partial updates are not *(owner, as interpreted above)*.
2. A before B, one spec, two plans and PRs *(owner)*.
3. COOP/COEP is expected. The library guides users through docs, example configs and a dev warning *(owner)*.
4. SAB render views are dropped. Shared memory is used only for the Worker's threads *(my call, with measured costs)*.
5. Threads run only inside the Worker scene *(my call)*.
6. Warm start keeps values bit-identical *(my call)*.

## 5. Left to the builder

- **Warm-start storage.** One `u32` per vertex is enough: the triangle id, or none. When a mesh is edited, the cache has to be invalidated or remapped through the edit (for example, drop entries for removed triangles).
- **Search seeding.** Two options: a closest-point query that takes an initial upper bound, or a query that starts from a known triangle. Keep the public `closest_point` unchanged; an internal or extra variant is fine.
- **Step budget.** Choose the budget unit (time or vertex count) and its default. The Worker's 4 ms yield budget from 9a is a reasonable starting point.
- **Nightly pin.** Choose the nightly version and the exact `wasm-bindgen-rayon` version.
- **Pool size.** Choose the default pool size cap.
- **Threaded tests.** Decide whether the Node `worker_threads` harness can run the threaded artifact. Node supports `SharedArrayBuffer` and nested workers. If it cannot, run the threaded tests and benches in a headless browser.
- **Plan split.** Split each plan into tasks.

## 6. Decisions after the builder check (all *my call*; they override earlier sections)

1. **Other requests during a paused recompute.**
   - At each step boundary the Worker looks at the queue.
   - It **abandons** the recompute when a queued request makes it stale:
     - a newer drag move or `setPlacement` of a mesh the recompute depends on (the coloured mesh or its target);
     - an edit, repair or compact of either mesh;
     - `undo` or `redo`;
     - removal of either mesh or of the field.
   - Every other request (toggles, queries, other meshes) **waits** until the recompute completes. 9a's arrival order and "one result per batch, in order" are kept.
   - The abandoned work is redone at the newest pose after the stale-making request has run.
   - Waiting is bounded by one recompute, which part B shortens by the core count.
2. **One pose is published atomically.** All fields that a pose makes due are stepped as one unit: the step state belongs to the batch of due fields, not to one field. Their values and colours are applied together when the last one finishes. Abandoning drops the whole unit, so no change records appear for any of its fields.
3. **Warm start is the leading idea, not a fixed requirement.**
   - Part A starts by profiling the 1M wide drag (sequential, native): where do the candidate search, closest-point queries and sign tests spend their time?
   - Any **exact** optimisation the profile points to is in scope without asking. Examples: warm start; skipping the near-triangle search when the candidate set already covers every vertex in range; a faster sign path.
   - Approximations stay out (§2 non-goals).
   - **Target for A** (single-threaded, native, 1M wide drag): at least 3× faster per pose than today's ~0.4 s.
   - If A cannot reach that, record the profile and the result in `docs/benchmarks.md` and continue to B, which multiplies the gain.
4. **#48 scope.**
   - This work covers the drag path only: the per-pose recompute of due fields.
   - Steppable field builds, sign-data builds, edits and `addMesh` stay in #48, which stays open. The PR says "part of #48", not "closes".
5. **Browsers and fallback.**
   - Targets: current Chrome/Edge, Firefox and Safari (their last two major versions).
   - If threaded start-up fails on an isolated page, the Worker falls back to the single-threaded artifact and warns once (dev builds). Failures include no nested workers, `initThreadPool` rejecting, and shared memory failing to allocate. The fallback is required and tested.
   - The test matrix is headless Chromium in CI. Firefox and WebKit are checked once by hand or with Playwright if it is available, and the results are recorded.
6. **Bundlers.**
   - Goal: no configuration beyond the COOP/COEP headers in Vite and webpack 5, tested against an `npm pack` install, not the workspace link.
   - If a bundler cannot load the nested worker helpers without configuration, document the minimal setting (for example Vite `optimizeDeps.exclude`) in the README next to the header guidance, and test that setting.

## 7. Testing

- **Warm start.** A property test compares warm and cold results. It runs randomised drags and edits on both meshes and checks bit equality of values and colours, for both sign modes and for finite and infinite `maxDistance`.
- **Stepping.** Stepping to completion must equal a one-shot `update_fields`. Abandoning must leave the previous values and colours untouched, and leave no change records.
- **Worker.** A drag storm with a wide range must end with exactly one completed recompute per kept pose. Abandoned poses must publish nothing. Statuses must follow the 9a/9b rules.
- **Threads.** Threaded and single-threaded must match bit for bit on the same op sequence.
- **Fallback.** On a non-isolated page, the scene runs single-threaded and warns once (dev builds only).
- **Examples.** Both examples build in CI with the headers configured. Their Vite servers send the COOP/COEP headers; check this with a fetch in CI or a test.

## 8. Part A implementation notes

Part A is on `feat/fast-mesh-drag-speed`, from `f64cc25` on (part of #48). The numbers are in `docs/benchmarks.md`, "Wide-range drag"; the profile is `docs/superpowers/notes/2026-10-21-wide-drag-profile.md`.

### Result

- **The 3× target (§6.3) is met, with about 29×.** Native, single-threaded, 1M wide drag (`maxDistance` 8, `scene wide drag 1M`, poses 0, 16, 32 and 48), at `4fbb41d`: 13.89 s to 484 ms per pose for the field on A (28.7×), and 27.77 s to 960 ms for fields on A and B (28.9×). Re-run at `16cf700`: 483 and 997 ms. Contact drag (1M, `maxDistance` 0.08): 1.74 to 1.48 ms per pose.
- **What did it.** The profile put 96.6% of a pose in the near-triangle search, which listed every triangle pair within the margin. The search now gives up past one near pair per vertex slot of A (`triangles_near_limited`), and the update then recomputes every vertex of A. This is exact: a vertex's value depends only on its position, the target and `maxDistance`, and the plan compares each value bit for bit.
- **Warm start was tried and dropped** (`67503c0`, reverted in `acfef5d`). It was within noise (453 against 457 ms per pose). The cached triangle was the answer for only about a quarter of the points, and the query's cost is the descent from the root, which a tighter bound does not shorten. Its exactness also rested on a rounding bound that is not proven on the sliver path of `closest_on_triangle`. §6.3 made it optional, so §4 decision 6 and the warm-start items of §5 and §7 do not apply to part A.
- **Worker (wasm, one thread).**
  - The 9a wide drag went from 402 ms to 72 ms per move.
  - Final run (`6acbdf3`, lock step, median): a pose takes 71 ms (9a terrain) and 635 ms (native terrain) at 1M, and 631 ms and 6.68 s at 10M.
  - During a continuous 60 Hz drag, with the starvation guard (200 ms), the colours update every 252 ms (9a terrain) and 851 ms (native terrain) at 1M, median, and every 714 ms and 7.61 s at 10M, where a recompute outlasts the guard and almost none is dropped. When the drag ends right after its last move, the last pose's colours arrive 133 ms and 635 ms after it at 1M, and 1.06 s and 9.89 s at 10M: about one recompute, but about 1.5 at 10M on the native terrain (the extra time is not profiled). A pointer that pauses without ending may wait up to two recomputes (818 ms and 12.4 s on the native terrain, 1M and 10M). Before the guard (Task 5), no pose finished during the drag, and the last arrived 81 and 683 ms after it.
  - The longest Worker block is opening a step unit: about 60 ms at 1M and 0.5 to 0.65 s at 10M (up to 1.0 s seen during the continuous 10M drag).
- **Step API.**
  - Rust (additive): `Scene::set_defer_fields`, `field_step(FieldBudget)`, `abandon_field_step`, `field_step_pending` and `field_step_meshes`.
  - Wasm bindings: `set_defer_fields`, `field_step`, `abandon_field_step` and `field_step_meshes`.
  - TS: an `@internal` `fieldSteps(scene)` in `scene.ts`, for the Worker engine only. The public TS API is unchanged.
  - Step budget: 4,096 vertices per step (`STEP_POINTS`), with 9a's 4 ms yield budget.

### Rulings

- The 3× target is measured on the native `scene wide drag 1M` bench against Task 1's baseline (14.95 s per pose, field on A), not against the 9a JS figure, which uses a different terrain. Both numbers are reported. The committed 4-pose bench later re-recorded the baseline as 13.89 s.
- The committed wide bench must finish in reasonable time. It runs 4 of the 64 poses, so a criterion run takes minutes, not hours.
- Warm start is dropped: `67503c0` is reverted with its tests and docs. §6.3 makes it optional, it gives no measurable gain, and it carries an unproven exactness risk; the closest-point cost is the root descent. A later per-thread profile may want it back.
- The ~38 ms unit-open cost stays unsplit in part A. The near-search abort is already bounded, and B's threads shorten it.
- The Worker engine does not call `take_changes` or queries while a unit it keeps is open. Changes are taken only after a unit completes or is abandoned, which abandon-on-mutate enforces.
- A compute or finish error drops only that field from the unit, mirroring `update_due`'s per-field isolation. The other fields still apply.
- Open-time BVH refits and sign rebuilds that survive an abandon are the documented exception to "no trace". They equal what any query would do, and results never depend on them, but `field_status().sign` may reflect a newer build than the values.
- A drag `cancel` of a unit mesh is stale-making, because it restores a different pose, so it abandons. `clearHistory` and colouring changes wait, because they do not change field inputs.
- The step API stays internal to the Worker engine and is not public on the TS `Scene` (constraint "API stability"). The bindings follow the WasmScene snake_case naming.
- Field errors at unit completion stay swallowed, as in 9a's eager behaviour. The failing fields stay stale, and the field shows `ready` rather than `error`.
- Before opening a unit, the engine peeks at the queue. It does not open one while a stale-maker for the last abandoned unit's meshes is queued.
- After a completed unit, an already-queued stale-maker still costs one unit open (about 38 ms at 1M wide). This is accepted: it is bounded and does not grow a backlog.
- **Starvation guard** (the owner chose "hybrid"). Once more than a guard (default 200 ms, `EngineOptions.staleGuardMs`) has passed since the last completed (published) unit, a queued drag move no longer abandons the unit: it completes and is published, and the newest pose follows. Edits, repairs, compactions, removals, `setPlacement`, undo, redo and drag cancels still always abandon. Colours stay exact for one pose; during a sustained drag they update about every guard plus one recompute.
- **A drag `end` lifts the starvation guard.** With a unit mesh's `end` queued, drag moves of the unit's meshes abandon again, so a unit kept for an older pose is dropped and the final pose follows within one recompute after release. An `end` with no newer move queued leaves the unit alone, because it is already at the final pose (Review Focus 5). A pointer that pauses without `end` may still wait up to two recomputes; that is accepted.
- **The engine's yield admits incoming messages in Node `worker_threads` and in browsers.** The engine yields with `setImmediate` where it exists (Node), else with a `MessageChannel` tick (browsers), else `setTimeout(0)`; the bench harness no longer patches it. Checked with a Node test on real `worker_threads` and in headless Chrome.

### Left for part B (deferred minors that matter there)

- **The unit open is one call that cannot be split.** It takes about 40 ms native and 60 ms in wasm at 1M wide, and 0.55 to 0.6 s in wasm at 10M (up to 1.06 s seen). It is the Worker's longest block during a drag. Threads in the near search and the preparation shorten it.
- **There is no cross-field parallelism within a step.** The fields of a unit are stepped one after another. With threads, a step should split across fields as well as across points.
- **Failing fields stay due** and are retried, at the cost of one unit open, at every later batch end.
- **One unit open per completed unit** is spent when a stale-maker is already queued, because the pre-open peek knows only the last abandoned unit's meshes.
- **Mid-range fields near one pair per slot** can switch between the near search and the full recompute from pose to pose. That is about a 9% cost jump, exact either way.
- **During a sustained drag colours update about every guard plus one recompute** (the starvation guard). Part B's speed-up shortens the recompute, and so the interval.
- **The yield in other browsers.** The `MessageChannel` tick was checked in headless Chrome only; Firefox and Safari are not checked yet. Part B's threaded Worker should re-check the yield wherever it runs.
- `fields.rs` is about 3,000 lines and should be split.

## 9. Part B implementation notes

Part B is on `feat/fast-mesh-threads`, from `6fe0685` on (part of #48). The numbers are in `docs/benchmarks.md`, "Threaded Worker"; the user-facing guide is the README's "Multithreading (COOP/COEP)".

### Result

- **What shipped.** A second, threaded build of the same Rust engine (`dist/wasm-threads/`), loaded only by the Worker entry on a cross-origin isolated page; `createWorkerScene({ threads })` and `WorkerScene.threads`; a fallback to the single-threaded build whenever the threaded one cannot start. Results are bit-identical to the single-threaded engine: every change record, mirror buffer, placement, field value and section result of a 54-frame op sequence (two 360k-triangle terrains, wide colouring both ways, drags, an edit, a filled section, two undos) compare equal byte for byte, with the same digest in Chrome, Firefox and WebKit.
- **Speed (headless Chrome 155, i5-13600K, 8 threads against `threads: 1`, wide drag, `maxDistance` 8, median).**
  - Longest Worker block per pose (the unit open): 47.3 → 23.5 ms (1M, 9a terrain), 50.9 → 21.0 ms (1M, native terrain), 671 → 380 ms (10M, 9a), 825 → 397 ms (10M, native). 1.8 to 2.4×.
  - Lock step, move to colours: 60.5 → 37.5 ms, 623 → 140 ms, 907 → 637 ms, 10.26 → 2.97 s. 1.4 to 4.5×.
  - Update interval during a sustained 60 Hz drag: 252 → 222 ms (the 200 ms guard dominates), 792 → 428 ms, 915 → 637 ms, 10.4 → 2.96 s.
  - Release to the last pose's colours: 69.0 → 34.7 ms, 611 → 298 ms, 912 → 615 ms, 10.05 → 2.71 s.
- **Against §6.1's "shortens by the core count":** the gain is not 8×. The recompute's per-vertex part scales well (native terrain, 4.5× at 1M), but the unit open (the field's preparation and its near search) scales only about 2×, and on the 9a terrain it is most of a recompute. Splitting the open further is left (below).
- **Single-threaded browser against Node.** At 1M the browser matches part A's Node `worker_threads` numbers; at 10M it is slower (907 ms against 631 ms and 10.3 s against 6.68 s per lock-step pose, not profiled). Threaded numbers are compared only with single-threaded ones from the same browser run.

### Toolchain, sizes and memory

- **Nightly pin:** `nightly-2026-10-15` (rustc 1.101.0-nightly), in one place: `NIGHTLY` in `scripts/build-wasm.mjs` (`--print-nightly` for CI and the release workflow, which share an install action; the Rust cache is keyed on it). Built with `-Z build-std=panic_abort,std`, `+atomics,+bulk-memory,+mutable-globals`, and the link args `--shared-memory --max-memory=4294967296 --import-memory` plus the TLS exports (this nightly does not add a shared memory on its own).
- **wasm-bindgen-rayon 1.3.0** works with the `=0.2.129` wasm-bindgen pin: no bump.
- **Sizes (release):** single-threaded `.wasm` 772,026 bytes (307 KB gzip -9), threaded 975,880 bytes (368 KB); glue 41.5 KB and 45.8 KB; the rayon Worker helper 4.5 KB. Bundlers emit the threaded chunk, `.wasm` and helper into the build output; pages that run single-threaded never fetch them.
- **Memory:** the threaded build imports a shared memory with 18 pages initial (1.125 MiB) and 65,536 pages maximum (4 GiB). Measured: about 18 MB once the pool has started (thread stacks); it grew to about 560 MB in the browser memory check, and to 2.4 GB in the 10M wide-drag scene (2.1 GB single-threaded).

### Rulings

- Local `pnpm test`/build without the pinned nightly skip the threaded artifact with a loud warning (its tests skip). With `CI` set, and in release/prepublish, a missing nightly or threaded artifact is a hard failure. Cost if wrong: a contributor's local run does not exercise the threaded build.
- A 10 s `initThreadPool` timeout falls back to single-threaded. It prevents a hung start-up: a nested Worker that fails to load never reports ready, and fires no `error` event on the outer Worker in Chrome, Firefox or WebKit. Cost if wrong: a very slow device runs single-threaded.
- When threaded start fails after the glue was imported (instantiation succeeded; the pool failed or timed out), the Worker reports it, hands back the batches it received, and the client terminates that Worker and starts a fresh one with `threads: 1`, resending the held batches. Such a scene stays at `threads: 1` for later restarts (no repeated 10 s waits). Cost if wrong: one extra Worker start on the failure path.
- An explicit `threads` option is clamped to [1, 64]; a non-integer is rejected (`INVALID_ARGUMENT`). Cost if wrong: none.
- The rayon helper's `import('../../..')` is rewritten to `../../../fast_mesh_wasm.js` at build time, guarded so that an upstream change fails the build. It is needed for pages without a bundler (a browser cannot import a directory). Cost if wrong: a wasm-bindgen-rayon upgrade needs the guard updated.
- The examples and docs set COOP/COEP with a small Vite middleware plugin, not `server.headers`/`preview.headers`: Vite leaves those off 304 responses, and WebKit then refuses the revalidated Worker script. `scripts/browser-check.mjs` (check `headers`) guards the 304 case. Cost if wrong: none.

### Also decided in the tasks

- The development warning (dev builds or `debug`) is logged at most once per scene, also across restarts.
- `debugDump` is not used for the browser comparison (it needs a non-public client export); the check compares the public mirrors, `fetchFieldValues`, section results and change records instead.
- No bundler setting is needed: Vite build and dev, webpack 5 (all from an `npm pack` install) and a page without a bundler start with 8 threads (`scripts/bundler-check.mjs`).
- Part A's open point, the yield in other browsers, is closed: during a continuous 1M drag, messages arrive mid-unit and units are abandoned in Chrome, Firefox and WebKit, single-threaded and threaded.

### Known issues and left open

- **WebKit.** Playwright's WebKit 27.2 on Windows crashed its web process in 2 of about 20 dev-server runs of the restart check (a threaded Worker crashed on purpose, then `restart()`); 0 of about 11 preview runs. The cause is unknown. Real Safari on macOS is unverified.
- **The unit open scales only about 2×** with 8 threads and is the longest Worker block (about 21 to 24 ms at 1M, 0.4 s at 10M). Further splitting of the field preparation and near search is not done.
- **No cross-field parallelism within a step** (part A's deferred item): the fields of a unit still step one after another.
- **A crash on a pool thread is not detected.** With the threaded engine, a Rust panic or abort on a rayon pool thread (a nested Worker) ends only that thread; the engine Worker stays blocked waiting for it, so the scene stops answering and is never poisoned (the single-threaded engine reports `INSTANCE_POISONED`). `restart()` and `terminate()` still work; `threads: 1` gives guaranteed poisoning. The README says so; detection is a follow-up issue.
- **The old Worker during the pool-failure handback.** An `error` event from it now poisons the scene (final fix wave), so held batches never wait on a crashed Worker. One that stops without an `error` event would still leave them waiting (the same class as other silent Worker hangs before a reply).
- The browser check's nested-failure case costs a fixed 10 s (the pool timeout), and the bundler check needs npm registry access in CI.
- Part A's other deferred items (failing fields stay due, one unit open per completed unit with a stale-maker queued, the mid-range switch, splitting `fields.rs`) are unchanged.
