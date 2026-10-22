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
