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
