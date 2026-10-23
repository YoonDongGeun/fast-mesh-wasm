---
"fast-mesh-wasm": minor
---

The Worker scene no longer blocks its queue for the whole of a long engine call (#48). The engine computes the same results, bit for bit. What changes is latency, and when a large new mesh's buffers reach the main thread.

- **Field builds and field updates run in steps.** This covers the targets' sign data, the near search and the values.
  - The Worker yields every few milliseconds, so batches sent meanwhile are received and coalesced.
  - A field add for a display that is already superseded is dropped mid-build, as it was before the build started.
  - Longest Worker block at 1M triangles: 118 ms to 6 ms for a field build, and 97 ms to 5 ms for an edit with a field measured to the edited mesh. At 10M: 1.4 to 3.4 s to 14 to 18 ms.
- **A large mesh is added in the background (a change of reply order).** From 65,536 triangles, its BVH is built in steps while later requests that do not need it run. A toggle sent right after a 1M `addMesh` is answered in about 8 ms instead of 1.18 s, and after a 10M add in 71 ms instead of 16 s.
  - These requests still wait for the new mesh: requests that name it (or a field not built yet that is measured on or to it), any add or removal of a mesh, and queries of every mesh (`raycast`, `closestPoint` without a mesh, `findContacts` without pairs). They then build it in one call. So the engine's state is the same as in arrival order.
  - Replies can now come in a different order. A reply to an unrelated request sent after the add can arrive before the mesh. The mesh's `mesh:<id>` status reads `computing` and its views are empty until it lands. Its buffers arrive with that status turning `ready`: with a later result, or on their own if the Worker is idle.
  - New `WorkerSceneOptions.backgroundAddTriangles` sets the threshold. `Infinity` turns background adds off, and every add then comes back with its own batch's result.
- **Field add errors.** A field add now fails, and the field is not kept, when computing its first values fails (out of memory). Before, the add succeeded, the field stayed registered but stale, and the error only showed later. The same applies to a field replayed by `restart()`. A field whose deferred build fails (its name is taken by then, say) now stays known: the error is reported each time it is needed, and removing it works.
