---
"fast-mesh-wasm": patch
---

The Worker scene no longer blocks its queue for the whole of a long engine call (#48). Results are unchanged, bit for bit. Only latency and the time a new mesh's buffers arrive change.

- **Field builds and field updates run in steps.** This covers the targets' sign data, the near search and the values. The Worker yields every few milliseconds. Batches sent meanwhile are received and coalesced, and a field add for a superseded display is dropped mid-build, as it was before the build started.
  - Longest Worker block at 1M triangles: 118 ms to 6 ms for a field build, and 97 ms to 5 ms for an edit with a field measured to the edited mesh. At 10M: 1.4 to 3.4 s to 14 to 18 ms.
- **A large mesh is added in the background.** This applies from 65,536 triangles. Its BVH is built in steps while later requests that do not depend on it run. A toggle sent right after a 1M `addMesh` is answered in about 8 ms instead of 1.18 s, and after a 10M one in 71 ms instead of 16 s.
  - Requests that name the new mesh, add or remove a mesh, or query every mesh (`raycast`, `closestPoint` without a mesh, `findContacts` without pairs) still wait for it. So the engine computes exactly what running the requests in order would.
  - The mesh's `mesh:<id>` status reads `computing` until it lands. Its buffers arrive with that status: with a later result, or on their own if the Worker is idle. A reply to an unrelated request sent after the add can now arrive first.
