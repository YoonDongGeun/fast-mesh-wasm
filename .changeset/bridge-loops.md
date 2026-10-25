---
"fast-mesh-wasm": minor
---

Bridge-loops repair op. `scene.repair(mesh, { op: "bridgeLoops", loops: [a, b], maxEdges })` stitches a minimum-area triangle tube between two boundary loops picked from `analyze(mesh).loops`; the loops must wind in opposite directions. Each call is one undo entry; bad selections report `INVALID_ARGUMENT` and unknown ids report `STALE_LOOPS`.

Mesh join. `scene.join(target, source, { loopA, loopB, maxEdges })` copies the source mesh's geometry into the target (keeping its world shape), bridges the two loops and removes the source. The copy and the bridge undo as one step; the removal is permanent. Also on `WorkerScene` (async, the source stays usable until the join lands).
