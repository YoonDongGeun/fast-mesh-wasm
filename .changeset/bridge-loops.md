---
"fast-mesh-wasm": minor
---

Bridge-loops repair op. `scene.repair(mesh, { op: "bridgeLoops", loops: [a, b], maxEdges })` stitches a minimum-area triangle tube between two boundary loops picked from `analyze(mesh).loops`; the loops must wind in opposite directions. Each call is one undo entry; bad selections report `INVALID_ARGUMENT` and unknown ids report `STALE_LOOPS`.
