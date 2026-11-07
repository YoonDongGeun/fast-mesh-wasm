---
"fast-mesh-wasm": minor
---

Keep a large distance field live during a gesture: `setFieldInteraction(points)` on `Scene` and `WorkerScene` updates about `points` vertices closest to the contact per update and makes the field exact when it ends. Stepped field updates (the Worker) now refit pseudonormal sign data after a small position-only edit instead of rebuilding them, and a queued drag move no longer drops a recompute under way by default.
