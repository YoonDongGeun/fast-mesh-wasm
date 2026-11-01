---
"fast-mesh-wasm": patch
---

Starting a background add's BVH build no longer zeroes its per-triangle leaf table in one call: `BvhBuild::new` only reserves the table and the prim scan fills it with the same pass, so the store-to-BVH transition stays short even on a large mesh (it was 1.9 ms at 1M and 12.6 ms at 10M in one step) and the built tree is unchanged bit for bit. A toggle queued behind a field build or an edit's recompute still waits for it, by arrival order; letting unrelated requests run during an open field unit stays a spec-first design change.
