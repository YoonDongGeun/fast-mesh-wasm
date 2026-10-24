---
"fast-mesh-wasm": patch
---

A large mesh added to a Worker scene in the background no longer blocks the Worker while its inputs copy into wasm memory and its store builds: the upload runs in chunks with yields between them, then the store builds in steps, bit-identical to the one-call build, before the BVH build starts as before. In back-to-back runs the longest Worker block for a 1M-triangle add fell from 18.3–22.6 ms to 15.7–17.8 ms (now a BVH build step either way), and at 10M from 184–207 ms to 21.0–38.9 ms; a toggle sent right after the add is answered in about 8 ms at 1M. The winding-moments build lists its leaves incrementally too, instead of one traversal up front. Results are unchanged.
