---
"fast-mesh-wasm": patch
---

A large mesh added to a Worker scene in the background no longer blocks the Worker when it lands: its normals are computed, the last stage of its BVH build runs, and its buffers are copied out, all in steps before it lands (the longest Worker block for a 1M-triangle add falls from about 20 to 11 ms, and at 10M from 150-200 to about 100 ms, the add's own batch). Stepped near searches now size their steps from the BVH's real subtree counts, so steps stay short on unbalanced trees too. Results are unchanged.
