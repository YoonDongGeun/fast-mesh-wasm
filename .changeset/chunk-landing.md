---
"fast-mesh-wasm": patch
---

A large mesh added to a Worker scene in the background no longer blocks the Worker when it lands: its normals are computed, the last stage of its BVH build runs, and its buffers are copied out, all in steps before it lands. In back-to-back runs the longest Worker block for a 1M-triangle add fell from 21.6–25.2 ms to 14.0–16.3 ms, and at 10M from 184–207 ms to 103–108 ms (now the add's own batch). Stepped near searches now size their steps from the BVH's real subtree counts, so steps stay short on unbalanced trees too. Results are unchanged.
