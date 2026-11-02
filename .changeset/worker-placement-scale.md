---
"fast-mesh-wasm": patch
---

`WorkerScene` placements now carry scale end to end: `setPlacement` and drag moves accept `scale`, `placement()` always returns it, and restarts replay it, so the worker mirror matches `Scene` (whose `placement()` already included it).
