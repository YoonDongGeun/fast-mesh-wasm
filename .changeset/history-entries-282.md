---
"fast-mesh-wasm": minor
---

Entries of the application in the undo history (#282). `recordEntry({ undo, redo, drop, bytes, coalesce })` on `Scene`, `WorkerScene` and the store records state that is not the scene's (a selection, a list of curves, a title) in order with mesh edits: it joins the open step, merges with the entry before it under the same `coalesce` key, is cancelled with its step, counts `bytes` against the history budget, and `drop` is called once when the history lets go of it, so payloads do not leak. On a `WorkerScene` the functions run on the main thread as results arrive, in the history's order whatever is in flight.
