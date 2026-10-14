---
"fast-mesh-wasm": minor
---

Easier JavaScript API (breaking):

- `computeBounds` now returns a plain object `{ min, max, center, size }` with `[x, y, z]` arrays. The `Bounds` class, `free()` and `using` are gone.
- Inputs accept `Uint16Array` indices (three.js), `Float64Array` and plain `number[]`; `Float32Array`/`Uint32Array` are still used without copying.
- Calling a function before `await init()` in the browser throws a clear error instead of an internal one.
- Fixed: the Node.js entry no longer gets dropped by bundlers that honor `sideEffects` (webpack server builds, e.g. Next.js), which made every call crash.
- `InitInput` is exported, and the Node.js `init()` accepts the same argument as the browser one.
