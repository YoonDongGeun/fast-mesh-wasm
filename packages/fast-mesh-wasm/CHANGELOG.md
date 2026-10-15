# fast-mesh-wasm

## 0.2.0

### Minor Changes

- [#9](https://github.com/YoonDongGeun/fast-mesh-wasm/pull/9) [`d7fed6d`](https://github.com/YoonDongGeun/fast-mesh-wasm/commit/d7fed6da89c1010cd3cc4bf9e14c02ac74ec4783) Thanks [@YoonDongGeun](https://github.com/YoonDongGeun)! - Easier JavaScript API (breaking):
  
  - `computeBounds` now returns a plain object `{ min, max, center, size }` with `[x, y, z]` arrays. The `Bounds` class, `free()` and `using` are gone.
  - Inputs accept `Uint16Array` indices (three.js), `Float64Array` and plain `number[]`; `Float32Array`/`Uint32Array` are still used without copying.
  - Calling a function before `await init()` in the browser throws a clear error instead of an internal one.
  - Fixed: the Node.js entry no longer gets dropped by bundlers that honor `sideEffects` (webpack server builds, e.g. Next.js), which made every call crash.
  - `InitInput` is exported, and the Node.js `init()` accepts the same argument as the browser one.

## 0.1.0

### Minor Changes

- [`bf94aa5`](https://github.com/YoonDongGeun/fast-mesh-wasm/commit/bf94aa5088886d69886f9e02b535e162cffe9065) Thanks [@YoonDongGeun](https://github.com/YoonDongGeun)! - Initial release: `computeVertexNormals` and `computeBounds`, with ESM entry points for browsers/bundlers and Node.js/Bun.
