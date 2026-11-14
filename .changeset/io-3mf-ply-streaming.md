---
"fast-mesh-wasm": minor
---

Add pure-TypeScript streaming mesh codecs and format expansion (`zip.ts`, `threemf.ts`, `ply.ts`, `stl.ts`, `obj.ts`, `stream.ts`), adding 3MF import/export with `<basematerials>`/`<m:colorgroup>` vertex colors and Production Extension support, binary/ASCII PLY import with UV round-trips, binary/ASCII STL import with spatial-hash vertex welding, OBJ vertex-color import/export, and bounded `ReadableStream`/`Blob` loading with `AbortSignal` for `Scene` and `WorkerScene`.
