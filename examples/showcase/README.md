# fast-mesh-wasm showcase

A small mesh tool built on the library: load meshes, move them, undo. Vite + React + React Three
Fiber over a `SceneStore` (`fast-mesh-wasm/store`), drawn with `<FastMesh>` (`fast-mesh-wasm/r3f`).

```sh
pnpm build                                          # the library (once)
pnpm --filter fast-mesh-wasm-example-showcase dev   # or: build, then preview
```

- **Load.** Open or drop STL (binary and ASCII), OBJ and PLY (ASCII and binary little-endian) files.
  They are read and parsed in a Worker (`src/load/parser.worker.ts`); the page only calls
  `store.addMesh` with the transferred arrays. Each file shows its progress. STL corners with
  bit-identical positions share a vertex.
- **Demo meshes.** A (a blobby sphere), B (a rounded box penetrating A) and C (a shell with two holes
  and a floating piece) are generated in the same Worker when the app opens.
- **Mesh list.** Name, triangle count, visibility, opacity; select, focus the camera, remove.
- **Move.** Click a mesh to select it; the gizmo (three's `TransformControls`, through
  `attachTransformControls`) moves (<kbd>T</kbd>) or rotates (<kbd>R</kbd>) it, one undo step per drag.
  Undo and Redo, or <kbd>Ctrl</kbd>+<kbd>Z</kbd> and <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>Z</kbd>.
- **Status bar.** The scene's status, the engine's thread count (`WorkerScene.threads`), whether the
  page is cross-origin isolated, and on an error its message with Restart.

URL options:

| Option | Effect |
| --- | --- |
| `?local` | The synchronous `Scene` on the main thread instead of the Worker scene |
| `?big` | Demo meshes A and B at about a million triangles each |
| `?threads=N` | Ask for N engine threads |

The dev and preview servers send the cross-origin isolation headers (COOP/COEP, a middleware in
`vite.config.ts`), so the Worker scene runs its multithreaded engine.

## Checks

```sh
pnpm --filter fast-mesh-wasm-example-showcase test   # parsers and demo meshes (vitest)
node scripts/showcase-check.mjs --threads 4          # a scripted session in headless Chrome (on the build; --build builds first)
node scripts/showcase-check.mjs --big                # plus a 1M-triangle STL upload (main thread < 50 ms)
```
