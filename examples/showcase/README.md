# fast-mesh-wasm showcase

A small mesh tool built on the library: load meshes, move them, repair them, and see how two meshes
touch as a contact heatmap. Vite + React + React Three Fiber over a `SceneStore`
(`fast-mesh-wasm/store`), drawn with `<FastMesh>` (`fast-mesh-wasm/r3f`).

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
- **Repair.** For the selected mesh, `analyze` lists its holes (with their sizes), orientation
  conflicts, components (floating pieces), non-manifold edges and vertices, and degenerate triangles.
  The buttons run the library's repairs through `store.repair`, each one undo step: Weld (within a
  distance), Remove degenerate, Orient, Remove small components (below a triangle count) and Fill
  holes (up to a size, optionally faired to follow the curvature). Each shows its report while the
  analysis refreshes; a repair, its undo and its redo all re-analyse.
- **Collision.** Pick meshes A and B (the first two meshes by default): each is coloured by its
  signed distance to the other with `contactHeatmap`, penetration from dark red (*Penetration
  depth*) to red, white within ± *Contact tolerance*, and the gap from green to blue up to *Colour
  from (mm)*. *Colour from* 0 colours penetration only. Every vertex's distance is exact by default;
  *Fast mode (large meshes)* bounds the fields at the larger of depth and colour from, so a deeper
  penetration shows the mesh colour. Readouts, refreshed about four times a second while a mesh
  moves and again when it stops:
  - **Max penetration**: the deepest negative value of either colouring
    (`store.fetchColoringValues`); in fast mode, "≥" when it reaches the bound.
  - **Min gap**: `meshDistance`, 0 when the surfaces touch.
  - **Intersecting**: `intersects`.
  - **In contact**: each mesh's triangles within the tolerance of the other
    (`findContacts({ margin: tolerance })`).

  When a target is open or inconsistently oriented (`fieldStatus`), the panel says so: penetration
  then follows its surface normals; Orient or Fill holes give a closed result. Removing A or B stops
  the colouring and resets its picker.
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
pnpm --filter fast-mesh-wasm-example-showcase test   # parsers, demo meshes, panel arithmetic (vitest)
node scripts/showcase-check.mjs --threads 4          # a scripted session in headless Chrome (on the build; --build builds first)
node scripts/showcase-check.mjs --big                # plus a 1M-triangle STL upload (main thread < 50 ms)
```

The scripted session selects and drags A with the gizmo and undoes it; checks the heatmap of A and
B and its readouts while A is dragged away; picks the open mesh C as B and expects the open-mesh
warning; fills C's holes while it is coloured and undoes it; removes B while it is coloured; loads
STL, OBJ and PLY files (and a broken one); then repeats a short session with `?local`. It fails on
any console error.
