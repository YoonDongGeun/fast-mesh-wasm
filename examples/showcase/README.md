# fast-mesh-wasm showcase

A small mesh tool built on the library: load meshes, move them, repair them, and see how two meshes
touch as a contact heatmap. Vite + React + React Three Fiber over a `SceneStore`
(`fast-mesh-wasm/store`), drawn with `<FastMesh>` (`fast-mesh-wasm/r3f`).

**[Live demo](https://yoondonggeun.github.io/fast-mesh-wasm/)**, published from `main` by
`.github/workflows/pages.yml`. GitHub Pages cannot send the COOP/COEP headers that threads need, so
there a small service worker (`public/coi-sw.js`, registered in `index.html`) adds them and the page
reloads once on the first visit. Where service workers are unavailable the scene runs
single-threaded. The dev and preview servers send the headers themselves, so the worker is not
registered there. `SHOWCASE_BASE=/fast-mesh-wasm/` sets the build's public path (default `/`);
`node scripts/showcase-pages-check.mjs` checks that build in Chrome (see Checks).

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
- **Mesh list.** Name, picture, triangle count, visibility, opacity; select, focus the camera, remove.
  A picture is drawn from the mesh's render views by a small software z-buffer
  (`src/panels/mesh-preview.ts`), in slices of a few milliseconds at idle, and redrawn once edits
  or turns have stopped for 400 ms and no pointer is pressed on the viewport. A mesh with nothing left
  to draw shows an empty state. The built-in models' buttons show committed pictures
  (`public/previews`, about 15 KB in all, regenerated with `node scripts/showcase-previews.mjs --build`
  when a model changes), so showing the list loads no model.
- **Move.** Click a mesh to select it; the gizmo (three's `TransformControls`, through
  `attachTransformControls`) moves (<kbd>T</kbd>) or rotates (<kbd>R</kbd>) it, one undo step per drag.
  Undo and Redo, or <kbd>Ctrl</kbd>+<kbd>Z</kbd> and <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>Z</kbd>.
- **Repair.** For the selected mesh, `analyze` lists its holes (with their sizes), orientation
  conflicts, components (floating pieces), non-manifold edges and vertices, and degenerate triangles.
  The buttons run the library's repairs through `store.repair`, each one undo step: Weld (within a
  distance), Remove degenerate, Orient, Remove small components (below a triangle count) and Fill
  holes (up to 64 edges by default; an empty limit fills any size, an open mesh's outer boundary
  included; optionally faired to follow the curvature). Each shows its report while the analysis
  refreshes; a repair, its undo and its redo all re-analyse, and an undo or redo clears the report.
  Auto repair runs the library's whole pipeline with those settings as one undo step
  (`store.repair(mesh, { kind: "auto", ... })`).
- **Collision.** Pick meshes A and B (the first two meshes by default): each is coloured by its
  signed distance to the other with `contactHeatmap`, penetration from dark red (*Penetration
  depth*) to red, white within ± *Contact tolerance*, and the gap from green to blue up to *Colour
  from*. *Colour from* 0 colours penetration only. Every vertex's distance is exact once the pointer
  stops. While a gizmo drag or a sculpt stroke moves a large mesh (`scene.setFieldInteraction`, about
  8,192 vertices a field per update), the vertices closest to the contact stay exact, a sweep brings the
  others along, and they are all recomputed about 200 ms after the pointer stops or at once on release:
  a few seconds at 1M triangles, in steps, while the page stays live. *Fast mode (large meshes)* bounds the fields at twice the larger of depth and colour from (so the
  coloured area's outer edge still softens with *Boundary softness*), and a deeper
  penetration shows the mesh colour. The colourings are drawn with fragment shading
  (`<FastMesh shading="fragment">`): each pixel is coloured from its interpolated distance, so the
  edge of the coloured area and the contact band are crisp iso-lines rather than blurred across a
  triangle. *Boundary softness* widens them into a blend that many distance units wide (0, the
  default, is a hard edge). Readouts, refreshed at most about four times a second as the
  meshes change, paused during a gizmo drag or a sculpt stroke (the colours stay live; reading field
  values would force stale poses up to date) and refreshed when it ends:
  - **Max penetration**: the deepest negative value of either colouring
    (`store.fetchColoringValues`). In fast mode it is measured within the bound (the panel says
    so), and reads "≥ bound" while the meshes intersect and the deepest value is within 10% of the
    bound, or no vertex inside the other mesh is within it.
  - **Min gap**: `meshDistance`, 0 when the surfaces touch.
  - **Intersecting**: `intersects`.
  - **In contact**: each mesh's triangles within the tolerance of the other
    (`findContacts({ margin: tolerance })`).

  When A or B is open (boundary loops, open chains or open components) or inconsistently oriented
  (orientation conflicts), from `analyze` of it when the pair changes or its triangles change, the
  panel says so: penetration measured to it then follows its surface normals; Orient or Fill holes
  give a closed result. Removing A or B stops the colouring and resets its picker.
- **Collision: Group mode.** Tick several meshes (the first three by default) into a contact group
  (`store.setContactGroup`), measured to each other in *All pairs* or in the pairs ticked in a small
  matrix. Each member gets one field measured to all of its partners and one colour scale, the same
  heatmap and sliders as the pair: each vertex shows its distance to the nearest partner surface (or
  the deepest penetration). Moving a member recomputes only its own field and its partners'. A
  readout per pair, throttled and paused during drags like the pair's: *Min gap* (`meshDistance`),
  *Hit* (`intersects`) and *Max depth*, the deepest negative value of either member at the vertices
  the other is nearest to (`store.fetchColoringValues` and the field's `nearest` buffer, whose
  indices follow `store.coloringTargets`; a vertex inside two partners counts for the deeper one
  only), read "≥ bound" in fast mode under the same rule as the pair. Hovering a member shows which
  partner the vertex under the pointer is nearest to and how far: the vertex is found with the
  engine's BVH `raycast` (one at a time, at most every 50 ms), and the `nearest` buffer is read
  through `fetchFieldNearest`, or `fieldNearest` with `?local`, when the member is first hovered
  and after it changes, never during a drag. Open or inconsistently oriented members get the same
  warning as the pair. Removing a member keeps the others coloured with their remaining partners.
- **Status bar.** The scene's status, the engine's thread count (`WorkerScene.threads`), whether the
  page is cross-origin isolated, and on an error its message with Restart.
- **Units.** The engine is unitless; this app reads its world unit as the millimetre, the unit DICOM
  speaks, and every stored value stays in it. Two separate things convert:
  - *Display unit* (mm or in, the button in the status bar): every length, area and volume readout
    and length input follows, at the factor, its square or its cube. Text only: no mesh, slider
    value or colouring moves. Mesh and section exports (STL, DXF, ...) stay in mm; the collision
    CSV follows, and its header names the unit.
  - *Import scale* (*Assume files in*, or any factor, above the mesh list): a file's numbers are
    multiplied once as it loads, so its mesh is in mm from then on. It applies to the next files
    opened or dropped, shows on the mesh's row, and does not persist. DICOM series and samples
    never take it: their spacing is already in mm.

  The display unit is per project with a global default: a project file stores its own (`units`),
  a file without one takes the default, and picking a unit by hand also sets the default. Opening a
  project never changes the default. The file stores each mesh's positions in mm, and its
  `importScale` as provenance only, so reopening cannot scale a mesh twice.
- **Settings.** Display prefs (theme, panel visibility and widths, wireframe, boundary softness,
  repair overlays, collision mode, the last-used heatmap sliders, the default display unit) persist in localStorage across
  reloads under versioned `fm.showcase.*` keys; Reset settings in the status bar restores the
  defaults. Meshes and projects stay file-based.

URL options:

| Option | Effect |
| --- | --- |
| `?local` | The synchronous `Scene` on the main thread instead of the Worker scene: queries, repairs and `analyze` (the repair panel's, and the collision warnings') then run on the main thread, so they block the page on a large mesh. The exact depth (the default) recomputes every coloured vertex on the main thread on every drag frame, so dragging large meshes stutters: use *Fast mode* there |
| `?big` | Demo meshes A and B at about a million triangles each; `?big=N` at about N |
| `?threads=N` | Ask for N engine threads |

The dev and preview servers send the cross-origin isolation headers (COOP/COEP, a middleware in
`vite.config.ts`), so the Worker scene runs its multithreaded engine.

## Checks

```sh
pnpm --filter fast-mesh-wasm-example-showcase test   # parsers, demo meshes, panel arithmetic (vitest)
node scripts/showcase-check.mjs --threads 4          # a scripted session in headless Chrome (on the build; --build builds first)
node scripts/showcase-check.mjs --big                # plus a 1M-triangle STL upload (main thread < 50 ms)
node scripts/showcase-pages-check.mjs                # the Pages build: no headers, the service worker, the fallback
node scripts/showcase-touch-check.mjs                # phone metrics, real multi-touch input (--shots <dir> saves screenshots)
node scripts/showcase-heatmap-bench.mjs              # frame and colour update times of a drag and a stroke at 100k and 1M
```

The scripted session selects and drags A with the gizmo and undoes it; checks the heatmap of A and
B and its readouts while A is dragged away; picks the open mesh C as B and expects the open-mesh
warning; fills C's holes while it is coloured and undoes it; colours A, B and C as a contact group
(colours on all three, the per-pair readouts, the hover tip, the pair matrix) and removes C from it;
removes B while it is coloured; loads STL, OBJ and PLY files (and a broken one); drives the units
(inch cubes through the import scale, readouts in both units, a project round trip); then repeats a
short session with `?local` (the group's readouts included). It fails on any console error.

The mask session (`scripts/showcase-mask-check.mjs`, `--big` adds the cost at 1M triangles) compares
screenshots: the tint appears where painted, follows invert, hide and clear, reads on light and dark
meshes, and leaves a heatmap's pixels alone. A dab on the mask's edge moves free vertices only and
leaves masked positions byte-identical. The optional stroke-end remesh stands down where it could reach the mask; live splits run over it, and a split between masked vertices joins the mask.

The touch session runs with phone metrics and multi-touch input through `Input.dispatchTouchEvent`,
and asserts on the camera: one finger orbits, two pinch and pan, a second finger never starts a tool
(gizmo, slice and section handles), a tap selects and shows the hover tip in Group mode, panels stay
reachable in portrait, landscape and on a tablet, and a mouse still behaves as before. Sculpting and
mask paint are mouse tools for now: their panel is hidden on a coarse pointer.
