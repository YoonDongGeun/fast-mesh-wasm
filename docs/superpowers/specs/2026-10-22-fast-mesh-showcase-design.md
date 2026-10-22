# fast-mesh: showcase app and contact heatmap

- **Status:** draft for review
- **Builds on:**
  - the Worker scene (9a);
  - the store, React and R3F layers (9b);
  - the wide-range drag work (parts A and B: stepped recompute, threads).
- **Labels:** *(owner)* marks what the owner said. *(my call)* marks choices made without asking.

## 1. Intent

The owner wants a showcase app that works like a real tool built on the library *(owner)*:

- **Load meshes.** Upload mesh files (several).
- **Move meshes.** Move each mesh on its own.
- **Repair meshes.** Run repair on them.
- **Inspect collisions.** See the collision between two meshes as a heatmap.

The owner also wants depth-weighted collision colouring: the deeper a mesh penetrates, the darker it is coloured *(owner)*. The owner wants the collision tolerance to be adjustable too *(owner)*.

The BVH is exact, and its boxes are only an acceleration structure. So a tolerance has to come from the existing query parameters (`margin`, `maxDistance`) and from the colour mapping, not from resizing BVH boxes *(my call, explained to the owner)*.

## 2. Goals

1. **The `examples/showcase` app.** It is a new Vite + React + R3F app. The existing `examples/react` stays as the minimal example *(my call)*.
   - **Load.**
     - Drag and drop, or a file picker, for STL (binary and ASCII), OBJ and PLY (ASCII and binary little-endian). Several files can be loaded. *(my call: those three formats)*
     - Parsing runs off the main thread when that is cheap. Otherwise it parses on the main thread, then does `addMesh` with transfer.
     - Large files: show a progress or busy state, and the UI must never freeze. 9a measured `addMesh` at 1M: 9.3 ms main-thread block.
     - When the app opens, generated demo meshes are already loaded, so it is useful before any upload.
   - **Mesh list panel.**
     - Name, triangle count, a visibility toggle and an opacity slider for each mesh.
     - Select a mesh, remove it, and focus the camera on it.
   - **Move.**
     - Click a mesh to select it. A gizmo moves it, with translate and rotate modes, through `attachTransformControls` and `onBind`.
     - Undo and redo buttons with their enabled state, plus Ctrl+Z and Ctrl+Shift+Z.
   - **Repair panel** for the selected mesh.
     - Analyze shows the counts: boundary loops (holes) with their sizes, flipped or inconsistent orientation, components (floating pieces), non-manifold edges and vertices, and degenerate triangles.
     - Buttons run the library's repair ops: weld, remove degenerate, orient, remove small components (with a size threshold), and fill holes. Fill holes takes a max-size filter and a "fair (follow curvature)" toggle.
     - Show each op's report, re-analyze afterwards, and make every repair undoable.
     - Show "computing" while it runs.
   - **Collision panel.**
     - Pick two meshes, A and B. Both are coloured with the contact heatmap against each other.
     - Controls:
       - **Penetration depth range:** the depth that maps to the darkest red.
       - **Gap range:** the distance that maps to the far blue.
       - **Contact tolerance:** values within ± tolerance count as contact and are shown white.
     - Readouts (async queries):
       - the maximum penetration depth;
       - the minimum gap (`meshDistance`);
       - whether the meshes intersect (`intersects`);
       - a contacts count (`findContacts({ margin: tolerance })`).
     - The heatmap updates live while dragging. It uses the latest-pose-wins behaviour from part A.
   - **Status bar.** It shows:
     - the overall status;
     - the thread count (`WorkerScene.threads`), and whether the page is cross-origin isolated;
     - the error if there is one, with Restart.
   - **Engine.** A `WorkerScene` store, threaded with the COOP/COEP middleware plugin from part B. `?local` switches to the synchronous `Scene`.
2. **Library: a contact heatmap preset** *(my call: in the library, not only the example)*.
   - The function is `contactHeatmap(options): Omit<DistanceColoring, "distanceTo">`, exported from `fast-mesh-wasm/store`. It is an alias-free helper the app spreads into `setDisplay({ coloring: { distanceTo, ...contactHeatmap(opts) } })`.
   - Options: `{ depth: number, gap: number, tolerance?: number, colors?: { deep?, shallow?, contact?, near?, far? } }`.
   - It builds the stops and `range = [-depth, gap]`, with `maxDistance = max(depth, gap)`.
   - The colours run from deep red, through red, to white at 0 (or across [-tolerance, +tolerance]), then from green to blue. Out of range is clamped below and gives the base colour above.
   - Default colours *(my call)*: deep `#7f0000`, shallow `#ff6b6b`, contact `#ffffff`, near `#4caf50`, far `#1e3a8a`.
   - `tolerance` is drawn as a flat white band. Because of that, stops must stay strictly increasing; when the tolerance is 0, the band collapses to one stop.
   - Pure TS, unit tested, documented in the README.
3. **Queries the readouts need.**
   - The maximum penetration depth is the minimum value of the field on A and on B (`fetchFieldValues`, min over finite values).
   - If fetching the whole field is too costly per frame at 1M, the app throttles the readouts to about 4 Hz and to drag end. *(my call)*

## 3. Non-goals

- **Hosting.** Public hosting (GitHub Pages, and so on) is left for later; it is a local example *(my call; asked, not decided)*. It is built in CI and checked in headless Chrome.
- **No SDF grid.** SDF grid (approximate) distance fields are out of scope; all colouring stays exact.
- **Editing tools.** No sculpting, no boolean operations, no file export (export may be a later small addition).
- **No BVH box tuning.** BVH internals stay as they are (see §1).

## 4. Constraints

- **Library API.** Only the additive preset. No breaking changes. The changeset is a minor bump for `contactHeatmap`.
- **Example code.** The parsers (OBJ, PLY, STL) live in the example, not the library.
- **Interaction.** The UI must never freeze. Long operations (repair, fill-holes fairing on large meshes) show busy states. All repair ops are undoable through the engine's history.
- **Scale.** Works with 1M-triangle meshes. Check the drag and the heatmap at 1M in the bench or browser check.
- **Tests.**
  - Unit tests for `contactHeatmap`.
  - Parser unit tests (small fixtures).
  - A browser check (headless Chrome via `scripts/chrome-cdp.mjs`) that loads the built showcase and runs a scripted session: load two fixtures, drag, heatmap, one repair, undo. It checks for no console errors and that the readouts update.
  - The CI builds it.
- **Workspace.** Done in its own clone and branch, like the parallel issue work, so it runs alongside other work.

## 5. Left to the builder

- Layout and visual styling. It should be clean and readable, not over-designed.
- The component structure inside the example.
- The parser implementation details.
- Throttling details for the readouts.

## 6. Decisions after the builder check (they override earlier sections)

1. **Penetration beyond the depth slider** *(owner chose option c)*. A field marks vertices beyond its `maxDistance` as far, with no sign. A slider-bounded field would therefore show deep interiors as base grey. So `contactHeatmap` takes `exactDepth?: boolean`:
   - `true` gives `maxDistance: Infinity`. Every vertex then gets its true signed distance, and anything deeper than `depth` clamps to the darkest red. The cost is a full recompute on every update.
   - `false` gives `maxDistance = max(depth, gap)`, the fast contact path. Deeper vertices show the base colour, and the UI labels them "beyond range".
   - The showcase defaults to `true`, with a "Fast mode (large meshes)" toggle that switches to `false`. The library default is `false`, because the library is general-purpose and the fast path is the safe default.
2. **Sign** *(my call)*. `contactHeatmap` passes `sign` through, defaulting to `"auto"`. Auto uses pseudonormals for open meshes, so penetration still has a side.
   - The collision panel reads `fieldStatus` (async) for A and B. When the target is open or inconsistently oriented, it shows a warning: "B is open: penetration side follows its surface normals; repair → orient/fill holes for a closed result".
3. **Field access** *(my call; additive library API)*. Add `SceneStore.fetchColoringValues(mesh): Promise<Float32Array | null>`. It returns the values of the store-owned field behind the mesh's current colouring, or `null` when there is none. It is the async `fetchFieldValues` on Worker scenes and a copy on local ones.
   - Readouts: maximum penetration = `-min(values)` over finite negative values. In fast mode, if any value sits at the bound, show "≥ depth". Minimum gap comes from `meshDistance`.
   - Readouts throttle to about 4 Hz during drags and refresh on drag end.
4. **Tolerance** *(my call)*. It is one number, used for two things:
   - the symmetric white band [-tolerance, +tolerance] in the stops;
   - the `findContacts({ margin: tolerance })` readout.

   Changing it changes only the stops. The field is reused and only the colours are recomputed.
5. **Validation** *(my call)*. `contactHeatmap` throws `FastMeshError("INVALID_ARGUMENT")` unless `depth > 0`, `gap > 0` and `0 ≤ tolerance < min(depth, gap)`. The UI sliders clamp to these limits. Stops stay strictly increasing. At `tolerance = 0` the white band collapses to a single stop at 0. Below the range is `clamp` and above is `base`.
6. **Analyze** *(checked in code)*. `analyze` already exists on `Scene` (sync) and `WorkerScene` (async), and it is not a history step. The app owns its scene and calls it directly. The store is not involved. Repairs go through `store.repair`, so they are undoable.
7. **Loading** *(my call)*.
   - Files are parsed in a dedicated parser Worker inside the example. It returns transferable typed arrays, and the app calls `store.addMesh`.
   - The STL parser merges bit-identical vertex positions, since STL is triangle soup. Near-duplicates are left to the Weld button.
   - Progress shows per file.
8. **Demo meshes** *(my call)*. Three meshes are generated on start:
   - A: a closed blobby sphere;
   - B: a closed rounded box that penetrates A;
   - C: a mesh with a hole and a floating piece, so repair has something to show.

   `?big` scales A and B to about 1M triangles each for scale checks.
9. **Selection** *(my call)*. Collision A and B default to the first two meshes. If either is removed, its picker resets and colouring stops. Colouring persists across repairs; the field recomputes. Each mesh has one colouring, so picking a mesh as A or B replaces any colouring it had.
10. **CI** *(my call)*. CI runs the browser check against `vite preview` with the COOP/COEP plugin and asserts `threads > 1`. It also runs a short `?local` pass. The app uses the workspace package and three's `TransformControls` from `three/addons`, as in `examples/react`, with no new gizmo dependency.

## 7. Owner additions (2026-10-23)

1. **Colouring start distance** *(owner)*. Colouring begins within a chosen distance, for example "from 1 mm", or "from 0 mm", which colours penetration only.
   - `contactHeatmap` accepts `gap ≥ 0` (with `depth > 0`). When `gap = 0`, everything at or beyond 0 above the tolerance band shows the base colour.
   - The showcase exposes `gap` as the slider "Colour from (mm)".
2. **Crisp, adjustable boundary between coloured and uncoloured areas** *(owner)*. Vertex colours interpolate across triangles, so the edge between coloured and base-coloured vertices blurs over a triangle's width. The fix is fragment shading *(my call)*:
   - **Engine.** It exposes, per mesh, the scalar values of the field behind its colouring as a render buffer, `values()` (f32 per vertex slot). It uses the same dirty ranges as the colours. In Rust this is a view of the colouring field's channel, so no extra copy is made.
   - **Worker client.** It mirrors that buffer like the other render buffers, with dirty-range transfer.
   - **`bindMesh` (`fast-mesh-wasm/three`).** It gains `shading?: "vertex" | "fragment"`, default `"vertex"`, which keeps today's behaviour. In `"fragment"` mode the material colours each fragment from the interpolated value through the rule's colormap: a LUT texture, the range, below/above handling, the base colour, and the center.
     - Far values (Infinity) and NaN are encoded so that they interpolate to base without colouring their neighbours.
     - `edgeSoftness?: number`, in distance units, sets the width of the blend at the coloured/uncoloured boundary and at the tolerance band. 0 means a hard edge.
   - **R3F.** `<FastMesh>` passes `shading` and `edgeSoftness` through.
   - **Showcase.** The collision panel gets a "Boundary softness" slider, and fragment shading is on by default.
   - **Exactness.** The per-fragment colour at a vertex equals the vertex-mode colour at that vertex. This is tested.

## 8. Multi-mesh contact groups (owner, 2026-10-23)

The owner's case: meshes A–D, with contact relationships A↔B, B↔C, A↔C and A↔D. Moving A must update everything related to A. B↔C must not be recomputed when A moves. Each mesh shows one colour scale, whichever partner it touches *(owner: the same colours for every partner)*.

1. **Model: a relation graph with three layers** *(my call)*.
   - **Edges (pairs).** Each ordered pair X→Y is a distance relation: X's vertices measured to Y. The engine already marks only the fields that involve a moved or edited mesh as due. It also skips pairs whose root boxes are farther apart than the margin.
   - **Nodes (meshes).** A mesh's value is the reduction of its edges: per vertex, the minimum signed distance over all of its partners (the nearest surface, or the deepest penetration). Per vertex, the engine also keeps which partner produced the value (`nearest: u32` partner index; NaN or Infinity → none), so a UI can say "this red area touches C".
   - **Display.** One colouring per mesh, so the heatmap preset applies unchanged.
2. **Engine: multi-target distance fields** *(my call)*.
   - `DistanceField` gains `to: Vec<MeshId>`; the single-target form stays as sugar.
   - **Per vertex:** the closest point is searched over every target's BVH. The best distance found so far is the upper bound for the next target, so adding partners costs little. The result is exact and bit-identical to taking the min over separate single-target fields, with ties broken by the lowest partner index.
   - **Recompute:** when any target or the source moves or is edited, the field is due. Candidates come from the union of each target's near set, using the existing near-search limit per target.
   - **Removing a target** removes it from the set. When the set becomes empty, the field is removed.
3. **Store API** *(my call)*. `store.setContactGroup(id, { meshes: MeshHandle[], pairs?: [MeshHandle, MeshHandle][] | "all", heatmap: ContactHeatmapOptions } | null)`.
   - The store derives each member's partner set from `pairs`. `"all"` means every other member, with root-box culling.
   - For each member it creates one multi-target field, with `distanceTo` as an array, and colours the member with the heatmap.
   - Moving one member recomputes only the fields of that member and of its partners. Pairs that do not involve the moved mesh (B↔C when A moves) are untouched, because neither B's nor C's field has A… unless A is in their partner set, in which case only those fields recompute. That is the correct dependency.
   - `fetchColoringValues` works per member.
4. **Showcase.**
   - The collision panel gets a "Group" mode: tick the meshes, and choose "all pairs" or edit the pair list in a small matrix.
   - Readouts are given per pair. They are min gap and max depth through `meshDistance`, throttled.
   - A tooltip on hover shows the nearest partner, read from the `nearest` buffer through the existing query path. This is optional if it is costly.
