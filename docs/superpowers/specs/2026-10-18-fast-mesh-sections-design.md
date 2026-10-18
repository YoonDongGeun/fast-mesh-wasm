# fast-mesh Phase 6: Sections

- **Status:** draft for review
- **Parent spec:** `2026-10-14-fast-mesh-engine-design.md`, §9 (Sections) and §15 (phase 6). Where this document differs from §9, this document wins.
- **Builds on:**
  - Phase 2 BVH (`fast-mesh-bvh`)
  - Phase 3 queries (`fast-mesh-query`)
  - Phase 4 Scene (`fast-mesh-scene`)
  - Phase 5a fields (PR #34), which established subscriber reports and the "follow the mesh's update mode" rule
- **Labels:** what the owner said is marked *(owner)*. Choices made without asking are marked *(my call)*.

## 1. Intent

The owner wants to place a plane at any angle and see the cross-section of everything the plane passes through *(owner)*. The point is to look at meshes that are hidden from outside, and check two things *(owner)*:
- their shapes;
- the distances between them.

Distances are judged by looking at the section lines, not by reading numbers *(owner)*.

This is a personal, general-purpose library. Any meshes can be sectioned: scans, CAD parts, open or closed surfaces.

Success means:
- dragging a plane through scenes of million-triangle meshes gives section lines that keep up with the pointer;
- the lines stay correct while meshes are moved or edited.

## 2. Goals, non-goals, anti-goals

### Goals
1. **Plane slicing query** (`fast-mesh-query`). Given one mesh, its BVH and a plane in the mesh's frame, return the contours where the plane cuts the surface. This works without a Scene.
2. **Sections in the Scene.**
   - Any number of sections, each with its own plane and set of selected meshes.
   - Each selected mesh gets its own contours.
   - Changing the plane, or moving or editing a selected mesh, updates only what is affected.
3. **Contours.**
   - Closed polylines for closed surfaces; open polylines where the plane leaves an open surface.
   - Points are given in world coordinates and in the plane's own 2D coordinates.
4. **Real-time drag** *(my call, accepted by the owner)*. Re-slicing a 1M-triangle mesh after a plane move takes about a millisecond (§9).
5. **Cheap measurements** *(my call, accepted)*. Per contour: length, and signed area when closed. Per mesh: totals. On demand: the minimum in-plane gap between two meshes' contours.
6. **Fill** *(my call, accepted)*. An optional triangulation of each mesh's closed contours, holes included. It is meaningful for closed meshes only.

### Non-goals (this phase)
- **Drawing anything.** Clipping planes, line rendering and fill materials belong to Phase 8 (three.js) *(owner accepted)*.
- **Undo of plane changes.** A section is view state, not mesh data *(my call)*.
- **Other cutting shapes.** No thick slabs, curved cuts, or sections through boolean combinations of meshes.
- **Colouring contours by a distance field.** Possible later: contour points know their triangle, so a field could be sampled there.

### Anti-goals
- **No stalls on plane moves.** Moving the plane must never trigger work proportional to the whole mesh. It must not touch BVHs, fields, colours or contacts.
- **No broken contours from rounding.** A plane through a vertex, along an edge, or flush with a face must not produce gaps, duplicate points or broken chains on a closed, edge-manifold mesh.
- **No hidden recomputation.** A section must never recompute a mesh whose changes cannot reach the plane.

## 3. Constraints
- Same rules as earlier phases:
  - storage is `f32` mesh-local, each mesh has an `f64` origin, and placements are `f64` rigid;
  - no `unsafe`; MSRV 1.88; edition 2024;
  - no heap allocation in BVH traversal (fixed stack);
  - fallible allocation (`try_reserve`) for outputs, with `OutOfMemory` errors;
  - errors are typed;
  - everything stays single-threaded.
- New dependencies only with a reason. Triangulation is written in-tree (§5.3) so Phase 7 can reuse it.
- Placements are rigid, so plane distances in mesh-local space equal world distances.

## 4. The slicing query (`fast-mesh-query`)

### 4.1 Plane and side classification
- `Plane { origin: [f64; 3], normal: [f64; 3] }`:
  - The normal must be finite and non-zero. It is normalised on construction (`Plane::new`), and an invalid normal returns `InvalidInput`.
  - The signed distance of a point `p` is `s(p) = n·p − n·origin`, computed in `f64`.
- **Side of a vertex.** A vertex is `+` when `s ≥ 0` and `−` otherwise. A vertex exactly on the plane therefore counts as `+` (a symbolic perturbation that pushes the plane slightly to the `−` side).
  - Consequence: no vertex is ever *on* the plane, every crossing lies strictly inside an edge, and each triangle the plane crosses gives exactly one segment.
  - A face flush with the plane gives no segment. The contour runs along its neighbours instead, which is still one closed chain *(my call)*.
- **Determinism.** Side is a pure function of the vertex's stored position and the plane. Every triangle sharing a vertex therefore agrees on its side, whatever rounding happens.

### 4.2 Crossing points
- An edge whose endpoints have opposite sides gets one crossing point, computed from its endpoints in canonical order (smaller vertex id first):
  - `t = s(a) / (s(a) − s(b))`
  - `p = a + t·(b − a)`
  - The result is identical whichever triangle asks.
- Crossing points are identified by their edge `(min id, max id)`, not by their coordinates. Chaining is purely topological, so it is immune to coincident or near-coincident points.

### 4.3 Traversal
- Walk the BVH, visiting only nodes whose box straddles the plane.
- The box test is conservative: inflate by a few ulps of the box magnitude, so no crossing triangle is skipped.
- Triangles whose vertices are all on one side give no segment.
- Cost: proportional to the triangles near the plane (about √n for a typical surface) plus the straddling nodes. Never O(n).

### 4.4 Orientation and chaining
- **Segment direction.** Each segment is oriented using its triangle's winding: walking along it, the triangle's `+` side region is on the left, seen from the plane normal. For a consistently oriented closed mesh this makes outer contours counter-clockwise and holes clockwise, in the plane's 2D frame.
- **Chaining.** Join segment ends that share an edge key.
  - **Boundary edge** (one incident triangle): the chain ends there, giving an open polyline.
  - **Non-manifold edge** (more than two incident triangles crossing): the chain is split there. Each piece is reported open, with `manifold: false`.
  - **Mismatched orientation:** where two adjacent triangles disagree, the chain still joins (undirected) and the contour reports `oriented: false`.
- **Determinism.** Output order is deterministic. Contours start at their smallest edge key; closed contours rotate to start there.

### 4.5 Output

```rust
pub struct Slice { pub contours: Vec<Contour> }
pub struct Contour {
    pub points: Vec<[f64; 3]>,      // in the frame the plane was given in
    pub edges: Vec<(VertexId, VertexId)>, // the crossed edge of each point
    pub triangles: Vec<TriangleId>, // the triangle of each segment (len = points − 1, or = points if closed)
    pub closed: bool,
    pub oriented: bool,
    pub manifold: bool,
}
pub fn slice(mesh: &Mesh, bvh: &Bvh, plane: &Plane, out: &mut Slice) -> Result<()>;
```

- Fails with `StaleBvh` or `MeshMismatch` like every other query.
- Allocation is only into `out`, which is reused across calls.
- The plane is given in the mesh's local frame, origin included. The Scene does the world-to-local mapping (§6.2).

## 5. Derived results

### 5.1 Plane frame
- Each plane has a 2D frame `(u, v)`.
- **Default:** `u` is the world axis least parallel to `n`, made orthogonal to `n` and normalised, and `v = n × u`. It is deterministic and stable while the normal changes slightly *(my call)*.
- `Plane::with_axes(origin, u, v)` lets callers choose the frame. `n = u × v`, and the axes must be orthonormal within 1e-6.

### 5.2 Measurements
- **Contour length:** the sum of segment lengths. Closed contours include the closing segment.
- **Closed-contour area:** signed, by the shoelace formula in `(u, v)`. Positive means counter-clockwise.
- **Per-mesh totals:**
  - `length`: the sum over all contours;
  - `area`: the sum of signed areas over closed, oriented contours. For a consistently oriented closed mesh this is the material area of the cut.
- **Gap** `section_gap(a, b)`: the minimum 2D distance between any segment of mesh a's contours and any of mesh b's, both points included. It is 0 when the contours cross.
  - Computed on demand and not cached.
  - Must not be O(n·m) brute force: use a 2D grid or a sweep over segments (builder's choice).
  - This is an in-plane distance. It equals the 3D mesh distance only when the closest points happen to lie in the plane.

### 5.3 Fill (optional per section)
- Applies to closed contours only. Open contours are never filled.
- **Nesting:**
  - Oriented contours: counter-clockwise contours are outers, and each clockwise contour belongs to the smallest outer containing it.
  - Unoriented contours: even-odd nesting by containment depth.
- **Triangulation:** ear clipping with hole bridging (the earcut approach), in `(u, v)`. It lives in a new public module `fast_mesh_query::triangulate` so Phase 7 hole filling can reuse it.
- **Output:** vertices (world and `(u, v)`) plus `[u32; 3]` triangles, counter-clockwise when seen from the plane normal.
- **Self-intersecting contours** (a self-intersecting mesh) never panic. The fill may be imperfect, and `fill_ok: false` reports it.

## 6. Scene integration (`fast-mesh-scene`)

### 6.1 API

```rust
pub struct SectionDesc { pub plane: Plane, pub meshes: Vec<MeshId>, pub fill: bool }

scene.add_section(desc) -> Result<SectionId>
scene.remove_section(id) -> Result<()>
scene.set_section_plane(id, plane) -> Result<()>
scene.set_section_meshes(id, meshes) -> Result<()>
scene.set_section_fill(id, fill) -> Result<()>
scene.update_sections() -> Result<()>          // computes every stale entry; returns the first error

scene.section(id, mesh) -> Result<&SectionResult>   // &self
scene.section_status(id, mesh) -> Result<SectionStatus> // Current | Stale
scene.section_gap(id, a, b) -> Result<Option<Gap>>  // &self; None if either has no contours

pub struct SectionResult {
    pub contours: Vec<SectionContour>, // world points + (u, v) points + Contour flags
    pub length: f64,
    pub area: f64,
    pub fill: Option<SectionFill>,
    pub fill_ok: bool,
}
pub struct Gap { pub distance: f64, pub a: [f64; 3], pub b: [f64; 3] }
```

- `SectionId` is generational, like `MeshId`.
- **Errors:**
  - unknown mesh, duplicate mesh in `meshes`, or invalid plane: `InvalidInput`;
  - unknown section: `InvalidInput` (same rule as `FieldId`).

### 6.2 Coordinates
- For each selected mesh, the world plane maps into the mesh's local `f32` frame. This uses the mesh's placement inverse and origin, in `f64`.
- The resulting contour points map back to world coordinates the same way.
- Contours are therefore computed exactly as in §4, with no `f32` loss beyond the stored vertices.

### 6.3 When results update *(my call, mirrors Phase 5a §11.2)*
- **Timing follows the mesh's update mode:**
  - *Eager:* entries update immediately after the change that staled them. A failed Eager update is swallowed and leaves the entry Stale.
  - *Lazy / Manual:* entries update on `update_sections()`, or on `update(id)` / `flush()` for that mesh.
- **Reading:** readers take `&self` and return the last computed result. `section_status` says whether it is current.
- **Plane or fill change** (`set_section_plane`, `set_section_fill`): stales every entry of that section. It touches nothing else.
- **Selection change** (`set_section_meshes`): added meshes start Stale, and removed meshes drop their entries.
- **Placement change of a selected mesh:** stales that mesh's entries in every section selecting it.
- **Edit, undo or redo of a selected mesh:** stales an entry only if the change can reach the plane. Let `S` be the vertex set of the entry's crossed edges. The entry stales iff one of these holds:
  1. a changed vertex is in `S`;
  2. a changed vertex not in `S` now has an incident triangle with vertices on both sides;
  3. an added or removed triangle has (or had) vertices on both sides.

  Why this suffices: before the change, no edge at a vertex outside `S` crossed the plane, so the triangles around it lay on one side. A recompute redoes the whole mesh, since §4.3 makes that cheap; there is no splicing.
- **Compact:** full recompute of that mesh's entries.
- **Corrupted history or a `None` change set:** full recompute.
- **Mesh removal:** the mesh is dropped from every section, and reported.
- **Stale BVH** (Manual mode): the entry stays Stale, and `update_sections` returns `StaleBvh` for it, as fields do.

### 6.4 Reports
- `MeshChanges` gains `sections_changed: Vec<SectionId>`: the sections whose result for this mesh changed since the last call, each once.
- A removed section is reported on each mesh it selected.
- Subscribers receive these through the existing fan-out.
- `field:` remains the only reserved channel prefix. Sections store nothing in mesh channels.

### 6.5 History
- Sections are not in history: undo does not move planes.
- Undo of a mesh edit reaches sections through the normal change sets.

## 7. Errors
- No new error variants are expected.
- `InvalidInput` covers bad planes, ids and selections.
- `StaleBvh`, `MeshMismatch` and `OutOfMemory` keep their meanings.

## 8. Testing
- **Brute-force oracle:** classify every triangle with the same side rule. The segment multiset must equal the BVH query's. Use random meshes, random planes, and planes snapped through vertices, edges and faces.
- **Topology properties:** on closed edge-manifold meshes (sphere, torus, cube, the star prism fixture), every contour is closed, oriented and manifold. This includes:
  - planes exactly through vertices;
  - planes along edges;
  - planes flush with faces;
  - axis-aligned planes through cube faces.
- **Analytic checks:**
  - the area of a high-resolution sphere cut at height h is ≈ π(r²−h²);
  - a box cut has rectangle length and area;
  - a torus cut yields two contours, or one with a hole, with the expected signs.
- **Open meshes:**
  - a terrain cut gives open polylines that end on boundary edges;
  - a non-manifold fixture gives split chains with `manifold: false`.
- **Fill:**
  - fill area equals the summed contour area within 1e-9 relative;
  - a ring has one hole;
  - nested islands are handled;
  - degenerate and self-intersecting inputs never panic (proptest).
- **Gap:** compare with the brute-force segment–segment minimum. Two concentric circles give `r₂ − r₁`.
- **Scene:**
  - property test (like `fields_match_from_scratch`) with random sequences of plane moves, placements, edits, undo/redo, mode switches, compaction and selection changes; every current entry must equal a from-scratch slice;
  - meshes at non-zero origins (~1000);
  - a test that a far-away edit leaves the entry untouched (counts as no recompute).
- **Reports:** `sections_changed` lists exactly the changed sections.

## 9. Benchmarks
- Slice of 1M and 10M triangle meshes (terrain and sphere) at several plane positions. Target: about 1 ms at 1M.
- Plane drag: 64 poses across a 1M mesh, re-slice per pose.
- Fill at 1M.
- Gap between two 1M meshes' contours.
- Edit far from the plane: should cost nearly nothing.
- Record in `docs/benchmarks.md`.

## 10. Left to the builder
- Internal data structures for chaining (hash map vs sort by edge key) and for the gap search.
- How `S` is stored per entry (sorted vec, bitset, hash set) and how criteria 2–3 of §6.3 are checked efficiently.
- The exact box-inflation term in §4.3, provided it is conservative.
- Whether `SectionResult` keeps `(u, v)` points always or computes them on demand.
- Splitting into plan tasks.

## 11. Decisions after the builder check (all *my call*; they override earlier sections)

1. **Coincident points.** When the plane passes exactly through a vertex, every crossed edge at that vertex gets `t = 0`. Their points share coordinates, and zero-length segments join them.
   - The query keeps them. Each crossed edge appears exactly once, which is what "no duplicate points" in §2 means: no *topological* duplicates.
   - The parallel `points` / `edges` / `triangles` arrays in §4.5 stay as specified.
   - Length and area are unaffected by zero-length segments.
   - Fill merges consecutive points with equal `(u, v)` before triangulating.
   - Tests must accept zero-length segments at snapped vertices. Gaps, breaks and open chains on closed edge-manifold meshes still fail.
2. **Unknown ids.**
   - Add `SceneError::UnknownSection`, following `UnknownField`. This overrides §6.1 and §7.
   - A stale or unknown `MeshId` anywhere in the section API returns `UnknownMesh`, as fields and contacts do.
   - A live mesh that the section does not select returns `InvalidInput`.
   - A duplicate mesh in `meshes`, or an invalid plane, returns `InvalidInput`.
3. **`fill_ok` is best effort.** It is `false` in two cases: ear clipping fails to finish, or the fill area differs from the summed contour area by more than 1e-6 relative. There is no separate self-intersection sweep. A bad fill can go unreported; the docs say so.
4. **Before the first computation.**
   - `section(id, mesh)` returns an empty `SectionResult` (no contours, zero length and area, no fill), and `section_status` returns `Stale`.
   - `add_section` and `set_section_meshes` never fail with `StaleBvh`. For Eager meshes they compute immediately, and a failure is swallowed (§6.3). Lazy and Manual meshes wait for `update_sections`, `update(id)` or `flush`.
5. **Triangle changes with no vertex moved** (for example an edge flip) count as removed plus added triangles under criterion 3 of §6.3. Each entry keeps enough state, such as its crossed-triangle set, to evaluate a removed triangle's former side.
6. **`sections_changed`** follows the semantics of `fields_changed`: each section once per call, in the order first reported.
