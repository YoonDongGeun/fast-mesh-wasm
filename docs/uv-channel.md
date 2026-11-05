# UV channel

A mesh can carry one set of texture coordinates: `scene.addMesh(positions, indices, { uvs })` (2 finite floats per vertex), read back zero-copy with `scene.views(mesh).uvs()`, which is `null` for a mesh without UVs. In Rust it is a source channel named `"uv"` ([`Mesh::add_uvs`], [`Mesh::uvs`]), so it is journaled and undoable like any source channel. A mesh without UVs has no channel: it costs no memory and takes the same code paths as before.

Values are stored as given, in the convention of the file they came from. glTF's `v` points down from the top of the image, so a texture for glTF UVs needs `texture.flipY = false` in three.js, as `GLTFLoader` sets.

## What each operation does

Every operation either keeps the UVs aligned with the vertices or drops the channel. **Dropping is explicit**: `views(mesh).uvs()` turns `null`, the change record reports `uvs: "all"`, the Rust report says `uvs_dropped`, and undo brings the values back exactly. UVs are never left stale.

| Operation | UVs | What happens |
| --- | --- | --- |
| `weld` | Preserved | A pair merges only if its positions are within the tolerance **and** its UVs agree within `1e-6` per component, so the two sides of a seam stay apart. Without UVs, the signature and results are unchanged. `analyze` still reports seam vertices as coincident: it is about positions. |
| `removeDegenerate` | Preserved | Only removes; survivors keep their UVs. |
| `removeSmallComponents` | Preserved | Only removes. An edge also joins across a seam (exactly coincident corners), so the UV islands of one surface count as one component. |
| `orient` | Preserved | Only flips winding. |
| `fillHoles` | Interpolated | New interior vertices take the weighted mean of their creation parents, starting from the rim. A loop with a seam edge is skipped with reason `"uvSeam"`, unless you list it in `loops`; then its patch blends the two sides. |
| `repair(mesh, { kind: "auto" })` | Preserved and interpolated | One undo step of `weld`, `removeDegenerate`, `orient`, `removeSmallComponents` and `fillHoles`, each as in its row: seams stay, UV islands stay, and only holes without a seam edge are filled. It never drops UVs. |
| `bridgeLoops` | Preserved | Adds triangles between existing vertices. |
| `remesh` | Interpolated | Split and collapse vertices interpolate their parents; existing vertices do not move. Boundary vertices, so seams, are locked against collapses. |
| `subdivide` | Interpolated | New vertices take their edge's UV midpoint (linear, not Loop's stencil), and old vertices keep their UV even though they move. |
| `simplify` | Interpolated | A survivor takes the UV at its new position's parameter along the collapsed edge. The error metric ignores UVs, so textures can stretch. With `preserveBoundary` (the default) seam vertices are left alone; without it the two sides of a seam may collapse differently and open a crack. |
| `inflate`, `smooth`, `relax`, `grab`, `offset` | Preserved | Moves vertices; the texture moves with them. |
| `shell` | Preserved | The inner surface copies its outer vertices' UVs. Wall triangles repeat the rim's UVs, so they sample a line of the texture. |
| `cutByPlane` | Interpolated | Both sides keep their UVs. Crossing vertices interpolate along the cut edge, and cap vertices from the rim. The new mesh gets its own channel. |
| `join` | Preserved or dropped | With UVs on both meshes they carry over. With UVs on one, the result has none: there is nothing to give the other side's vertices. |
| `boolean` | Dropped | The result mixes split pieces and the other mesh's surface, which has no coordinates in this mesh's texture space. An open mesh with UVs is read with exactly coincident vertices merged, so seams on a closed surface are accepted, for either mesh; undo brings back the split vertices. A boolean of a mesh with itself changes nothing and keeps them. |
| `compact` | Preserved | Remapped with the slots. Afterwards earlier drops cannot be undone, as for all history. |
| `edit` | Your call | `addVertices` gives new vertices the UV `(0, 0)`, and `setPositions` leaves UVs alone. Set them in the same edit with `{ op: "setUvs", vertices, uvs }`. |

The tests are one per row: `crates/fast-mesh-ops/src/uv_table.rs` and `crates/fast-mesh-scene/src/uv_tests.rs`.

## Seams

A seam is two vertices at exactly one position with different UVs. They are separate vertices, so `analyze` counts the seam as an open boundary, and `orient` treats each UV island as its own open piece: it makes winding consistent within an island but cannot turn a closed surface outward across seams. To treat such a mesh as closed for those, add it without `uvs` and weld it.

`Scene.boolean` needs no such step. Lower down, Rust's `boolean_plan` takes meshes as they are and rejects a seamed one as open.

Seams are recognized by exact coincidence. Sides merely close to each other count as a real boundary.

## Files

`importGLTF` and `importGLTFAsync` read `TEXCOORD_0` (float, or normalized `u8`/`u16`) into `ImportedMesh.uvs`, including external buffers and Draco primitives. `exportGLTF` writes it as a float `VEC2` accessor, with or without `draco: true`. Draco quantizes UVs to 16 bits by default (`dracoOptions.quantizationBits.uv`). `TEXCOORD_1` and up are skipped, and so are materials and textures. STL, OBJ and PLY exports do not carry UVs.

## three.js

`bindMesh` adds a `uv` attribute while the mesh has UVs. Pass `map` (or a `material` with a `map`) to draw a texture. The map is taken off the material while the mesh has no UVs, as after a boolean, and put back on undo, rather than stretching one texel over the surface.

[`Mesh::add_uvs`]: https://docs.rs/fast-mesh-store/latest/fast_mesh_store/struct.Mesh.html#method.add_uvs
[`Mesh::uvs`]: https://docs.rs/fast-mesh-store/latest/fast_mesh_store/struct.Mesh.html#method.uvs
