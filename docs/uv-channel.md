# UV channel

A mesh can carry one set of texture coordinates: `scene.addMesh(positions, indices, { uvs })` (2 finite floats per vertex), read back zero-copy with `scene.views(mesh).uvs()`, which is `null` for a mesh without UVs. In Rust it is a source channel named `"uv"` ([`Mesh::add_uvs`], [`Mesh::uvs`]), so it is journaled and undoable like any source channel. A mesh without UVs has no channel: it costs no memory and takes the same code paths as before.

Values are stored as given, in the convention of the file they came from. glTF's `v` points down from the top of the image, so a texture for glTF UVs needs `texture.flipY = false` in three.js, as `GLTFLoader` sets.

## What each operation does

Every operation either keeps the UVs aligned with the vertices or drops the channel. **Dropping is explicit**: `views(mesh).uvs()` turns `null`, the change record reports `uvs: "all"`, the Rust report says `uvs_dropped`, and undo brings the values back exactly. UVs are never left stale.

| Operation | UVs | What happens |
| --- | --- | --- |
| `weld` | Preserved | A pair merges only if its positions are within the tolerance **and** its UVs agree within `1e-6` per component, so the two sides of a seam stay apart. Without UVs, the signature and results are unchanged. `analyze` still reports seam vertices as coincident: it is about positions. |
| `removeDegenerate`, `removeSmallComponents` | Preserved | Only removes; survivors keep their UVs. |
| `orient` | Preserved | Only flips winding. |
| `fillHoles` | Interpolated | New interior vertices take the weighted mean of their creation parents, starting from the rim. A patch across a seam blends the two sides. |
| `bridgeLoops` | Preserved | Adds triangles between existing vertices. |
| `remesh` | Interpolated | Split and collapse vertices interpolate their parents; existing vertices do not move. Boundary vertices, so seams, are locked against collapses. |
| `subdivide` | Interpolated | New vertices take their edge's UV midpoint (linear, not Loop's stencil), and old vertices keep their UV even though they move. |
| `simplify` | Interpolated | A survivor takes the UV at its new position's parameter along the collapsed edge. The error metric ignores UVs, so textures can stretch. With `preserveBoundary` (the default) seam vertices are left alone; without it the two sides of a seam may collapse differently and open a crack. |
| `inflate`, `smooth`, `relax`, `grab`, `offset` | Preserved | Moves vertices; the texture moves with them. |
| `shell` | Preserved | The inner surface copies its outer vertices' UVs. Wall triangles repeat the rim's UVs, so they sample a line of the texture. |
| `cutByPlane` | Interpolated | Both sides keep their UVs. Crossing vertices interpolate along the cut edge, and cap vertices from the rim. The new mesh gets its own channel. |
| `join` | Preserved or dropped | With UVs on both meshes they carry over. With UVs on one, the result has none: there is nothing to give the other side's vertices. |
| `boolean` | Dropped | The result mixes split pieces and the other mesh's surface, which has no coordinates in this mesh's texture space. A boolean of a mesh with itself changes nothing and keeps them. |
| `compact` | Preserved | Remapped with the slots. Afterwards earlier drops cannot be undone, as for all history. |
| `edit` | Your call | `addVertices` gives new vertices the UV `(0, 0)`, and `setPositions` leaves UVs alone. Set them in the same edit with `{ op: "setUvs", vertices, uvs }`. |

The tests are one per row: `crates/fast-mesh-ops/src/uv_table.rs` and `crates/fast-mesh-scene/src/uv_tests.rs`.

## Seams

A seam is two vertices at one position with different UVs. They are separate vertices, so to everything topological the seam is a boundary: `analyze` counts it as open, `fillHoles` on it would add a patch, and `boolean`, which needs a closed mesh, rejects it. To treat such a mesh as closed, ignoring its texture, add it without `uvs` and weld it.

## Files

`importGLTF` and `importGLTFAsync` read `TEXCOORD_0` (float, or normalized `u8`/`u16`) into `ImportedMesh.uvs`, including external buffers and Draco primitives. `exportGLTF` writes it as a float `VEC2` accessor, with or without `draco: true`. Draco quantizes UVs to 16 bits by default (`dracoOptions.quantizationBits.uv`). `TEXCOORD_1` and up are skipped, and so are materials and textures. STL, OBJ and PLY exports do not carry UVs.

## three.js

`bindMesh` adds a `uv` attribute while the mesh has UVs. Pass `map` (or a `material` with a `map`) to draw a texture. The map is taken off the material while the mesh has no UVs, as after a boolean, and put back on undo, rather than stretching one texel over the surface.

[`Mesh::add_uvs`]: https://docs.rs/fast-mesh-store/latest/fast_mesh_store/struct.Mesh.html#method.add_uvs
[`Mesh::uvs`]: https://docs.rs/fast-mesh-store/latest/fast_mesh_store/struct.Mesh.html#method.uvs
