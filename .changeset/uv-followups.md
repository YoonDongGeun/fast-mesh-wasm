---
"fast-mesh-wasm": patch
---

UV follow-ups. `importGLTF` no longer fails on a signed, sparse or otherwise unsupported `TEXCOORD_0`: the geometry imports, `uvs` is `null` and the new `ImportedMesh.uvsSkipped` says why. `TEXCOORD_0` now honours `normalized` as glTF defines it: non-normalized `u8`/`u16` are no longer divided as if normalized, and those forms and `i8`/`i16` are read when the asset declares `KHR_mesh_quantization`, and skipped otherwise. `cutByPlane` caps across UV seams, so a mesh with UVs that is closed by position gives two parts closed by position, where it used to stay open with no error; meshes without UVs are unchanged. `quantizationBits.texCoord` is now a deprecated alias of `uv`. The UV channel docs now cover what `offset`, `shell`, `subdivide`, `inflate` and `smooth` do to seams.
