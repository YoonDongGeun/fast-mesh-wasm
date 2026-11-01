# IO: export / import

Exporters read `scene.views(mesh)` (positions, indices, normals, colors),
so meshes export exactly as they are — edited, repaired, or colored.
Importers return plain arrays ready for `scene.addMesh`.

## Export one mesh

```ts
import { exportGLTF, exportOBJ, exportPLY, exportSTL } from "fast-mesh-wasm/io";

const stl: Uint8Array = exportSTL(scene, mesh); // binary STL by default
const stlAscii: Uint8Array = exportSTL(scene, mesh, { binary: false, name: "part" });
const obj: Uint8Array = exportOBJ(scene, mesh, { name: "part" });
const ply: Uint8Array = exportPLY(scene, mesh); // binary little-endian
const glb: Uint8Array = exportGLTF(scene, mesh); // binary .glb
const gltf: Uint8Array = exportGLTF(scene, mesh, { binary: false }); // embedded-base64 .gltf JSON
```

Options shared by every exporter (`ExportOptions`):

- `applyPlacement` (default `true`): map through `scene.matrix4(mesh)`
  so the file holds world coordinates. Pass `false` for the local
  render coordinates in `views(mesh).positions()`.
- Binary formats default on; ASCII variants are the same call with
  `binary: false`.

Details worth knowing:

- Only live triangles are written; dead slots are compacted away
  (a triangle slot reading `0, 0, 0` is dead).
- Colors ride along where the mesh has them (OBJ `v r g b`, PLY
  `red green blue alpha`, glTF `COLOR_0`); normals follow through the
  inverse-transpose rotation.
- glTF writes one node / mesh / `TRIANGLES` primitive — Draco
  compression is not applied.

## Import glTF

```ts
import { importGLTF, importGLTFAsync } from "fast-mesh-wasm/io";

// Binary .glb, or .gltf JSON with embedded base64 buffers.
for (const part of importGLTF(bytes)) {
  // { name, positions (world, Float64Array), indices, placement, colors }
  const mesh = scene.addMesh(part.positions, part.indices);
}
```

- One scene mesh per glTF (node, primitive); node transforms are baked
  into `positions`, and also returned as `placement` (column-major).
  `TRIANGLE_STRIP` / `TRIANGLE_FAN` are expanded; anything else throws.
- External (non-embedded) `.bin` buffers and sparse accessors are not
  supported; animations, skins, and materials beyond per-vertex colors
  are skipped.
- `KHR_draco_mesh_compression` primitives throw under `importGLTF` —
  decode them with `importGLTFAsync`, which takes the same bytes plus
  optional Draco options:

```ts
const parts = await importGLTFAsync(bytes, {
  // decoderModule, // a ready draco3d module (reuse across calls), or
  // wasmBinary,    // the decoder .wasm bytes for bundlers/workers
});
```

## Standalone Draco

```ts
import { decodeDraco } from "fast-mesh-wasm/io";

const { positions, indices } = await decodeDraco(drcBytes);
const mesh = scene.addMesh(positions, indices); // triangle meshes only
```

Needs the `draco3d` package installed unless you pass a decoder module
(or its `.wasm`) through the same options as `importGLTFAsync`.
