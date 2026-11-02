# IO: export / import

Exporters read `scene.views(mesh)` (positions, indices, normals, colors),
so meshes export exactly as they are — edited, repaired, or colored.
Importers return plain arrays ready for `scene.addMesh`.

## Export one mesh

```ts
import { init, Scene } from "fast-mesh-wasm";
import { exportGLTF, exportOBJ, exportPLY, exportSTL } from "fast-mesh-wasm/io";

await init();
const scene = new Scene();

// A unit box: 8 vertices, 12 triangles, outward winding.
const boxPositions = new Float64Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1]);
const boxIndices = new Uint32Array([
  0, 3, 2, 0, 2, 1, // bottom (z = 0)
  4, 5, 6, 4, 6, 7, // top (z = 1)
  0, 1, 5, 0, 5, 4, // front (y = 0)
  2, 3, 7, 2, 7, 6, // back (y = 1)
  0, 4, 7, 0, 7, 3, // left (x = 0)
  1, 2, 6, 1, 6, 5, // right (x = 1)
]);
const mesh = scene.addMesh(boxPositions, boxIndices);

const stl: Uint8Array = exportSTL(scene, mesh); // binary STL by default
const stlAscii: Uint8Array = exportSTL(scene, mesh, { binary: false, name: "part" });
const obj: Uint8Array = exportOBJ(scene, mesh, { name: "part" });
const ply: Uint8Array = exportPLY(scene, mesh); // binary little-endian
const glb: Uint8Array = exportGLTF(scene, mesh); // binary .glb
const gltf: Uint8Array = exportGLTF(scene, mesh, { binary: false }); // embedded-base64 .gltf JSON
if (stl.length === 0 || obj.length === 0 || ply.length === 0 || glb.length === 0 || gltf.length === 0) {
  throw new Error("expected bytes from every exporter");
}
const ascii = new TextDecoder().decode(stlAscii);
if (!ascii.startsWith("solid part")) throw new Error("expected ASCII STL to name the solid");
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
- glTF writes one node / mesh / `TRIANGLES` primitive — uncompressed
  by default, or Draco-compressed (`KHR_draco_mesh_compression`) with
  `{ draco: true }`, which makes the call async.

## Import glTF

```ts
import { importGLTF, importGLTFAsync } from "fast-mesh-wasm/io";

// Binary .glb, or .gltf JSON with embedded base64 buffers.
for (const part of importGLTF(glb)) {
  // { name, positions (world, Float64Array), indices, placement, colors }
  const imported = scene.addMesh(part.positions, part.indices);
  if (scene.triangleCount(imported) !== 12) throw new Error("expected the box back");
}
```

- One scene mesh per glTF (node, primitive); node transforms are baked
  into `positions`, and also returned as `placement` (column-major).
  `TRIANGLE_STRIP` / `TRIANGLE_FAN` are expanded; anything else throws.
- External (non-embedded) `.bin` buffers load through `importGLTFAsync`'s
  `resolveBuffer` (see below); sparse accessors are not supported.
  Animations, skins, and materials beyond per-vertex colors are skipped.
- `KHR_draco_mesh_compression` primitives throw under `importGLTF` —
  decode them with `importGLTFAsync`, which takes the same bytes plus
  optional Draco options:

```ts
const parts = await importGLTFAsync(glb, {
  // decoderModule, // a ready draco3d module (reuse across calls), or
  // wasmBinary,    // the decoder .wasm bytes for bundlers/workers
  // resolveBuffer, // external .bin bytes by URI (see below)
});
if (parts.length !== 1 || parts[0].indices.length !== 36) throw new Error("expected the box back");
scene.free();
```

## Standalone Draco

```ts no-run
import { decodeDraco } from "fast-mesh-wasm/io";

// drcBytes: your own .drc file's bytes (decodeDraco only runs on real
// Draco bitstreams, so this block is not executed by the snippet test).
declare const drcBytes: Uint8Array;
const { positions, indices } = await decodeDraco(drcBytes);
const mesh = scene.addMesh(positions, indices); // triangle meshes only
```

Needs the `draco3d` package installed unless you pass a decoder module
(or its `.wasm`) through the same options as `importGLTFAsync`.
Encoding mirrors it: `encodeDraco` writes standalone `.drc` bytes and
`exportGLTF` takes `{ draco: true }`, both sharing the encoder module
(or its `.wasm`) through `encoderModule` / `wasmBinary`, plus
quantization, compression level, and face-order options:

```ts
import { decodeDraco, encodeDraco } from "fast-mesh-wasm/io";

// The same box through Draco: standalone bytes, then compressed glTF.
const drc: Uint8Array = await encodeDraco(boxPositions, boxIndices);
if (new TextDecoder().decode(drc.subarray(0, 5)) !== "DRACO") throw new Error("expected Draco bytes");
const decoded = await decodeDraco(drc);
if (decoded.indices.length !== 36) throw new Error("expected 36 corners back");
const dracoScene = new Scene();
const dracoMesh = dracoScene.addMesh(boxPositions, boxIndices);
const glbDraco: Uint8Array = await exportGLTF(dracoScene, dracoMesh, { draco: true });
const fromDraco = await importGLTFAsync(glbDraco);
if (fromDraco.length !== 1 || fromDraco[0].indices.length !== 36) throw new Error("expected the box back");
dracoScene.free();
```

## External .bin buffers

A `.gltf` that references sibling files (`"uri": "model.bin"`) loads through
`importGLTFAsync` with a `resolveBuffer` callback. The importer never fetches:
the callback is the only way external bytes enter, so it is also where size
caps belong — reject URIs outside the asset's directory and buffers past a
byte budget before returning them.

```ts
// The sibling file's bytes stand in for a fetch / FileReader / fs read.
const extPositions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
const extBin = new Uint8Array(extPositions.buffer);
const extDoc = new TextEncoder().encode(
  JSON.stringify({
    asset: { version: "2.0" },
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
    accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: "VEC3" }],
    bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: extBin.length }],
    buffers: [{ byteLength: extBin.length, uri: "tri.bin" }],
  }),
);
const extParts = await importGLTFAsync(extDoc, {
  resolveBuffer: (uri) => {
    if (uri !== "tri.bin") throw new Error(`unexpected buffer ${uri}`);
    return extBin;
  },
});
if (extParts.length !== 1 || extParts[0].indices.length !== 3) {
  throw new Error("expected the triangle back");
}
```
