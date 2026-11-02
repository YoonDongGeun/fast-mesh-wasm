# Load → repair → color → export

The whole session in one flow: a part arrives as glTF bytes, gets
cleaned, gets painted by its distance to a neighbour, and ships as STL.
Each stage links to the recipe that covers it in depth.

## Load

In your app the bytes come from disk or the network; here they
round-trip through the exporter so the snippet runs standalone.

```ts
import { init, Scene } from "fast-mesh-wasm";
import { exportGLTF, exportSTL, importGLTF } from "fast-mesh-wasm/io";

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

// The load path: glTF bytes in, one mesh per (node, primitive) out.
// The part arrives with its bottom face missing (10 of the 12 triangles).
const template = scene.addMesh(boxPositions, boxIndices.slice(6));
const [part] = importGLTF(exportGLTF(scene, template));
scene.removeMesh(template);
const mesh = scene.addMesh(part.positions, part.indices);
if (scene.triangleCount(mesh) !== 10) throw new Error("expected the open box back");

// Its neighbour, shifted half a unit over so the two interpenetrate.
const neighbour = scene.addMesh(
  boxPositions.map((v, i) => (i % 3 === 0 ? v + 0.5 : v)),
  boxIndices,
);
```

See [IO: export / import](io-export-import.md) for the formats and options.

## Repair

```ts
const report = scene.analyze(mesh);
if (report.loops.length !== 1) throw new Error("expected the open bottom");

const [, filled] = scene.repair(mesh, [{ op: "orient" }, { op: "fillHoles", maxEdges: 64 }]);
if (filled.filled.length !== 1) throw new Error("expected the bottom filled");

const m = scene.measure(mesh);
if (!m.closed || m.volume === null || Math.abs(m.volume - 1) > 1e-6) {
  throw new Error("expected a watertight unit box");
}
```

See [Auto-repair](auto-repair.md) for the full pipeline and the loop ids.

## Color

```ts
const field = scene.addDistanceField({ name: "contact", on: mesh, to: neighbour, maxDistance: 2.0 });
scene.setColoring(mesh, { field, map: "turbo", range: [-0.5, 1] });

const colors = scene.views(mesh).colors();
if (colors === null || colors.length !== 4 * scene.vertexSlots(mesh)) {
  throw new Error("expected RGBA bytes for every vertex slot");
}
```

See [Contact heatmap of two meshes](contact-heatmap.md) for reading the
field back, and [Custom colormap + fragment shading](custom-colormap.md)
for custom maps.

## Export

```ts
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const stl = exportSTL(scene, mesh); // binary STL, world coordinates
const out = mkdtempSync(join(tmpdir(), "fast-mesh-"));
writeFileSync(join(out, "part.stl"), stl);
if (stl.length === 0) throw new Error("expected STL bytes");
scene.free();
```
