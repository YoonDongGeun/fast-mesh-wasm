# Custom colormap + fragment shading

Any field (distance, thickness, curvature) can drive vertex colors
through a custom map, rendered per-pixel so range edges are crisp
iso-lines instead of triangle-width gradients.

## Setup

A unit box; its wall thickness (about 1 through the faces) drives the map.

```ts
import { init, Scene } from "fast-mesh-wasm";

await init();
const scene = new Scene();

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
```

## Custom stops

A colormap is a built-in name or stops — `[t, rgba]` pairs, at least
two, `t` in [0, 1] strictly increasing, interpolated linearly:

```ts
const field = scene.addThicknessField({
  name: "wall",
  on: mesh,
  maxDistance: 5.0, // thicker vertices hold Infinity
});

scene.setColoring(mesh, {
  field,
  map: {
    stops: [
      [0.0, [49, 54, 149, 255]], // thin: deep blue
      [0.5, [245, 245, 245, 255]], // nominal: near white
      [1.0, [165, 0, 38, 255]], // thick: deep red
    ],
  },
  range: [0, 5.0], // mapped onto the colormap
  center: 2.5, // pin a value to the middle of the map (optional)
  below: "clamp", // under-range: map end color | "base"
  above: "base", // over-range + Infinity: base color | map end
  base: [200, 200, 200, 255],
  nan: [128, 128, 128, 255], // open-sheet rims hold NaN for thickness
});
```

`center` makes the mapping piecewise linear (two halves) — useful for
diverging maps where a nominal value must sit mid-scale. Curvature
fields (`addCurvatureField`, signed mean curvature in 1/world-units:
positive on domes, negative in bowls) pair well with it: center at 0.

## Fragment shading (three.js)

Vertex coloring interpolates colors across each triangle, blurring the
edge between colored and base-colored vertices over a triangle's width.
Fragment shading colors each fragment from its interpolated field value
instead, so range edges and contact-band boundaries render as exact
iso-lines (optionally softened):

```ts
import { bindMesh } from "fast-mesh-wasm/three";

const bound = bindMesh(scene, mesh, {
  shading: "fragment",
  edgeSoftness: 0.05, // blend width in field distance units; 0 = hard edge
});
// bound.shading reports "fragment" (falls back to "vertex" with a warning
// when the scene or material cannot support it).
if (bound.shading !== "fragment") throw new Error("expected fragment shading");
bound.setEdgeSoftness(0.1); // adjust later, no recompile.
bound.dispose();
scene.free();
```

Requirements: a scene with `coloring(mesh)` and `views(mesh).values`
(`Scene` and `WorkerScene` have both), and a standard material
(`MeshStandardMaterial` / Phong / Lambert / Basic — each
fragment-shaded binding needs its own material, `material.clone()`).
Far (`Infinity`) and NaN values never tint their neighbours; the switch
is a hard edge halfway between vertices.
