# Contact heatmap of two meshes

Color one mesh by its distance to another: a signed distance field plus
a `setColoring` rule. Penetration, near-contact, and far regions each
get their own color.

## Setup

Two unit boxes, the second shifted half a unit over so they interpenetrate.

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

const meshA = scene.addMesh(boxPositions, boxIndices);
const meshB = scene.addMesh(
  boxPositions.map((v, i) => (i % 3 === 0 ? v + 0.5 : v)),
  boxIndices,
);
```

## Paint the gap

```ts
// 1. Distance field on mesh A, measured to mesh B.
const field = scene.addDistanceField({
  name: "contact",
  on: meshA,
  to: meshB, // or [meshB, meshC]: nearest surface wins
  maxDistance: 2.0, // farther vertices hold Infinity
  sign: "auto", // | "pseudonormal" | "winding"
});

// 2. Paint it: the contact band gets the map, far/NaN get the base grey.
scene.setColoring(meshA, {
  field,
  map: "turbo", // | "viridis" | "coolwarm" | "grayscale" | custom stops
  range: [0, 2.0],
  below: "clamp", // values under lo: map end color (default)
  above: "base", // Infinity (far): the base color (default)
  base: [200, 200, 200, 255],
  nan: [128, 128, 128, 255], // dead/unused vertices
});
```

## Reading the result

- `scene.fieldValues(field)` → per-vertex `Float32Array` (signed:
  negative inside the target when the sign data agrees).
- `scene.fieldNearest(field)` → per-vertex `Uint32Array`: the index in
  `to` each vertex measured to (`0xffffffff` when none).
- `scene.fieldStatus(field)` → `{ current, sign }`: whether the values
  match the meshes as they are now, and what the sign data found out
  (`mode`, `closed`, `consistentOrientation`, counts).
- `scene.findContacts({ margin: 2.0, pairs: [[meshA, meshB]] })` →
  per-pair `{ intersecting, distance, trianglesA, trianglesB }` for the
  triangle-level counterpart of the same band.

```ts
const values = scene.fieldValues(field);
if (values.length !== scene.vertexSlots(meshA)) throw new Error("expected one value per vertex slot");
const nearest = scene.fieldNearest(field);
if (nearest.length !== values.length || !nearest.every((n) => n === 0)) {
  throw new Error("expected every vertex measured to the only target");
}
const status = scene.fieldStatus(field);
if (!status.current || !status.sign.closed) throw new Error("expected a current field on a closed target");

const [contact] = scene.findContacts({ margin: 2.0, pairs: [[meshA, meshB]] });
if (!contact.intersecting || contact.distance !== 0) throw new Error("expected the boxes to intersect");
if (contact.trianglesA.length === 0 || contact.trianglesB.length === 0) {
  throw new Error("expected triangles in the contact band");
}
```

## Keep it fresh

Fields are eager by default: edits recompute them. After moving a mesh,
check `fieldStatus(field).current` before trusting colors in a stale
pipeline, and clear with `scene.setColoring(meshA, null)`.

```ts
scene.setColoring(meshA, null);
scene.free();
```

See [Custom colormap + fragment shading](custom-colormap.md) for custom
stops and per-pixel rendering of this same band.
