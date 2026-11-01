# Contact heatmap of two meshes

Color one mesh by its distance to another: a signed distance field plus
a `setColoring` rule. Penetration, near-contact, and far regions each
get their own color.

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
- `scene.fieldNearest(field)` → per-vertex target mesh index (which of
  `to` each vertex measured to).
- `scene.fieldStatus(field)` → `{ current, sign }`: whether the values
  match the meshes as they are now, and what the sign data found out
  (`mode`, `closed`, `consistentOrientation`, counts).
- `scene.findContacts({ margin: 2.0, pairs: [[meshA, meshB]] })` →
  per-pair `{ intersecting, distance, trianglesA, trianglesB }` for the
  triangle-level counterpart of the same band.

## Keep it fresh

Fields are eager by default: edits recompute them. After moving a mesh,
check `fieldStatus(field).current` before trusting colors in a stale
pipeline, and clear with `scene.setColoring(meshA, null)`.

See [Custom colormap + fragment shading](custom-colormap.md) for custom
stops and per-pixel rendering of this same band.
