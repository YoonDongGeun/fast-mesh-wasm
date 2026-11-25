// Where a mesh is, for what is centred on it. A mesh has an `f64` origin
// (`Item.center`): the point subtracted from its positions before they are
// stored as `f32`. It is part of the mesh's own coordinates, before the
// placement, and `scene.matrix4` already has it in: that takes local
// positions, relative to the origin, to the world. So `matrix4` times
// `Item.center` counts the origin twice; `worldCenter` is the mesh's middle.
import type { MeshHandle, SceneLike, Vec3 } from "fast-mesh-wasm";

/** The mesh's origin (`[0, 0, 0]` on a scene that does not tell). */
export function meshOrigin(scene: object, mesh: MeshHandle): Vec3 {
  const { origin } = scene as { origin?: (mesh: MeshHandle) => Readonly<Vec3> };
  const o = origin?.call(scene, mesh) ?? [0, 0, 0];
  return [o[0], o[1], o[2]];
}

/**
 * The centre of the mesh's bounding box in local coordinates, as it is now
 * (edits move it): that of a loaded mesh starts at its origin, `[0, 0, 0]`.
 * Also that point for a mesh without triangles, or with none mirrored yet.
 */
export function localCenter(scene: Pick<SceneLike, "bounds">, mesh: MeshHandle): Vec3 {
  const c = scene.bounds?.(mesh)?.center ?? [0, 0, 0];
  return [c[0], c[1], c[2]];
}

/**
 * Where {@link localCenter} is in the world, through the placement.
 *
 * @throws when the mesh is gone.
 */
export function worldCenter(scene: Pick<SceneLike, "bounds" | "matrix4">, mesh: MeshHandle): Vec3 {
  const m = scene.matrix4(mesh);
  const [x, y, z] = localCenter(scene, mesh);
  return [
    m[0] * x + m[4] * y + m[8] * z + m[12],
    m[1] * x + m[5] * y + m[9] * z + m[13],
    m[2] * x + m[6] * y + m[10] * z + m[14],
  ];
}
