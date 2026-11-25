// What the overlays draw around. Buffers on the GPU are `f32`, which far from
// the world origin holds coordinates only to its steps there (0.008 at 1e5);
// an object's position is not, so what is far away is drawn as offsets from a
// point near it, with the object there.
import type { Vec3 } from "fast-mesh-wasm";

/** The spacing of the points to draw around: offsets stay below half of it, plus what is drawn. */
const STEP = 1024;

/** A point near `p` to draw around: the world origin for what is near that, so it is drawn at its coordinates. */
export function drawOrigin(p: Readonly<Vec3>): Vec3 {
  return [0, 1, 2].map((k) => Math.round(p[k] / STEP) * STEP + 0) as Vec3;
}
