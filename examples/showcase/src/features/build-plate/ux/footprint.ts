// What a mesh takes up on the build plate (#266): the world box around the
// vertices its live triangles use, as it is placed now. The box is read from
// the geometry itself, so a turned mesh has its own projection, not the box
// around its turned local box. Read in slices that yield to the event loop:
// a scene of many large meshes stays interactive while it runs.
import type { MeshHandle, SceneLike, Vec3 } from "fast-mesh-wasm";
import { type Pause, yieldToEventLoop } from "../../select/ux/face-criteria";
import type { WorldBounds } from "../../transform/ux/placement-math";

/** The scene calls a footprint is read through. */
export type FootprintScene = Pick<SceneLike, "views" | "matrix4" | "onChanges">;

/** How long one slice runs before it yields, in milliseconds. */
const SLICE_BUDGET_MS = 8;
/** Triangles, or vertices, between two looks at the clock. */
const CHUNK = 1 << 16;
/** How often a mesh that changes while it is read is read again. */
const MAX_PASSES = 4;

/** Runs steps in slices of {@link SLICE_BUDGET_MS}, over all the meshes of one read. */
class Slices {
  #start = performance.now();
  readonly #pause: Pause;

  constructor(pause: Pause) {
    this.#pause = pause;
  }

  /** Whether the slice's time is used up; when it is, waits for the event loop. */
  async next(): Promise<boolean> {
    if (performance.now() - this.#start < SLICE_BUDGET_MS) {
      return false;
    }
    await this.#pause();
    this.#start = performance.now();
    return true;
  }
}

/**
 * One mesh's world box at `matrix`, `null` without live triangles. Fetches
 * the views again after every pause (an engine call in between may have
 * moved them); `undefined` when their size changed meanwhile.
 */
async function readBox(
  scene: FootprintScene,
  mesh: MeshHandle,
  m: ArrayLike<number>,
  slices: Slices,
): Promise<WorldBounds | null | undefined> {
  const views = () => {
    const v = scene.views(mesh);
    return { positions: v.positions(), indices: v.indices() };
  };
  let { positions, indices } = views();
  const vertexCount = Math.floor(positions.length / 3);
  const triangleCount = Math.floor(indices.length / 3);
  const refetched = () => {
    ({ positions, indices } = views());
    return positions.length >= 3 * vertexCount && Math.floor(indices.length / 3) === triangleCount;
  };

  const used = new Uint8Array(vertexCount);
  for (let t = 0; t < triangleCount; ) {
    for (const end = Math.min(t + CHUNK, triangleCount); t < end; t++) {
      const a = indices[3 * t];
      const b = indices[3 * t + 1];
      const c = indices[3 * t + 2];
      // A removed triangle slot reads `0, 0, 0`.
      if (a !== 0 || b !== 0 || c !== 0) {
        used[a] = 1;
        used[b] = 1;
        used[c] = 1;
      }
    }
    if ((await slices.next()) && !refetched()) {
      return undefined;
    }
  }

  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let v = 0; v < vertexCount; ) {
    for (const end = Math.min(v + CHUNK, vertexCount); v < end; v++) {
      if (used[v] === 0) {
        continue;
      }
      const x = positions[3 * v];
      const y = positions[3 * v + 1];
      const z = positions[3 * v + 2];
      const wx = m[0] * x + m[4] * y + m[8] * z + m[12];
      const wy = m[1] * x + m[5] * y + m[9] * z + m[13];
      const wz = m[2] * x + m[6] * y + m[10] * z + m[14];
      if (!Number.isFinite(wx + wy + wz)) {
        continue;
      }
      if (wx < min[0]) min[0] = wx;
      if (wx > max[0]) max[0] = wx;
      if (wy < min[1]) min[1] = wy;
      if (wy > max[1]) max[1] = wy;
      if (wz < min[2]) min[2] = wz;
      if (wz > max[2]) max[2] = wz;
    }
    if ((await slices.next()) && !refetched()) {
      return undefined;
    }
  }
  return min[0] <= max[0] ? { min, max } : null;
}

function sameMatrix(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
  for (let k = 0; k < 16; k++) {
    if (a[k] !== b[k]) {
      return false;
    }
  }
  return true;
}

/**
 * The world boxes of `meshes` as they are placed when this resolves, `null`
 * for one without live triangles. A mesh that is moved or edited while the
 * boxes are read is read again.
 *
 * @throws when the meshes keep changing.
 */
export async function readWorldBoxes(
  scene: FootprintScene,
  meshes: readonly MeshHandle[],
  pause: Pause = yieldToEventLoop,
): Promise<Map<MeshHandle, WorldBounds | null>> {
  const boxes = new Map<MeshHandle, WorldBounds | null>();
  const matrices = new Map<MeshHandle, Float64Array>();
  const stale = new Set(meshes);
  const watched = new Set(meshes);
  const slices = new Slices(pause);
  // Edits show in the change records; moves in the matrices, compared below.
  const off = scene.onChanges((changes) => {
    for (const c of changes) {
      if (watched.has(c.mesh) && !c.removed && (c.positions.length > 0 || c.indices.length > 0)) {
        stale.add(c.mesh);
      }
    }
  });
  try {
    for (let pass = 0; stale.size > 0; pass++) {
      if (pass === MAX_PASSES) {
        throw new Error("The meshes changed while they were read. Try again once they are still.");
      }
      for (const mesh of [...stale]) {
        stale.delete(mesh);
        const matrix = scene.matrix4(mesh).slice();
        const box = await readBox(scene, mesh, matrix, slices);
        if (box === undefined) {
          stale.add(mesh);
        } else {
          boxes.set(mesh, box);
          matrices.set(mesh, matrix);
        }
      }
      for (const [mesh, matrix] of matrices) {
        if (!sameMatrix(matrix, scene.matrix4(mesh))) {
          stale.add(mesh);
        }
      }
    }
  } finally {
    off();
  }
  return boxes;
}
