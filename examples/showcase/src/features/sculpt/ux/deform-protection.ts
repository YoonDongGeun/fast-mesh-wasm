// What a free-form deformation may move (#193): the `FreeformProtection` a
// solver gets from the sculpt mask, explicit vertex weights and the pin box.
// Critical Wave-6 semantics: **`weight = 0` means FROZEN and `weight = 1`
// means FULL MOVE**.
import type { FreeformProtection, MeshHandle } from "fast-mesh-wasm";
import type { Vec3Tuple } from "./ffd-lattice";
import type { SculptMask } from "./sculpt";

export interface PinBoxProtection {
  min: Vec3Tuple;
  max: Vec3Tuple;
  positions: ArrayLike<number>;
  liveIds?: ArrayLike<number>;
}

export interface ProtectionBuildInput {
  mask?: SculptMask | null;
  activeMesh?: MeshHandle | null;
  protectMask?: boolean;
  frozenVertices?: readonly number[];
  weights?: readonly number[] | Float32Array;
  pinBox?: PinBoxProtection | null;
}

/**
 * Converts a Showcase `SculptMask` on `activeMesh` into a `FreeformProtection` option.
 * Critical Wave-6 semantics: **`weight = 0` means FROZEN and `weight = 1` means FULL MOVE**.
 */
export function buildFreeformProtectionFromMask(
  mask: SculptMask | null | undefined,
  activeMesh?: MeshHandle | null,
): FreeformProtection | undefined {
  if (!mask || mask.vertices.length === 0) {
    return undefined;
  }
  if (activeMesh !== undefined && activeMesh !== null && mask.mesh !== activeMesh) {
    return undefined;
  }
  const vertices = new Uint32Array(mask.vertices);
  // 0 = completely frozen, 1 = full move.
  const weights = new Float32Array(vertices.length);
  return { vertices, weights };
}

/**
 * Builds a unified `FreeformProtection` object from sculpt mask, explicit weights, and/or pin box.
 * Critical Wave-6 semantics: **`weight = 0` means FROZEN and `weight = 1` means FULL MOVE**.
 */
export function buildProtectionOption(input: ProtectionBuildInput): FreeformProtection | undefined {
  const byVertex = new Map<number, number>();
  const setMinWeight = (v: number, w: number) => {
    const clamped = Math.max(0, Math.min(1, w));
    const prev = byVertex.get(v);
    byVertex.set(v, prev === undefined ? clamped : Math.min(prev, clamped));
  };

  if (input.protectMask !== false && input.mask) {
    const fromMask = buildFreeformProtectionFromMask(input.mask, input.activeMesh);
    if (fromMask) {
      for (let i = 0; i < fromMask.vertices.length; i++) {
        setMinWeight(fromMask.vertices[i], fromMask.weights ? fromMask.weights[i] : 0);
      }
    }
  }

  if (input.frozenVertices && input.frozenVertices.length > 0) {
    for (let i = 0; i < input.frozenVertices.length; i++) {
      const v = input.frozenVertices[i];
      if (!Number.isInteger(v) || v < 0) {
        throw new Error(`Invalid protected vertex index: ${v}`);
      }
      const w = input.weights ? Number(input.weights[i] ?? 0) : 0;
      if (!Number.isFinite(w)) {
        throw new Error(`Invalid protection weight at index ${i}: ${w}`);
      }
      setMinWeight(v, w);
    }
  }

  if (input.pinBox) {
    const { min, max, positions, liveIds } = input.pinBox;
    const minX = Math.min(min[0], max[0]);
    const maxX = Math.max(min[0], max[0]);
    const minY = Math.min(min[1], max[1]);
    const maxY = Math.max(min[1], max[1]);
    const minZ = Math.min(min[2], max[2]);
    const maxZ = Math.max(min[2], max[2]);
    const slotCount = Math.floor(positions.length / 3);
    const iterCount = liveIds ? liveIds.length : slotCount;
    for (let i = 0; i < iterCount; i++) {
      const v = liveIds ? liveIds[i] : i;
      const base = v * 3;
      if (base + 2 >= positions.length) continue;
      const x = positions[base];
      const y = positions[base + 1];
      const z = positions[base + 2];
      if (x >= minX && x <= maxX && y >= minY && y <= maxY && z >= minZ && z <= maxZ) {
        setMinWeight(v, 0);
      }
    }
  }

  if (byVertex.size === 0) {
    return undefined;
  }

  const sortedIds = Array.from(byVertex.keys()).sort((a, b) => a - b);
  const vertices = Uint32Array.from(sortedIds);
  const weights = new Float32Array(sortedIds.length);
  for (let i = 0; i < sortedIds.length; i++) {
    weights[i] = byVertex.get(sortedIds[i]) ?? 0;
  }
  return { vertices, weights };
}

/**
 * The protection a viewport drag solves under: the sculpt mask on `activeMesh`
 * alone, frozen, while the panel's "Protect sculpt mask" is on.
 */
export function dragProtection(
  protectMask: boolean,
  mask: SculptMask | null,
  activeMesh: MeshHandle,
): FreeformProtection | undefined {
  return protectMask ? buildFreeformProtectionFromMask(mask, activeMesh) : undefined;
}
