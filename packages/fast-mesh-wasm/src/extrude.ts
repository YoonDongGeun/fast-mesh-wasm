// Extrusions: the types `Scene.extrudeFaces` and `Scene.extrudeBase` (and
// their `WorkerScene` twins) take and return, and the argument checks and
// decoding they share.
import type { Vec3 } from "./api.js";
import { FastMeshError } from "./core.js";
import type { LoopId } from "./scene.js";

function invalid(message: string): FastMeshError {
  return new FastMeshError("INVALID_ARGUMENT", message);
}

/**
 * Which way `extrudeFaces` moves the selection:
 * - `"region"`: each region of the selection (its triangles connected through
 *   shared edges) moves as one rigid piece along its area-weighted average
 *   normal. A flat region gains exactly its area times the distance in volume.
 * - `"vertex"`: each vertex moves along the average normal of its selected
 *   triangles, so the faces follow the surface like an offset.
 * - a world vector (any length): everything moves along it.
 */
export type ExtrudeDirection = "region" | "vertex" | Readonly<Vec3>;

/** Options for `extrudeFaces`. */
export interface ExtrudeFacesOptions {
  /** How far the faces move, in world units; negative pushes them in. Finite and not zero. */
  distance: number;
  /** Which way they move (default `"region"`). */
  direction?: ExtrudeDirection;
  /** Look for self-intersections the result would have (default `true`). */
  checkSelfIntersections?: boolean;
}

/** What `extrudeFaces` did. */
export interface ExtrudeFacesResult {
  /** Regions of the selection, each extruded as one piece with walls of its own. */
  regions: number;
  /** Selected triangles, each counted once. They keep their slots and are the moved faces. */
  capTriangles: number;
  /** Vertices that moved in place: those inside the selection. */
  movedVertices: number;
  /** New vertices: copies of the vertices on the selection's boundary. */
  addedVertices: number;
  /** Triangle slots of the new side walls, ascending. */
  wallTriangles: Uint32Array;
  /**
   * Triangle slots of every pair that touches, crosses or folds flat after
   * the extrusion with a moved or new triangle among them, ascending; `null`
   * with `checkSelfIntersections: false`.
   */
  selfIntersections: Uint32Array | null;
}

/**
 * Options for `extrudeBase`. Give the base plane's place as `height` or as
 * `offset`, not both.
 */
export interface ExtrudeBaseOptions {
  /** The world direction the walls run in, any length (default `[0, -1, 0]`, down with y up). */
  direction?: Readonly<Vec3>;
  /**
   * The base plane's world height, measured from the world origin along the
   * opposite of `direction`: with the default direction, the plane `y = height`.
   */
  height?: number;
  /** How far the base plane is past the mesh's furthest vertex along `direction`, in world units. */
  offset?: number;
  /**
   * The boundary loops to extrude: `"largest"` (default) is the one with the
   * longest perimeter in the world, `"all"` every one, where outlines nested on the plane
   * become holes in the cap, or the ids of some, from `analyze(mesh).loops`.
   */
  loops?: "largest" | "all" | readonly LoopId[];
  /** Refine each cap to the density of its walls, up to 20,000 new vertices (default `true`). */
  refine?: boolean;
  /** Look for self-intersections the result would have (default `true`). */
  checkSelfIntersections?: boolean;
}

/** What `extrudeBase` did. */
export interface ExtrudeBaseResult {
  /** The loops extruded, as they were before. */
  loops: LoopId[];
  /** The base plane's world height, as {@link ExtrudeBaseOptions.height} measures it. */
  height: number;
  /** New vertices on the plane's outlines. */
  baseVertices: number;
  /** New vertices inside the caps. */
  capVertices: number;
  /** New side wall triangles. */
  wallTriangles: number;
  /** New cap triangles. */
  capTriangles: number;
  /** Whether a cap's refinement stopped at its vertex budget. */
  capped: boolean;
  /**
   * Whether a surface was turned around: it faced the way the walls run, so
   * its solid would have been inside out.
   */
  flipped: boolean;
  /**
   * Triangle slots of every pair that touches, crosses or folds flat after
   * the extrusion with a new triangle among them, ascending; `null` with
   * `checkSelfIntersections: false`.
   */
  selfIntersections: Uint32Array | null;
}

const NO_VEC3 = new Float64Array(0);

function vec3(value: unknown, what: string): Float64Array {
  const v = value as ArrayLike<unknown> | null | undefined;
  const out = new Float64Array(3);
  if (v === null || typeof v !== "object" || v.length !== 3) {
    throw invalid(`${what} must be 3 finite numbers`);
  }
  for (let k = 0; k < 3; k++) {
    const c = v[k];
    if (typeof c !== "number" || !Number.isFinite(c)) {
      throw invalid(`${what} must be 3 finite numbers`);
    }
    out[k] = c;
  }
  return out;
}

function flag(value: unknown, what: string): boolean {
  if (value === undefined) {
    return true;
  }
  if (typeof value !== "boolean") {
    throw invalid(`${what} must be a boolean`);
  }
  return value;
}

function finite(value: unknown, what: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw invalid(`${what} must be a finite number`);
  }
  return value;
}

function checkOptions(options: unknown): void {
  if (options === null || typeof options !== "object") {
    throw invalid("options must be an object");
  }
}

/** The binding's arguments for `extrudeFaces`, checked. */
export function extrudeFacesArgs(options: ExtrudeFacesOptions): {
  distance: number;
  mode: number;
  direction: Float64Array;
  check: boolean;
} {
  checkOptions(options);
  const distance = finite(options.distance, "options.distance");
  if (distance === 0) {
    throw invalid("options.distance must not be 0");
  }
  const check = flag(options.checkSelfIntersections, "options.checkSelfIntersections");
  const way = options.direction ?? "region";
  if (way === "region" || way === "vertex") {
    return { distance, mode: way === "region" ? 0 : 1, direction: NO_VEC3, check };
  }
  if (typeof way === "string") {
    throw invalid('options.direction must be "region", "vertex" or 3 finite numbers');
  }
  return { distance, mode: 2, direction: vec3(way, "options.direction"), check };
}

const LOOP_CODES: Record<"largest" | "all", number> = { largest: 0, all: 1 };
const U32_LIMIT = 0x1_0000_0000;

/** Loop ids as the binding's words: `minVertex, len, hash low, hash high` each. */
function loopWords(ids: readonly LoopId[]): Uint32Array {
  const out = new Uint32Array(4 * ids.length);
  ids.forEach((id, i) => {
    const { minVertex, len, hash } = id ?? ({} as LoopId);
    if (typeof hash !== "bigint" || hash < 0n || hash >= 1n << 64n) {
      throw invalid("a loop id's hash must be a bigint in 0..2^64");
    }
    for (const v of [minVertex, len]) {
      if (!Number.isInteger(v) || v < 0 || v >= U32_LIMIT) {
        throw invalid("a loop id's minVertex and len must be u32 integers");
      }
    }
    out.set([minVertex, len, Number(hash & 0xffffffffn), Number(hash >> 32n)], 4 * i);
  });
  return out;
}

/** The binding's arguments for `extrudeBase`, checked. */
export function extrudeBaseArgs(options: ExtrudeBaseOptions): {
  direction: Float64Array;
  plane: number;
  value: number;
  which: number;
  loops: Uint32Array;
  refine: boolean;
  check: boolean;
} {
  checkOptions(options);
  const { height, offset } = options;
  if ((height === undefined) === (offset === undefined)) {
    throw invalid("give options.height or options.offset, not both");
  }
  const plane = height !== undefined ? 0 : 1;
  const value = plane === 0 ? finite(height, "options.height") : finite(offset, "options.offset");
  const loops = options.loops ?? "largest";
  if (typeof loops === "string" ? !Object.hasOwn(LOOP_CODES, loops) : !Array.isArray(loops)) {
    throw invalid('options.loops must be "largest", "all" or a list of loop ids');
  }
  return {
    direction: vec3(options.direction ?? [0, -1, 0], "options.direction"),
    plane,
    value,
    which: typeof loops === "string" ? LOOP_CODES[loops] : 2,
    loops: loopWords(typeof loops === "string" ? [] : loops),
    refine: flag(options.refine, "options.refine"),
    check: flag(options.checkSelfIntersections, "options.checkSelfIntersections"),
  };
}

/** The result in a binding's `extrude_faces` words. */
export function decodeExtrudeFaces(raw: Uint32Array): ExtrudeFacesResult {
  const walls = 6 + raw[5];
  return {
    regions: raw[0],
    capTriangles: raw[1],
    movedVertices: raw[2],
    addedVertices: raw[3],
    wallTriangles: raw.slice(6, walls),
    selfIntersections: raw[4] !== 0 ? raw.slice(walls) : null,
  };
}

/** The result in a binding's `extrude_base` values. */
export function decodeExtrudeBase(raw: Float64Array): ExtrudeBaseResult {
  const loops: LoopId[] = [];
  let at = 9;
  for (let i = 0; i < raw[8]; i++, at += 4) {
    const hash = BigInt(raw[at + 2]) | (BigInt(raw[at + 3]) << 32n);
    loops.push({ minVertex: raw[at], len: raw[at + 1], hash });
  }
  return {
    loops,
    height: raw[0],
    baseVertices: raw[1],
    capVertices: raw[2],
    wallTriangles: raw[3],
    capTriangles: raw[4],
    capped: raw[5] !== 0,
    flipped: raw[6] !== 0,
    selfIntersections: raw[7] !== 0 ? Uint32Array.from(raw.subarray(at)) : null,
  };
}
