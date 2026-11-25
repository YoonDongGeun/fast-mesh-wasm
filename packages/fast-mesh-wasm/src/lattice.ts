// Lattice infill: the types `Scene.latticeFill`, `Scene.latticePlan` and
// `Scene.latticePreview` (and their `WorkerScene` twins) take and return, and
// the argument checks and decoding they share.
import { FastMeshError } from "./core.js";

/** The default `maxVoxels` of a lattice: 256 x 256 x 256. */
export const DEFAULT_LATTICE_MAX_VOXELS = 16_777_216;
/** The default `maxVoxels` of a lattice preview: a million. */
export const DEFAULT_LATTICE_PREVIEW_VOXELS = 1_000_000;

function invalid(message: string): FastMeshError {
  return new FastMeshError("INVALID_ARGUMENT", message);
}

/**
 * The surface a lattice's walls follow:
 * - `"gyroid"`: no straight lines and no mirror planes, about as stiff in
 *   every direction.
 * - `"schwarzP"`: a cubic grid of round chambers joined by necks.
 * - `"diamond"`: the most wall of the three for the same cell size and wall
 *   thickness.
 */
export type LatticeKind = "gyroid" | "schwarzP" | "diamond";

/**
 * What a lattice fills:
 * - `"inside"`: the inside of the mesh, its material.
 * - `"cavity"`: the cavities the mesh encloses, such as that of a hollowed
 *   mesh. A cavity that a hole opens to the outside is not enclosed: fill it
 *   before making the hole.
 * - `"auto"`: the cavities when they make up at least a tenth of what the
 *   surfaces around them enclose (a hollowed mesh), else the inside: a few
 *   bubbles do not make a mesh hollow.
 */
export type LatticeRegion = "auto" | "inside" | "cavity";

/** Options for `latticeFill`, `latticePlan` and `latticePreview`; lengths in world units. */
export interface LatticeOptions {
  /** The surface the walls follow (default `"gyroid"`). */
  kind?: LatticeKind;
  /** The edge length of one cell. Positive and finite. */
  cellSize: number;
  /** The thickness of the walls. Positive, and at most a third of `cellSize`. */
  wallThickness: number;
  /**
   * How thick a solid shell is left inside the region's boundary before the
   * lattice starts; not negative, default `0`. With a shell the mesh's own
   * surface is kept as it is; without one, filling the inside turns the whole
   * mesh into an open lattice. In a cavity the shell comes on top of the wall
   * that is there.
   */
  shellThickness?: number;
  /** What is filled (default `"auto"`). */
  region?: LatticeRegion;
  /**
   * The voxel size. Without it, it follows from the lengths: 2.5 voxels to a
   * wall, at least 16 to a cell and 2 to a shell. At least two voxels must
   * make a wall and the shell of an inside, and eight a cell.
   */
  voxelSize?: number;
  /**
   * The largest voxel grid accepted, a whole number of at least 64. For
   * `latticeFill` and `latticePlan`, default
   * {@link DEFAULT_LATTICE_MAX_VOXELS}: a fill over it is refused before
   * anything is allocated. For `latticePreview`, default
   * {@link DEFAULT_LATTICE_PREVIEW_VOXELS}: the preview gets coarser to fit.
   */
  maxVoxels?: number;
  /**
   * Connected pieces of the result that enclose less than this share of a
   * cell's volume are left out: loose bits of wall at the surface of an open
   * lattice, and pockets too small to matter in a shell. Not negative,
   * default `0.01`.
   */
  minFragment?: number;
}

/** The voxel grid of a lattice and what to expect of it, known before it runs. */
export interface LatticePlan {
  /** What is filled. */
  region: "inside" | "cavity";
  /** Grid nodes along x, y and z. */
  dims: [number, number, number];
  /** The voxel size in world units. */
  voxelSize: number;
  /** The thickness of the walls: as asked for, except in a preview on a grid too coarse for it. */
  wallThickness: number;
  /** Grid nodes in all. */
  voxels: number;
  /** Estimated triangles of the result. */
  triangles: number;
  /**
   * The estimated peak memory of the grid, the extraction and the result, in
   * bytes. The mesh and its search tree come on top.
   */
  bytes: number;
  /** The limit the plan was checked against. */
  maxVoxels: number;
  /** Whether `voxels <= maxVoxels`; `latticeFill` refuses the plan otherwise. */
  fits: boolean;
  /** The share of space the walls fill, without any shell, from 0 to 1. */
  solidFraction: number;
  /** The world volume of the region that is filled. */
  regionVolume: number;
  /** Whether the grid covers only the middle of the mesh (a preview of many small cells). */
  cropped: boolean;
}

/** What `latticeFill` did. */
export interface LatticeReport {
  /** Vertices added. */
  vertices: number;
  /** Triangles added. */
  triangles: number;
  /** Loose pieces left out (see `minFragment`). */
  fragments: number;
  /**
   * The world volume of material in the region now, walls and shell. Divided
   * by `plan.regionVolume` it is the share that is left.
   */
  solidVolume: number;
  /** Whether the mesh's UVs were dropped (undo restores them). */
  uvsDropped: boolean;
  /** The voxel grid used. */
  plan: LatticePlan;
}

/** What `latticePreview` built. */
export interface LatticePreview {
  /** The walls' positions (xyz per vertex), in the mesh's local frame. */
  positions: Float32Array;
  /** The walls' triangles (3 vertices each), closed and wound outward; empty without walls. */
  indices: Uint32Array;
  /** The grid used, with the thickness the walls are drawn with. */
  plan: LatticePlan;
}

/** Internal: {@link LatticeOptions} with every default filled in. */
export type CheckedLatticeOptions = Required<Omit<LatticeOptions, "voxelSize">> & Pick<LatticeOptions, "voxelSize">;

const KINDS: Record<LatticeKind, number> = { gyroid: 0, schwarzP: 1, diamond: 2 };
const REGIONS: Record<LatticeRegion, number> = { auto: 0, inside: 1, cavity: 2 };
const PLAN_VALUES = 14;

function oneOf<K extends string>(codes: Record<K, number>, value: unknown, name: string, fallback: K): K {
  const v = value ?? fallback;
  if (typeof v !== "string" || !Object.hasOwn(codes, v)) {
    throw invalid(`options.${name} must be one of ${Object.keys(codes).join(", ")}`);
  }
  return v as K;
}

function length(value: unknown, name: string, fallback: number | undefined, zero: boolean): number {
  const v = value ?? fallback;
  if (typeof v !== "number" || !Number.isFinite(v) || v < 0 || (!zero && v === 0)) {
    throw invalid(`options.${name} must be a finite number ${zero ? "and not negative" : "above 0"}`);
  }
  return v;
}

/** Internal: the options with their defaults, checked; a plain object, safe to post to a Worker. */
export function checkLatticeOptions(options: LatticeOptions, defaultVoxels: number): CheckedLatticeOptions {
  if (options === null || typeof options !== "object") {
    throw invalid("options must be an object");
  }
  const cellSize = length(options.cellSize, "cellSize", undefined, false);
  const wallThickness = length(options.wallThickness, "wallThickness", undefined, false);
  const shellThickness = length(options.shellThickness, "shellThickness", 0, true);
  const minFragment = length(options.minFragment, "minFragment", 0.01, true);
  const voxelSize =
    options.voxelSize === undefined ? undefined : length(options.voxelSize, "voxelSize", undefined, false);
  const maxVoxels = options.maxVoxels ?? defaultVoxels;
  if (!Number.isInteger(maxVoxels) || maxVoxels < 64 || maxVoxels > 0xffff_ffff) {
    throw invalid("options.maxVoxels must be a whole number from 64 to 4294967295");
  }
  return {
    kind: oneOf(KINDS, options.kind, "kind", "gyroid"),
    cellSize,
    wallThickness,
    shellThickness,
    region: oneOf(REGIONS, options.region, "region", "auto"),
    ...(voxelSize === undefined ? {} : { voxelSize }),
    maxVoxels,
    minFragment,
  };
}

/** Internal: checked options as the binding takes them. */
export function encodeLatticeOptions(o: CheckedLatticeOptions): Float64Array {
  return Float64Array.of(
    KINDS[o.kind],
    o.cellSize,
    o.wallThickness,
    o.shellThickness,
    REGIONS[o.region],
    o.voxelSize ?? 0,
    o.maxVoxels,
    o.minFragment,
  );
}

/** Internal: a plan from the binding's values. */
export function decodeLatticePlan(raw: ArrayLike<number>, at = 0): LatticePlan {
  return {
    region: raw[at] === 2 ? "cavity" : "inside",
    dims: [raw[at + 1], raw[at + 2], raw[at + 3]],
    voxelSize: raw[at + 4],
    wallThickness: raw[at + 5],
    voxels: raw[at + 6],
    triangles: raw[at + 7],
    bytes: raw[at + 8],
    maxVoxels: raw[at + 9],
    fits: raw[at + 10] !== 0,
    solidFraction: raw[at + 11],
    regionVolume: raw[at + 12],
    cropped: raw[at + 13] !== 0,
  };
}

/** Internal: the report in a binding's `lattice_fill` values. */
export function decodeLatticeReport(raw: ArrayLike<number>): LatticeReport {
  return {
    vertices: raw[0],
    triangles: raw[1],
    fragments: raw[2],
    solidVolume: raw[3],
    uvsDropped: raw[4] !== 0,
    plan: decodeLatticePlan(raw, 5),
  };
}

/** Internal: the preview in a binding's `lattice_preview` words. */
export function decodeLatticePreview(raw: Uint32Array): LatticePreview {
  const v = raw[0];
  const t = raw[1];
  const head = 2 + 2 * PLAN_VALUES;
  const bytes = (from: number, to: number): ArrayBuffer => {
    const out = new Uint8Array(4 * (to - from));
    out.set(new Uint8Array(raw.buffer, raw.byteOffset + 4 * from, out.byteLength));
    return out.buffer;
  };
  return {
    positions: new Float32Array(bytes(head, head + 3 * v)),
    indices: raw.slice(head + 3 * v, head + 3 * v + 3 * t),
    plan: decodeLatticePlan(new Float64Array(bytes(2, head))),
  };
}
