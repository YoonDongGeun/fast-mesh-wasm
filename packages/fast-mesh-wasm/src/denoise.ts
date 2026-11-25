// Denoising and Delaunay flips: the types `Scene.denoise` and
// `Scene.delaunayFlips` (and their `WorkerScene` twins) take and return, and
// the argument checks and decoding they share.
import type { Indices } from "./api.js";
import { FastMeshError } from "./core.js";
import { normalizeU32Selection } from "./wave6.js";

function invalid(message: string): FastMeshError {
  return new FastMeshError("INVALID_ARGUMENT", message);
}

/** Options for `denoise`. */
export interface DenoiseOptions {
  /** Rounds of normal filtering and vertex fitting, 0 to 100 (default 5). 0 changes nothing. */
  iterations?: number;
  /**
   * How far each round moves the surface's normals towards their filtered
   * value, in `[0, 1]` (default 0.5). 0 changes nothing.
   */
  strength?: number;
  /**
   * The angle between two triangles' normals, in degrees in `(0, 180]`, from
   * which they no longer smooth each other (default 45): edges sharper than
   * it are kept.
   */
  featureAngle?: number;
  /**
   * Triangles to denoise, by id (default: the whole mesh). Only vertices
   * whose triangles are all listed move; an empty list changes nothing.
   */
  region?: Indices;
}

/** Options for `delaunayFlips`. */
export interface DelaunayFlipOptions {
  /**
   * How far, in degrees in `[0, 90]`, the flips may take the surface from the
   * original (default 10): the largest angle across a flipped edge, and
   * between a new triangle's normal and those of the original triangles it
   * replaces. Edges sharper than it never flip.
   */
  dihedralLimit?: number;
  /**
   * Triangles to work on, by id (default: the whole mesh). Only edges
   * between two listed triangles flip; an empty list changes nothing.
   */
  region?: Indices;
}

/** Readouts of a surface's triangle shape and smoothness, over the region when there is one. */
export interface SurfaceStats {
  /** Triangles measured. */
  triangles: number;
  /** The smallest corner angle, in degrees; 0 when there are no triangles. */
  minAngle: number;
  /**
   * The mean angle between the normals of triangles that share an edge, in
   * degrees; 0 when there are none. Noise raises it; flat faces bring it to
   * 0, apart from the edges along real features.
   */
  meanNormalDeviation: number;
}

/** What `denoise` did. */
export interface DenoiseReport {
  /** Vertices whose position changed. */
  moved: number;
  /** The largest distance a vertex moved, in the mesh's own units (before its placement's scale). */
  maxDisplacement: number;
  /** Rounds run. */
  iterations: number;
  /** Vertices left where they were because a triangle using them would have turned over. */
  reverted: number;
  /** The surface before. */
  before: SurfaceStats;
  /** The surface after. */
  after: SurfaceStats;
}

/** What `delaunayFlips` did. */
export interface DelaunayFlipReport {
  /** Edges flipped. */
  flips: number;
  /** Passes over the edges that flipped at least one. */
  passes: number;
  /** `false` when the pass limit (1000) ran out with flips left. */
  converged: boolean;
  /** The surface before. */
  before: SurfaceStats;
  /** The surface after. */
  after: SurfaceStats;
}

function checkObject(options: unknown): void {
  if (options !== undefined && (options === null || typeof options !== "object")) {
    throw invalid("options must be an object");
  }
}

/** A finite number, or nothing; the engine checks its range. */
function checkNumber(v: number | undefined, what: string): void {
  if (v !== undefined && (typeof v !== "number" || !Number.isFinite(v))) {
    throw invalid(`options.${what} must be a finite number`);
  }
}

function checkRegion(region: Indices | undefined): Uint32Array | undefined {
  return region === undefined ? undefined : normalizeU32Selection(region, "options.region");
}

/** A checked copy of the options, safe to keep and to post: the region is a fresh `Uint32Array`. */
export function checkDenoiseOptions(
  options: DenoiseOptions | undefined,
): Omit<DenoiseOptions, "region"> & { region?: Uint32Array } {
  checkObject(options);
  const { iterations, strength, featureAngle, region } = options ?? {};
  checkNumber(iterations, "iterations");
  checkNumber(strength, "strength");
  checkNumber(featureAngle, "featureAngle");
  return { iterations, strength, featureAngle, region: checkRegion(region) };
}

/** A checked copy of the options, as {@link checkDenoiseOptions}. */
export function checkDelaunayFlipOptions(
  options: DelaunayFlipOptions | undefined,
): Omit<DelaunayFlipOptions, "region"> & { region?: Uint32Array } {
  checkObject(options);
  const { dihedralLimit, region } = options ?? {};
  checkNumber(dihedralLimit, "dihedralLimit");
  return { dihedralLimit, region: checkRegion(region) };
}

function statsAt(raw: ArrayLike<number>, at: number): SurfaceStats {
  return { triangles: raw[at], minAngle: raw[at + 1], meanNormalDeviation: raw[at + 2] };
}

/** The binding's `[moved, maxDisplacement, iterations, reverted, before x 3, after x 3]`. */
export function decodeDenoiseReport(raw: ArrayLike<number>): DenoiseReport {
  return {
    moved: raw[0],
    maxDisplacement: raw[1],
    iterations: raw[2],
    reverted: raw[3],
    before: statsAt(raw, 4),
    after: statsAt(raw, 7),
  };
}

/** The binding's `[flips, passes, converged, before x 3, after x 3]`. */
export function decodeDelaunayFlipReport(raw: ArrayLike<number>): DelaunayFlipReport {
  return {
    flips: raw[0],
    passes: raw[1],
    converged: raw[2] !== 0,
    before: statsAt(raw, 3),
    after: statsAt(raw, 6),
  };
}
