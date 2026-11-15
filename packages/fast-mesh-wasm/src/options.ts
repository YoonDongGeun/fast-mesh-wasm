// Wasm-free input normalization and validation helpers shared between
// `Scene`, `WorkerScene`, domain builders, and the Worker `Engine`.
import { FastMeshError } from "./core.js";
import type { Vec3 } from "./api.js";
import type { MeshHandle } from "./handles.js";
import type {
  BooleanMethod,
  BooleanOp,
  BooleanOptions,
  CoarseAlignMethod,
  CsgNode,
  DeviationOptions,
  GeodesicMethod,
  GeodesicOptions,
  IcpObjective,
  LandmarkPair,
  OverhangOptions,
  QualityMetric,
  RegistrationOptions,
  SurfacePathOptions,
  SurfacePathPoint,
  TriangleQualityOptions,
  TrimCurveOptions,
} from "./scene.js";

function invalid(message: string): FastMeshError {
  return new FastMeshError("INVALID_ARGUMENT", message);
}

function checkFiniteNumber(v: unknown, what: string): number {
  if (typeof v !== "number" || !Number.isFinite(v)) {
    throw invalid(`${what} must be a finite number`);
  }
  return v;
}

function checkOptionalNumber(v: number | undefined, what: string, min?: number): number | undefined {
  if (v === undefined) {
    return undefined;
  }
  if (typeof v !== "number" || Number.isNaN(v)) {
    throw invalid(`${what} must be a number`);
  }
  if (min !== undefined && v < min) {
    throw invalid(`${what} must be >= ${min}`);
  }
  return v;
}

function checkOptionalPositive(v: number | undefined, what: string, inclusiveZero = false): number | undefined {
  if (v === undefined) {
    return undefined;
  }
  if (typeof v !== "number" || !Number.isFinite(v) || (inclusiveZero ? v < 0 : v <= 0)) {
    throw invalid(`${what} must be finite and ${inclusiveZero ? ">= 0" : "> 0"}`);
  }
  return v;
}

function checkOptionalInt(v: number | undefined, what: string, min: number): number | undefined {
  if (v === undefined) {
    return undefined;
  }
  if (!Number.isInteger(v) || v < min) {
    throw invalid(`${what} must be an integer >= ${min}`);
  }
  return v;
}

export function normalizeVec3(v: Readonly<Vec3>, what: string): Vec3 {
  const a = v as readonly number[] | undefined;
  if (a?.length !== 3) {
    throw invalid(`${what} must be 3 finite numbers`);
  }
  return [
    checkFiniteNumber(a[0], what),
    checkFiniteNumber(a[1], what),
    checkFiniteNumber(a[2], what),
  ];
}

export function codeIn<T extends string>(table: Record<T, number>, value: T, what: string): number {
  const code = table[value];
  if (code === undefined) {
    throw invalid(`invalid ${what} "${String(value)}"`);
  }
  return code;
}

export const BOOLEAN_OP_CODES: Record<BooleanOp, number> = {
  union: 0,
  subtract: 1,
  intersect: 2,
};

export const BOOLEAN_METHOD_CODES: Record<BooleanMethod, number> = {
  manifold: 0,
  bvhSdf: 1,
  fhEdt: 2,
};

export const QUALITY_METRIC_CODES: Record<QualityMetric, number> = {
  radiusRatio: 0,
  minAngleDeg: 1,
  aspectRatio: 2,
};

export const GEODESIC_METHOD_CODES: Record<GeodesicMethod, number> = {
  dijkstra: 0,
  heatMethod: 1,
};

export const COARSE_ALIGN_CODES: Record<CoarseAlignMethod, number> = {
  none: 0,
  pcaCentroid: 1,
  orientedBoundingBox: 2,
};

export const ICP_OBJECTIVE_CODES: Record<IcpObjective, number> = {
  pointToPoint: 0,
  pointToPlane: 1,
  symmetricNormal: 2,
};

export interface NormalizedBooleanOptions {
  op: BooleanOp;
  method: BooleanMethod;
  coplanarTolerance?: number;
  snapSlivers: boolean;
  voxelPitch?: number;
  maxTriangles?: number;
}

/** Validates and normalizes `BooleanOp` + `BooleanOptions` arguments. */
export function normalizeBooleanArgs(
  opOrOptions: BooleanOp | (BooleanOptions & { op: BooleanOp }),
  options?: Omit<BooleanOptions, "op">,
): NormalizedBooleanOptions {
  const op: BooleanOp = typeof opOrOptions === "string" ? opOrOptions : opOrOptions?.op;
  codeIn(BOOLEAN_OP_CODES, op, "op");
  const opts: BooleanOptions = typeof opOrOptions === "string" ? (options ?? {}) : opOrOptions;
  const method = opts.method ?? "manifold";
  codeIn(BOOLEAN_METHOD_CODES, method, "method");
  if (opts.snapSlivers !== undefined && typeof opts.snapSlivers !== "boolean") {
    throw invalid("snapSlivers must be a boolean");
  }
  return {
    op,
    method,
    coplanarTolerance: checkOptionalPositive(opts.coplanarTolerance, "coplanarTolerance", true),
    snapSlivers: opts.snapSlivers ?? true,
    voxelPitch: checkOptionalPositive(opts.voxelPitch, "voxelPitch", false),
    maxTriangles: checkOptionalInt(opts.maxTriangles, "maxTriangles", 1),
  };
}

/** Resolves either `(meshes, tree, options)` or handle-leaf `(tree, options)` CSG batch inputs. */
export function normalizeCsgBatchArgs(
  meshesOrTree: readonly MeshHandle[] | CsgNode,
  treeOrOptions?: CsgNode | BooleanOptions,
  maybeOptions?: BooleanOptions,
): { meshes: MeshHandle[]; tree: CsgNode; options: NormalizedBooleanOptions } {
  let explicitMeshes: readonly MeshHandle[] | undefined;
  let root: CsgNode;
  let rawOptions: BooleanOptions | undefined;

  if (Array.isArray(meshesOrTree)) {
    explicitMeshes = meshesOrTree;
    if (treeOrOptions === undefined || typeof treeOrOptions !== "object" || !("kind" in treeOrOptions)) {
      throw invalid("CSG batch requires a CsgNode tree");
    }
    root = treeOrOptions as CsgNode;
    rawOptions = maybeOptions;
  } else {
    root = meshesOrTree as CsgNode;
    rawOptions = treeOrOptions as BooleanOptions | undefined;
  }

  const normalizedOpts = normalizeBooleanArgs({ ...(rawOptions ?? {}), op: rawOptions?.op ?? "union" });
  const meshes: MeshHandle[] = explicitMeshes ? [...explicitMeshes] : [];
  const handleIndex = new Map<MeshHandle, number>();
  meshes.forEach((m, i) => {
    if (!handleIndex.has(m)) {
      handleIndex.set(m, i);
    }
  });

  const compile = (node: CsgNode): CsgNode => {
    if (node == null || typeof node !== "object") {
      throw invalid("CSG node must be an object");
    }
    if (node.kind === "leaf") {
      if ("mesh" in node && node.mesh !== undefined) {
        let idx = handleIndex.get(node.mesh);
        if (idx === undefined) {
          if (explicitMeshes !== undefined) {
            throw invalid("CSG leaf mesh handle is not present in meshes array");
          }
          idx = meshes.length;
          meshes.push(node.mesh);
          handleIndex.set(node.mesh, idx);
        }
        return { kind: "leaf", meshIndex: idx };
      }
      if (!("meshIndex" in node) || !Number.isInteger(node.meshIndex) || node.meshIndex < 0 || node.meshIndex >= meshes.length) {
        throw invalid("CSG leaf meshIndex out of range");
      }
      return { kind: "leaf", meshIndex: node.meshIndex };
    }
    if (node.kind === "op") {
      codeIn(BOOLEAN_OP_CODES, node.op, "node.op");
      return { kind: "op", op: node.op, left: compile(node.left), right: compile(node.right) };
    }
    throw invalid(`invalid CSG node kind "${String((node as { kind?: unknown }).kind)}"`);
  };

  const tree = compile(root);
  if (meshes.length === 0) {
    throw invalid("CSG batch requires at least one mesh");
  }
  return { meshes, tree, options: normalizedOpts };
}

/** Encodes a normalized `CsgNode` (index leaves only) into prefix notation for `WasmScene.boolean_batch`. */
export function encodeCsgTree(tree: CsgNode, out: number[] = []): number[] {
  if (tree.kind === "leaf") {
    if (!("meshIndex" in tree)) {
      throw invalid("CSG leaf must carry meshIndex");
    }
    out.push(0, tree.meshIndex);
    return out;
  }
  out.push(1, codeIn(BOOLEAN_OP_CODES, tree.op, "node.op"));
  encodeCsgTree(tree.left, out);
  encodeCsgTree(tree.right, out);
  return out;
}

/** Normalizes either `TrimCurveOptions` or `(points | path, options)` arguments. */
export function normalizeTrimCurveArgs(
  optionsOrCurve: TrimCurveOptions | readonly (Readonly<Vec3> | SurfacePathPoint)[],
  maybeOptions?: Omit<TrimCurveOptions, "points">,
): { points: Vec3[]; closed: boolean; deleteInside: boolean; snapTolerance?: number } {
  const opts: TrimCurveOptions = Array.isArray(optionsOrCurve)
    ? {
        ...(maybeOptions ?? {}),
        points: (optionsOrCurve as readonly (Readonly<Vec3> | SurfacePathPoint)[]).map((item) =>
          Array.isArray(item) ? (item as Readonly<Vec3>) : (item as SurfacePathPoint).position,
        ),
      }
    : (optionsOrCurve as TrimCurveOptions);

  if (!opts || !Array.isArray(opts.points) || opts.points.length < 2) {
    throw invalid("trimByCurve requires at least 2 points");
  }
  if (opts.closed !== undefined && typeof opts.closed !== "boolean") {
    throw invalid("closed must be a boolean");
  }
  if (opts.deleteInside !== undefined && typeof opts.deleteInside !== "boolean") {
    throw invalid("deleteInside must be a boolean");
  }
  return {
    points: opts.points.map((p, i) => normalizeVec3(p, `points[${i}]`)),
    closed: opts.closed ?? false,
    deleteInside: opts.deleteInside ?? true,
    snapTolerance: checkOptionalPositive(opts.snapTolerance, "snapTolerance", true),
  };
}

export function normalizeOverhangOptions(options: OverhangOptions = {}): {
  buildDirection: Vec3;
  criticalAngleDeg?: number;
} {
  const buildDirection =
    options.buildDirection === undefined
      ? ([0, 0, 1] as Vec3)
      : normalizeVec3(options.buildDirection, "buildDirection");
  if (buildDirection[0] === 0 && buildDirection[1] === 0 && buildDirection[2] === 0) {
    throw invalid("buildDirection must be non-zero");
  }
  const criticalAngleDeg = checkOptionalNumber(options.criticalAngleDeg, "criticalAngleDeg", 0);
  if (criticalAngleDeg !== undefined && criticalAngleDeg > 90) {
    throw invalid("criticalAngleDeg must be in [0, 90]");
  }
  return { buildDirection, criticalAngleDeg };
}

export function normalizeQualityArgs(
  metricOrOptions: QualityMetric | TriangleQualityOptions = "radiusRatio",
  sliverThreshold?: number,
): { metric: QualityMetric; sliverThreshold: number } {
  const metric: QualityMetric =
    typeof metricOrOptions === "string" ? metricOrOptions : (metricOrOptions.metric ?? "radiusRatio");
  codeIn(QUALITY_METRIC_CODES, metric, "metric");
  const rawThreshold = typeof metricOrOptions === "string" ? sliverThreshold : metricOrOptions.sliverThreshold;
  const fallback = metric === "aspectRatio" ? 10 : metric === "minAngleDeg" ? 10 : 0.1;
  const threshold = checkOptionalPositive(rawThreshold, "sliverThreshold", true) ?? fallback;
  return { metric, sliverThreshold: threshold };
}

export function normalizeDeviationOptions(options: DeviationOptions = {}): {
  signed: boolean;
  maxDistance?: number;
} {
  if (options.signed !== undefined && typeof options.signed !== "boolean") {
    throw invalid("signed must be a boolean");
  }
  let maxDistance = options.maxDistance;
  if (maxDistance !== undefined && maxDistance !== Number.POSITIVE_INFINITY) {
    maxDistance = checkOptionalPositive(maxDistance, "maxDistance", false);
  }
  return { signed: options.signed ?? false, maxDistance };
}

export function normalizeGeodesicOptions(options: GeodesicOptions = {}): {
  method: GeodesicMethod;
  maxDistance?: number;
  timeStepScale?: number;
} {
  const method = options.method ?? "dijkstra";
  codeIn(GEODESIC_METHOD_CODES, method, "method");
  let maxDistance = options.maxDistance;
  if (maxDistance !== undefined && maxDistance !== Number.POSITIVE_INFINITY) {
    maxDistance = checkOptionalPositive(maxDistance, "maxDistance", true);
  }
  const timeStepScale = checkOptionalPositive(options.timeStepScale, "timeStepScale", false);
  return { method, maxDistance, timeStepScale };
}

export function normalizeSourceVertices(sources: number | readonly number[]): Uint32Array {
  const arr = typeof sources === "number" ? [sources] : sources;
  if (!Array.isArray(arr) && !ArrayBuffer.isView(arr)) {
    throw invalid("sources must be a vertex index or array of vertex indices");
  }
  if (arr.length === 0) {
    throw invalid("sources must contain at least one vertex index");
  }
  const out = new Uint32Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (!Number.isInteger(v) || v < 0 || v >= 0x1_0000_0000) {
      throw invalid(`sources[${i}] must be a non-negative u32 integer`);
    }
    out[i] = v;
  }
  return out;
}

export function normalizeSurfacePathTarget(targetVertex: number): number {
  if (!Number.isInteger(targetVertex) || targetVertex < 0 || targetVertex >= 0x1_0000_0000) {
    throw invalid("targetVertex must be a non-negative u32 integer");
  }
  return targetVertex;
}

export function checkSurfacePathOptions(options: SurfacePathOptions = {}): {
  distances?: Float32Array;
  geodesic: ReturnType<typeof normalizeGeodesicOptions>;
} {
  if (options.distances !== undefined && !(options.distances instanceof Float32Array)) {
    throw invalid("distances must be a Float32Array");
  }
  return {
    distances: options.distances,
    geodesic: normalizeGeodesicOptions(options),
  };
}

export function normalizeLandmarkPairs(
  sameMesh: boolean,
  pairs: readonly LandmarkPair[],
  allowScaling = false,
): { sourcePoints: Float64Array; targetPoints: Float64Array; allowScaling: boolean } {
  if (sameMesh) {
    throw invalid("cannot align a mesh to itself");
  }
  if (typeof allowScaling !== "boolean") {
    throw invalid("allowScaling must be a boolean");
  }
  if (!Array.isArray(pairs) || pairs.length < 3) {
    throw invalid("alignByLandmarks requires at least 3 landmark pairs");
  }
  const sourcePoints = new Float64Array(pairs.length * 3);
  const targetPoints = new Float64Array(pairs.length * 3);
  for (let i = 0; i < pairs.length; i++) {
    const pair = pairs[i];
    if (!pair || typeof pair !== "object") {
      throw invalid(`pairs[${i}] must be an object with source and target`);
    }
    const s = normalizeVec3(pair.source, `pairs[${i}].source`);
    const t = normalizeVec3(pair.target, `pairs[${i}].target`);
    sourcePoints.set(s, i * 3);
    targetPoints.set(t, i * 3);
  }
  return { sourcePoints, targetPoints, allowScaling };
}

export function normalizeRegistrationOptions(sameMesh: boolean, options: RegistrationOptions = {}): {
  coarseMethod: CoarseAlignMethod;
  objective: IcpObjective;
  maxIterations?: number;
  tolerance?: number;
  overlap?: number;
  maxDistance?: number;
  targetRmse?: number;
  sampleCount?: number;
  andersonWindow?: number;
} {
  if (sameMesh) {
    throw invalid("cannot register a mesh to itself");
  }
  const coarseMethod = options.coarseMethod ?? "none";
  codeIn(COARSE_ALIGN_CODES, coarseMethod, "coarseMethod");
  const objective = options.objective ?? "pointToPlane";
  codeIn(ICP_OBJECTIVE_CODES, objective, "objective");
  const overlap = checkOptionalNumber(options.overlap, "overlap");
  if (overlap !== undefined && !(overlap > 0 && overlap <= 1)) {
    throw invalid("overlap must be in (0, 1]");
  }
  return {
    coarseMethod,
    objective,
    maxIterations: checkOptionalInt(options.maxIterations, "maxIterations", 0),
    tolerance: checkOptionalPositive(options.tolerance, "tolerance", true),
    overlap,
    maxDistance: checkOptionalPositive(options.maxDistance, "maxDistance", false),
    targetRmse: checkOptionalPositive(options.targetRmse, "targetRmse", true),
    sampleCount: checkOptionalInt(options.sampleCount, "sampleCount", 1),
    andersonWindow: checkOptionalInt(options.andersonWindow, "andersonWindow", 0),
  };
}
