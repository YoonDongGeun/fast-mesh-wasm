// Wave 6 (#233, #193, #194, #234, #236) Wasm-free types, option normalizers,
// and flat-slice codecs shared across Scene, WorkerScene, domain facades,
// stores, and entry points.
import type { Indices, Placement, PlacementInput, Positions, Quat, Vec3 } from "./api.js";
import { FastMeshError } from "./core.js";
import type { MeshHandle } from "./handles.js";
import { rotationMatrix } from "./handles.js";
import {
  BOOLEAN_METHOD_CODES,
  checkU32Int,
  codeIn,
  normalizeU32List,
  normalizeVec3,
} from "./options.js";
import type { BooleanMethod } from "./scene.js";

function invalid(message: string): FastMeshError {
  return new FastMeshError("INVALID_ARGUMENT", message);
}

function checkFinite(value: number, what: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw invalid(`${what} must be a finite number`);
  }
  return value;
}

function checkPositive(value: number, what: string): number {
  const v = checkFinite(value, what);
  if (v <= 0) {
    throw invalid(`${what} must be positive`);
  }
  return v;
}

function checkNonNegative(value: number, what: string): number {
  const v = checkFinite(value, what);
  if (v < 0) {
    throw invalid(`${what} must be non-negative`);
  }
  return v;
}

function checkBoolean(value: boolean | undefined, fallback: boolean, what: string): boolean {
  if (value === undefined) {
    return fallback;
  }
  if (typeof value !== "boolean") {
    throw invalid(`${what} must be a boolean`);
  }
  return value;
}

function toF32Array(values: Float32Array | ArrayLike<number>, what: string): Float32Array {
  const out = values instanceof Float32Array ? values : Float32Array.from(values);
  for (let i = 0; i < out.length; i++) {
    if (!Number.isFinite(out[i])) {
      throw invalid(`${what}[${i}] must be finite`);
    }
  }
  return out;
}

// ============================================================================
// #233 Face Selection Types & Codecs
// ============================================================================

/** Screen-space face selection shape in normalized device coordinates (`[-1, 1]`). */
export type ScreenSelectionShape =
  | {
      kind: "rect";
      minNdc: readonly [number, number];
      maxNdc: readonly [number, number];
    }
  | {
      kind: "lasso";
      polygonNdc: readonly (readonly [number, number])[] | Float32Array | Float64Array;
    };

/** World-space camera and visibility options for {@link Scene.selectFacesScreen}. */
export interface ScreenSelectOptions {
  /** Column-major 4×4 view-projection matrix (16 finite numbers). */
  viewProj: ReadonlyArray<number> | Float32Array | Float64Array;
  cameraOrigin: Readonly<Vec3>;
  cameraForward: Readonly<Vec3>;
  isOrthographic?: boolean;
  visibleOnly?: boolean;
  rejectBackfaces?: boolean;
  includeTouching?: boolean;
}

/** Optional normal cone filter for spherical face selection. */
export interface FaceSelectionNormalCone {
  normal: Readonly<Vec3>;
  angleRad: number;
}

/** Topological adjacency rule for growing/shrinking face selections. */
export type AdjacencyKind = "edge" | "vertex";

/** Summary of a destructive face deletion or split edit. */
export interface FaceEditReport {
  removedTriangles: number;
  removedVertices: number;
}

/** Result of splitting selected faces off into a detached scene mesh. */
export interface SplitFacesReport {
  detached: MeshHandle;
  removedTriangles: number;
  removedVertices: number;
}

export interface NormalizedScreenSelectArgs {
  shapeKind: number;
  shapeData: Float64Array;
  viewProj: Float64Array;
  cameraOrigin: Vec3;
  cameraForward: Vec3;
  isOrthographic: boolean;
  visibleOnly: boolean;
  rejectBackfaces: boolean;
  includeTouching: boolean;
}

export const ADJACENCY_CODES: Record<AdjacencyKind, number> = {
  edge: 0,
  vertex: 1,
};

export function normalizeScreenSelectArgs(
  shape: ScreenSelectionShape,
  options: ScreenSelectOptions,
): NormalizedScreenSelectArgs {
  if (shape == null || typeof shape !== "object") {
    throw invalid("shape must be an object");
  }
  let shapeKind: number;
  let shapeData: Float64Array;
  if (shape.kind === "rect") {
    if (shape.minNdc?.length !== 2 || shape.maxNdc?.length !== 2) {
      throw invalid("rect shape requires 2-element minNdc and maxNdc");
    }
    shapeKind = 0;
    shapeData = Float64Array.from([
      checkFinite(shape.minNdc[0], "minNdc[0]"),
      checkFinite(shape.minNdc[1], "minNdc[1]"),
      checkFinite(shape.maxNdc[0], "maxNdc[0]"),
      checkFinite(shape.maxNdc[1], "maxNdc[1]"),
    ]);
  } else if (shape.kind === "lasso") {
    shapeKind = 1;
    const poly = shape.polygonNdc;
    if (poly instanceof Float32Array || poly instanceof Float64Array) {
      if (poly.length < 6 || poly.length % 2 !== 0) {
        throw invalid("lasso polygonNdc must contain at least 3 (x, y) pairs");
      }
      shapeData = Float64Array.from(poly);
      for (let i = 0; i < shapeData.length; i++) {
        checkFinite(shapeData[i], `polygonNdc[${i}]`);
      }
    } else if (Array.isArray(poly)) {
      if (poly.length < 3) {
        throw invalid("lasso polygonNdc requires at least 3 vertices");
      }
      shapeData = new Float64Array(poly.length * 2);
      for (let i = 0; i < poly.length; i++) {
        const pt = poly[i];
        if (!Array.isArray(pt) || pt.length !== 2) {
          throw invalid(`polygonNdc[${i}] must be [x, y]`);
        }
        shapeData[i * 2] = checkFinite(pt[0], `polygonNdc[${i}][0]`);
        shapeData[i * 2 + 1] = checkFinite(pt[1], `polygonNdc[${i}][1]`);
      }
    } else {
      throw invalid("lasso polygonNdc must be an array of [x, y] points or a typed array");
    }
  } else {
    throw invalid('shape.kind must be "rect" or "lasso"');
  }

  if (options == null || typeof options !== "object") {
    throw invalid("options must be an object");
  }
  const vp = options.viewProj;
  if (vp == null || vp.length !== 16) {
    throw invalid("viewProj must have 16 elements");
  }
  const viewProj = new Float64Array(16);
  for (let i = 0; i < 16; i++) {
    viewProj[i] = checkFinite(vp[i], `viewProj[${i}]`);
  }
  return {
    shapeKind,
    shapeData,
    viewProj,
    cameraOrigin: normalizeVec3(options.cameraOrigin, "cameraOrigin"),
    cameraForward: normalizeVec3(options.cameraForward, "cameraForward"),
    isOrthographic: checkBoolean(options.isOrthographic, false, "isOrthographic"),
    visibleOnly: checkBoolean(options.visibleOnly, true, "visibleOnly"),
    rejectBackfaces: checkBoolean(options.rejectBackfaces, true, "rejectBackfaces"),
    includeTouching: checkBoolean(options.includeTouching, true, "includeTouching"),
  };
}

export function decodeFaceEditReport(raw: Float64Array): FaceEditReport {
  return {
    removedTriangles: raw[0],
    removedVertices: raw[1],
  };
}

// ============================================================================
// #193 Free-Form Deformation Types & Codecs
// ============================================================================

/** Basis function for Free-Form Deformation lattices. */
export type FfdBasis = "bernstein" | "cubicBSpline";
export type FfdBasisInput = FfdBasis | "bspline";
export type FfdOutsideMode = "ignore" | "clamp";
export type ArapWeights = "cotangent" | "uniform";
export type ArapSolver = "auto" | "direct" | "pcg";
export type KelvinletScale = "single" | "bi" | "tri";

/** World-space oriented box frame for an FFD lattice. */
export interface FfdFrameData {
  center: Vec3;
  axes: [Vec3, Vec3, Vec3];
  halfExtents: Vec3;
}

/** 3D Free-Form Deformation control lattice and control-point offsets. */
export interface FfdLatticeData {
  dims: [number, number, number];
  basis: FfdBasis;
  frame: FfdFrameData;
  /** Flat `[dx0, dy0, dz0, ...]` control point displacements (`length === 3 * nx * ny * nz`). */
  offsets: Float64Array;
}

/** Alias for {@link FfdLatticeData}. */
export type FfdLattice = FfdLatticeData;

/**
 * Optional vertex protection weights shielding pinned regions from freeform displacement.
 * Matches `VertexProtection`: weight `0` holds a vertex frozen in place,
 * `1` keeps full displacement, and unset holds all listed vertices frozen (`0`).
 */
export interface FreeformProtection {
  vertices: Indices;
  weights?: Float32Array | ArrayLike<number>;
}

/** Optional spatial region, vertex subset, and protection filter shared across freeform deformers. */
export interface FreeformScope {
  region?: { center: Readonly<Vec3>; radius: number };
  vertices?: Indices;
  protect?: FreeformProtection;
}

/** Options for {@link Scene.applyFfd}. */
export interface FfdOptions extends FreeformScope {
  lattice: FfdLatticeData;
  outsideMode?: FfdOutsideMode;
}

/** An anchor vertex constraint for ARAP surface deformation. */
export interface ArapAnchorInput {
  vertex: number;
  position?: Readonly<Vec3>;
}

/** A target handle constraint for ARAP surface deformation. */
export interface ArapHandleInput {
  vertex: number;
  target: Readonly<Vec3>;
}

/** Options for {@link Scene.applyArap}. */
export interface ArapOptions extends FreeformScope {
  anchors: readonly ArapAnchorInput[];
  handles: readonly ArapHandleInput[];
  weights?: ArapWeights;
  solver?: ArapSolver;
  iterations?: number;
  tolerance?: number;
}

/** Load specification for volumetric Kelvinlet brushes. */
export type KelvinletLoad =
  | { kind: "grab"; force: Readonly<Vec3> }
  | { kind: "scale"; magnitude: number }
  | { kind: "twist"; torque: Readonly<Vec3> }
  | { kind: "pinch"; forceDipole: readonly [Readonly<Vec3>, Readonly<Vec3>, Readonly<Vec3>] | ReadonlyArray<number> };

/** Options for {@link Scene.applyKelvinlet}. */
export interface KelvinletOptions extends FreeformScope {
  center: Readonly<Vec3>;
  load: KelvinletLoad;
  epsilon?: number;
  nu?: number;
  mu?: number;
  scale?: KelvinletScale;
  scaleAlpha?: number;
  symmetry?: { x?: boolean; y?: boolean; z?: boolean };
}

/** Pinned vertex target for XPBD soft-body simulation. */
export interface XpbdPinInput {
  vertex: number;
  target: Readonly<Vec3>;
}

/** Compliant handle constraint for XPBD soft-body simulation. */
export interface XpbdHandleInput {
  vertex: number;
  target: Readonly<Vec3>;
  compliance?: number;
}

/** Options for {@link Scene.applyXpbd}. */
export interface XpbdOptions {
  substeps?: number;
  iterations?: number;
  dt?: number;
  stretchCompliance?: number;
  bendCompliance?: number;
  volumeCompliance?: number;
  damping?: number;
  gravity?: Readonly<Vec3>;
  pins?: readonly XpbdPinInput[];
  handles?: readonly XpbdHandleInput[];
  selfCollision?: boolean;
  collisionThickness?: number;
  protect?: FreeformProtection;
}

/** Outcome statistics returned by all freeform deformation operators. */
export interface FreeformReport {
  verticesEvaluated: number;
  verticesMoved: number;
  maxDisplacement: number;
  iterations: number;
  converged: boolean;
  selfCollisionContactsResolved: number;
}

export const FFD_BASIS_CODES: Record<FfdBasisInput, number> = {
  bernstein: 0,
  cubicBSpline: 1,
  bspline: 1,
};

const FFD_OUTSIDE_CODES: Record<FfdOutsideMode, number> = { ignore: 0, clamp: 1 };
const ARAP_WEIGHT_CODES: Record<ArapWeights, number> = { cotangent: 0, uniform: 1 };
const ARAP_SOLVER_CODES: Record<ArapSolver, number> = { auto: 0, direct: 1, pcg: 2 };
const KELVINLET_SCALE_CODES: Record<KelvinletScale, number> = { single: 0, bi: 1, tri: 2 };

function normalizeProtection(protect: FreeformProtection | undefined): {
  protectedVerts: Uint32Array | undefined;
  protectionWeights: Float32Array | undefined;
} {
  if (protect === undefined) {
    return { protectedVerts: undefined, protectionWeights: undefined };
  }
  const verts = Uint32Array.from(normalizeU32List(protect.vertices, "protect.vertices"));
  let weights: Float32Array | undefined;
  if (protect.weights !== undefined) {
    weights = toF32Array(protect.weights, "protect.weights");
    if (weights.length !== verts.length) {
      throw invalid("protect.weights must have the same length as protect.vertices");
    }
  }
  return { protectedVerts: verts, protectionWeights: weights };
}

function normalizeScope(scope: FreeformScope): {
  regionCenter: Float64Array | undefined;
  regionRadius: number;
  vertices: Uint32Array | undefined;
  protectedVerts: Uint32Array | undefined;
  protectionWeights: Float32Array | undefined;
} {
  let regionCenter: Float64Array | undefined;
  let regionRadius = 0;
  if (scope.region !== undefined) {
    regionCenter = Float64Array.from(normalizeVec3(scope.region.center, "region.center"));
    regionRadius = checkNonNegative(scope.region.radius, "region.radius");
  }
  const vertices =
    scope.vertices === undefined ? undefined : Uint32Array.from(normalizeU32List(scope.vertices, "vertices"));
  return {
    regionCenter,
    regionRadius,
    vertices,
    ...normalizeProtection(scope.protect),
  };
}

export function normalizeFfdLatticeDims(dims: readonly [number, number, number]): Uint32Array {
  if (dims?.length !== 3) {
    throw invalid("dims must be [nx, ny, nz]");
  }
  return Uint32Array.from([
    checkU32Int(dims[0], "dims[0]"),
    checkU32Int(dims[1], "dims[1]"),
    checkU32Int(dims[2], "dims[2]"),
  ]);
}

export function decodeFfdLattice(raw: Float64Array): FfdLatticeData {
  return {
    dims: [raw[0], raw[1], raw[2]],
    basis: raw[3] === 1 ? "cubicBSpline" : "bernstein",
    frame: {
      center: [raw[4], raw[5], raw[6]],
      axes: [
        [raw[7], raw[8], raw[9]],
        [raw[10], raw[11], raw[12]],
        [raw[13], raw[14], raw[15]],
      ],
      halfExtents: [raw[16], raw[17], raw[18]],
    },
    offsets: Float64Array.from(raw.subarray(19)),
  };
}

export interface NormalizedFfdArgs {
  dims: Uint32Array;
  basis: number;
  frame15: Float64Array;
  offsets: Float64Array;
  outsideMode: number;
  regionCenter: Float64Array | undefined;
  regionRadius: number;
  vertices: Uint32Array | undefined;
  protectedVerts: Uint32Array | undefined;
  protectionWeights: Float32Array | undefined;
}

export function normalizeFfdArgs(options: FfdOptions): NormalizedFfdArgs {
  if (options == null || typeof options !== "object" || options.lattice == null) {
    throw invalid("ffd options require a lattice");
  }
  const lat = options.lattice;
  const dims = normalizeFfdLatticeDims(lat.dims);
  const basis = codeIn(FFD_BASIS_CODES, lat.basis, "lattice.basis");
  const c = normalizeVec3(lat.frame.center, "lattice.frame.center");
  if (!Array.isArray(lat.frame.axes) || lat.frame.axes.length !== 3) {
    throw invalid("lattice.frame.axes must be 3 vectors");
  }
  const ax0 = normalizeVec3(lat.frame.axes[0], "lattice.frame.axes[0]");
  const ax1 = normalizeVec3(lat.frame.axes[1], "lattice.frame.axes[1]");
  const ax2 = normalizeVec3(lat.frame.axes[2], "lattice.frame.axes[2]");
  const he = normalizeVec3(lat.frame.halfExtents, "lattice.frame.halfExtents");
  const frame15 = Float64Array.from([...c, ...ax0, ...ax1, ...ax2, ...he]);
  const expectedOffsets = dims[0] * dims[1] * dims[2] * 3;
  if (lat.offsets?.length !== expectedOffsets) {
    throw invalid(`lattice.offsets must have length ${expectedOffsets}`);
  }
  const offsets = Float64Array.from(lat.offsets);
  for (let i = 0; i < offsets.length; i++) {
    checkFinite(offsets[i], `lattice.offsets[${i}]`);
  }
  const outsideMode = codeIn(FFD_OUTSIDE_CODES, options.outsideMode ?? "ignore", "outsideMode");
  return {
    dims,
    basis,
    frame15,
    offsets,
    outsideMode,
    ...normalizeScope(options),
  };
}

export interface NormalizedArapArgs {
  anchorVerts: Uint32Array;
  anchorPos: Float64Array;
  anchorHasPos: Uint8Array;
  handleVerts: Uint32Array;
  handleTargets: Float64Array;
  weights: number;
  solver: number;
  iterations: number;
  tolerance: number;
  regionCenter: Float64Array | undefined;
  regionRadius: number;
  vertices: Uint32Array | undefined;
  protectedVerts: Uint32Array | undefined;
  protectionWeights: Float32Array | undefined;
}

export function normalizeArapArgs(options: ArapOptions): NormalizedArapArgs {
  if (options == null || typeof options !== "object") {
    throw invalid("arap options must be an object");
  }
  if (!Array.isArray(options.anchors) || !Array.isArray(options.handles)) {
    throw invalid("arap options require anchors and handles arrays");
  }
  const anchorVerts = new Uint32Array(options.anchors.length);
  const anchorPos = new Float64Array(options.anchors.length * 3);
  const anchorHasPos = new Uint8Array(options.anchors.length);
  for (let i = 0; i < options.anchors.length; i++) {
    const a = options.anchors[i];
    anchorVerts[i] = checkU32Int(a.vertex, `anchors[${i}].vertex`);
    if (a.position !== undefined) {
      const p = normalizeVec3(a.position, `anchors[${i}].position`);
      anchorHasPos[i] = 1;
      anchorPos[i * 3] = p[0];
      anchorPos[i * 3 + 1] = p[1];
      anchorPos[i * 3 + 2] = p[2];
    }
  }
  const handleVerts = new Uint32Array(options.handles.length);
  const handleTargets = new Float64Array(options.handles.length * 3);
  for (let i = 0; i < options.handles.length; i++) {
    const h = options.handles[i];
    handleVerts[i] = checkU32Int(h.vertex, `handles[${i}].vertex`);
    const t = normalizeVec3(h.target, `handles[${i}].target`);
    handleTargets[i * 3] = t[0];
    handleTargets[i * 3 + 1] = t[1];
    handleTargets[i * 3 + 2] = t[2];
  }
  return {
    anchorVerts,
    anchorPos,
    anchorHasPos,
    handleVerts,
    handleTargets,
    weights: codeIn(ARAP_WEIGHT_CODES, options.weights ?? "cotangent", "weights"),
    solver: codeIn(ARAP_SOLVER_CODES, options.solver ?? "auto", "solver"),
    iterations: checkU32Int(options.iterations ?? 10, "iterations"),
    tolerance: checkNonNegative(options.tolerance ?? 1e-4, "tolerance"),
    ...normalizeScope(options),
  };
}

export interface NormalizedKelvinletArgs {
  center: Float64Array;
  loadKind: number;
  loadData: Float64Array;
  epsilon: number;
  nu: number;
  mu: number;
  scaleMode: number;
  scaleAlpha: number;
  symX: boolean;
  symY: boolean;
  symZ: boolean;
  regionCenter: Float64Array | undefined;
  regionRadius: number;
  vertices: Uint32Array | undefined;
  protectedVerts: Uint32Array | undefined;
  protectionWeights: Float32Array | undefined;
}

export function normalizeKelvinletArgs(options: KelvinletOptions): NormalizedKelvinletArgs {
  if (options == null || typeof options !== "object" || options.load == null) {
    throw invalid("kelvinlet options require center and load");
  }
  const center = Float64Array.from(normalizeVec3(options.center, "center"));
  let loadKind: number;
  let loadData: Float64Array;
  const load = options.load;
  switch (load.kind) {
    case "grab":
      loadKind = 0;
      loadData = Float64Array.from(normalizeVec3(load.force, "load.force"));
      break;
    case "scale":
      loadKind = 1;
      loadData = Float64Array.from([checkFinite(load.magnitude, "load.magnitude")]);
      break;
    case "twist":
      loadKind = 2;
      loadData = Float64Array.from(normalizeVec3(load.torque, "load.torque"));
      break;
    case "pinch": {
      loadKind = 3;
      const d = load.forceDipole;
      if (d?.length === 3 && Array.isArray(d[0])) {
        const rows = d as readonly [Readonly<Vec3>, Readonly<Vec3>, Readonly<Vec3>];
        loadData = Float64Array.from([
          ...normalizeVec3(rows[0], "forceDipole[0]"),
          ...normalizeVec3(rows[1], "forceDipole[1]"),
          ...normalizeVec3(rows[2], "forceDipole[2]"),
        ]);
      } else if (d?.length === 9) {
        loadData = Float64Array.from(d as ReadonlyArray<number>);
        for (let i = 0; i < 9; i++) {
          checkFinite(loadData[i], `forceDipole[${i}]`);
        }
      } else {
        throw invalid("pinch forceDipole must be 3x3 vectors or 9 numbers");
      }
      break;
    }
    default:
      throw invalid('load.kind must be "grab", "scale", "twist" or "pinch"');
  }
  return {
    center,
    loadKind,
    loadData,
    epsilon: checkPositive(options.epsilon ?? 0.5, "epsilon"),
    nu: checkFinite(options.nu ?? 0.3, "nu"),
    mu: checkPositive(options.mu ?? 1.0, "mu"),
    scaleMode: codeIn(KELVINLET_SCALE_CODES, options.scale ?? "single", "scale"),
    scaleAlpha: checkFinite(options.scaleAlpha ?? 2.0, "scaleAlpha"),
    symX: checkBoolean(options.symmetry?.x, false, "symmetry.x"),
    symY: checkBoolean(options.symmetry?.y, false, "symmetry.y"),
    symZ: checkBoolean(options.symmetry?.z, false, "symmetry.z"),
    ...normalizeScope(options),
  };
}

export interface NormalizedXpbdArgs {
  substeps: number;
  iterations: number;
  dt: number;
  stretchCompliance: number;
  bendCompliance: number;
  volumeCompliance: number;
  damping: number;
  gravity: Float64Array;
  pinVerts: Uint32Array;
  pinTargets: Float64Array;
  handleVerts: Uint32Array;
  handleTargets: Float64Array;
  handleCompliances: Float64Array;
  selfCollision: boolean;
  collisionThickness: number;
  protectedVerts: Uint32Array | undefined;
  protectionWeights: Float32Array | undefined;
}

export function normalizeXpbdArgs(options: XpbdOptions): NormalizedXpbdArgs {
  if (options == null || typeof options !== "object") {
    throw invalid("xpbd options must be an object");
  }
  const pins = options.pins ?? [];
  const handles = options.handles ?? [];
  const pinVerts = new Uint32Array(pins.length);
  const pinTargets = new Float64Array(pins.length * 3);
  for (let i = 0; i < pins.length; i++) {
    pinVerts[i] = checkU32Int(pins[i].vertex, `pins[${i}].vertex`);
    const t = normalizeVec3(pins[i].target, `pins[${i}].target`);
    pinTargets[i * 3] = t[0];
    pinTargets[i * 3 + 1] = t[1];
    pinTargets[i * 3 + 2] = t[2];
  }
  const handleVerts = new Uint32Array(handles.length);
  const handleTargets = new Float64Array(handles.length * 3);
  const handleCompliances = new Float64Array(handles.length);
  for (let i = 0; i < handles.length; i++) {
    handleVerts[i] = checkU32Int(handles[i].vertex, `handles[${i}].vertex`);
    const t = normalizeVec3(handles[i].target, `handles[${i}].target`);
    handleTargets[i * 3] = t[0];
    handleTargets[i * 3 + 1] = t[1];
    handleTargets[i * 3 + 2] = t[2];
    handleCompliances[i] = checkNonNegative(handles[i].compliance ?? 0, `handles[${i}].compliance`);
  }
  return {
    substeps: checkU32Int(options.substeps ?? 4, "substeps"),
    iterations: checkU32Int(options.iterations ?? 8, "iterations"),
    dt: checkPositive(options.dt ?? 1 / 60, "dt"),
    stretchCompliance: checkNonNegative(options.stretchCompliance ?? 0, "stretchCompliance"),
    bendCompliance: checkNonNegative(options.bendCompliance ?? 0.05, "bendCompliance"),
    volumeCompliance:
      options.volumeCompliance === undefined ? Number.NaN : checkNonNegative(options.volumeCompliance, "volumeCompliance"),
    damping: checkNonNegative(options.damping ?? 0.02, "damping"),
    gravity: Float64Array.from(normalizeVec3(options.gravity ?? [0, 0, 0], "gravity")),
    pinVerts,
    pinTargets,
    handleVerts,
    handleTargets,
    handleCompliances,
    selfCollision: checkBoolean(options.selfCollision, false, "selfCollision"),
    collisionThickness: checkNonNegative(options.collisionThickness ?? 0.01, "collisionThickness"),
    ...normalizeProtection(options.protect),
  };
}

export function decodeFreeformReport(raw: Float64Array): FreeformReport {
  return {
    verticesEvaluated: raw[0],
    verticesMoved: raw[1],
    maxDisplacement: raw[2],
    iterations: raw[3],
    converged: raw[4] !== 0,
    selfCollisionContactsResolved: raw[5],
  };
}

// ============================================================================
// #194 Inter-Object Collision Types & Codecs
// ============================================================================

export type CollisionMode = "passThrough" | "rigid" | "cut" | "softDeform";
export type CollisionRole = "symmetric" | "movingIsCutter" | "movingIsTarget";
export type CollisionDistanceBackend = "bvhSdf" | "fhEdt";
export type CutTrigger = "onRelease" | "liveStep";
export type SoftDeformSolver = "sdfLaplacian" | "kelvinlet" | "xpbd";
export type MaterialMemory = "elastic" | "plastic";

/** Collision response configuration for the scene default or an unordered mesh pair. */
export interface CollisionRule {
  mode: CollisionMode;
  role: CollisionRole;
  distanceBackend: CollisionDistanceBackend;
  cutMethod: BooleanMethod;
  cutTrigger: CutTrigger;
  softSolver: SoftDeformSolver;
  materialMemory: MaterialMemory;
  compliance: number;
  stiffness: number;
  recoveryRate: number;
  contactMargin: number;
  selfCollisionGuard: boolean;
  rigidSlide: boolean;
}

export type CollisionRuleInput = Partial<CollisionRule>;

export const DEFAULT_COLLISION_RULE: Readonly<CollisionRule> = Object.freeze({
  mode: "passThrough",
  role: "symmetric",
  distanceBackend: "bvhSdf",
  cutMethod: "manifold",
  cutTrigger: "onRelease",
  softSolver: "sdfLaplacian",
  materialMemory: "elastic",
  compliance: Math.fround(0.05),
  stiffness: 0.5,
  recoveryRate: 1.0,
  contactMargin: 0.0,
  selfCollisionGuard: true,
  rigidSlide: true,
});

/** Per-pair collision outcome from a {@link moveWithCollision} or commit step. */
export interface CollisionPairOutcome<M = MeshHandle> {
  other: M;
  mode: CollisionMode;
  role: CollisionRole;
  intersecting: boolean;
  minDistance: number;
  contactBlocked: boolean;
  slidden: boolean;
  cutExecuted: boolean;
  previewOnly: boolean;
  softDeformedVertices: number;
  elasticRecoveredVertices: number;
  selfCollisionGuardTriggered: boolean;
}

/** Step-level collision detection and response report. */
export interface CollisionStepReport<M = MeshHandle> {
  appliedPlacement: Placement;
  appliedQuaternion: Quat;
  activePairs: CollisionPairOutcome<M>[];
  modifiedMeshes: M[];
  minClearance: number;
  cutsExecuted: number;
  softVerticesDeformed: number;
  elasticRecoveryCount: number;
  selfIntersectionGuardTriggered: boolean;
}

const COLLISION_MODE_CODES: Record<CollisionMode, number> = {
  passThrough: 0,
  rigid: 1,
  cut: 2,
  softDeform: 3,
};
const COLLISION_MODES: readonly CollisionMode[] = ["passThrough", "rigid", "cut", "softDeform"];

const COLLISION_ROLE_CODES: Record<CollisionRole, number> = {
  symmetric: 0,
  movingIsCutter: 1,
  movingIsTarget: 2,
};
const COLLISION_ROLES: readonly CollisionRole[] = ["symmetric", "movingIsCutter", "movingIsTarget"];

const COLLISION_DIST_CODES: Record<CollisionDistanceBackend, number> = { bvhSdf: 0, fhEdt: 1 };
const COLLISION_DISTS: readonly CollisionDistanceBackend[] = ["bvhSdf", "fhEdt"];

const BOOLEAN_METHOD_NAMES: readonly BooleanMethod[] = ["manifold", "bvhSdf", "fhEdt"];

const CUT_TRIGGER_CODES: Record<CutTrigger, number> = { onRelease: 0, liveStep: 1 };
const CUT_TRIGGERS: readonly CutTrigger[] = ["onRelease", "liveStep"];

const SOFT_SOLVER_CODES: Record<SoftDeformSolver, number> = { sdfLaplacian: 0, kelvinlet: 1, xpbd: 2 };
const SOFT_SOLVERS: readonly SoftDeformSolver[] = ["sdfLaplacian", "kelvinlet", "xpbd"];

const MEMORY_CODES: Record<MaterialMemory, number> = { elastic: 0, plastic: 1 };
const MEMORIES: readonly MaterialMemory[] = ["elastic", "plastic"];

export function normalizeCollisionRule(input: CollisionRuleInput, base: CollisionRule = DEFAULT_COLLISION_RULE): CollisionRule {
  if (input == null || typeof input !== "object") {
    throw invalid("collision rule must be an object");
  }
  const recoveryRaw = checkFinite(input.recoveryRate ?? base.recoveryRate, "recoveryRate");
  if (recoveryRaw < 0 || recoveryRaw > 1) {
    throw invalid("recoveryRate must be in [0, 1]");
  }
  const merged: CollisionRule = {
    mode: input.mode ?? base.mode,
    role: input.role ?? base.role,
    distanceBackend: input.distanceBackend ?? base.distanceBackend,
    cutMethod: input.cutMethod ?? base.cutMethod,
    cutTrigger: input.cutTrigger ?? base.cutTrigger,
    softSolver: input.softSolver ?? base.softSolver,
    materialMemory: input.materialMemory ?? base.materialMemory,
    compliance: Math.fround(checkNonNegative(input.compliance ?? base.compliance, "compliance")),
    stiffness: Math.fround(checkNonNegative(input.stiffness ?? base.stiffness, "stiffness")),
    recoveryRate: Math.fround(recoveryRaw),
    contactMargin: checkNonNegative(input.contactMargin ?? base.contactMargin, "contactMargin"),
    selfCollisionGuard: checkBoolean(input.selfCollisionGuard, base.selfCollisionGuard, "selfCollisionGuard"),
    rigidSlide: checkBoolean(input.rigidSlide, base.rigidSlide, "rigidSlide"),
  };
  codeIn(COLLISION_MODE_CODES, merged.mode, "mode");
  codeIn(COLLISION_ROLE_CODES, merged.role, "role");
  codeIn(COLLISION_DIST_CODES, merged.distanceBackend, "distanceBackend");
  codeIn(BOOLEAN_METHOD_CODES, merged.cutMethod, "cutMethod");
  codeIn(CUT_TRIGGER_CODES, merged.cutTrigger, "cutTrigger");
  codeIn(SOFT_SOLVER_CODES, merged.softSolver, "softSolver");
  codeIn(MEMORY_CODES, merged.materialMemory, "materialMemory");
  return merged;
}

export function encodeCollisionRule(rule: CollisionRule): Float64Array {
  return Float64Array.from([
    COLLISION_MODE_CODES[rule.mode],
    COLLISION_ROLE_CODES[rule.role],
    COLLISION_DIST_CODES[rule.distanceBackend],
    BOOLEAN_METHOD_CODES[rule.cutMethod],
    CUT_TRIGGER_CODES[rule.cutTrigger],
    SOFT_SOLVER_CODES[rule.softSolver],
    MEMORY_CODES[rule.materialMemory],
    rule.compliance,
    rule.stiffness,
    rule.recoveryRate,
    rule.contactMargin,
    rule.selfCollisionGuard ? 1 : 0,
    rule.rigidSlide ? 1 : 0,
    0,
    0,
    0,
  ]);
}

export function decodeCollisionRule(flat: Float64Array): CollisionRule {
  return {
    mode: COLLISION_MODES[flat[0]] ?? "passThrough",
    role: COLLISION_ROLES[flat[1]] ?? "symmetric",
    distanceBackend: COLLISION_DISTS[flat[2]] ?? "bvhSdf",
    cutMethod: BOOLEAN_METHOD_NAMES[flat[3]] ?? "manifold",
    cutTrigger: CUT_TRIGGERS[flat[4]] ?? "onRelease",
    softSolver: SOFT_SOLVERS[flat[5]] ?? "sdfLaplacian",
    materialMemory: MEMORIES[flat[6]] ?? "elastic",
    compliance: flat[7],
    stiffness: flat[8],
    recoveryRate: flat[9],
    contactMargin: flat[10],
    selfCollisionGuard: flat[11] !== 0,
    rigidSlide: flat[12] !== 0,
  };
}

export function normalizePlacementInputArgs(placement: PlacementInput): {
  rotation: Float64Array;
  translation: Float64Array;
  scale: Float64Array | undefined;
} {
  if (placement == null || typeof placement !== "object") {
    throw invalid("placement must be an object");
  }
  const r = rotationMatrix(placement.rotation);
  if (!r.every(Number.isFinite)) {
    throw invalid("rotation must be finite numbers");
  }
  const t = normalizeVec3(placement.translation, "translation");
  const s =
    placement.scale === undefined
      ? undefined
      : Float64Array.from(normalizeVec3(placement.scale, "scale"));
  return {
    rotation: Float64Array.from(r),
    translation: Float64Array.from(t),
    scale: s,
  };
}

export function decodeCollisionStepReport<T>(
  raw: Float64Array,
  mapHandle: (index: number, generation: number) => T,
  scale: Vec3 = [1, 1, 1],
): CollisionStepReport<T> {
  const quat: Quat = [raw[3], raw[4], raw[5], raw[6]];
  const appliedPlacement: Placement = {
    rotation: rotationMatrix(quat),
    translation: [raw[0], raw[1], raw[2]],
    scale: [...scale],
  };
  let cur = 7;
  const pairCount = raw[cur++];
  const activePairs: CollisionPairOutcome<T>[] = [];
  for (let i = 0; i < pairCount; i++) {
    const other = mapHandle(raw[cur], raw[cur + 1]);
    activePairs.push({
      other,
      mode: COLLISION_MODES[raw[cur + 2]] ?? "passThrough",
      role: COLLISION_ROLES[raw[cur + 3]] ?? "symmetric",
      intersecting: raw[cur + 4] !== 0,
      minDistance: raw[cur + 5],
      contactBlocked: raw[cur + 6] !== 0,
      slidden: raw[cur + 7] !== 0,
      cutExecuted: raw[cur + 8] !== 0,
      previewOnly: raw[cur + 9] !== 0,
      softDeformedVertices: raw[cur + 10],
      elasticRecoveredVertices: raw[cur + 11],
      selfCollisionGuardTriggered: raw[cur + 12] !== 0,
    });
    cur += 13;
  }
  const modCount = raw[cur++];
  const modifiedMeshes: T[] = [];
  for (let i = 0; i < modCount; i++) {
    modifiedMeshes.push(mapHandle(raw[cur], raw[cur + 1]));
    cur += 2;
  }
  return {
    appliedPlacement,
    appliedQuaternion: quat,
    activePairs,
    modifiedMeshes,
    minClearance: raw[cur],
    cutsExecuted: raw[cur + 1],
    softVerticesDeformed: raw[cur + 2],
    elasticRecoveryCount: raw[cur + 3],
    selfIntersectionGuardTriggered: raw[cur + 4] !== 0,
  };
}

// ============================================================================
// #234 Primitive Fit, Contours & Point Cloud Types, Codecs & Standalone API
// ============================================================================

export type PrimitiveKind = "plane" | "sphere" | "cylinder" | "cone";
export type PrimitiveKindInput = PrimitiveKind | "auto";
export type ContourExportFormat = "svg" | "json" | "csv";
export type PointCloudFormat = "xyz" | "pcd" | "ply";
export type SurfaceReconstructMethod = "implicitMlsMarchingCubes" | "ballPivoting";

export type FittedPrimitive =
  | { kind: "plane"; origin: Vec3; normal: Vec3 }
  | { kind: "sphere"; center: Vec3; radius: number }
  | { kind: "cylinder"; axisOrigin: Vec3; axisDir: Vec3; radius: number; height: number }
  | { kind: "cone"; apex: Vec3; axisDir: Vec3; halfAngleRad: number; height: number };

export interface FittedPrimitiveReport {
  primitive: FittedPrimitive;
  rmsError: number;
  maxError: number;
  inlierCount: number;
  inlierRatio: number;
  residuals: Float32Array;
}

export interface FitPrimitiveOptions {
  subsetTris?: Indices;
  kind?: PrimitiveKindInput;
  inlierThreshold?: number;
  ransacIterations?: number;
  lmIterations?: number;
  seed?: number;
  minInlierRatio?: number;
  normalWeight?: number;
}

export interface ContourLoop {
  closed: boolean;
  perimeter: number;
  enclosedArea: number;
  centroid2d: [number, number];
  centroid3d: Vec3;
  minCaliperDiameter: number;
  maxCaliperDiameter: number;
  /** The points in the plane, `u, v` each: around the slice's `planeOrigin`, along its axes. */
  points2d: Float64Array;
  /**
   * The points in world coordinates, `x, y, z` each, rounded to `f32`: far
   * from the world origin that is by up to half of its steps there (0.004 at
   * 1e5). `points3d64` has them as precise as the mesh.
   */
  points3d: Float32Array;
  /**
   * The points in world coordinates, `x, y, z` each, as precise as the mesh
   * wherever it is. Every slice the SDK returns has them; without them in a
   * loop of yours, an export takes `points3d`.
   */
  points3d64?: Float64Array;
}

export interface ContourSlice {
  /** The plane's point in world coordinates: the one asked for, to a few `f32` steps of the mesh's local positions. */
  planeOrigin: Vec3;
  planeNormal: Vec3;
  uAxis: Vec3;
  vAxis: Vec3;
  totalPerimeter: number;
  totalArea: number;
  loops: ContourLoop[];
}

/**
 * Packed WebAssembly-ready point cloud buffer (`[count, hasNormals, hasColors, ...positions, ...normals, ...colors]`).
 *
 * The positions are `f32`, which far from zero holds them only to its steps
 * there (0.008 at 1e5). Keep a cloud there around an `origin` near it:
 * {@link packPointCloud} takes one, and {@link parsePointCloudText} and
 * {@link parsePointCloudBytes} find one with `center: true`.
 */
export interface PackedPointCloud {
  readonly data: Float32Array;
  readonly count: number;
  readonly hasNormals: boolean;
  readonly hasColors: boolean;
  /**
   * The world point the positions are relative to (`f64` numbers). Zero when
   * left out: they are then world positions. Everything that takes a cloud
   * counts it in, and what makes one of another keeps it.
   */
  readonly origin?: Readonly<Vec3>;
}

/** Options of {@link parsePointCloudText} and {@link parsePointCloudBytes}. */
export interface ParsePointCloudOptions {
  /**
   * Read the positions in `f64` and keep them around the centre of their
   * bounds, which is the cloud's `origin`, so that points far from zero keep
   * their digits. Defaults to `false`: `f32` world positions.
   */
  center?: boolean;
}

export interface OutlierReport {
  keptCount: number;
  removedCount: number;
  meanDistance: number;
  stdDev: number;
  threshold: number;
}

export interface PointCloudFilterResult {
  cloud: PackedPointCloud;
  report: OutlierReport;
}

export interface SurfaceReconstructOptions {
  method?: SurfaceReconstructMethod;
  voxelSize?: number;
  ballRadius?: number;
  kNeighbors?: number;
  normals?: boolean;
}

const PRIMITIVE_KIND_CODES: Record<PrimitiveKindInput, number> = {
  plane: 0,
  sphere: 1,
  cylinder: 2,
  cone: 3,
  auto: 4,
};

export const CONTOUR_EXPORT_CODES: Record<ContourExportFormat, number> = { svg: 0, json: 1, csv: 2 };
export const POINT_CLOUD_FORMAT_CODES: Record<PointCloudFormat, number> = { xyz: 0, pcd: 1, ply: 2 };
const RECONSTRUCT_METHOD_CODES: Record<SurfaceReconstructMethod, number> = {
  implicitMlsMarchingCubes: 0,
  ballPivoting: 1,
};

export interface NormalizedFitPrimitiveArgs {
  subsetTris: Uint32Array | undefined;
  kind: number;
  inlierThreshold: number;
  ransacIterations: number;
  lmIterations: number;
  seed: number;
  minInlierRatio: number;
  normalWeight: number;
}

export function normalizeFitPrimitiveArgs(options: FitPrimitiveOptions = {}): NormalizedFitPrimitiveArgs {
  if (options == null || typeof options !== "object") {
    throw invalid("fitPrimitive options must be an object");
  }
  const subsetTris =
    options.subsetTris === undefined
      ? undefined
      : Uint32Array.from(normalizeU32List(options.subsetTris, "subsetTris"));
  return {
    subsetTris,
    kind: codeIn(PRIMITIVE_KIND_CODES, options.kind ?? "auto", "kind"),
    inlierThreshold:
      options.inlierThreshold === undefined ? Number.NaN : checkPositive(options.inlierThreshold, "inlierThreshold"),
    ransacIterations:
      options.ransacIterations === undefined ? Number.NaN : checkU32Int(options.ransacIterations, "ransacIterations"),
    lmIterations: options.lmIterations === undefined ? Number.NaN : checkU32Int(options.lmIterations, "lmIterations"),
    seed: options.seed === undefined ? Number.NaN : checkNonNegative(options.seed, "seed"),
    minInlierRatio:
      options.minInlierRatio === undefined ? Number.NaN : checkNonNegative(options.minInlierRatio, "minInlierRatio"),
    normalWeight:
      options.normalWeight === undefined ? Number.NaN : checkNonNegative(options.normalWeight, "normalWeight"),
  };
}

export function encodeFittedPrimitive(primitive: FittedPrimitive): Float64Array {
  if (primitive == null || typeof primitive !== "object") {
    throw invalid("primitive must be an object");
  }
  switch (primitive.kind) {
    case "plane": {
      const o = normalizeVec3(primitive.origin, "primitive.origin");
      const n = normalizeVec3(primitive.normal, "primitive.normal");
      return Float64Array.from([0, o[0], o[1], o[2], n[0], n[1], n[2], 0, 0]);
    }
    case "sphere": {
      const c = normalizeVec3(primitive.center, "primitive.center");
      return Float64Array.from([1, c[0], c[1], c[2], checkPositive(primitive.radius, "primitive.radius"), 0, 0, 0, 0]);
    }
    case "cylinder": {
      const o = normalizeVec3(primitive.axisOrigin, "primitive.axisOrigin");
      const d = normalizeVec3(primitive.axisDir, "primitive.axisDir");
      return Float64Array.from([
        2,
        o[0],
        o[1],
        o[2],
        d[0],
        d[1],
        d[2],
        checkPositive(primitive.radius, "primitive.radius"),
        checkNonNegative(primitive.height, "primitive.height"),
      ]);
    }
    case "cone": {
      const a = normalizeVec3(primitive.apex, "primitive.apex");
      const d = normalizeVec3(primitive.axisDir, "primitive.axisDir");
      return Float64Array.from([
        3,
        a[0],
        a[1],
        a[2],
        d[0],
        d[1],
        d[2],
        checkPositive(primitive.halfAngleRad, "primitive.halfAngleRad"),
        checkNonNegative(primitive.height, "primitive.height"),
      ]);
    }
    default:
      throw invalid('primitive.kind must be "plane", "sphere", "cylinder" or "cone"');
  }
}

export function decodeFittedPrimitive(raw: Float64Array, at = 0): FittedPrimitive {
  switch (raw[at]) {
    case 0:
      return {
        kind: "plane",
        origin: [raw[at + 1], raw[at + 2], raw[at + 3]],
        normal: [raw[at + 4], raw[at + 5], raw[at + 6]],
      };
    case 1:
      return {
        kind: "sphere",
        center: [raw[at + 1], raw[at + 2], raw[at + 3]],
        radius: raw[at + 4],
      };
    case 2:
      return {
        kind: "cylinder",
        axisOrigin: [raw[at + 1], raw[at + 2], raw[at + 3]],
        axisDir: [raw[at + 4], raw[at + 5], raw[at + 6]],
        radius: raw[at + 7],
        height: raw[at + 8],
      };
    case 3:
      return {
        kind: "cone",
        apex: [raw[at + 1], raw[at + 2], raw[at + 3]],
        axisDir: [raw[at + 4], raw[at + 5], raw[at + 6]],
        halfAngleRad: raw[at + 7],
        height: raw[at + 8],
      };
    default:
      throw invalid(`unknown fitted primitive kind code ${raw[at]}`);
  }
}

export function decodeFitPrimitiveReport(raw: Float64Array): FittedPrimitiveReport {
  const n = raw[13];
  return {
    primitive: decodeFittedPrimitive(raw, 0),
    rmsError: raw[9],
    maxError: raw[10],
    inlierCount: raw[11],
    inlierRatio: raw[12],
    residuals: Float32Array.from(raw.subarray(14, 14 + n)),
  };
}

export function decodeContourSlices(raw: Float64Array): ContourSlice[] {
  let cur = 0;
  const sliceCount = raw[cur++];
  const slices: ContourSlice[] = [];
  for (let s = 0; s < sliceCount; s++) {
    const planeOrigin: Vec3 = [raw[cur], raw[cur + 1], raw[cur + 2]];
    const planeNormal: Vec3 = [raw[cur + 3], raw[cur + 4], raw[cur + 5]];
    const uAxis: Vec3 = [raw[cur + 6], raw[cur + 7], raw[cur + 8]];
    const vAxis: Vec3 = [raw[cur + 9], raw[cur + 10], raw[cur + 11]];
    const totalPerimeter = raw[cur + 12];
    const totalArea = raw[cur + 13];
    const loopCount = raw[cur + 14];
    cur += 15;
    const loops: ContourLoop[] = [];
    for (let l = 0; l < loopCount; l++) {
      const closed = raw[cur] !== 0;
      const perimeter = raw[cur + 1];
      const enclosedArea = raw[cur + 2];
      const centroid2d: [number, number] = [raw[cur + 3], raw[cur + 4]];
      const centroid3d: Vec3 = [raw[cur + 5], raw[cur + 6], raw[cur + 7]];
      const minCaliperDiameter = raw[cur + 8];
      const maxCaliperDiameter = raw[cur + 9];
      const npts = raw[cur + 10];
      cur += 11;
      const points2d = Float64Array.from(raw.subarray(cur, cur + npts * 2));
      cur += npts * 2;
      const points3d64 = raw.slice(cur, cur + npts * 3);
      const points3d = Float32Array.from(points3d64);
      cur += npts * 3;
      loops.push({
        closed,
        perimeter,
        enclosedArea,
        centroid2d,
        centroid3d,
        minCaliperDiameter,
        maxCaliperDiameter,
        points2d,
        points3d,
        points3d64,
      });
    }
    slices.push({
      planeOrigin,
      planeNormal,
      uAxis,
      vAxis,
      totalPerimeter,
      totalArea,
      loops,
    });
  }
  return slices;
}

export function encodeContourSlices(slices: readonly ContourSlice[]): Float64Array {
  const out: number[] = [slices.length];
  for (const s of slices) {
    out.push(
      ...s.planeOrigin,
      ...s.planeNormal,
      ...s.uAxis,
      ...s.vAxis,
      s.totalPerimeter,
      s.totalArea,
      s.loops.length,
    );
    for (const l of s.loops) {
      const points3d = l.points3d64 ?? l.points3d;
      const npts = Math.min(Math.floor(l.points2d.length / 2), Math.floor(points3d.length / 3));
      out.push(
        l.closed ? 1 : 0,
        l.perimeter,
        l.enclosedArea,
        l.centroid2d[0],
        l.centroid2d[1],
        ...l.centroid3d,
        l.minCaliperDiameter,
        l.maxCaliperDiameter,
        npts,
      );
      for (let i = 0; i < npts * 2; i++) {
        out.push(l.points2d[i]);
      }
      for (let i = 0; i < npts * 3; i++) {
        out.push(points3d[i]);
      }
    }
  }
  return Float64Array.from(out);
}

export function wrapPackedPointCloud(data: Float32Array, origin?: Readonly<Vec3>): PackedPointCloud {
  if (data.length < 3) {
    throw invalid("packed point cloud header must have at least 3 elements");
  }
  const count = data[0];
  const hasNormals = data[1] === 1;
  const hasColors = data[2] === 1;
  const expected = 3 + count * (3 + (hasNormals ? 3 : 0) + (hasColors ? 3 : 0));
  if (!Number.isInteger(count) || count < 0 || data.length !== expected) {
    throw invalid("packed point cloud buffer length mismatch");
  }
  return { data, count, hasNormals, hasColors, ...(origin === undefined ? {} : { origin }) };
}

/** The checked `origin` of a cloud, when it has one (a bare buffer has none). */
export function cloudOrigin(cloud: PackedPointCloud | Float32Array): Vec3 | undefined {
  const origin = cloud instanceof Float32Array ? undefined : cloud?.origin;
  return origin === undefined ? undefined : normalizeVec3(origin, "cloud.origin");
}

export function cloudBuffer(cloud: PackedPointCloud | Float32Array): Float32Array {
  if (cloud instanceof Float32Array) {
    return wrapPackedPointCloud(cloud).data;
  }
  if (cloud == null || !(cloud.data instanceof Float32Array)) {
    throw invalid("cloud must be a PackedPointCloud or packed Float32Array");
  }
  return wrapPackedPointCloud(cloud.data).data;
}

/** World positions as `f32` offsets from `origin`, taken in `f64`. */
function offsetsFrom(positions: Positions, origin: Readonly<Vec3>): Float32Array {
  const out = new Float32Array(positions.length);
  for (let i = 0; i < out.length; i++) {
    out[i] = positions[i] - origin[i % 3];
  }
  return out;
}

/**
 * Packs position (and optional normal/color) buffers into a {@link PackedPointCloud}.
 *
 * With `origin`, the positions (world coordinates, in a `Float64Array` or an
 * array to have more than `f32`) are stored relative to it, and it becomes the
 * cloud's `origin`: give a point near a cloud that is far from zero.
 */
export function packPointCloud(
  positions: Positions,
  normals?: Positions,
  colors?: Positions,
  origin?: Readonly<Vec3>,
): PackedPointCloud {
  const at = origin === undefined ? undefined : normalizeVec3(origin, "origin");
  const pos = toF32Array(at === undefined ? positions : offsetsFrom(positions, at), "positions");
  if (pos.length % 3 !== 0) {
    throw invalid("positions length must be a multiple of 3");
  }
  const n = pos.length / 3;
  const norm = normals === undefined ? undefined : toF32Array(normals, "normals");
  const col = colors === undefined ? undefined : toF32Array(colors, "colors");
  if (norm !== undefined && norm.length !== pos.length) {
    throw invalid("normals length must match positions length");
  }
  if (col !== undefined && col.length !== pos.length) {
    throw invalid("colors length must match positions length");
  }
  const total = 3 + n * (3 + (norm ? 3 : 0) + (col ? 3 : 0));
  const data = new Float32Array(total);
  data[0] = n;
  data[1] = norm ? 1 : 0;
  data[2] = col ? 1 : 0;
  let offset = 3;
  data.set(pos, offset);
  offset += pos.length;
  if (norm) {
    data.set(norm, offset);
    offset += norm.length;
  }
  if (col) {
    data.set(col, offset);
  }
  return {
    data,
    count: n,
    hasNormals: norm !== undefined,
    hasColors: col !== undefined,
    ...(at === undefined ? {} : { origin: at }),
  };
}

/** Unpacks a {@link PackedPointCloud} into separate typed arrays. The positions are relative to `origin`. */
export function unpackPointCloud(cloud: PackedPointCloud | Float32Array): {
  count: number;
  positions: Float32Array;
  normals: Float32Array | null;
  colors: Float32Array | null;
  /** The cloud's `origin`: `[0, 0, 0]` for world positions. */
  origin: Vec3;
} {
  const buf = cloudBuffer(cloud);
  const meta = wrapPackedPointCloud(buf);
  let at = 3;
  const positions = Float32Array.from(buf.subarray(at, at + meta.count * 3));
  at += meta.count * 3;
  const normals = meta.hasNormals ? Float32Array.from(buf.subarray(at, at + meta.count * 3)) : null;
  if (meta.hasNormals) {
    at += meta.count * 3;
  }
  const colors = meta.hasColors ? Float32Array.from(buf.subarray(at, at + meta.count * 3)) : null;
  return { count: meta.count, positions, normals, colors, origin: cloudOrigin(cloud) ?? [0, 0, 0] };
}

export function checkPositiveArg(value: number, what: string): number {
  return checkPositive(value, what);
}

export interface NormalizedReconstructArgs {
  method: number;
  voxelSize: number;
  ballRadius: number;
  kNeighbors: number;
  normals: boolean;
}

export function normalizeReconstructArgs(options: SurfaceReconstructOptions = {}): NormalizedReconstructArgs {
  if (options == null || typeof options !== "object") {
    throw invalid("reconstructPointCloud options must be an object");
  }
  return {
    method: codeIn(RECONSTRUCT_METHOD_CODES, options.method ?? "implicitMlsMarchingCubes", "method"),
    voxelSize: options.voxelSize === undefined ? Number.NaN : checkPositive(options.voxelSize, "voxelSize"),
    ballRadius: options.ballRadius === undefined ? Number.NaN : checkPositive(options.ballRadius, "ballRadius"),
    kNeighbors: options.kNeighbors === undefined ? Number.NaN : checkU32Int(options.kNeighbors, "kNeighbors"),
    normals: checkBoolean(options.normals, true, "normals"),
  };
}

// ============================================================================
// #236 Hollow, Drain Holes & Place Mesh Types & Codecs
// ============================================================================

export type HollowBackend = "bvhSdf" | "fhEdt";
export type PlaceUpAxis = "negZ" | "posZ" | "posY" | "negY";
export type PlaceMode = "keepSeparate" | "appendShell" | "embedUnion" | "engraveSubtract";

export interface HollowOptions {
  wallThickness: number;
  backend?: HollowBackend;
  voxelSize?: number;
  maxVoxels?: number;
  maxCavityTriangles?: number;
  minCavityVolume?: number;
  wallModulation?: {
    vertices: Indices;
    weights: Float32Array | ArrayLike<number>;
  };
}

export interface HollowReport {
  cavitiesCreated: number;
  cavityVertices: number;
  cavityTriangles: number;
  cavityVolume: number;
  voxelSize: number;
  uvsDropped: boolean;
}

export interface DrainHoleOptions {
  surfacePoint: Readonly<Vec3>;
  surfaceNormal: Readonly<Vec3>;
  diameter: number;
  depth?: number;
  outerChamfer?: number;
  taperAngleDeg?: number;
  segments?: number;
  booleanMethod?: BooleanMethod;
  fallbackBvhSdf?: boolean;
  generatePlug?: boolean;
  plugClearance?: number;
  plugNormals?: boolean;
}

export interface DrainHoleReport {
  holesPunched: number;
  keptTriangles: number;
  usedFallback: boolean;
  plugs: MeshHandle[];
  methodUsed: BooleanMethod;
  uvsDropped: boolean;
}

export interface PlaceTransform {
  rotation: number[];
  translation: Vec3;
  scale: number;
}

export interface PlaceMeshOptions {
  hitPoint: Readonly<Vec3>;
  hitNormal: Readonly<Vec3>;
  /**
   * The point of the stamp that goes to the hit point, in the stamp's own
   * coordinates: those it was added with, before its placement (of which only
   * the scale counts). Defaults to their zero; for a stamp more than 1024
   * times its size away from that, such as one modelled far from the world
   * origin, to the middle of its underside (the side of its bounds away from
   * `upAxis`).
   */
  stampAnchor?: Readonly<Vec3>;
  upAxis?: PlaceUpAxis;
  twistAngleRad?: number;
  uniformScale?: number;
  embedDepth?: number;
  mode?: PlaceMode;
  conformToSurface?: boolean;
  booleanMethod?: BooleanMethod;
  normals?: boolean;
}

export interface PlaceMeshReport {
  /** Takes the stamp's own coordinates, times its placement's scale, to the world: scaled, then turned, then moved. */
  transform: PlaceTransform;
  keptTriangles: number;
  usedFallback: boolean;
  conformMaxDisplacement: number;
  added: MeshHandle | null;
  uvsDropped: boolean;
}

const HOLLOW_BACKEND_CODES: Record<HollowBackend, number> = { bvhSdf: 0, fhEdt: 1 };
const PLACE_UP_AXIS_CODES: Record<PlaceUpAxis, number> = { negZ: 0, posZ: 1, posY: 2, negY: 3 };
const PLACE_MODE_CODES: Record<PlaceMode, number> = {
  keepSeparate: 0,
  appendShell: 1,
  embedUnion: 2,
  engraveSubtract: 3,
};

export interface NormalizedHollowArgs {
  wallThickness: number;
  backend: number;
  voxelSize: number;
  maxVoxels: number;
  maxCavityTriangles: number;
  minCavityVolume: number;
  modulationVerts: Uint32Array | undefined;
  modulationWeights: Float32Array | undefined;
}

export function normalizeHollowArgs(options: HollowOptions): NormalizedHollowArgs {
  if (options == null || typeof options !== "object") {
    throw invalid("hollow options must be an object");
  }
  let modulationVerts: Uint32Array | undefined;
  let modulationWeights: Float32Array | undefined;
  if (options.wallModulation !== undefined) {
    modulationVerts = Uint32Array.from(normalizeU32List(options.wallModulation.vertices, "wallModulation.vertices"));
    modulationWeights = toF32Array(options.wallModulation.weights, "wallModulation.weights");
    if (modulationVerts.length !== modulationWeights.length) {
      throw invalid("wallModulation.vertices and weights must have the same length");
    }
  }
  return {
    wallThickness: checkPositive(options.wallThickness, "wallThickness"),
    backend: codeIn(HOLLOW_BACKEND_CODES, options.backend ?? "bvhSdf", "backend"),
    voxelSize: options.voxelSize === undefined ? Number.NaN : checkPositive(options.voxelSize, "voxelSize"),
    maxVoxels: options.maxVoxels === undefined ? Number.NaN : checkU32Int(options.maxVoxels, "maxVoxels"),
    maxCavityTriangles:
      options.maxCavityTriangles === undefined
        ? Number.NaN
        : checkU32Int(options.maxCavityTriangles, "maxCavityTriangles"),
    minCavityVolume:
      options.minCavityVolume === undefined ? Number.NaN : checkNonNegative(options.minCavityVolume, "minCavityVolume"),
    modulationVerts,
    modulationWeights,
  };
}

export function decodeHollowReport(raw: Float64Array): HollowReport {
  return {
    cavitiesCreated: raw[0],
    cavityVertices: raw[1],
    cavityTriangles: raw[2],
    cavityVolume: raw[3],
    voxelSize: raw[4],
    uvsDropped: raw[5] !== 0,
  };
}

export interface NormalizedDrainHoleArgs {
  surfacePoint: Float64Array;
  surfaceNormal: Float64Array;
  diameter: number;
  depth: number;
  outerChamfer: number;
  taperAngleDeg: number;
  segments: number;
  booleanMethod: number;
  fallbackBvhSdf: boolean;
  generatePlug: boolean;
  plugClearance: number;
  plugNormals?: boolean;
}

export function normalizeDrainHoleArgs(options: DrainHoleOptions): NormalizedDrainHoleArgs {
  if (options == null || typeof options !== "object") {
    throw invalid("drainHole options must be an object");
  }
  return {
    surfacePoint: Float64Array.from(normalizeVec3(options.surfacePoint, "surfacePoint")),
    surfaceNormal: Float64Array.from(normalizeVec3(options.surfaceNormal, "surfaceNormal")),
    diameter: checkPositive(options.diameter, "diameter"),
    depth: options.depth === undefined ? Number.NaN : checkPositive(options.depth, "depth"),
    outerChamfer: checkNonNegative(options.outerChamfer ?? 0, "outerChamfer"),
    taperAngleDeg: checkFinite(options.taperAngleDeg ?? 0, "taperAngleDeg"),
    segments: checkU32Int(options.segments ?? 32, "segments"),
    booleanMethod: codeIn(BOOLEAN_METHOD_CODES, options.booleanMethod ?? "manifold", "booleanMethod"),
    fallbackBvhSdf: checkBoolean(options.fallbackBvhSdf, true, "fallbackBvhSdf"),
    generatePlug: checkBoolean(options.generatePlug, false, "generatePlug"),
    plugClearance: checkNonNegative(options.plugClearance ?? 0.1, "plugClearance"),
    ...(options.plugNormals === undefined ? {} : { plugNormals: checkBoolean(options.plugNormals, true, "plugNormals") }),
  };
}

export function decodeDrainHoleReport<T>(
  raw: Float64Array,
  mapHandle: (index: number, generation: number) => T,
): {
  holesPunched: number;
  keptTriangles: number;
  usedFallback: boolean;
  plugs: T[];
  methodUsed: BooleanMethod;
  uvsDropped: boolean;
} {
  const plugCount = raw[3];
  const plugs: T[] = [];
  let cur = 4;
  for (let i = 0; i < plugCount; i++) {
    plugs.push(mapHandle(raw[cur], raw[cur + 1]));
    cur += 2;
  }
  return {
    holesPunched: raw[0],
    keptTriangles: raw[1],
    usedFallback: raw[2] !== 0,
    plugs,
    methodUsed: BOOLEAN_METHOD_NAMES[raw[cur]] ?? "manifold",
    uvsDropped: raw[cur + 1] !== 0,
  };
}

export interface NormalizedPlaceMeshArgs {
  hitPoint: Float64Array;
  hitNormal: Float64Array;
  upAxis: number;
  twistAngleRad: number;
  uniformScale: number;
  embedDepth: number;
  mode: number;
  conformToSurface: boolean;
  booleanMethod: number;
  normals?: boolean;
  stampAnchor: Float64Array | undefined;
}

export function normalizePlaceMeshArgs(options: PlaceMeshOptions): NormalizedPlaceMeshArgs {
  if (options == null || typeof options !== "object") {
    throw invalid("placeMesh options must be an object");
  }
  return {
    hitPoint: Float64Array.from(normalizeVec3(options.hitPoint, "hitPoint")),
    hitNormal: Float64Array.from(normalizeVec3(options.hitNormal, "hitNormal")),
    upAxis: codeIn(PLACE_UP_AXIS_CODES, options.upAxis ?? "posZ", "upAxis"),
    twistAngleRad: checkFinite(options.twistAngleRad ?? 0, "twistAngleRad"),
    uniformScale: checkPositive(options.uniformScale ?? 1, "uniformScale"),
    embedDepth: checkFinite(options.embedDepth ?? 0, "embedDepth"),
    mode: codeIn(PLACE_MODE_CODES, options.mode ?? "keepSeparate", "mode"),
    conformToSurface: checkBoolean(options.conformToSurface, false, "conformToSurface"),
    booleanMethod: codeIn(BOOLEAN_METHOD_CODES, options.booleanMethod ?? "manifold", "booleanMethod"),
    ...(options.normals === undefined ? {} : { normals: checkBoolean(options.normals, true, "normals") }),
    stampAnchor:
      options.stampAnchor === undefined
        ? undefined
        : Float64Array.from(normalizeVec3(options.stampAnchor, "stampAnchor")),
  };
}

export function decodePlaceMeshReport<T>(
  raw: Float64Array,
  mapHandle: (index: number, generation: number) => T,
): {
  transform: PlaceTransform;
  keptTriangles: number;
  usedFallback: boolean;
  conformMaxDisplacement: number;
  added: T | null;
  uvsDropped: boolean;
} {
  const added = raw[16] >= 0 ? mapHandle(raw[16], raw[17]) : null;
  return {
    transform: {
      rotation: Array.from(raw.subarray(0, 9)),
      translation: [raw[9], raw[10], raw[11]],
      scale: raw[12],
    },
    keptTriangles: raw[13],
    usedFallback: raw[14] !== 0,
    conformMaxDisplacement: raw[15],
    added,
    uvsDropped: raw[18] !== 0,
  };
}

export function normalizeU32Selection(selectedTris: Indices, what = "selectedTris"): Uint32Array {
  return Uint32Array.from(normalizeU32List(selectedTris, what));
}

export function normalizeSelectInSphereArgs(
  center: Readonly<Vec3>,
  radius: number,
  normalCone?: FaceSelectionNormalCone,
): { center: Float64Array; radius: number; normal: Float64Array | undefined; coneAngleRad: number } {
  return {
    center: Float64Array.from(normalizeVec3(center, "center")),
    radius: checkNonNegative(radius, "radius"),
    normal: normalCone === undefined ? undefined : Float64Array.from(normalizeVec3(normalCone.normal, "normalCone.normal")),
    coneAngleRad: normalCone === undefined ? 0 : checkNonNegative(normalCone.angleRad, "normalCone.angleRad"),
  };
}

export function normalizeSelectSmallIslandsArgs(
  maxFaceCount: number,
  maxArea?: number,
): { maxFaceCount: number; maxArea: number } {
  return {
    maxFaceCount: checkU32Int(maxFaceCount, "maxFaceCount"),
    maxArea: maxArea === undefined ? Number.NaN : checkNonNegative(maxArea, "maxArea"),
  };
}

export function normalizeGrowShrinkArgs(
  selectedTris: Indices,
  rings = 1,
  adjacency: AdjacencyKind = "edge",
): { selectedTris: Uint32Array; rings: number; adjacency: number } {
  return {
    selectedTris: normalizeU32Selection(selectedTris),
    rings: checkU32Int(rings, "rings"),
    adjacency: codeIn(ADJACENCY_CODES, adjacency, "adjacency"),
  };
}

export function normalizeVertexMaskSelectArgs(
  mask: Float32Array | ArrayLike<number>,
  threshold = 0.5,
  requireAllVertices = false,
): { mask: Float32Array; threshold: number; requireAllVertices: boolean } {
  return {
    mask: toF32Array(mask, "mask"),
    threshold: checkFinite(threshold, "threshold"),
    requireAllVertices: checkBoolean(requireAllVertices, false, "requireAllVertices"),
  };
}

export function normalizeBuildFfdLatticeArgs(
  dims: readonly [number, number, number],
  basis: FfdBasisInput = "bernstein",
  padding = 0,
): { dims: Uint32Array; basis: number; padding: number } {
  return {
    dims: normalizeFfdLatticeDims(dims),
    basis: codeIn(FFD_BASIS_CODES, basis, "basis"),
    padding: checkNonNegative(padding, "padding"),
  };
}

export function normalizeSliceContoursArgs(
  origin: Readonly<Vec3>,
  normal: Readonly<Vec3>,
): { origin: Float64Array; normal: Float64Array } {
  return {
    origin: Float64Array.from(normalizeVec3(origin, "origin")),
    normal: Float64Array.from(normalizeVec3(normal, "normal")),
  };
}

export function normalizeSliceContourStackArgs(
  origin: Readonly<Vec3>,
  normal: Readonly<Vec3>,
  spacing: number,
  count: number,
): { origin: Float64Array; normal: Float64Array; spacing: number; count: number } {
  return {
    ...normalizeSliceContoursArgs(origin, normal),
    spacing: checkPositive(spacing, "spacing"),
    count: checkU32Int(count, "count"),
  };
}

export function normalizeAlignToPrimitiveArgs(
  primitive: FittedPrimitive,
  targetAxis: Readonly<Vec3>,
  targetOrigin: Readonly<Vec3>,
): { prim9: Float64Array; targetAxis: Float64Array; targetOrigin: Float64Array } {
  return {
    prim9: encodeFittedPrimitive(primitive),
    targetAxis: Float64Array.from(normalizeVec3(targetAxis, "targetAxis")),
    targetOrigin: Float64Array.from(normalizeVec3(targetOrigin, "targetOrigin")),
  };
}

export function decodePlacement15(raw: Float64Array): Placement {
  return {
    rotation: Array.from(raw.subarray(0, 9)),
    translation: [raw[9], raw[10], raw[11]],
    scale: [raw[12], raw[13], raw[14]],
  };
}

export function checkU32Arg(value: number, what: string): number {
  return checkU32Int(value, what);
}

export function checkNonNegativeArg(value: number, what: string): number {
  return checkNonNegative(value, what);
}

export function checkBooleanArg(value: boolean | undefined, fallback: boolean, what: string): boolean {
  return checkBoolean(value, fallback, what);
}
