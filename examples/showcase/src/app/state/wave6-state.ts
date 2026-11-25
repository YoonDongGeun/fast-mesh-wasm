import type {
  AdjacencyKind,
  ArapWeights,
  BooleanMethod,
  CollisionDistanceBackend,
  CollisionMode,
  CollisionRole,
  ContourSlice,
  CutTrigger,
  FfdBasis,
  FfdLatticeData,
  FittedPrimitiveReport,
  KelvinletScale,
  MaterialMemory,
  PackedPointCloud,
  PlaceUpAxis,
  SoftDeformSolver,
} from "fast-mesh-wasm";
import { DEFAULT_TEXT_PARAMS, type TextParams } from "../../features/mesh-edit/ux/text-mesh";

export interface Wave6SelectState {
  activeTool: null | "lasso" | "box" | "brush" | "crease" | "shell";
  combineMode: "replace" | "add" | "subtract" | "intersect";
  visibleOnly: boolean;
  backfaceCull: boolean;
  brushRadius: number;
  creaseAngleDeg: number;
  islandMaxFaces: number;
  adjacency: AdjacencyKind;
  selectedByMesh: Record<number, Uint32Array>;
}

export interface Wave6DeformState {
  algorithm: "ffd" | "arap" | "kelvinlet" | "xpbd";
  activeTool: null | "ffdCage" | "arapPin" | "arapHandle" | "kelvinletDrag" | "xpbdPin";
  ffdDims: [number, number, number];
  ffdBasis: FfdBasis;
  ffdLattice: FfdLatticeData | null;
  arapAnchors: number[];
  arapHandles: Array<{ vertex: number; target: [number, number, number] }>;
  arapWeights: ArapWeights;
  kelvinletKind: "grab" | "scale" | "twist" | "pinch";
  kelvinletScale: KelvinletScale;
  kelvinletNu: number;
  kelvinletEpsilon: number;
  kelvinletSymX: boolean;
  kelvinletSymY: boolean;
  kelvinletSymZ: boolean;
  xpbdSelfCollision: boolean;
  xpbdThickness: number;
  xpbdStretchCompliance: number;
  xpbdBendCompliance: number;
  xpbdSubsteps: number;
  xpbdProtectMask: boolean;
  lastReport: string | null;
}

export interface Wave6CollisionState {
  mode: CollisionMode;
  role: CollisionRole;
  distanceBackend: CollisionDistanceBackend;
  cutBackend: BooleanMethod;
  cutTrigger: CutTrigger;
  softSolver: SoftDeformSolver;
  materialMemory: MaterialMemory;
  rigidSlide: boolean;
  compliance: number;
  recoveryRate: number;
  preventSelfIntersection: boolean;
  lastStepSummary: string | null;
}

export interface Wave6MetrologyState {
  fittedPrimitive: FittedPrimitiveReport | null;
  showPrimitiveOverlay: boolean;
  contourSlices: ContourSlice[] | null;
  showContoursOverlay: boolean;
  pointCloud: PackedPointCloud | null;
  showPointCloudOverlay: boolean;
  pointCloudReport: string | null;
}

export interface Wave6PrintPrepState {
  activeTool: null | "drainHole" | "placeStamp";
  drainParams: {
    diameter: number;
    depth: number;
    outerChamfer: number;
    taperDeg: number;
    generatePlug: boolean;
  };
  placeParams: {
    stampSource: "peg" | "hexBoss" | "dome" | "cube" | "sceneMesh" | "text";
    stampMeshId: number | null;
    /** The text a `"text"` stamp spells, and its letter height, depth and spacing (#260). */
    text: TextParams;
    mode: "embedUnion" | "engraveSubtract" | "keepSeparate";
    upAxis: PlaceUpAxis;
    embedDepth: number;
    uniformScale: number;
    twistDeg: number;
    conformToSurface: boolean;
  };
}

export interface Wave6State {
  select: Wave6SelectState;
  deform: Wave6DeformState;
  collision: Wave6CollisionState;
  metrology: Wave6MetrologyState;
  printPrep: Wave6PrintPrepState;
}

export function createDefaultWave6State(): Wave6State {
  return {
    select: {
      activeTool: null,
      combineMode: "replace",
      visibleOnly: true,
      backfaceCull: true,
      brushRadius: 0.25,
      creaseAngleDeg: 35,
      islandMaxFaces: 24,
      adjacency: "edge",
      selectedByMesh: {},
    },
    deform: {
      algorithm: "ffd",
      activeTool: null,
      ffdDims: [3, 3, 3],
      ffdBasis: "bernstein",
      ffdLattice: null,
      arapAnchors: [],
      arapHandles: [],
      arapWeights: "cotangent",
      kelvinletKind: "grab",
      kelvinletScale: "bi",
      kelvinletNu: 0.35,
      kelvinletEpsilon: 0.4,
      kelvinletSymX: false,
      kelvinletSymY: false,
      kelvinletSymZ: false,
      xpbdSelfCollision: true,
      xpbdThickness: 0.02,
      xpbdStretchCompliance: 0.0,
      xpbdBendCompliance: 0.02,
      xpbdSubsteps: 4,
      xpbdProtectMask: true,
      lastReport: null,
    },
    collision: {
      mode: "passThrough",
      role: "symmetric",
      distanceBackend: "bvhSdf",
      cutBackend: "manifold",
      cutTrigger: "onRelease",
      softSolver: "sdfLaplacian",
      materialMemory: "elastic",
      rigidSlide: true,
      compliance: Math.fround(0.05),
      recoveryRate: 1.0,
      preventSelfIntersection: true,
      lastStepSummary: null,
    },
    metrology: {
      fittedPrimitive: null,
      showPrimitiveOverlay: false,
      contourSlices: null,
      showContoursOverlay: false,
      pointCloud: null,
      showPointCloudOverlay: false,
      pointCloudReport: null,
    },
    printPrep: {
      activeTool: null,
      drainParams: {
        diameter: 4.0,
        depth: 12.0,
        outerChamfer: 0.5,
        taperDeg: 0,
        generatePlug: false,
      },
      placeParams: {
        stampSource: "peg",
        stampMeshId: null,
        text: { ...DEFAULT_TEXT_PARAMS },
        mode: "embedUnion",
        upAxis: "posZ",
        embedDepth: 0.5,
        uniformScale: 1.0,
        twistDeg: 0,
        conformToSurface: false,
      },
    },
  };
}

export const DEFAULT_WAVE6_STATE: Wave6State = createDefaultWave6State();
