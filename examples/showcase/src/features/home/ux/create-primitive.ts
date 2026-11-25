// The Create dialog's behaviour (#259): the shapes it offers, each one's
// parameters (lengths in millimetres, resolutions as counts) with defaults
// scaled to the scene, the shape they describe for the engine's
// `generatePrimitive`, their checks and triangle count, the default name,
// where the generator runs, and adding the result as one undo step.
//
// Generation runs where the engine does: in the Worker scene's Worker
// (`WorkerScene.generatePrimitive`, so the page loads no wasm of its own),
// or on the main thread with `?local` (the module is initialised already).
//
// Adding: the mesh is added at the chosen centre, with the rotation baked
// into the positions, as one undo step: undo removes it (and its row) and
// redo brings it back.
import type { MeshHandle, PrimitiveMesh, PrimitiveShape, PrimitiveShapeKind, SceneLike } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { bounds, type Vec3 } from "../../../io/mesh-data";
import type { AppState, Item } from "../../../app/state/app-state";
import { ensureWave6Wasm } from "../../../engine/wave6-wasm-lazy";
import { isWorkerScene } from "../../../engine/workbench-ops";
import { eulerDegToMat3 } from "../../transform/ux/transform-math";

/** A length (millimetres) or a resolution (a whole number). */
export interface ParamSpec {
  key: string;
  label: string;
  type: "length" | "count";
  /** The default: a length as a multiple of the dialog's base size, a count as is. */
  value: number;
  /** Lengths: whether 0 is allowed (a cone's apex). Counts: the smallest value. */
  min: number;
  /** Counts only: the largest value. */
  max?: number;
}

/** A shape button: its kind, label and parameters, in the order they show. */
export interface ShapeSpec {
  kind: PrimitiveShapeKind;
  label: string;
  params: readonly ParamSpec[];
}

const length = (key: string, label: string, value: number, min = 1): ParamSpec => ({
  key,
  label,
  type: "length",
  value,
  min,
});
const count = (key: string, label: string, value: number, min: number, max: number): ParamSpec => ({
  key,
  label,
  type: "count",
  value,
  min,
  max,
});

/** A length that may be 0 (`min: 0`); others must be positive (`min: 1`). */
const ZERO_OK = 0;
const SIDES = 8192;
const BANDS = 4096;

/** The shapes the dialog offers, in button order. */
export const CREATE_SHAPES: readonly ShapeSpec[] = [
  {
    kind: "box",
    label: "Box",
    params: [
      length("sizeX", "Width (X)", 1),
      length("sizeY", "Height (Y)", 1),
      length("sizeZ", "Depth (Z)", 1),
      count("segX", "Segments X", 1, 1, 2048),
      count("segY", "Segments Y", 1, 1, 2048),
      count("segZ", "Segments Z", 1, 1, 2048),
    ],
  },
  {
    kind: "cylinder",
    label: "Cylinder",
    params: [
      length("radius", "Radius", 0.5),
      length("height", "Height", 1),
      count("segments", "Sides", 48, 3, SIDES),
      count("rings", "Height segments", 1, 1, BANDS),
      count("capRings", "Cap rings", 1, 1, BANDS),
    ],
  },
  {
    kind: "cone",
    label: "Cone",
    params: [
      length("radius", "Base radius", 0.5, ZERO_OK),
      length("topRadius", "Top radius", 0, ZERO_OK),
      length("height", "Height", 1),
      count("segments", "Sides", 48, 3, SIDES),
      count("rings", "Height segments", 1, 1, BANDS),
      count("capRings", "Cap rings", 1, 1, BANDS),
    ],
  },
  {
    kind: "uvSphere",
    label: "Sphere",
    params: [length("radius", "Radius", 0.5), count("segments", "Segments", 48, 3, SIDES), count("rings", "Rings", 24, 2, BANDS)],
  },
  {
    kind: "icosphere",
    label: "Icosphere",
    params: [length("radius", "Radius", 0.5), count("subdivisions", "Subdivisions", 3, 0, 9)],
  },
  {
    kind: "torus",
    label: "Torus",
    params: [
      length("majorRadius", "Ring radius", 0.375),
      length("minorRadius", "Tube radius", 0.125),
      count("segments", "Ring segments", 48, 3, SIDES),
      count("tubeSegments", "Tube segments", 24, 3, SIDES),
    ],
  },
  {
    kind: "arrow",
    label: "Arrow",
    params: [
      length("length", "Length", 1),
      length("shaftRadius", "Shaft radius", 0.05),
      length("headRadius", "Head radius", 0.1),
      length("headLength", "Head length", 0.25),
      count("segments", "Sides", 32, 3, SIDES),
    ],
  },
  {
    kind: "prism",
    label: "Prism",
    params: [count("sides", "Sides", 6, 3, 1024), length("radius", "Radius", 0.5), length("height", "Height", 1)],
  },
  {
    kind: "plane",
    label: "Plane",
    params: [
      length("width", "Width (X)", 1),
      length("depth", "Depth (Z)", 1),
      count("segX", "Segments X", 10, 1, BANDS),
      count("segZ", "Segments Z", 10, 1, BANDS),
    ],
  },
];

/** The spec of `kind`. */
export function shapeSpec(kind: PrimitiveShapeKind): ShapeSpec {
  const spec = CREATE_SHAPES.find((s) => s.kind === kind);
  if (spec === undefined) {
    throw new Error(`no shape ${kind}`);
  }
  return spec;
}

/** One shape's parameter values, by spec key: millimetres and counts. */
export type ShapeParams = Readonly<Record<string, number>>;

/** The most triangles the dialog creates, to keep the page responsive. */
export const MAX_CREATE_TRIANGLES = 10_000_000;

/** How long parameter edits settle before the preview regenerates. */
export const PREVIEW_DEBOUNCE_MS = 120;

/** 1, 2 or 5 times a power of ten, the nearest to `x` (on a log scale). */
export function niceSize(x: number): number {
  if (!(x > 0) || !Number.isFinite(x)) {
    return 1;
  }
  const p = 10 ** Math.floor(Math.log10(x));
  const m = x / p;
  const step = m < Math.sqrt(2) ? 1 : m < Math.sqrt(10) ? 2 : m < Math.sqrt(50) ? 5 : 10;
  return step * p;
}

/**
 * The dialog's base size in millimetres: twice the median radius (half
 * bounding-box diagonal) of the listed meshes, rounded to 1, 2 or 5 times a
 * power of ten, so a new shape is about their size; 2 mm in an empty scene.
 */
export function defaultBaseSize(items: readonly Pick<Item, "radius">[]): number {
  const radii = items.map((i) => i.radius).filter((r) => r > 0 && Number.isFinite(r));
  if (radii.length === 0) {
    return 2;
  }
  radii.sort((a, b) => a - b);
  return niceSize(2 * radii[Math.floor(radii.length / 2)]);
}

/** The default values of `kind` at base size `size` (millimetres). */
export function defaultParams(kind: PrimitiveShapeKind, size: number): ShapeParams {
  const out: Record<string, number> = {};
  for (const p of shapeSpec(kind).params) {
    out[p.key] = p.type === "length" ? p.value * size : p.value;
  }
  return out;
}

/** The engine shape `params` describe (lengths in millimetres, the app's world unit). */
export function toShape(kind: PrimitiveShapeKind, params: ShapeParams): PrimitiveShape {
  const p = params;
  switch (kind) {
    case "box":
      return { kind, size: [p.sizeX, p.sizeY, p.sizeZ], segments: [p.segX, p.segY, p.segZ] };
    case "plane":
      return { kind, size: [p.width, p.depth], segments: [p.segX, p.segZ] };
    case "cylinder":
      return { kind, radius: p.radius, height: p.height, segments: p.segments, rings: p.rings, capRings: p.capRings };
    case "cone":
      return {
        kind,
        radius: p.radius,
        topRadius: p.topRadius,
        height: p.height,
        segments: p.segments,
        rings: p.rings,
        capRings: p.capRings,
      };
    case "uvSphere":
      return { kind, radius: p.radius, segments: p.segments, rings: p.rings };
    case "icosphere":
      return { kind, radius: p.radius, subdivisions: p.subdivisions };
    case "torus":
      return {
        kind,
        majorRadius: p.majorRadius,
        minorRadius: p.minorRadius,
        segments: p.segments,
        tubeSegments: p.tubeSegments,
      };
    case "arrow":
      return {
        kind,
        length: p.length,
        shaftRadius: p.shaftRadius,
        headRadius: p.headRadius,
        headLength: p.headLength,
        segments: p.segments,
      };
    case "prism":
      return { kind, sides: p.sides, radius: p.radius, height: p.height };
  }
}

/** The triangles `params` make, as the engine builds them (without building them). */
export function estimateTriangles(kind: PrimitiveShapeKind, p: ShapeParams): number {
  // A shape of revolution: a ring per profile point off the axis, and
  // 2 · sides triangles per ring (the bands between them and the fans at
  // the poles).
  const lathe = (sides: number, rings: number) => 2 * sides * rings;
  switch (kind) {
    case "box":
      return 4 * (p.segX * p.segY + p.segY * p.segZ + p.segZ * p.segX);
    case "plane":
      return 2 * p.segX * p.segZ;
    case "cylinder":
      return lathe(p.segments, p.rings - 1 + 2 * p.capRings);
    case "cone":
      return lathe(p.segments, p.rings - 1 + (p.radius > 0 ? p.capRings : 0) + (p.topRadius > 0 ? p.capRings : 0));
    case "uvSphere":
      return lathe(p.segments, p.rings - 1);
    case "icosphere":
      return 20 * 4 ** p.subdivisions;
    case "torus":
      return lathe(p.segments, p.tubeSegments);
    case "arrow":
      // Tail rim, shaft top, head rim.
      return lathe(p.segments, 3);
    case "prism":
      return lathe(p.sides, 2);
  }
}

/**
 * Why `params` cannot be created, in words, or `null` when they can: each
 * value in range, the shape's own rules (a torus tube inside its ring, an
 * arrow head wider than its shaft), and at most {@link MAX_CREATE_TRIANGLES}.
 */
export function paramsProblem(kind: PrimitiveShapeKind, p: ShapeParams): string | null {
  for (const spec of shapeSpec(kind).params) {
    const v = p[spec.key];
    if (typeof v !== "number" || !Number.isFinite(v)) {
      return `${spec.label} must be a number.`;
    }
    if (spec.type === "length") {
      if (spec.min === ZERO_OK ? v < 0 : v <= 0) {
        return `${spec.label} must be ${spec.min === ZERO_OK ? "zero or more" : "above zero"}.`;
      }
    } else if (!Number.isInteger(v) || v < spec.min || v > (spec.max ?? Infinity)) {
      return `${spec.label} must be a whole number from ${spec.min} to ${spec.max}.`;
    }
  }
  if (kind === "cone" && p.radius === 0 && p.topRadius === 0) {
    return "A cone needs a base or a top radius above zero.";
  }
  if (kind === "torus" && p.minorRadius >= p.majorRadius) {
    return "The tube radius must be below the ring radius.";
  }
  if (kind === "arrow" && p.headRadius <= p.shaftRadius) {
    return "The head radius must be above the shaft radius.";
  }
  if (kind === "arrow" && p.headLength >= p.length) {
    return "The head length must be below the length.";
  }
  const triangles = estimateTriangles(kind, p);
  if (triangles > MAX_CREATE_TRIANGLES) {
    return `${triangles.toLocaleString("en")} triangles: at most ${MAX_CREATE_TRIANGLES.toLocaleString("en")}.`;
  }
  return null;
}

/** `base`, or `base 2`, `base 3`, ... : the first not among `names`. */
export function uniqueName(base: string, names: readonly string[]): string {
  const taken = new Set(names);
  if (!taken.has(base)) {
    return base;
  }
  let n = 2;
  while (taken.has(`${base} ${n}`)) {
    n++;
  }
  return `${base} ${n}`;
}

/**
 * Generates `shape` where the engine runs: in the Worker scene's Worker, or
 * on the main thread for the synchronous `Scene` (`?local`).
 */
export function generateShape(scene: SceneLike, shape: PrimitiveShape): Promise<PrimitiveMesh> {
  if (isWorkerScene(scene)) {
    return scene.generatePrimitive(shape);
  }
  return ensureWave6Wasm().then((m) => m.generatePrimitive(shape));
}

/** `positions` rotated by `rotationDeg` (Euler XYZ, as the transform panel) and moved to `centre`, in `f64`. */
export function placePositions(positions: Float32Array, centre: Readonly<Vec3>, rotationDeg: Readonly<Vec3>): Float64Array {
  const r = eulerDegToMat3([rotationDeg[0], rotationDeg[1], rotationDeg[2]]);
  const out = new Float64Array(positions.length);
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i];
    const y = positions[i + 1];
    const z = positions[i + 2];
    out[i] = centre[0] + r[0] * x + r[1] * y + r[2] * z;
    out[i + 1] = centre[1] + r[3] * x + r[4] * y + r[5] * z;
    out[i + 2] = centre[2] + r[6] * x + r[7] * y + r[8] * z;
  }
  return out;
}

/** Where and as what a created mesh lands. */
export interface CreatePlacement {
  name: string;
  /** World millimetres: the mesh origin, where the gizmo sits. */
  centre: Vec3;
  /** Euler XYZ degrees, baked into the positions. */
  rotationDeg: Vec3;
}

/**
 * Settles once the scene has `mesh` (a Worker scene adds it later): resolves
 * also when it is gone again by then (undone), rejects when its add failed.
 */
export function landed(store: SceneStore, mesh: MeshHandle): Promise<void> {
  return new Promise((resolve, reject) => {
    let off: (() => void) | undefined;
    const check = (): boolean => {
      const state = store.getSnapshot().meshes.get(mesh);
      if (state?.status === "computing") {
        return false;
      }
      off?.();
      if (state?.status === "error") {
        reject(state.error ?? new Error("The mesh could not be added."));
      } else {
        resolve();
      }
      return true;
    };
    if (!check()) {
      off = store.subscribe(check);
    }
  });
}

/**
 * Adds `mesh` to the scene and the list as one undo step (see the file
 * comment), with its origin at the centre. The item lists at once; `done`
 * settles when the mesh has landed, and on failure it is removed again.
 *
 * @throws when the scene refuses the mesh at once (nothing is added).
 */
export function addPrimitive(
  store: SceneStore,
  app: AppState,
  mesh: PrimitiveMesh,
  place: CreatePlacement,
): { item: Item; done: Promise<void> } {
  const world = placePositions(mesh.positions, place.centre, place.rotationDeg);
  const handle: MeshHandle = store.addMesh(world, mesh.indices, { origin: place.centre, undoable: true });
  const item = app.listMesh(handle, {
    name: place.name,
    triangles: mesh.indices.length / 3,
    center: [place.centre[0], place.centre[1], place.centre[2]],
    radius: bounds(world).radius,
  });
  const done = landed(store, handle).catch((e: unknown) => {
    app.removeMesh(store, item, { undoable: false });
    throw e;
  });
  return { item, done };
}
