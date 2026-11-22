// Face selection by criteria (#258): which faces of a mesh a criterion
// picks. Engine queries run where the scene runs (on the Worker in the
// app): self-intersections and components through `inspect`, contacts with
// other meshes through `findContacts`. The per-face and per-edge tests
// (area, facing, overhang, degenerate and sliver, boundary, sharp edges,
// orientation) run here, in slices over a snapshot of the mesh's buffers
// (`face-criteria.ts`), following the engine's definitions where it has one.
import type { AdjacencyKind, MeshHandle, SceneLike, Vec3 } from "fast-mesh-wasm";
import {
  buildEdgeAdjacency,
  type CameraPose,
  expandToComponents,
  type MeshGeometry,
  type Pause,
  selectBoundaryFaces,
  selectByArea,
  selectDegenerate,
  selectFacingCamera,
  selectInconsistentOrientation,
  selectOverhangs,
  selectSharpFaces,
  selectSlivers,
  unionSlots,
  yieldToEventLoop,
} from "./face-criteria";

/** A way of picking faces. */
export type FaceCriterion =
  | "area"
  | "facing"
  | "overhang"
  | "selfIntersecting"
  | "degenerate"
  | "orientation"
  | "boundary"
  | "sharp"
  | "contact";

/** The criteria in the order the panel lists them, with their labels. */
export const CRITERION_OPTIONS: readonly { id: FaceCriterion; label: string }[] = [
  { id: "area", label: "Area" },
  { id: "facing", label: "Facing the camera" },
  { id: "overhang", label: "Overhangs" },
  { id: "selfIntersecting", label: "Self-intersecting" },
  { id: "degenerate", label: "Degenerate or sliver" },
  { id: "orientation", label: "Inconsistently oriented" },
  { id: "boundary", label: "Boundary-adjacent" },
  { id: "sharp", label: "Sharp edges" },
  { id: "contact", label: "Touching other meshes" },
];

/** A build direction along a world axis. */
export type BuildAxis = "+x" | "-x" | "+y" | "-y" | "+z" | "-z";

export const BUILD_AXES: readonly BuildAxis[] = ["+z", "-z", "+y", "-y", "+x", "-x"];

/** The unit vector of a build axis. */
export function buildAxisVector(axis: BuildAxis): Vec3 {
  const sign = axis[0] === "-" ? -1 : 1;
  const i = axis[1] === "x" ? 0 : axis[1] === "y" ? 1 : 2;
  const v: Vec3 = [0, 0, 0];
  v[i] = sign;
  return v;
}

/** The criteria's settings. Lengths and areas are in millimetres (the world unit). */
export interface CriteriaParams {
  areaMode: "below" | "above";
  /** mm². */
  areaThreshold: number;
  /** Largest angle between a face's normal and the direction to the camera. */
  facingAngleDeg: number;
  /** How far past horizontal a face tilts down before it overhangs. */
  overhangAngleDeg: number;
  buildAxis: BuildAxis;
  /** Circumradius over twice the inradius at which a face is a sliver. */
  sliverAspect: number;
  /** Angle between the normals of two faces on an edge at which it is sharp. */
  sharpAngleDeg: number;
  /** mm: faces at most this far from another visible mesh. */
  contactMargin: number;
}

export const DEFAULT_CRITERIA_PARAMS: CriteriaParams = {
  areaMode: "below",
  areaThreshold: 0.01,
  facingAngleDeg: 45,
  overhangAngleDeg: 45,
  buildAxis: "+z",
  sliverAspect: 20,
  sharpAngleDeg: 45,
  contactMargin: 0,
};

/** What a criterion runs against. */
export interface CriterionTarget {
  scene: SceneLike;
  mesh: MeshHandle;
  /** The other visible meshes, for the contact test. */
  others: readonly MeshHandle[];
  /** The camera, for the facing test. */
  camera: CameraPose | null;
  /** Edge or vertex adjacency, for the boundary test. */
  adjacency: AdjacencyKind;
  pause?: Pause;
}

/** A copy of the mesh's render buffers and its world matrix (views can move under a later engine call). */
export function snapshotGeometry(scene: SceneLike, mesh: MeshHandle): MeshGeometry {
  const views = scene.views(mesh);
  return {
    positions: views.positions().slice(),
    indices: views.indices().slice(),
    matrix: Float64Array.from(scene.matrix4(mesh)),
  };
}

function requireInspect(scene: SceneLike): NonNullable<SceneLike["inspect"]> {
  const inspect = scene.inspect;
  if (!inspect) {
    throw new Error("The inspect domain is unavailable on this scene.");
  }
  return inspect;
}

/** The faces of `target.mesh` that `criterion` picks, ascending triangle slots. */
export async function findFacesByCriterion(
  criterion: FaceCriterion,
  params: CriteriaParams,
  target: CriterionTarget,
): Promise<Uint32Array> {
  const { scene, mesh } = target;
  const pause = target.pause ?? yieldToEventLoop;
  switch (criterion) {
    case "selfIntersecting":
      return Promise.resolve(requireInspect(scene).selfIntersectingTriangles(mesh));
    case "contact": {
      const others = target.others.filter((m) => m !== mesh);
      if (others.length === 0) {
        throw new Error("No other visible mesh to test contacts against.");
      }
      const contacts = await Promise.resolve(
        requireInspect(scene).findContacts({
          margin: Math.max(0, params.contactMargin),
          pairs: others.map((o): [MeshHandle, MeshHandle] => [mesh, o]),
        }),
      );
      const slots = Math.floor(scene.views(mesh).indices().length / 3);
      return unionSlots(
        contacts.map((c) => (c.a === mesh ? c.trianglesA : c.trianglesB)),
        slots,
      );
    }
    case "degenerate": {
      const geom = snapshotGeometry(scene, mesh);
      const degenerate = await selectDegenerate(geom, pause);
      const slivers = await selectSlivers(geom, params.sliverAspect, pause);
      return unionSlots([degenerate, slivers], Math.floor(geom.indices.length / 3));
    }
    case "area":
      return selectByArea(
        snapshotGeometry(scene, mesh),
        { mode: params.areaMode, threshold: params.areaThreshold },
        pause,
      );
    case "facing": {
      if (target.camera === null) {
        throw new Error("No camera to face.");
      }
      return selectFacingCamera(snapshotGeometry(scene, mesh), target.camera, params.facingAngleDeg, pause);
    }
    case "overhang":
      return selectOverhangs(
        snapshotGeometry(scene, mesh),
        buildAxisVector(params.buildAxis),
        params.overhangAngleDeg,
        pause,
      );
    case "orientation":
    case "boundary":
    case "sharp": {
      const geom = snapshotGeometry(scene, mesh);
      const vertexSlots = Math.floor(geom.positions.length / 3);
      const adjacency = await buildEdgeAdjacency(geom.indices, vertexSlots, pause);
      if (criterion === "orientation") {
        return selectInconsistentOrientation(geom, adjacency, pause);
      }
      if (criterion === "boundary") {
        return selectBoundaryFaces(geom.indices, vertexSlots, adjacency, target.adjacency, pause);
      }
      return selectSharpFaces(geom, adjacency, params.sharpAngleDeg, pause);
    }
  }
}

/** `selected` grown to the whole components (through shared edges) it touches, from the engine's analysis. */
export async function expandSelectionToComponents(
  scene: SceneLike,
  mesh: MeshHandle,
  selected: ArrayLike<number>,
): Promise<Uint32Array> {
  const analysis = await Promise.resolve(requireInspect(scene).analyze(mesh, { maxListed: 0 }));
  const slots = Math.floor(scene.views(mesh).indices().length / 3);
  return expandToComponents(analysis.components, selected, slots);
}

/** The line a criterion's run reports. */
export function criterionLabel(criterion: FaceCriterion): string {
  return CRITERION_OPTIONS.find((o) => o.id === criterion)?.label ?? criterion;
}
