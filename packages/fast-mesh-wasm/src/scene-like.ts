// The part of a scene the renderer adapters (`fast-mesh-wasm/three`, ...) depend
// on. `Scene` implements it; so can any other host that mirrors a scene
// (for example one running in a Worker). Handles stay opaque: adapters only
// pass them back to these methods. A host other than `Scene` hands out
// instances of the same nominal classes (`new MeshHandle()`,
// `new SectionHandle()`) and keeps its own map from them to its ids.
import type { Bounds, Indices, Placement, Vec3 } from "./api.js";
import type {
  SyncAlignDomain,
  SyncBooleanDomain,
  SyncDeformDomain,
  SyncInspectDomain,
  SyncRemeshDomain,
  SyncRepairDomain,
  SyncSculptDomain,
  WorkerAlignDomain,
  WorkerBooleanDomain,
  WorkerDeformDomain,
  WorkerInspectDomain,
  WorkerRemeshDomain,
  WorkerRepairDomain,
  WorkerSculptDomain,
} from "./domains/index.js";
import type { MeshViews } from "./views.js";
import type {
  BridgeStats,
  ColorRule,
  DragSession,
  DyntopoDabReport,
  DyntopoOptions,
  DyntopoReport,
  FillContinuityStats,
  FillHolesContinuityOptions,
  MeshChange,
  MeshHandle,
  PartialBridgeOptions,
  SculptBrushOp,
  SectionHandle,
  SectionPlane,
  SectionResult,
  VertexProtection,
} from "./scene.js";
import type {
  CollisionRule,
  CollisionRuleInput,
  CollisionStepReport,
} from "./wave6.js";

/** Domain-separated capability surface exposed by both `Scene` and `WorkerScene`. */
export interface SceneDomainsLike {
  readonly repair?: SyncRepairDomain | WorkerRepairDomain;
  readonly sculpt?: SyncSculptDomain | WorkerSculptDomain;
  readonly remesh?: SyncRemeshDomain | WorkerRemeshDomain;
  readonly boolean?: SyncBooleanDomain | WorkerBooleanDomain;
  readonly deform?: SyncDeformDomain | WorkerDeformDomain;
  readonly inspect?: SyncInspectDomain | WorkerInspectDomain;
  readonly align?: SyncAlignDomain | WorkerAlignDomain;
}

/**
 * What the adapters need from a scene: render buffers, placements, section
 * results, change notification and drags. All methods are synchronous.
 * `Scene` and `WorkerScene` also attach the seven optional domain namespaces.
 */
export interface SceneLike extends SceneDomainsLike {
  /**
   * Views of the mesh's render buffers. Slot counts follow from them, so
   * `positions()` must be exactly 3 × vertex slots long and `indices()`
   * exactly 3 × triangle slots long (normals 3, colours 4 and UVs 2 per vertex
   * slot, when present): `positions().length / 3` vertex slots and
   * `indices().length / 3` triangle slots.
   */
  views(mesh: MeshHandle): MeshViews;
  /** The column-major world matrix of the mesh's local render coordinates. */
  matrix4(mesh: MeshHandle): Float64Array;
  placement(mesh: MeshHandle): Placement;
  section(s: SectionHandle, mesh: MeshHandle): SectionResult;
  sectionPlane(s: SectionHandle): SectionPlane;
  /** Registers a change listener; returns the function that removes it. */
  onChanges(listener: (c: MeshChange[]) => void): () => void;
  /** Flushes pending changes to the listeners and returns them. */
  commitFrame(): MeshChange[];
  drag(mesh: MeshHandle): DragSession;
  /** Opens a nested history step when supported by the scene host. */
  beginStep?(): void;
  /** Ends one open history step nesting level when supported by the scene host. */
  endStep?(): void;
  /** Cancels and reverts the active history step when supported by the scene host. */
  cancelStep?(): void;
  /** Fills a single boundary loop with selectable G0/G1/G2 continuity. */
  fillHoleContinuity?(
    mesh: MeshHandle,
    loopIndex: number,
    options?: FillHolesContinuityOptions,
  ): FillContinuityStats | Promise<FillContinuityStats>;
  /** Fills all eligible holes with selectable G0/G1/G2 continuity. */
  fillAllHolesContinuity?(
    mesh: MeshHandle,
    options?: FillHolesContinuityOptions,
  ): FillContinuityStats | Promise<FillContinuityStats>;
  /** Bridges two boundary vertex arcs or index ranges. */
  bridgePartialArcs?(
    mesh: MeshHandle,
    loopA: Indices,
    loopB: Indices,
    options?: PartialBridgeOptions,
  ): BridgeStats | Promise<BridgeStats>;
  /** Applies localized dynamic topology remeshing inside sphere `(center, radius)`. */
  applyDyntopo?(
    mesh: MeshHandle,
    center: Readonly<Vec3>,
    radius: number,
    options?: DyntopoOptions,
    pinned?: Indices,
  ): DyntopoReport | Promise<DyntopoReport>;
  /** Applies localized dynamic topology followed by a brush dab. */
  dyntopoDab?(
    mesh: MeshHandle,
    dab: SculptBrushOp,
    options?: DyntopoOptions,
    protection?: VertexProtection,
  ): DyntopoDabReport | Promise<DyntopoDabReport>;
  /**
   * The rule the mesh is coloured by, or `null` without colouring. Optional:
   * `bindMesh`'s fragment shading needs it (with `views(mesh).values`) and
   * falls back to vertex colours on a host without it.
   */
  coloring?(mesh: MeshHandle): ColorRule | null;
  /**
   * The box around the vertices the mesh's live triangles use, in local
   * render coordinates, or `null` without live triangles. Optional:
   * `bindMesh` takes its bounds from it and walks the mirrors on a host
   * without it.
   */
  bounds?(mesh: MeshHandle): Bounds | null;
  setDefaultCollisionRule?(rule: CollisionRuleInput): void;
  defaultCollisionRule?(): CollisionRule | Promise<CollisionRule>;
  setCollisionRule?(a: MeshHandle, b: MeshHandle, rule: CollisionRuleInput): void;
  clearCollisionRule?(a: MeshHandle, b: MeshHandle): boolean | Promise<boolean>;
  collisionRule?(a: MeshHandle, b: MeshHandle): CollisionRule | Promise<CollisionRule>;
  hasActiveCollisionResponse?(mesh: MeshHandle): boolean | Promise<boolean>;
  moveWithCollision?(
    mesh: MeshHandle,
    placement: import("./api.js").PlacementInput,
    isRelease?: boolean,
  ): CollisionStepReport | Promise<CollisionStepReport>;
  commitCollisionInteraction?(mesh: MeshHandle): CollisionStepReport | Promise<CollisionStepReport>;
}
