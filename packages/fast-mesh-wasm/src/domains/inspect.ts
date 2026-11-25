import type { Bounds, Placement, Vec3 } from "../api.js";
import { FastMeshError } from "../core.js";
import type {
  Analysis,
  AnalyzeOptions,
  ClosestHit,
  ClosestPointOptions,
  ColorRule,
  Contact,
  CurvatureFieldDesc,
  DeviationOptions,
  DeviationStats,
  DistanceFieldDesc,
  FieldHandle,
  FieldStatus,
  FindContactsOptions,
  GeodesicOptions,
  MeasureResult,
  MeshDistance,
  MeshHandle,
  OverhangOptions,
  OverhangReport,
  QualityMetric,
  QualityReport,
  RaycastOptions,
  RayHit,
  SectionDesc,
  SectionGap,
  SectionHandle,
  SectionPlane,
  SectionResult,
  SurfacePathOptions,
  SurfacePathPoint,
  ThicknessFieldDesc,
  TriangleQualityOptions,
} from "../scene.js";
import type {
  ContourSlice,
  FitPrimitiveOptions,
  FittedPrimitive,
  FittedPrimitiveReport,
  PackedPointCloud,
  SurfaceReconstructOptions,
} from "../wave6.js";

/** Return type parameterized over synchronous (`Scene`) vs worker (`WorkerScene`) hosts. */
export type InspectReturn<Async extends boolean, T> = Async extends true ? Promise<T> : T;

/** Low-level host backend used by {@link createInspectDomain}. */
export interface InspectBackend<Async extends boolean = false> {
  analyze(mesh: MeshHandle, options?: AnalyzeOptions): InspectReturn<Async, Analysis>;
  measure(mesh: MeshHandle): InspectReturn<Async, MeasureResult>;
  bounds(mesh: MeshHandle): Bounds | null;
  raycast(origin: Readonly<Vec3>, direction: Readonly<Vec3>, options?: RaycastOptions): InspectReturn<Async, RayHit | null>;
  closestPoint(point: Readonly<Vec3>, options?: ClosestPointOptions): InspectReturn<Async, ClosestHit | null>;
  meshDistance(
    a: MeshHandle,
    b: MeshHandle,
    options?: { maxDistance?: number },
  ): InspectReturn<Async, MeshDistance | null>;
  intersects(a: MeshHandle, b: MeshHandle): InspectReturn<Async, boolean>;
  selfIntersectingTriangles(mesh: MeshHandle): InspectReturn<Async, Uint32Array>;
  findContacts(options?: FindContactsOptions): InspectReturn<Async, Contact[]>;
  overhangs(mesh: MeshHandle, options?: OverhangOptions): InspectReturn<Async, OverhangReport>;
  triangleQuality(
    mesh: MeshHandle,
    metricOrOptions?: QualityMetric | TriangleQualityOptions,
    sliverThreshold?: number,
  ): InspectReturn<Async, QualityReport>;
  meshDeviation(source: MeshHandle, target: MeshHandle, options?: DeviationOptions): InspectReturn<Async, DeviationStats>;
  geodesicDistances(
    mesh: MeshHandle,
    sources: number | readonly number[],
    options?: GeodesicOptions,
  ): InspectReturn<Async, Float32Array>;
  shortestSurfacePath(
    mesh: MeshHandle,
    sources: number | readonly number[],
    targetVertex: number,
    options?: SurfacePathOptions,
  ): InspectReturn<Async, SurfacePathPoint[]>;
  releaseGeodesics?: () => InspectReturn<Async, void>;
  addDistanceField(desc: DistanceFieldDesc): FieldHandle;
  addThicknessField?: (desc: ThicknessFieldDesc) => FieldHandle;
  addCurvatureField?: (desc: CurvatureFieldDesc) => FieldHandle;
  removeField(field: FieldHandle): void;
  fieldValues(field: FieldHandle): InspectReturn<Async, Float32Array>;
  fieldNearest(field: FieldHandle): InspectReturn<Async, Uint32Array>;
  fieldStatus(field: FieldHandle): InspectReturn<Async, FieldStatus>;
  fieldTargetStatus(field: FieldHandle): InspectReturn<Async, FieldStatus["sign"][]>;
  setColoring(mesh: MeshHandle, rule: ColorRule | null): void;
  coloring(mesh: MeshHandle): ColorRule | null;
  addSection(desc: SectionDesc): SectionHandle;
  setSectionPlane(section: SectionHandle, plane: SectionPlane): void;
  sectionPlane(section: SectionHandle): SectionPlane;
  setSectionMeshes(section: SectionHandle, meshes: readonly MeshHandle[]): void;
  setSectionFill(section: SectionHandle, fill: boolean): void;
  removeSection(section: SectionHandle): void;
  section(section: SectionHandle, mesh: MeshHandle): SectionResult;
  sectionGap(section: SectionHandle, a: MeshHandle, b: MeshHandle): InspectReturn<Async, SectionGap | null>;
  updateSections?: () => void;
  fitPrimitive(mesh: MeshHandle, options?: FitPrimitiveOptions): InspectReturn<Async, FittedPrimitiveReport>;
  alignToFittedPrimitive(
    mesh: MeshHandle,
    primitive: FittedPrimitive,
    targetAxis: Readonly<Vec3>,
    targetOrigin: Readonly<Vec3>,
  ): InspectReturn<Async, Placement>;
  sliceContours(mesh: MeshHandle, origin: Readonly<Vec3>, normal: Readonly<Vec3>): InspectReturn<Async, ContourSlice>;
  sliceContourStack(
    mesh: MeshHandle,
    origin: Readonly<Vec3>,
    normal: Readonly<Vec3>,
    spacing: number,
    count: number,
  ): InspectReturn<Async, ContourSlice[]>;
  reconstructPointCloud(
    cloud: PackedPointCloud | Float32Array,
    options?: SurfaceReconstructOptions,
  ): InspectReturn<Async, MeshHandle>;
}

/** Metrology, defect analysis, raycasting, printability/quality/geodesic fields, surface paths, and cross-section inspection domain. */
export interface InspectDomain<Async extends boolean = false> {
  analyze(mesh: MeshHandle, options?: AnalyzeOptions): InspectReturn<Async, Analysis>;
  measure(mesh: MeshHandle): InspectReturn<Async, MeasureResult>;
  bounds(mesh: MeshHandle): Bounds | null;
  raycast(origin: Readonly<Vec3>, direction: Readonly<Vec3>, options?: RaycastOptions): InspectReturn<Async, RayHit | null>;
  closestPoint(point: Readonly<Vec3>, options?: ClosestPointOptions): InspectReturn<Async, ClosestHit | null>;
  meshDistance(
    a: MeshHandle,
    b: MeshHandle,
    options?: { maxDistance?: number },
  ): InspectReturn<Async, MeshDistance | null>;
  intersects(a: MeshHandle, b: MeshHandle): InspectReturn<Async, boolean>;
  /**
   * The live triangle slots of `mesh` that intersect or touch another of its
   * own triangles, ascending, each once. Triangles sharing a vertex are never
   * tested against each other and degenerate triangles are skipped. The
   * mesh's BVH is brought up to date first; a stale `"manual"` mesh throws
   * `STALE_BVH`.
   */
  selfIntersectingTriangles(mesh: MeshHandle): InspectReturn<Async, Uint32Array>;
  findContacts(options?: FindContactsOptions): InspectReturn<Async, Contact[]>;
  overhangs(mesh: MeshHandle, options?: OverhangOptions): InspectReturn<Async, OverhangReport>;
  triangleQuality(
    mesh: MeshHandle,
    metricOrOptions?: QualityMetric | TriangleQualityOptions,
    sliverThreshold?: number,
  ): InspectReturn<Async, QualityReport>;
  meshDeviation(source: MeshHandle, target: MeshHandle, options?: DeviationOptions): InspectReturn<Async, DeviationStats>;
  /**
   * The distance along the surface from the nearest of the vertices
   * `sources` to every vertex slot, in world units: `Infinity` where no
   * source reaches, on unused slots and past `maxDistance`.
   *
   * `"heatMethod"` first prepares operators of the mesh, about half the time
   * of a query. They are kept (about 130 bytes a triangle) for later queries
   * of the same mesh, until it changes, another mesh is queried, or
   * {@link InspectDomain.releaseGeodesics}. On a `WorkerScene` a query runs
   * in steps and can be cancelled (`options.signal`).
   */
  geodesicDistances(
    mesh: MeshHandle,
    sources: number | readonly number[],
    options?: GeodesicOptions,
  ): InspectReturn<Async, Float32Array>;
  /**
   * Frees what `"heatMethod"` queries keep for the next one (see
   * {@link InspectDomain.geodesicDistances}).
   */
  releaseGeodesics(): InspectReturn<Async, void>;
  shortestSurfacePath(
    mesh: MeshHandle,
    sources: number | readonly number[],
    targetVertex: number,
    options?: SurfacePathOptions,
  ): InspectReturn<Async, SurfacePathPoint[]>;
  findSurfacePath(
    mesh: MeshHandle,
    sources: number | readonly number[],
    targetVertex: number,
    options?: SurfacePathOptions,
  ): InspectReturn<Async, SurfacePathPoint[]>;
  addDistanceField(desc: DistanceFieldDesc): FieldHandle;
  addThicknessField(desc: ThicknessFieldDesc): FieldHandle;
  addCurvatureField(desc: CurvatureFieldDesc): FieldHandle;
  removeField(field: FieldHandle): void;
  fieldValues(field: FieldHandle): InspectReturn<Async, Float32Array>;
  fieldNearest(field: FieldHandle): InspectReturn<Async, Uint32Array>;
  fieldStatus(field: FieldHandle): InspectReturn<Async, FieldStatus>;
  fieldTargetStatus(field: FieldHandle): InspectReturn<Async, FieldStatus["sign"][]>;
  setColoring(mesh: MeshHandle, rule: ColorRule | null): void;
  coloring(mesh: MeshHandle): ColorRule | null;
  addSection(desc: SectionDesc): SectionHandle;
  setSectionPlane(section: SectionHandle, plane: SectionPlane): void;
  sectionPlane(section: SectionHandle): SectionPlane;
  setSectionMeshes(section: SectionHandle, meshes: readonly MeshHandle[]): void;
  setSectionFill(section: SectionHandle, fill: boolean): void;
  removeSection(section: SectionHandle): void;
  section(section: SectionHandle, mesh: MeshHandle): SectionResult;
  sectionGap(section: SectionHandle, a: MeshHandle, b: MeshHandle): InspectReturn<Async, SectionGap | null>;
  updateSections(): void;
  fitPrimitive(mesh: MeshHandle, options?: FitPrimitiveOptions): InspectReturn<Async, FittedPrimitiveReport>;
  alignToFittedPrimitive(
    mesh: MeshHandle,
    primitive: FittedPrimitive,
    targetAxis: Readonly<Vec3>,
    targetOrigin: Readonly<Vec3>,
  ): InspectReturn<Async, Placement>;
  sliceContours(mesh: MeshHandle, origin: Readonly<Vec3>, normal: Readonly<Vec3>): InspectReturn<Async, ContourSlice>;
  sliceContourStack(
    mesh: MeshHandle,
    origin: Readonly<Vec3>,
    normal: Readonly<Vec3>,
    spacing: number,
    count: number,
  ): InspectReturn<Async, ContourSlice[]>;
  reconstructPointCloud(
    cloud: PackedPointCloud | Float32Array,
    options?: SurfaceReconstructOptions,
  ): InspectReturn<Async, MeshHandle>;
}

/** Synchronous inspect domain exposed by `Scene.inspect`. */
export type SyncInspectDomain = InspectDomain<false>;

/** Asynchronous inspect domain exposed by `WorkerScene.inspect`. */
export type WorkerInspectDomain = InspectDomain<true>;

/** Builds an {@link InspectDomain} backed by `backend`. */
export function createInspectDomain<Async extends boolean>(backend: InspectBackend<Async>): InspectDomain<Async> {
  const shortestSurfacePath = (
    mesh: MeshHandle,
    sources: number | readonly number[],
    targetVertex: number,
    options?: SurfacePathOptions,
  ) => backend.shortestSurfacePath(mesh, sources, targetVertex, options);

  return {
    analyze: (mesh: MeshHandle, options?: AnalyzeOptions) => backend.analyze(mesh, options),
    measure: (mesh: MeshHandle) => backend.measure(mesh),
    bounds: (mesh: MeshHandle) => backend.bounds(mesh),
    raycast: (origin: Readonly<Vec3>, direction: Readonly<Vec3>, options?: RaycastOptions) =>
      backend.raycast(origin, direction, options),
    closestPoint: (point: Readonly<Vec3>, options?: ClosestPointOptions) => backend.closestPoint(point, options),
    meshDistance: (a: MeshHandle, b: MeshHandle, options?: { maxDistance?: number }) =>
      backend.meshDistance(a, b, options),
    intersects: (a: MeshHandle, b: MeshHandle) => backend.intersects(a, b),
    selfIntersectingTriangles: (mesh: MeshHandle) => backend.selfIntersectingTriangles(mesh),
    findContacts: (options?: FindContactsOptions) => backend.findContacts(options),
    overhangs: (mesh: MeshHandle, options?: OverhangOptions) => backend.overhangs(mesh, options),
    triangleQuality: (
      mesh: MeshHandle,
      metricOrOptions?: QualityMetric | TriangleQualityOptions,
      sliverThreshold?: number,
    ) => backend.triangleQuality(mesh, metricOrOptions, sliverThreshold),
    meshDeviation: (source: MeshHandle, target: MeshHandle, options?: DeviationOptions) =>
      backend.meshDeviation(source, target, options),
    geodesicDistances: (mesh: MeshHandle, sources: number | readonly number[], options?: GeodesicOptions) =>
      backend.geodesicDistances(mesh, sources, options),
    releaseGeodesics: () => backend.releaseGeodesics?.() as InspectReturn<Async, void>,
    shortestSurfacePath,
    findSurfacePath: shortestSurfacePath,
    addDistanceField: (desc: DistanceFieldDesc) => backend.addDistanceField(desc),
    addThicknessField: (desc: ThicknessFieldDesc) => {
      if (backend.addThicknessField === undefined) {
        throw new FastMeshError(
          "INVALID_ARGUMENT",
          "addThicknessField requires synchronous Scene (worker RPC not enabled)",
        );
      }
      return backend.addThicknessField(desc);
    },
    addCurvatureField: (desc: CurvatureFieldDesc) => {
      if (backend.addCurvatureField === undefined) {
        throw new FastMeshError(
          "INVALID_ARGUMENT",
          "addCurvatureField requires synchronous Scene (worker RPC not enabled)",
        );
      }
      return backend.addCurvatureField(desc);
    },
    removeField: (field: FieldHandle) => backend.removeField(field),
    fieldValues: (field: FieldHandle) => backend.fieldValues(field),
    fieldNearest: (field: FieldHandle) => backend.fieldNearest(field),
    fieldStatus: (field: FieldHandle) => backend.fieldStatus(field),
    fieldTargetStatus: (field: FieldHandle) => backend.fieldTargetStatus(field),
    setColoring: (mesh: MeshHandle, rule: ColorRule | null) => backend.setColoring(mesh, rule),
    coloring: (mesh: MeshHandle) => backend.coloring(mesh),
    addSection: (desc: SectionDesc) => backend.addSection(desc),
    setSectionPlane: (section: SectionHandle, plane: SectionPlane) => backend.setSectionPlane(section, plane),
    sectionPlane: (section: SectionHandle) => backend.sectionPlane(section),
    setSectionMeshes: (section: SectionHandle, meshes: readonly MeshHandle[]) =>
      backend.setSectionMeshes(section, meshes),
    setSectionFill: (section: SectionHandle, fill: boolean) => backend.setSectionFill(section, fill),
    removeSection: (section: SectionHandle) => backend.removeSection(section),
    section: (section: SectionHandle, mesh: MeshHandle) => backend.section(section, mesh),
    sectionGap: (section: SectionHandle, a: MeshHandle, b: MeshHandle) => backend.sectionGap(section, a, b),
    updateSections: () => backend.updateSections?.(),
    fitPrimitive: (mesh: MeshHandle, options?: FitPrimitiveOptions) => backend.fitPrimitive(mesh, options),
    alignToFittedPrimitive: (
      mesh: MeshHandle,
      primitive: FittedPrimitive,
      targetAxis: Readonly<Vec3>,
      targetOrigin: Readonly<Vec3>,
    ) => backend.alignToFittedPrimitive(mesh, primitive, targetAxis, targetOrigin),
    sliceContours: (mesh: MeshHandle, origin: Readonly<Vec3>, normal: Readonly<Vec3>) =>
      backend.sliceContours(mesh, origin, normal),
    sliceContourStack: (
      mesh: MeshHandle,
      origin: Readonly<Vec3>,
      normal: Readonly<Vec3>,
      spacing: number,
      count: number,
    ) => backend.sliceContourStack(mesh, origin, normal, spacing, count),
    reconstructPointCloud: (cloud: PackedPointCloud | Float32Array, options?: SurfaceReconstructOptions) =>
      backend.reconstructPointCloud(cloud, options),
  };
}
