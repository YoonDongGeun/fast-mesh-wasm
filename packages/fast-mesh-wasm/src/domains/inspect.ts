import type { Bounds, Vec3 } from "../api.js";
import { FastMeshError } from "../core.js";
import type {
  Analysis,
  AnalyzeOptions,
  ClosestHit,
  ClosestPointOptions,
  ColorRule,
  Contact,
  CurvatureFieldDesc,
  DistanceFieldDesc,
  FieldHandle,
  FieldStatus,
  FindContactsOptions,
  MeasureResult,
  MeshDistance,
  MeshHandle,
  RaycastOptions,
  RayHit,
  SectionDesc,
  SectionGap,
  SectionHandle,
  SectionPlane,
  SectionResult,
  ThicknessFieldDesc,
} from "../scene.js";

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
  findContacts(options?: FindContactsOptions): InspectReturn<Async, Contact[]>;
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
}

/** Metrology, defect analysis, raycasting, distance/thickness/curvature fields, and cross-section inspection domain. */
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
  findContacts(options?: FindContactsOptions): InspectReturn<Async, Contact[]>;
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
}

/** Synchronous inspect domain exposed by `Scene.inspect`. */
export type SyncInspectDomain = InspectDomain<false>;

/** Asynchronous inspect domain exposed by `WorkerScene.inspect`. */
export type WorkerInspectDomain = InspectDomain<true>;

/** Builds an {@link InspectDomain} backed by `backend`. */
export function createInspectDomain<Async extends boolean>(backend: InspectBackend<Async>): InspectDomain<Async> {
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
    findContacts: (options?: FindContactsOptions) => backend.findContacts(options),
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
  };
}
