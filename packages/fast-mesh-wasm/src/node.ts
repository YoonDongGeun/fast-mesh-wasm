// Node.js and Bun entry. The .wasm file is read from disk and instantiated
// synchronously on import, so `init()` is only kept for API parity with the browser.
// This module has a side effect on purpose; package.json lists it under "sideEffects"
// so bundlers never drop it.
import { readFileSync } from "node:fs";
import { type InitInput, markInitialized } from "./api.js";
import { initSync } from "./wasm/fast_mesh_wasm.js";

markInitialized(
  initSync({ module: readFileSync(new URL("./wasm/fast_mesh_wasm_bg.wasm", import.meta.url)) }),
);

/**
 * Resolves immediately: in Node.js and Bun the module is ready as soon as it is imported.
 * Accepts the same argument as the browser `init()` so the same code runs everywhere;
 * the argument is ignored.
 */
export function init(_input?: InitInput | Promise<InitInput>): Promise<void> {
  return Promise.resolve();
}

export {
  type Bounds,
  computeBounds,
  computeVertexNormals,
  type Indices,
  type InitInput,
  type Mat3,
  type Placement,
  type PlacementInput,
  type Positions,
  type Positions64,
  type Quat,
  type UpdateMode,
  type Vec3,
} from "./api.js";
export { FastMeshError, type FastMeshErrorCode } from "./core.js";
export type { SceneLike } from "./scene-like.js";
export {
  type AddMeshOptions,
  type Analysis,
  type AnalyzeOptions,
  type BoundaryLoop,
  type ClosestHit,
  type ClosestPointOptions,
  type Colormap,
  type ColorRule,
  type Contact,
  type DegenerateReport,
  type DistanceFieldDesc,
  type DragOptions,
  type DragSession,
  type EditOp,
  type EditRef,
  type EditResult,
  type FieldHandle,
  type FieldStatus,
  type FilledLoop,
  type FillReport,
  type FindContactsOptions,
  type LoopId,
  type MeshChange,
  type MeshComponent,
  type MeshDistance,
  type MeshHandle,
  type MeshViews,
  type OrientReport,
  type OutOfRange,
  type Range,
  type RaycastOptions,
  type RayHit,
  type RemoveReport,
  type RepairOp,
  type RepairReport,
  type RepairReportOf,
  type Rgba,
  Scene,
  type SceneOptions,
  type SectionContour,
  type SectionDesc,
  type SectionGap,
  type SectionHandle,
  type SectionPlane,
  type SectionResult,
  type SignMode,
  type WeldReport,
} from "./scene.js";
