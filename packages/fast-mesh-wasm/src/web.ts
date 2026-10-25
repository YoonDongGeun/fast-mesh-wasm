// Browser and bundler entry (Vite, webpack, Next.js client, Deno, ...).
// The glue locates the .wasm with `new URL("fast_mesh_wasm_bg.wasm", import.meta.url)`,
// which bundlers recognize and copy as an asset.
import { type InitInput, markInitialized } from "./api.js";
import wasmInit from "./wasm/fast_mesh_wasm.js";

let ready: Promise<void> | undefined;

/**
 * Loads the WebAssembly module. Await this once before calling any other function.
 * Calling it again returns the same promise.
 *
 * @param input - Optional custom source for the `.wasm` file (URL, Response, bytes or module).
 *   Leave empty to load the file shipped with this package.
 */
export function init(input?: InitInput | Promise<InitInput>): Promise<void> {
  ready ??= wasmInit(input === undefined ? undefined : { module_or_path: input }).then(
    markInitialized,
  );
  return ready;
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
  type BridgeReport,
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
  type InflateReport,
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
  type RemeshReport,
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
  type SmoothReport,
  type WeldReport,
} from "./scene.js";
