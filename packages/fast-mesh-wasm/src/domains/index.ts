export {
  type AlignBackend,
  type AlignDomain,
  type AlignReturn,
  createAlignDomain,
  type SyncAlignDomain,
  type WorkerAlignDomain,
} from "./align.js";
export {
  type BooleanBackend,
  type BooleanDomain,
  type BooleanReturn,
  createBooleanDomain,
  type OffsetOptions,
  type ShellOptions,
  type SyncBooleanDomain,
  type WorkerBooleanDomain,
} from "./boolean.js";
export {
  createDeformDomain,
  type DeformBackend,
  type DeformDomain,
  type DeformReturn,
  type ProtectedSculptOp,
  type SyncDeformDomain,
  type WorkerDeformDomain,
} from "./deform.js";
export {
  createInspectDomain,
  type InspectBackend,
  type InspectDomain,
  type InspectReturn,
  type SyncInspectDomain,
  type WorkerInspectDomain,
} from "./inspect.js";
export {
  createRemeshDomain,
  type RegionRemeshOptions,
  type RemeshBackend,
  type RemeshDomain,
  type RemeshReturn,
  type SimplifyOptions,
  type SubdivideOptions,
  type SyncRemeshDomain,
  type VoxelRemeshOptions,
  type WorkerRemeshDomain,
} from "./remesh.js";
export {
  type AutoRepairOptions,
  type BridgeLoopsOptions,
  type CleanOptions,
  type CleanReport,
  createRepairDomain,
  type DegenerateOptions,
  type FillHolesOptions,
  type RepairBackend,
  type RepairDomain,
  type RepairReturn,
  type SmallComponentOptions,
  type SyncRepairDomain,
  type WorkerRepairDomain,
} from "./repair.js";
export {
  type BeginStrokeOptions,
  type CaptureRestOptions,
  type CaptureStrokeOptions,
  type ClayOp,
  createSculptDomain,
  type CreaseOp,
  type FilteredSculptOp,
  type FlattenOp,
  type GrabOp,
  type InflateOp,
  type RelaxOp,
  type SculptBackend,
  type SculptDomain,
  type SculptReturn,
  type SmoothOp,
  type SplitStretchedOptions,
  type SyncSculptDomain,
  type TwistOp,
  type WorkerSculptDomain,
} from "./sculpt.js";

import type { AlignDomain } from "./align.js";
import type { BooleanDomain } from "./boolean.js";
import type { DeformDomain } from "./deform.js";
import type { InspectDomain } from "./inspect.js";
import type { RemeshDomain } from "./remesh.js";
import type { RepairDomain } from "./repair.js";
import type { SculptDomain } from "./sculpt.js";

/** Complete seven-domain API surface shared by `Scene` and `WorkerScene`. */
export interface SceneDomains<Async extends boolean = false> {
  readonly repair: RepairDomain<Async>;
  readonly sculpt: SculptDomain<Async>;
  readonly remesh: RemeshDomain<Async>;
  readonly boolean: BooleanDomain<Async>;
  readonly deform: DeformDomain<Async>;
  readonly inspect: InspectDomain<Async>;
  readonly align: AlignDomain<Async>;
}

/** Synchronous seven-domain surface implemented by `Scene`. */
export type SyncSceneDomains = SceneDomains<false>;

/** Asynchronous seven-domain surface implemented by `WorkerScene`. */
export type WorkerSceneDomains = SceneDomains<true>;
