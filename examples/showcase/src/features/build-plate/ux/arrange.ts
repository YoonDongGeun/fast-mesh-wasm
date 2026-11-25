// Arrange (#266): packs the footprints of meshes on the build plate and drops
// each onto it, as one undo step. A footprint is the mesh's own projection
// onto the plate (`readWorldBoxes`), so any placement counts as it is; the
// packing (`packRects`) turns meshes 90° about the up axis only. Meshes that
// do not fit, and locked ones, stay where they are and are reported; a
// locked mesh, and with `keepClear` every visible mesh that is not arranged,
// is kept clear of.
import type { MeshHandle, PlacementInput, Vec3 } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import type { AppState, Item } from "../../../app/state/app-state";
import { landed } from "../../home/ux/create-primitive";
import { hullSources } from "../../mesh-edit/ux/use-convex-hull";
import type { Pause } from "../../select/ux/face-criteria";
import {
  IDENTITY_MAT3,
  type Mat3,
  moveWorld,
  type PlacementLike,
  type UpAxis,
  type WorldBounds,
} from "../../transform/ux/placement-math";
import { readWorldBoxes } from "./footprint";
import { type PackBox, type PackedRect, packRects } from "./pack";
import { type BuildPlate, plateAxes, plateProblem } from "./plate";

/** A mesh to arrange: its world box and its placement. */
export interface PlatePart {
  box: WorldBounds;
  placement: PlacementLike;
}

/** Where Arrange puts the parts it could place, and those it could not, by index. */
export interface ArrangePlan {
  placed: { index: number; turned: boolean; placement: PlacementInput }[];
  oversized: number[];
  noRoom: number[];
}

/** The 90° turn about the up axis that takes the plate's width axis to its depth axis. */
export function quarterTurn(up: UpAxis): Mat3 {
  const { u, v, up: k } = plateAxes(up);
  const m: Mat3 = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  m[3 * v + u] = 1;
  m[3 * u + v] = -1;
  m[3 * k + k] = 1;
  return m;
}

/** The placement that rests a part on the plate with its footprint's lowest corner at `spot`, turned first when it says so. */
export function placeOnPlate(
  part: PlatePart,
  plate: Pick<BuildPlate, "origin" | "up">,
  spot: Pick<PackedRect, "x" | "y" | "turned">,
): PlacementInput {
  const { u, v, up } = plateAxes(plate.up);
  const { min, max } = part.box;
  // Turned, a point (p, q) of the plate goes to (-q, p).
  const low = spot.turned ? [-max[v], min[u]] : [min[u], min[v]];
  const dT: Vec3 = [0, 0, 0];
  dT[u] = plate.origin[0] + spot.x - low[0];
  dT[v] = plate.origin[1] + spot.y - low[1];
  dT[up] = -min[up];
  return moveWorld(part.placement, spot.turned ? quarterTurn(plate.up) : IDENTITY_MAT3, dT);
}

/** A world box's projection, in plate coordinates. */
export function platePrint(box: WorldBounds, plate: Pick<BuildPlate, "origin" | "up">): PackBox {
  const { u, v } = plateAxes(plate.up);
  return {
    x0: box.min[u] - plate.origin[0],
    y0: box.min[v] - plate.origin[1],
    x1: box.max[u] - plate.origin[0],
    y1: box.max[v] - plate.origin[1],
  };
}

/** Packs `parts` on the plate, keeping clear of `obstacles` (world boxes). */
export function planArrange(
  parts: readonly PlatePart[],
  plate: BuildPlate,
  obstacles: readonly WorldBounds[] = [],
): ArrangePlan {
  const sizes = parts.map((p) => {
    const b = platePrint(p.box, plate);
    return { w: b.x1 - b.x0, h: b.y1 - b.y0 };
  });
  const packed = packRects(sizes, {
    width: plate.width,
    depth: plate.depth,
    spacing: plate.spacing,
    turn: plate.turn,
    obstacles: obstacles.map((o) => platePrint(o, plate)),
  });
  return {
    placed: packed.placed.map((spot) => ({
      index: spot.index,
      turned: spot.turned,
      placement: placeOnPlate(parts[spot.index], plate, spot),
    })),
    oversized: packed.oversized,
    noRoom: packed.noRoom,
  };
}

/** What an Arrange did: counts, and the names of the meshes it left where they are. */
export interface ArrangeReport {
  placed: number;
  turned: number;
  /** Placed meshes whose placement changed. */
  moved: number;
  /** Larger than the plate. */
  oversized: string[];
  /** No room left on the plate. */
  noRoom: string[];
  locked: string[];
  /** Without triangles. */
  empty: string[];
}

/**
 * The meshes Arrange works on, in list order: the selected ones, or every
 * visible one.
 */
export function arrangeTargets(
  scope: "selected" | "all",
  items: readonly Item[],
  multi: readonly MeshHandle[],
  selection: MeshHandle | null,
  isVisible: (mesh: MeshHandle) => boolean,
): Item[] {
  if (scope === "all") {
    return items.filter((i) => isVisible(i.mesh));
  }
  const picked = hullSources(items, multi, items.find((i) => i.mesh === selection) ?? null);
  return items.filter((i) => picked.includes(i));
}

/** Whether a mesh shows (as last requested). */
export function meshVisible(store: SceneStore, mesh: MeshHandle): boolean {
  return store.getSnapshot().meshes.get(mesh)?.desired.visible ?? false;
}

/** The world boxes of `items`' meshes, once those still being added have landed. */
async function readBoxes(
  store: SceneStore,
  items: readonly Item[],
  pause: Pause | undefined,
): Promise<Map<MeshHandle, WorldBounds | null>> {
  const meshes = items.map((i) => i.mesh);
  // A failed add is the list's to report; here it reads as an empty mesh.
  await Promise.allSettled(
    meshes.filter((m) => store.scene.views(m).indices().length === 0).map((m) => landed(store, m)),
  );
  return readWorldBoxes(store.scene, meshes, pause);
}

/**
 * Arranges `items` on the plate (see the file comment) as one undo step.
 * Nothing is recorded when every mesh already lies where it would go.
 *
 * @throws when the plate has no size, or a mesh goes or keeps changing while it is read.
 */
export async function arrangeMeshes(params: {
  store: SceneStore;
  app: AppState;
  items: readonly Item[];
  plate: BuildPlate;
  pause?: Pause;
}): Promise<ArrangeReport> {
  const { store, app, items, plate, pause } = params;
  const problem = plateProblem(plate);
  if (problem !== null) {
    throw new Error(problem);
  }
  const movable = items.filter((i) => !i.locked);
  const locked = items.filter((i) => i.locked);
  const others = plate.keepClear
    ? app.getSnapshot().items.filter((i) => !items.includes(i) && meshVisible(store, i.mesh))
    : [];
  const boxes = await readBoxes(store, [...movable, ...locked, ...others], pause);
  const boxed = movable.flatMap((item) => {
    const box = boxes.get(item.mesh) ?? null;
    return box === null ? [] : [{ item, box, placement: store.scene.placement(item.mesh) }];
  });
  const obstacles = [...locked, ...others].flatMap((i) => boxes.get(i.mesh) ?? []);
  const plan = planArrange(boxed, plate, obstacles);

  // Less than a thousandth of the packing's unit is float noise: the mesh already lies there.
  const still = Math.max(plate.width, plate.depth) * 1e-9;
  const moves = plan.placed.filter(({ index, turned, placement }) => {
    const from = boxed[index].placement.translation;
    return turned || placement.translation.some((t, k) => Math.abs(t - from[k]) > still);
  });
  if (moves.length > 0) {
    store.beginStep();
    try {
      for (const p of moves) {
        store.setPlacement(boxed[p.index].item.mesh, p.placement);
      }
      store.endStep();
    } catch (err) {
      store.cancelStep();
      throw err;
    }
  }
  const names = (indices: readonly number[]) => indices.map((i) => boxed[i].item.name);
  return {
    placed: plan.placed.length,
    turned: plan.placed.filter((p) => p.turned).length,
    moved: moves.length,
    oversized: names(plan.oversized),
    noRoom: names(plan.noRoom),
    locked: locked.map((i) => i.name),
    empty: movable.filter((i) => (boxes.get(i.mesh) ?? null) === null).map((i) => i.name),
  };
}

/** `A, B, C and 4 more`. */
export function listNames(names: readonly string[], most = 5): string {
  const shown = names.slice(0, most).join(", ");
  return names.length > most ? `${shown} and ${names.length - most} more` : shown;
}

const meshes = (n: number) => `${n} mesh${n === 1 ? "" : "es"}`;

/** The report as the panel shows it: what was arranged, then a line per reason a mesh was left. */
export function describeArrange(r: ArrangeReport): { text: string; left: string[] } {
  const left = [
    r.oversized.length > 0 ? `Too big for the plate: ${listNames(r.oversized)}` : "",
    r.noRoom.length > 0 ? `No room left on the plate: ${listNames(r.noRoom)}` : "",
    r.locked.length > 0 ? `Locked: ${listNames(r.locked)}` : "",
    r.empty.length > 0 ? `No triangles: ${listNames(r.empty)}` : "",
  ].filter((line) => line !== "");
  const total = r.placed + r.oversized.length + r.noRoom.length + r.locked.length + r.empty.length;
  let text: string;
  if (r.placed === 0) {
    text = "Nothing was arranged";
  } else if (r.moved === 0) {
    text = `${meshes(r.placed)} already in place`;
  } else {
    const of = total > r.placed ? `${r.placed} of ${meshes(total)}` : meshes(r.placed);
    text = `Arranged ${of}${r.turned > 0 ? `, ${r.turned} turned 90°` : ""}`;
  }
  return { text, left };
}
