// Multi-clone (#266): N copies of a mesh in a grid on the floor or in a line
// along an axis, as one undo step. The copies take the source's geometry,
// UVs, origin and placement, moved by whole steps: its world box plus the
// spacing. They are plain meshes: no colouring, lock or source volume. The
// source stays where it is, as the array's first place; afterwards it and
// its copies are selected, so Arrange can follow.
import type { MeshHandle, Vec3 } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import type { AppState, Item } from "../../../app/state/app-state";
import { extractLiveMeshData } from "../../../engine/workbench-ops";
import { formatTriangles } from "../../../io/big-file";
import { landed } from "../../home/ux/create-primitive";
import type { Pause } from "../../select/ux/face-criteria";
import { axisVector, type SignedAxis, type UpAxis, type WorldBounds } from "../../transform/ux/placement-math";
import { readWorldBoxes } from "./footprint";
import { plateAxes } from "./plate";

/** The most copies one run makes. */
export const MAX_COPIES = 1000;
/** The most triangles one run adds. */
export const MAX_CLONE_TRIANGLES = 20_000_000;

export interface CloneLayout {
  pattern: "grid" | "linear";
  /** How many copies to make; the source is not counted. */
  copies: number;
  /** Grid: places per row, along X; `null` for a near-square grid. */
  columns: number | null;
  /** Linear: where the line runs. */
  axis: SignedAxis;
  /** The gap between neighbours, in millimetres. */
  spacing: number;
}

export const DEFAULT_CLONE_LAYOUT: CloneLayout = {
  pattern: "grid",
  copies: 5,
  columns: null,
  axis: "+x",
  spacing: 2,
};

/** The grid's places per row: as given, or the ones of a near-square grid. */
export function cloneColumns(layout: Pick<CloneLayout, "copies" | "columns">): number {
  return layout.columns ?? Math.ceil(Math.sqrt(layout.copies + 1));
}

/** Why `layout` cannot run on a mesh of `triangles` triangles, in words, or `null` when it can. */
export function cloneProblem(layout: CloneLayout, triangles: number): string | null {
  const { copies, columns, spacing } = layout;
  if (!Number.isInteger(copies) || copies < 1 || copies > MAX_COPIES) {
    return `Copies must be a whole number from 1 to ${MAX_COPIES}.`;
  }
  if (layout.pattern === "grid" && columns !== null && (!Number.isInteger(columns) || columns < 1)) {
    return "Columns must be a whole number, one or more.";
  }
  if (!(spacing >= 0 && Number.isFinite(spacing))) {
    return "The spacing must be zero or more.";
  }
  if (copies * triangles > MAX_CLONE_TRIANGLES) {
    return `${formatTriangles(copies * triangles)} triangles: at most ${formatTriangles(MAX_CLONE_TRIANGLES)}.`;
  }
  return null;
}

/**
 * Each copy's world offset from the source, whose world box is `size` large.
 * A grid fills rows along X, then grows along the floor's other axis (Y with
 * Z up, Z with Y up); a line runs along its axis.
 */
export function cloneOffsets(size: Readonly<Vec3>, layout: CloneLayout, up: UpAxis): Vec3[] {
  const offsets: Vec3[] = [];
  const { u, v } = plateAxes(up);
  const columns = cloneColumns(layout);
  const dir = axisVector(layout.axis);
  for (let k = 1; k <= layout.copies; k++) {
    const at: Vec3 = [0, 0, 0];
    if (layout.pattern === "grid") {
      at[u] = (k % columns) * (size[u] + layout.spacing);
      at[v] = Math.floor(k / columns) * (size[v] + layout.spacing);
    } else {
      for (let a = 0; a < 3; a++) {
        at[a] = dir[a] === 0 ? 0 : dir[a] * k * (size[a] + layout.spacing);
      }
    }
    offsets.push(at);
  }
  return offsets;
}

/** Names for `count` copies of `name`: `name copy`, `name copy 2`, ..., none among `taken`. */
export function cloneNames(name: string, count: number, taken: readonly string[]): string[] {
  const used = new Set(taken);
  const names: string[] = [];
  for (let n = 1; names.length < count; n++) {
    const next = n === 1 ? `${name} copy` : `${name} copy ${n}`;
    if (!used.has(next)) {
      names.push(next);
    }
  }
  return names;
}

/** The size of a world box. */
function boxSize(b: WorldBounds): Vec3 {
  return [b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]];
}

/**
 * Makes the copies (see the file comment) and selects them with the source.
 * Resolves with the copies once the scene has them.
 *
 * @throws when the layout cannot run, the mesh has no triangles, or the scene
 * refuses a copy (nothing is left of them then).
 */
export async function cloneMesh(params: {
  store: SceneStore;
  app: AppState;
  item: Item;
  layout: CloneLayout;
  up: UpAxis;
  pause?: Pause;
}): Promise<Item[]> {
  const { store, app, item, layout, up, pause } = params;
  const box = (await readWorldBoxes(store.scene, [item.mesh], pause)).get(item.mesh) ?? null;
  if (box === null) {
    throw new Error("The mesh has no triangles");
  }
  const source = extractLiveMeshData(store.scene, item.mesh);
  const triangles = source.indices.length / 3;
  const problem = cloneProblem(layout, triangles);
  if (problem !== null) {
    throw new Error(problem);
  }
  // As the source's own add: positions with its origin in them, subtracted again by the engine.
  const { origin, placement } = source;
  const positions = source.positions;
  for (let i = 0; i < positions.length; i++) {
    positions[i] += origin[i % 3];
  }
  const offsets = cloneOffsets(boxSize(box), layout, up);
  const names = cloneNames(item.name, offsets.length, app.getSnapshot().items.map((i) => i.name));

  const copies: Item[] = [];
  store.beginStep();
  try {
    offsets.forEach((offset, k) => {
      const copy = app.addMesh(
        store,
        {
          name: names[k],
          positions,
          indices: source.indices,
          ...(source.uvs ? { uvs: source.uvs } : {}),
          center: [origin[0], origin[1], origin[2]],
          radius: source.radius,
          volumeId: null,
          preset: null,
          importScale: item.importScale,
        },
        { undoable: true },
      );
      copies.push(copy);
      const t = placement.translation;
      store.setPlacement(copy.mesh, {
        rotation: placement.rotation as [number, number, number, number, number, number, number, number, number],
        translation: [t[0] + offset[0], t[1] + offset[1], t[2] + offset[2]],
        ...(placement.scale ? { scale: placement.scale } : {}),
      });
    });
    store.endStep();
  } catch (err) {
    store.cancelStep();
    app.syncMeshes(store);
    throw err;
  }
  const results = await Promise.allSettled(copies.map((c) => landed(store, c.mesh)));
  const failed = results.find((r) => r.status === "rejected");
  if (failed !== undefined) {
    for (const copy of copies) {
      if (app.getSnapshot().items.includes(copy)) {
        app.removeMesh(store, copy, { undoable: false });
      }
    }
    throw failed.reason;
  }
  selectAll(store, app, item.mesh, copies.map((c) => c.mesh));
  return copies;
}

/** Selects `primary` and `others`, those still in the scene. */
function selectAll(store: SceneStore, app: AppState, primary: MeshHandle, others: readonly MeshHandle[]): void {
  const { meshes } = store.getSnapshot();
  const all = [primary, ...others].filter((m) => meshes.has(m));
  if (all.length > 0) {
    app.setMulti(all);
    store.select(all[0]);
  }
}
