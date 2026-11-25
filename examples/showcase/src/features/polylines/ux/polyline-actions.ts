// What the polyline tools do to the scene (#268): keep curves as objects,
// edit, offset and remove them, and make meshes of them. Each is one undo
// step. Simplify and resample change the polylines themselves; offset adds
// new ones; a tube or an extrusion adds a mesh, as the Create dialog adds a
// shape, and leaves the polyline. A tool that cannot do what it is asked
// changes nothing and throws with the reason in words.
import type { Scene } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import type { AppSnapshot, AppState, InspectSection, Item } from "../../../app/state/app-state";
import type { ContourLike } from "../../../io/report";
import { formatLength, type LengthUnit } from "../../../shared/ux/units";
import { readSectionContours } from "../../home/ux/export-actions";
import { addPrimitive, MAX_CREATE_TRIANGLES } from "../../home/ux/create-primitive";
import { pathCurve, type SurfacePath } from "../../inspect/ux/surface-paths";
import { selectMesh } from "../../select/ux/selection";
import { extrudeGroup, type ExtrudeProblem, type ExtrudeSide, MAX_EXTRUDE_POINTS, planarGroups } from "./extrude-curves";
import { offsetCurve, type OffsetCorners, type OffsetProblem } from "./offset";
import {
  freshNames,
  makeCurve,
  MAX_POLYLINES_PER_ACTION,
  pointCount,
  POLYLINE_PALETTE,
  type Polyline,
  type PolylineCurve,
} from "./polyline";
import { simplifyCurve } from "./simplify";
import type { SolidMesh } from "./solid";
import { resampleCurve, type ResampleOptions } from "./spline";
import { buildTube, tubeProblem, tubeTriangles } from "./tube";

/** The polylines the tools work on: the selected ones, else the only one. */
export function activePolylines(s: Pick<AppSnapshot, "polylines" | "polylineSelection">): readonly Polyline[] {
  if (s.polylineSelection.length > 0) {
    return s.polylines.filter((p) => s.polylineSelection.includes(p.id));
  }
  return s.polylines.length === 1 ? s.polylines : [];
}

/** The palette colour fewest of `polylines` have (the first of equals). */
export function nextColor(polylines: readonly Pick<Polyline, "color">[]): number {
  const uses = POLYLINE_PALETTE.map((c) => polylines.filter((p) => p.color === c).length);
  return POLYLINE_PALETTE[uses.indexOf(Math.min(...uses))];
}

/** Throws when an action would make more polylines than the list is for. */
function checkCount(count: number): void {
  if (count > MAX_POLYLINES_PER_ACTION) {
    throw new Error(
      `That makes ${count.toLocaleString("en")} polylines: at most ${MAX_POLYLINES_PER_ACTION.toLocaleString("en")} at a time.`,
    );
  }
}

/**
 * Keeps `curves` as polylines named after `base`, in one colour, and selects
 * them: one undo step. Curves without two points apart are left out.
 */
export function keepCurves(app: AppState, store: SceneStore, curves: readonly ContourLike[], base: string): Polyline[] {
  const made = curves.flatMap((c) => makeCurve(c.points, c.closed) ?? []);
  if (made.length === 0) {
    throw new Error("There is no curve to keep.");
  }
  checkCount(made.length);
  const { polylines } = app.getSnapshot();
  const color = nextColor(polylines);
  const names = freshNames(base, made.length, polylines.map((p) => p.name));
  const kept = made.map((c, i): Polyline => ({ ...c, id: app.newPolylineId(), name: names[i], color, visible: true }));
  app.editPolylines(
    store,
    [...polylines, ...kept],
    kept.map((p) => p.id),
  );
  return kept;
}

/** What Keep says it did. */
export function describeKept(kept: readonly Polyline[]): string {
  return kept.length === 1 ? `Kept as "${kept[0].name}".` : `Kept ${kept.length} polylines, "${kept[0].name}" to "${kept[kept.length - 1].name}".`;
}

/** Keeps a surface path, closed or not. */
export function keepPath(app: AppState, store: SceneStore, path: SurfacePath | null, meshName: string): Polyline[] {
  if (path === null || path.legs.length === 0) {
    throw new Error("Draw a path first.");
  }
  return keepCurves(app, store, [{ points: pathCurve(path).flat(), closed: path.closed }], `${meshName} path`);
}

/** Keeps the section plane's contours, of every mesh it cuts. */
export function keepSection(
  app: AppState,
  store: SceneStore,
  scene: Scene | WorkerScene,
  section: Pick<InspectSection, "handle" | "plane">,
): Polyline[] {
  const read = readSectionContours(store, scene, section.handle, section.plane);
  if (read === null || read.contours.length === 0) {
    throw new Error("The section has no contours to keep yet.");
  }
  return keepCurves(app, store, read.contours, "Section");
}

/** {@link keepSection} with its report, or why there was nothing to keep. */
export function keepSectionText(...args: Parameters<typeof keepSection>): { text: string; error: boolean } {
  try {
    return { text: describeKept(keepSection(...args)), error: false };
  } catch (e) {
    return { text: e instanceof Error ? e.message : String(e), error: true };
  }
}

/** Removes polylines, as one undo step. */
export function removePolylines(app: AppState, store: SceneStore, ids: readonly number[]): void {
  const { polylines } = app.getSnapshot();
  const left = polylines.filter((p) => !ids.includes(p.id));
  if (left.length < polylines.length) {
    app.editPolylines(store, left);
  }
}

/** An edit's points before and after. */
export interface EditCounts {
  before: number;
  after: number;
  /** Nothing changed: no undo step was made. */
  unchanged: boolean;
}

/** Replaces the curves of `targets` by `edit`, as one undo step (none when nothing changes). */
function editCurves(
  app: AppState,
  store: SceneStore,
  targets: readonly Polyline[],
  edit: (curve: Polyline) => PolylineCurve,
): EditCounts {
  const made = new Map(targets.map((p) => [p.id, edit(p)]));
  const counts: EditCounts = { before: 0, after: 0, unchanged: true };
  for (const p of targets) {
    const c = made.get(p.id) as PolylineCurve;
    counts.before += pointCount(p);
    counts.after += pointCount(c);
    counts.unchanged &&= c === p;
  }
  if (!counts.unchanged) {
    app.editPolylines(
      store,
      app.getSnapshot().polylines.map((p) => {
        const c = made.get(p.id);
        return c === undefined ? p : { ...p, closed: c.closed, points: c.points };
      }),
    );
  }
  return counts;
}

/** Drops the points `targets` do without within `tolerance` (millimetres). */
export function simplifyPolylines(
  app: AppState,
  store: SceneStore,
  targets: readonly Polyline[],
  tolerance: number,
): EditCounts {
  if (!(tolerance > 0) || !Number.isFinite(tolerance)) {
    throw new Error("The tolerance must be above zero.");
  }
  return editCurves(app, store, targets, (p) => simplifyCurve(p, tolerance));
}

/** Replaces `targets` by evenly spaced points of their splines, smoothed first when asked. */
export function resamplePolylines(
  app: AppState,
  store: SceneStore,
  targets: readonly Polyline[],
  options: ResampleOptions,
): EditCounts {
  return editCurves(app, store, targets, (p) => resampleCurve(p, options));
}

/** Why a curve cannot be offset or extruded, in words. */
export function planarProblemText(problem: OffsetProblem | ExtrudeProblem, name: string, unit: LengthUnit): string {
  switch (problem.reason) {
    case "straight":
      return `"${name}" is a straight line: it lies in no one plane.`;
    case "bent":
      return `"${name}" is not planar: it strays ${formatLength(problem.deviation, unit)} from its plane.`;
    case "open":
      return `"${name}" is open: only closed polylines can be extruded.`;
    case "crossing":
      return `"${name}" crosses or touches itself, or another polyline of its plane: the inside is not clear.`;
    case "caps":
      return `The caps of "${name}" could not be filled.`;
    case "points":
      return `"${name}" has ${problem.points.toLocaleString("en")} points: at most ${MAX_EXTRUDE_POINTS.toLocaleString("en")}. Simplify it first.`;
  }
}

/**
 * Adds the curves at `distance` from each of `targets` as polylines of their
 * colour (outward, or to the left of an open one; negative: the other way),
 * and selects them: one undo step.
 */
export function offsetPolylines(
  app: AppState,
  store: SceneStore,
  targets: readonly Polyline[],
  distance: number,
  corners: OffsetCorners,
  unit: LengthUnit,
): Polyline[] {
  if (distance === 0 || !Number.isFinite(distance)) {
    throw new Error("The offset must not be zero.");
  }
  const { polylines } = app.getSnapshot();
  const made: Polyline[] = [];
  for (const p of targets) {
    const curves = offsetCurve(p, distance, corners);
    if (!Array.isArray(curves)) {
      throw new Error(planarProblemText(curves, p.name, unit));
    }
    checkCount(made.length + curves.length);
    const taken = [...polylines, ...made].map((q) => q.name);
    const names = freshNames(`${p.name} offset`, curves.length, taken);
    curves.forEach((c, i) => made.push({ ...c, id: app.newPolylineId(), name: names[i], color: p.color, visible: true }));
  }
  if (made.length === 0) {
    throw new Error(`Nothing is left at ${formatLength(Math.abs(distance), unit)}: a smaller offset fits.`);
  }
  app.editPolylines(
    store,
    [...polylines, ...made],
    made.map((p) => p.id),
  );
  return made;
}

/** Adds `solids` as meshes in one undo step, and selects the last. `done` settles once the engine has them. */
function addSolids(
  app: AppState,
  store: SceneStore,
  solids: readonly { mesh: SolidMesh; name: string }[],
): { items: Item[]; done: Promise<void> } {
  const items: Item[] = [];
  const landings: Promise<void>[] = [];
  store.beginStep();
  try {
    for (const { mesh, name } of solids) {
      const added = addPrimitive(store, app, mesh, { name, centre: mesh.center, rotationDeg: [0, 0, 0] });
      items.push(added.item);
      landings.push(added.done);
    }
    store.endStep();
  } catch (e) {
    store.cancelStep();
    app.syncMeshes(store);
    throw e;
  }
  selectMesh(app, (m) => store.select(m), store.getSnapshot().selection, items[items.length - 1].mesh, false);
  return { items, done: Promise.all(landings).then(() => undefined) };
}

/** What a tube or an extrusion added. */
export interface AddedSolids {
  items: Item[];
  triangles: number;
  /** Settles once the engine has the meshes. */
  done: Promise<void>;
}

/** Sweeps a circle along each of `targets`, as a closed mesh each: one undo step. */
export function tubePolylines(
  app: AppState,
  store: SceneStore,
  targets: readonly Polyline[],
  radius: number,
  sides: number,
): AddedSolids & { folds: number } {
  let left = MAX_CREATE_TRIANGLES;
  for (const p of targets) {
    const problem = tubeProblem(p, radius, sides, left);
    if (problem !== null) {
      throw new Error(problem);
    }
    left -= tubeTriangles(p, sides);
  }
  let folds = 0;
  let triangles = 0;
  const solids = targets.map((p) => {
    const mesh = buildTube(p, radius, sides);
    folds += mesh.folds;
    triangles += mesh.indices.length / 3;
    return { mesh, name: `${p.name} tube` };
  });
  return { ...addSolids(app, store, named(app, solids)), triangles, folds };
}

/**
 * Extrudes the closed planar `targets` into solids, one per plane they lie
 * in (a curve inside another is a hole): one undo step.
 */
export function extrudePolylines(
  app: AppState,
  store: SceneStore,
  targets: readonly Polyline[],
  depth: number,
  side: ExtrudeSide,
  unit: LengthUnit,
): AddedSolids {
  if (depth === 0 || !Number.isFinite(depth)) {
    throw new Error("The depth must not be zero.");
  }
  // One at a time first, so the message names the polyline at fault.
  for (const p of targets) {
    const alone = planarGroups([p]);
    if (!Array.isArray(alone)) {
      throw new Error(planarProblemText(alone, p.name, unit));
    }
  }
  const groups = planarGroups(targets);
  if (!Array.isArray(groups)) {
    throw new Error(planarProblemText(groups, targets[0].name, unit));
  }
  let triangles = 0;
  const solids = groups.map((g) => {
    const name = (g.curves[0] as Polyline).name;
    const mesh = extrudeGroup(g, depth, side);
    if ("reason" in mesh) {
      throw new Error(planarProblemText(mesh, name, unit));
    }
    triangles += mesh.indices.length / 3;
    return { mesh, name: `${name} extrusion` };
  });
  return { ...addSolids(app, store, named(app, solids)), triangles };
}

/** `solids` with names no mesh of the list (or one before) has. */
function named<T extends { name: string }>(app: AppState, solids: readonly T[]): T[] {
  const taken = new Set(app.getSnapshot().items.map((i) => i.name));
  return solids.map((s) => {
    const [name] = freshNames(s.name, 1, taken);
    taken.add(name);
    return { ...s, name };
  });
}
