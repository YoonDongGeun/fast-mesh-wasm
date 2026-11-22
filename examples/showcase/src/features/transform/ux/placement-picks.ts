// The placement helpers' surface picks (#256): Lay flat takes a face, Align
// to axis two points or a face, 3-2-1 six points (three on the primary
// plane, two along the secondary direction, one for the origin). The armed
// pick lives in the app state (`placementTool`); each click on the mesh adds
// its world point (and, for a face, the face's world normal), and the last
// one turns the picks into the new placement.
import type { MeshHandle, PlacementInput, Vec3 } from "fast-mesh-wasm";
import {
  alignToAxis,
  datumAlign,
  layFlat,
  type PlacedMesh,
  type SignedAxis,
  sub,
  type UpAxis,
} from "./placement-math";

/** What the armed pick collects for. */
export type PlacementPickKind = "layFlat" | "alignPoints" | "alignFace" | "datum321";

/** An armed placement pick on one mesh. */
export interface PlacementPick {
  kind: PlacementPickKind;
  mesh: MeshHandle;
  /** The world points picked so far. */
  points: readonly Vec3[];
  /** The world normal of the latest picked face (face picks only). */
  normal: Vec3 | null;
  /** The floor's axis (Lay flat and 3-2-1). */
  up: UpAxis;
  /** The axis the direction turns onto (Align to axis). */
  target: SignedAxis;
}

/** The latest helper's outcome, for the panel. */
export interface PlacementReport {
  text: string;
  error: boolean;
}

/** The placement helpers' app state: the armed pick and the latest report. */
export interface PlacementToolState {
  pick: PlacementPick | null;
  report: PlacementReport | null;
}

export const DEFAULT_PLACEMENT_TOOL: PlacementToolState = { pick: null, report: null };

/** How many clicks each pick needs. */
export const PICKS_NEEDED: Readonly<Record<PlacementPickKind, number>> = {
  layFlat: 1,
  alignPoints: 2,
  alignFace: 1,
  datum321: 6,
};

/** A click on the mesh: its world point, and the hit face's outward world normal. */
export interface PlacementHit {
  point: Vec3;
  normal: Vec3 | null;
}

/** What the next click should land on. */
export function pickPrompt(pick: PlacementPick): string {
  const k = pick.points.length;
  switch (pick.kind) {
    case "layFlat":
      return "Click the face to lay the mesh on.";
    case "alignFace":
      return `Click a face: its normal turns onto ${pick.target.toUpperCase()}.`;
    case "alignPoints":
      return `Click point ${k + 1} of 2: the direction from the first to the second turns onto ${pick.target.toUpperCase()}.`;
    case "datum321":
      if (k < 3) {
        return `Primary plane: click point ${k + 1} of 3.`;
      }
      if (k < 5) {
        return `Secondary direction: click point ${k - 2} of 2.`;
      }
      return "Origin: click the last point.";
  }
}

/**
 * Adds `hit` to `pick`: the pick with the hit added, and whether that was
 * its last click. A face pick without a normal (a degenerate face) is not
 * counted.
 */
export function addPlacementHit(
  pick: PlacementPick,
  hit: PlacementHit,
): { pick: PlacementPick; complete: boolean } {
  const face = pick.kind === "layFlat" || pick.kind === "alignFace";
  if (face && hit.normal === null) {
    return { pick, complete: false };
  }
  const next: PlacementPick = {
    ...pick,
    points: [...pick.points, [hit.point[0], hit.point[1], hit.point[2]]],
    normal: face ? hit.normal : pick.normal,
  };
  return { pick: next, complete: next.points.length >= PICKS_NEEDED[pick.kind] };
}

/** The placement a completed pick gives `mesh`. Throws on degenerate picks. */
export function placementForPick(mesh: PlacedMesh, pick: PlacementPick): PlacementInput {
  const p = pick.points;
  switch (pick.kind) {
    case "layFlat":
      return layFlat(mesh, requireNormal(pick), pick.up);
    case "alignFace":
      return alignToAxis(mesh, requireNormal(pick), pick.target);
    case "alignPoints":
      return alignToAxis(mesh, sub(p[1], p[0]), pick.target);
    case "datum321":
      return datumAlign(mesh, { plane: [p[0], p[1], p[2]], line: [p[3], p[4]], origin: p[5] }, pick.up);
  }
}

function requireNormal(pick: PlacementPick): Vec3 {
  if (pick.normal === null) {
    throw new Error("No face was picked");
  }
  return pick.normal;
}

/** Which group the pick's point `index` belongs to, for its marker colour: 3-2-1's plane (0), line (1) or origin (2); 0 otherwise. */
export function pickMarkerGroup(kind: PlacementPickKind, index: number): 0 | 1 | 2 {
  if (kind !== "datum321") {
    return 0;
  }
  return index < 3 ? 0 : index < 5 ? 1 : 2;
}

/** The outcome line of a completed pick. */
export function pickReport(pick: PlacementPick): string {
  switch (pick.kind) {
    case "layFlat":
      return `Laid flat on ${pick.up.toUpperCase()} = 0`;
    case "alignFace":
      return `Face normal aligned to ${pick.target.toUpperCase()}`;
    case "alignPoints":
      return `Direction aligned to ${pick.target.toUpperCase()}`;
    case "datum321":
      return "3-2-1 alignment applied";
  }
}

/** The placement commands the ribbon and the command palette offer, by the panel button they press. */
/** `shelf`: also on the Select ribbon shelf. */
export const PLACEMENT_COMMANDS: readonly {
  action: string;
  label: string;
  short: string;
  keywords: string;
  shelf: boolean;
}[] = [
  {
    action: "origin-center",
    label: "Placement: Move Centre to Origin",
    short: "To Origin",
    keywords: "placement move origin centre center bounding box zero",
    shelf: true,
  },
  {
    action: "origin-bottom",
    label: "Placement: Move Bottom Centre to Origin",
    short: "Bottom to Origin",
    keywords: "placement move origin bottom floor base zero",
    shelf: false,
  },
  {
    action: "lay-flat",
    label: "Placement: Lay Flat on a Face",
    short: "Lay Flat",
    keywords: "placement lay flat face down floor rest orient",
    shelf: true,
  },
  {
    action: "align-points",
    label: "Placement: Align Two Points to an Axis",
    short: "Align Axis",
    keywords: "placement align axis direction two points rotate orient",
    shelf: false,
  },
  {
    action: "datum-321",
    label: "Placement: 3-2-1 Alignment",
    short: "3-2-1",
    keywords: "placement 3-2-1 321 datum plane line point align origin",
    shelf: true,
  },
  {
    action: "apply-transform",
    label: "Placement: Apply Transform",
    short: "Apply Transform",
    keywords: "placement apply bake freeze transform reset identity vertices",
    shelf: true,
  },
];
