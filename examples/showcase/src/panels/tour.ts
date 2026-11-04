// First-run onboarding state: pure helpers (no DOM at import, so tests can
// use them). The tour starts automatically on a fresh profile, and stays off
// in headless runs (showcase-check.mjs drives `--headless=new` Chrome, whose
// user agent carries "HeadlessChrome") unless `?tour=1` forces it on.
// `?tour=0` (or `?notour`) forces it off.
import { readStorage, removeStorage, writeStorage } from "./settings";

export interface TourStep {
  /** The `[data-tour]` anchor this step points at. */
  target: string;
  title: string;
  body: string;
}

export const TOUR_STEPS: TourStep[] = [
  {
    target: "mesh-list",
    title: "Meshes",
    body: "Loaded meshes land here: pick one to select it, shift-click for the Join queue, or drop STL, OBJ, PLY or DICOM files anywhere.",
  },
  {
    target: "gizmo",
    title: "Gizmo",
    body: "Drag the gizmo to move the selection. T moves, R rotates, F frames it, Esc lets go.",
  },
  {
    target: "repair",
    title: "Repair",
    body: "Analyze the selected mesh, then fill holes, weld, or remesh it. Undo brings everything back.",
  },
  {
    target: "collision",
    title: "Collision",
    body: "Pick two meshes (or a group) to paint their penetration and gap live, as the gizmo drags.",
  },
];

// The tour card's state machine, pure so tests can drive it: Next from the
// last step (or Skip/Done anywhere) ends the tour (`null`), Back stays on
// the first step.
export function tourNext(index: number): number | null {
  return index + 1 >= TOUR_STEPS.length ? null : index + 1;
}

export function tourBack(index: number): number {
  return index <= 0 ? 0 : index - 1;
}

export type TourStart = "auto" | "forced" | "off";

export function resolveTourStart(search: string, agent: string, webdriver: boolean, seen: boolean): TourStart {
  const params = new URLSearchParams(search);
  if (params.get("tour") === "1") {
    return "forced";
  }
  if (params.get("tour") === "0" || params.has("notour")) {
    return "off";
  }
  if (webdriver || agent.includes("HeadlessChrome")) {
    return "off";
  }
  return seen ? "off" : "auto";
}

const SEEN_KEY = "fm.showcase.seen-tour";
/** The pre-#144 key: adopted once, then forgotten. */
const LEGACY_SEEN_KEY = "fast-mesh-showcase.seen-tour";

export function tourSeen(): boolean {
  if (readStorage(SEEN_KEY) === "1") {
    return true;
  }
  if (readStorage(LEGACY_SEEN_KEY) === "1") {
    writeStorage(SEEN_KEY, "1");
    removeStorage(LEGACY_SEEN_KEY);
    return true;
  }
  return false;
}

export function markTourSeen(): void {
  writeStorage(SEEN_KEY, "1");
  removeStorage(LEGACY_SEEN_KEY);
}
