// The one-line hint over the viewport: what the pointer does now. An armed
// tool speaks first (sculpt, mask paint, a Wave 6 tool, the brush centre
// pick), else selecting or moving, in touch words on a coarse pointer.
import type { AppSnapshot } from "../state/app-state";

/** A finger is the main pointer here: the hints speak of tapping and pinching. */
export const coarsePointer = typeof matchMedia === "function" && matchMedia("(pointer: coarse)").matches;

type Wave6 = AppSnapshot["wave6"];

export interface HintState {
  sculpting: boolean;
  maskPaint: boolean;
  selectTool: Wave6["select"]["activeTool"];
  deformTool: Wave6["deform"]["activeTool"];
  printTool: Wave6["printPrep"]["activeTool"];
  /** Waiting for a click to set the brush centre. */
  picking: boolean;
  /** A placement helper waits for clicks on the selected mesh (#256). */
  placementPick?: boolean;
  selected: boolean;
  coarse: boolean;
}

export function viewportHint(s: HintState): string {
  if (s.sculpting) {
    return "Hover any visible mesh to see brush · drag to sculpt · Esc disarms";
  }
  if (s.maskPaint) {
    return "Hover any visible mesh · drag to paint mask · Esc disarms";
  }
  if (s.selectTool !== null) {
    return `Use ${s.selectTool} selection on mesh · Esc disarms`;
  }
  if (s.deformTool !== null) {
    return "Drag deformation handle in viewport · Esc disarms";
  }
  if (s.printTool === "drainHole") {
    return "Click mesh surface to punch drain hole · Esc disarms";
  }
  if (s.printTool === "placeStamp") {
    return "Click mesh surface to align and place stamp · Esc disarms";
  }
  if (s.picking) {
    return "Click the mesh to set the brush center · Esc cancels";
  }
  if (s.placementPick === true) {
    return "Click the selected mesh to pick (the Transform panel says what) · Esc cancels";
  }
  if (!s.selected) {
    return s.coarse
      ? "Tap a mesh to select it · drag to rotate · pinch to zoom · two fingers to pan"
      : "Click a mesh to select it · drag to rotate · scroll to zoom · Ctrl+K search · drop files to load";
  }
  return s.coarse
    ? "Drag the gizmo to move · tap empty space to deselect"
    : "Drag the gizmo to move · T/R/S move/rotate/scale · F focus · Ctrl+K search · Esc deselect";
}
