// Arming Place Mesh with a text as its stamp (#260), apart from building that
// stamp (text-stamp.ts): the command palette arms it before the Mesh Edit
// chunk is here, and the viewport tool, once it is, takes the armed state.
import type { AppState } from "../../../app/state/app-state";
import type { TextParams } from "./text-params";

/** The two ways text goes onto a surface. */
export type TextStampMode = "embedUnion" | "engraveSubtract";

/**
 * Arms Place Mesh with `text` as its stamp, in `mode`, conforming to the
 * surface (so a long line follows a curve), with every other viewport tool
 * off, and shows the Mesh Edit panel that holds its settings.
 */
export function armTextStamp(app: AppState, text: TextParams, mode: TextStampMode): void {
  app.setSculpt(null);
  app.setMaskPaint(false);
  app.updateWave6((prev) => ({
    ...prev,
    select: { ...prev.select, activeTool: null },
    deform: { ...prev.deform, activeTool: null },
    printPrep: {
      ...prev.printPrep,
      activeTool: "placeStamp",
      placeParams: { ...prev.printPrep.placeParams, stampSource: "text", text: { ...text }, mode, conformToSurface: true },
    },
  }));
  app.setActiveTab("edit");
}
