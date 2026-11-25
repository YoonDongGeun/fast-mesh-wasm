// Text on a surface (#260), through Place Mesh: the text mesh is the
// stamp, standing on its base at z = 0 and centred on the picked point, so
// Embed & Union raises it out of the surface and Engrave cuts it in, as
// one undoable edit of the target (the Place Mesh op). This module builds
// that stamp, says in words what the settings will do, and arms the
// viewport tool with the text.
import type { AppState } from "../../../app/state/app-state";
import type { Wave6PrintPrepState } from "../../../app/state/wave6-state";
import { formatLength, type LengthUnit } from "../../../shared/ux/units";
import { loadTextFont } from "./text-font";
import { buildTextMesh, FOLLOW_EDGES_PER_LETTER, refineAcross, type TextParams, textParamsProblem } from "./text-mesh";

/**
 * The stamp Place Mesh puts down for `params`: the closed text mesh with
 * its base at z = 0 and its box centred on the origin, as engine buffers.
 * With `follow` its faces are cut small enough to follow a curved surface
 * (Conform to Surface moves vertices only). Loads the font on first use.
 */
export async function textStampGeometry(
  params: TextParams,
  options: { follow?: boolean } = {},
): Promise<{ positions: Float64Array; indices: Uint32Array }> {
  const problem = textParamsProblem(params);
  if (problem !== null) {
    throw new Error(problem);
  }
  const mesh = buildTextMesh(await loadTextFont(), params);
  if (mesh.indices.length === 0) {
    throw new Error("The text has no letters to place.");
  }
  const stamp = options.follow === true ? refineAcross(mesh, params.size / FOLLOW_EDGES_PER_LETTER) : mesh;
  return { positions: Float64Array.from(stamp.positions), indices: stamp.indices };
}

/** The two ways text goes onto a surface. */
export type TextStampMode = "embedUnion" | "engraveSubtract";

/**
 * What a text stamp will do with `place`'s settings, in words: how far
 * Embed & Union raises it (its scaled depth less the embed), or how deep
 * Engrave cuts (the embed, when the text still reaches the surface);
 * `problem` when it would not show. `null` for Keep Separate.
 */
export function textStampEffect(
  place: Wave6PrintPrepState["placeParams"],
  unit: LengthUnit,
): { text: string; problem: boolean } | null {
  const above = place.text.depth * Math.max(0.01, place.uniformScale) - place.embedDepth;
  if (place.mode === "embedUnion") {
    return above > 0
      ? { text: `Raises the text ${formatLength(above, unit)} out of the surface.`, problem: false }
      : { text: "The text sinks under the surface: make Depth larger or Embed smaller.", problem: true };
  }
  if (place.mode === "engraveSubtract") {
    if (!(place.embedDepth > 0)) {
      return { text: "Engrave cuts as deep as Embed: make Embed above zero.", problem: true };
    }
    return above > 0
      ? { text: `Cuts the text ${formatLength(place.embedDepth, unit)} into the surface.`, problem: false }
      : { text: "The text lies under the surface: make Depth larger than Embed.", problem: true };
  }
  return null;
}

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
