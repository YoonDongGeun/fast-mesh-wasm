// The command palette's Create entries (#259): one that opens the Create
// dialog on its last shape, and one per shape that opens it on that shape.
import type { AppState } from "../../../app/state/app-state";
import type { CommandPaletteItem } from "../../../shell/ux/command-registry";
import { createDialogOf } from "./create-dialog-state";
import { CREATE_SHAPES } from "./create-primitive";

/** Extra words a shape answers to in the palette's search. */
const SYNONYMS: Partial<Record<string, string>> = {
  box: "cube cuboid block",
  uvSphere: "ball globe uv",
  icosphere: "geodesic ball sphere",
  torus: "ring donut",
  prism: "hexagon polygon ngon",
  plane: "grid floor quad",
  arrow: "pointer direction",
};

export function createPrimitiveCommands(app: AppState): CommandPaletteItem[] {
  const dialog = createDialogOf(app);
  const words = "create add new primitive shape mesh";
  return [
    {
      id: "action:create-primitive",
      title: "Create Shape…",
      category: "Action",
      keywords: `${words} ${CREATE_SHAPES.map((s) => s.label.toLowerCase()).join(" ")}`,
      run: () => dialog.open(),
    },
    ...CREATE_SHAPES.map(
      (s): CommandPaletteItem => ({
        id: `create:${s.kind}`,
        title: `Create: ${s.label}`,
        category: "Action",
        keywords: `${words} ${s.label.toLowerCase()} ${SYNONYMS[s.kind] ?? ""}`,
        run: () => dialog.open(s.kind),
      }),
    ),
  ];
}
