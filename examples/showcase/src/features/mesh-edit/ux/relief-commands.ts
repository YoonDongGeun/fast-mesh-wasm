// The command palette's height map entries (#267): the Image to Relief dialog,
// and the export panel's Height Map section.
import type { AppState } from "../../../app/state/app-state";
import type { CommandPaletteItem } from "../../../shell/ux/command-registry";
import { focusElement } from "../../../shell/ux/panel-triggers";
import { reliefDialogOf } from "./relief-dialog-state";

/** Shows the export panel's Height Map section. */
export function showHeightMapExport(app: AppState): void {
  app.setActiveTab("home");
  focusElement('[data-action="export-height-map"]');
}

export function reliefCommands(app: AppState): CommandPaletteItem[] {
  const words = "height map heightmap depth image png 16-bit relief";
  return [
    {
      id: "action:image-to-relief",
      title: "Image to Relief…",
      category: "Action",
      keywords: `${words} import picture photo lithophane terrain emboss heightfield create solid`,
      run: () => reliefDialogOf(app).open(),
    },
    {
      id: "action:export-height-map",
      title: "Export Height Map…",
      category: "Action",
      keywords: `${words} export save download orthographic displacement`,
      run: () => showHeightMapExport(app),
    },
  ];
}
