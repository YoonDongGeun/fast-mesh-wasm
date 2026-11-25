// The build plate's command palette entries (#266): show or hide the plate,
// arrange (through the panel's own button), and go to Multi-Clone.
import type { AppState } from "../../../app/state/app-state";
import type { CommandPaletteItem } from "../../../shell/ux/command-registry";
import { focusElement, triggerClick } from "../../../shell/ux/panel-triggers";

/** The panel's controls the ribbon and the command palette reach for. */
export const ARRANGE_BUTTON = '[data-build-plate-action="arrange"]';
export const ARRANGE_SCOPE = "[data-build-plate-scope]";
export const CLONE_COPIES = '[data-multi-clone="copies"]';

export function toggleBuildPlate(app: AppState): void {
  app.setBuildPlate({ visible: !app.getSnapshot().buildPlate.visible });
}

export function buildPlateCommands(app: AppState, hasItems: boolean, hasSelection: boolean): CommandPaletteItem[] {
  return [
    {
      id: "action:build-plate-toggle",
      title: app.getSnapshot().buildPlate.visible ? "Build Plate: Hide" : "Build Plate: Show",
      category: "Action",
      keywords: "build plate bed platform show hide grid print floor",
      run: () => toggleBuildPlate(app),
    },
    {
      id: "action:build-plate-arrange",
      title: "Build Plate: Arrange Meshes",
      category: "Action",
      keywords: "build plate arrange pack nest layout footprint spacing drop floor print",
      disabled: !hasItems,
      run: () => {
        app.setActiveTab("home");
        triggerClick(ARRANGE_BUTTON);
      },
    },
    {
      id: "tool:multi-clone",
      title: "Multi-Clone: Grid or Linear Array",
      category: "Tool",
      keywords: "multi clone copies duplicate array grid linear pattern repeat rows columns spacing",
      disabled: !hasSelection,
      run: () => {
        app.setActiveTab("home");
        focusElement(CLONE_COPIES);
      },
    },
  ];
}
