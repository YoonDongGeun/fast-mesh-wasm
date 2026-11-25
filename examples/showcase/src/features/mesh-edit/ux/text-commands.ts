// The command palette's Text entries (#260): one that opens the Text
// dialog, and two that arm Place Mesh with the current text to emboss or
// engrave it on a clicked surface.
import type { AppState } from "../../../app/state/app-state";
import type { CommandPaletteItem } from "../../../shell/ux/command-registry";
import { textDialogOf } from "./text-dialog-state";
import { armTextStamp } from "./text-stamp";

export function textCommands(app: AppState): CommandPaletteItem[] {
  const words = "text letters lettering label font word type 3d";
  const textNow = () => app.getSnapshot().wave6.printPrep.placeParams.text;
  return [
    {
      id: "action:create-text",
      title: "Create Text…",
      category: "Action",
      keywords: `${words} create add new mesh solid extrude`,
      run: () => textDialogOf(app).open(),
    },
    {
      id: "tool:text-emboss",
      title: "Text: Emboss on Surface",
      category: "Tool",
      keywords: `${words} emboss raise relief stamp place mesh surface`,
      run: () => armTextStamp(app, textNow(), "embedUnion"),
    },
    {
      id: "tool:text-engrave",
      title: "Text: Engrave on Surface",
      category: "Tool",
      keywords: `${words} engrave cut carve deboss stamp place mesh surface`,
      run: () => armTextStamp(app, textNow(), "engraveSubtract"),
    },
  ];
}
