// The command palette's polyline entries (#268): each presses its button in
// the polyline panel (or the Inspect panel's Keep buttons), so the panel's
// fields and reports are the one path.
import type { AppState, WorkbenchTab } from "../../../app/state/app-state";
import type { CommandPaletteItem } from "../../../shell/ux/command-registry";
import { focusElement, triggerClick } from "../../../shell/ux/panel-triggers";
import { activePolylines } from "./polyline-actions";

/** A tool's palette entry: the button it presses, and what it works on. */
interface PolylineCommand {
  action: string;
  title: string;
  keywords: string;
}

/** The tools that work on the selected polylines (`data-polyline-action` in the panel). */
export const POLYLINE_COMMANDS: readonly PolylineCommand[] = [
  { action: "simplify", title: "Polyline: Simplify", keywords: "simplify reduce points tolerance decimate" },
  { action: "resample", title: "Polyline: Smooth & Resample", keywords: "smooth resample spline spacing even points" },
  { action: "offset", title: "Polyline: Offset", keywords: "offset parallel grow shrink outline planar" },
  { action: "tube", title: "Polyline: Tube", keywords: "tube pipe sweep wire radius solid mesh" },
  { action: "extrude", title: "Polyline: Extrude", keywords: "extrude solid profile depth closed planar mesh" },
  { action: "export-svg", title: "Polyline: Export SVG", keywords: "export svg save file drawing" },
  { action: "export-dxf", title: "Polyline: Export DXF", keywords: "export dxf save file drawing" },
];

/** The Keep buttons of the Inspect panel (`data-action`). */
const KEEP_COMMANDS: readonly PolylineCommand[] = [
  { action: "inspect-keep-contours", title: "Polyline: Keep Section Contours", keywords: "keep contour slice loop object" },
  { action: "section-keep", title: "Polyline: Keep Section Plane Contours", keywords: "keep section plane cut contour object" },
  { action: "path-keep", title: "Polyline: Keep Surface Path", keywords: "keep path geodesic loop object" },
];

const WORDS = "polyline curve line";

export function polylineCommands(app: AppState): CommandPaletteItem[] {
  const s = app.getSnapshot();
  const none = activePolylines(s).length === 0;
  const show = (tab: WorkbenchTab) => {
    // The panel shows on both tabs: stay on the one that is open.
    const now = app.getSnapshot().activeTab;
    app.setActiveTab(tab === "edit" && now === "inspect" ? now : tab);
  };
  return [
    {
      id: "polyline:panel",
      title: "Polylines: Simplify, Offset, Tube, Extrude",
      category: "Tool",
      keywords: `${WORDS} panel contour path objects`,
      run: () => {
        show("edit");
        focusElement("[data-panel='polylines']");
      },
    },
    ...POLYLINE_COMMANDS.map(
      (c): CommandPaletteItem => ({
        id: `polyline:${c.action}`,
        title: c.title,
        category: "Tool",
        keywords: `${WORDS} ${c.keywords}`,
        disabled: c.action.startsWith("export") ? s.polylines.length === 0 : none,
        run: () => {
          show("edit");
          triggerClick(`[data-polyline-action="${c.action}"]`);
        },
      }),
    ),
    ...KEEP_COMMANDS.map(
      (c): CommandPaletteItem => ({
        id: `polyline:${c.action}`,
        title: c.title,
        category: "Action",
        keywords: `${WORDS} ${c.keywords}`,
        run: () => {
          show("inspect");
          triggerClick(`[data-action="${c.action}"]`);
        },
      }),
    ),
  ];
}
