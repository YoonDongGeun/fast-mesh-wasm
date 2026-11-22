// The ribbon's eight workspace tabs (#228) and the right inspector's
// sections: each tab's labels, and which sections it shows.
import type { WorkbenchTab } from "../../app/state/app-state";

export interface WorkbenchTabMeta {
  id: WorkbenchTab;
  label: string;
  shortLabel: string;
  title: string;
}

export const WORKBENCH_TABS: readonly WorkbenchTabMeta[] = [
  { id: "home", label: "Home", shortLabel: "Home", title: "Scene overview, loading, export & layout" },
  { id: "select", label: "Select", shortLabel: "Select", title: "Object selection & Move/Rotate/Scale gizmos" },
  { id: "repair", label: "Mesh Repair", shortLabel: "Repair", title: "Mesh Healer, defect overlays & repair" },
  { id: "edit", label: "Mesh Edit", shortLabel: "Edit", title: "Booleans, plane cut, offset, shell, simplify & subdivide" },
  { id: "sculpt", label: "Sculpt & Deform", shortLabel: "Sculpt", title: "3D sculpt brushes, mask painting, symmetry & live split" },
  { id: "align", label: "Points & Align", shortLabel: "Align", title: "Landmark ICP alignment & clearance/collision analysis" },
  { id: "volume", label: "Voxel / CT", shortLabel: "Volume", title: "CT/DICOM volume rendering, MPR slices & presets" },
  { id: "inspect", label: "Inspect", shortLabel: "Inspect", title: "Ruler, thickness/point query, cross-section & clip box" },
];

export type InspectorSlot =
  | "home"
  | "project"
  | "transform"
  | "select"
  | "sculpt"
  | "deform"
  | "repair"
  | "modeling"
  | "points"
  | "align"
  | "collision"
  | "volume"
  | "inspect"
  | "export";

/** Returns which right-inspector sections are shown for `tab`. */
export function tabMatchesSlot(tab: WorkbenchTab, slot: InspectorSlot): boolean {
  switch (tab) {
    case "home":
      return slot === "home" || slot === "project" || slot === "export";
    case "select":
      return slot === "transform" || slot === "select";
    case "repair":
      return slot === "repair";
    case "edit":
      return slot === "modeling";
    case "sculpt":
      return slot === "sculpt" || slot === "deform";
    case "align":
      return slot === "points" || slot === "align" || slot === "collision";
    case "volume":
      return slot === "volume";
    case "inspect":
      return slot === "inspect";
  }
}

export function workbenchTabLabel(tab: WorkbenchTab): string {
  return WORKBENCH_TABS.find((t) => t.id === tab)?.label ?? "Home";
}
