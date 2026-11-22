// The command palette's registry (#235): every command it can run, built
// from the app state and the shell's callbacks. Workspace tabs, the six
// diagnostic shading modes, camera presets, sculpt brushes and the other
// tools, the shell actions, and a select and a lock command per loaded
// object.
import type { MeshHandle } from "fast-mesh-wasm";
import type { AppState, CameraPreset, DiagnosticShadingMode, Item, WorkbenchTab } from "../../app/state/app-state";
import { sculptBrush, type SculptTool } from "../../features/sculpt/ux/sculpt";
import { PLACEMENT_COMMANDS } from "../../features/transform/ux/placement-picks";
import { triggerClick } from "./panel-triggers";
import type { GizmoMode } from "../../viewport/ui/Viewport";

export interface CommandPaletteItem {
  id: string;
  title: string;
  category: "Tab" | "Tool" | "Shading" | "Camera" | "Object" | "Action";
  keywords: string;
  shortcut?: string;
  disabled?: boolean;
  run: () => void;
}

export interface CommandPaletteContext {
  app: AppState;
  items: readonly Item[];
  eligibleItems?: readonly Item[];
  selection: MeshHandle | null;
  canUndo: boolean;
  canRedo: boolean;
  ortho: boolean;
  units: "mm" | "in";
  onSelectItem: (mesh: MeshHandle) => void;
  onSetMode: (mode: GizmoMode) => void;
  onResetView: () => void;
  onUndo: () => void;
  onRedo: () => void;
  onOpenHelp: () => void;
}

export const TAB_LABELS: readonly { tab: WorkbenchTab; label: string; keywords: string }[] = [
  { tab: "home", label: "Home Workspace", keywords: "tab home open recent project overview" },
  { tab: "select", label: "Select & Transform", keywords: "tab select gizmo move rotate scale groups face lasso box brush shell split extract" },
  { tab: "repair", label: "Mesh Repair & Healer", keywords: "tab heal repair holes manifold slivers intersections" },
  { tab: "edit", label: "Mesh Edit & Modeling", keywords: "tab modeling offset shell decimate boolean remesh hollow drain hole place stamp" },
  { tab: "sculpt", label: "Sculpt & Deform", keywords: "tab sculpt inflate smooth flatten crease grab clay twist mask deform ffd arap kelvinlet xpbd self collision" },
  { tab: "align", label: "Points & Align", keywords: "tab align register point cloud sor ror poisson bpa reconstruct collision rule rigid cut soft distance gap pick" },
  { tab: "volume", label: "Voxel / CT Volume", keywords: "tab volume dicom mpr hu window slice" },
  { tab: "inspect", label: "Inspect & Measure", keywords: "tab inspect ruler measure section clip curvature thickness primitive fit plane sphere cylinder cone datum contour svg dxf" },
];

const SHADING_COMMANDS: readonly { mode: DiagnosticShadingMode; label: string; keywords: string }[] = [
  { mode: "smooth", label: "Shading: Smooth", keywords: "shading smooth phong normal" },
  { mode: "flat", label: "Shading: Flat", keywords: "shading flat faceted triangle normals" },
  { mode: "autoCrease", label: "Shading: Auto-Crease", keywords: "shading auto crease sharp dihedral angle edges" },
  { mode: "wireframeShaded", label: "Shading: Wireframe-on-Shaded", keywords: "shading wireframe shaded overlay mesh topology" },
  { mode: "matcap", label: "Shading: MatCap", keywords: "shading matcap clay sculpt studio" },
  { mode: "backfaceWarning", label: "Shading: Backface Warning", keywords: "shading backface warning inverted flipped winding red pink" },
];

const CAMERA_PRESETS: readonly { preset: CameraPreset; label: string; shortcut: string }[] = [
  { preset: "front", label: "Camera: Front View", shortcut: "1" },
  { preset: "back", label: "Camera: Back View", shortcut: "Shift+1" },
  { preset: "left", label: "Camera: Left View", shortcut: "3" },
  { preset: "right", label: "Camera: Right View", shortcut: "Shift+3" },
  { preset: "top", label: "Camera: Top View", shortcut: "7" },
  { preset: "bottom", label: "Camera: Bottom View", shortcut: "Shift+7" },
];

const SCULPT_TOOLS: readonly { tool: SculptTool; label: string; keywords: string }[] = [
  { tool: "inflate", label: "Sculpt: Inflate Brush", keywords: "sculpt inflate push dent brush" },
  { tool: "smooth", label: "Sculpt: Smooth Brush", keywords: "sculpt smooth taubin relax brush" },
  { tool: "flatten", label: "Sculpt: Flatten Brush", keywords: "sculpt flatten planar brush" },
  { tool: "crease", label: "Sculpt: Crease Brush", keywords: "sculpt crease pinch sharp groove brush" },
  { tool: "grab", label: "Sculpt: Grab Brush", keywords: "sculpt grab move drag deform" },
  { tool: "clay", label: "Sculpt: Clay Build Brush", keywords: "sculpt clay buildup brush" },
  { tool: "twist", label: "Sculpt: Twist Brush", keywords: "sculpt twist vortex rotate brush" },
];

/** Every command for `ctx`, in registry order (the palette ranks them per query). */
export function buildCommandPaletteItems(ctx: CommandPaletteContext): CommandPaletteItem[] {
  const { app, items, selection } = ctx;
  const cmds: CommandPaletteItem[] = [];

  for (const t of TAB_LABELS) {
    cmds.push({
      id: `tab:${t.tab}`,
      title: t.label,
      category: "Tab",
      keywords: t.keywords,
      run: () => app.setActiveTab(t.tab),
    });
  }

  for (const s of SHADING_COMMANDS) {
    cmds.push({
      id: `shading:${s.mode}`,
      title: s.label,
      category: "Shading",
      keywords: s.keywords,
      run: () => app.setShadingMode(s.mode),
    });
  }

  for (const c of CAMERA_PRESETS) {
    cmds.push({
      id: `camera:${c.preset}`,
      title: c.label,
      category: "Camera",
      keywords: `camera view ${c.preset} preset ortho focus`,
      shortcut: c.shortcut,
      run: () => app.requestCameraPreset(c.preset),
    });
  }

  const eligible = ctx.eligibleItems ?? items.filter((i) => !i.locked);
  const sculptTarget =
    (selection !== null ? eligible.find((i) => i.mesh === selection) : undefined) ?? eligible[0] ?? null;

  for (const st of SCULPT_TOOLS) {
    cmds.push({
      id: `sculpt:${st.tool}`,
      title: st.label,
      category: "Tool",
      keywords: st.keywords,
      disabled: sculptTarget === null,
      run: () => {
        if (sculptTarget === null) return;
        app.setActiveTab("sculpt");
        const snap = app.getSnapshot();
        const cur = snap.sculpt;
        const radius = cur?.mesh === sculptTarget.mesh ? cur.radius : sculptTarget.radius * 0.15;
        const strength = st.tool === "smooth" ? 0.5 : radius * 0.2;
        app.setSculpt(
          sculptBrush(st.tool, sculptTarget.mesh, radius, strength, "smooth", {
            methods: snap.sculptMethods,
            grabPoisson: snap.sculptGrabPoisson,
            filter: snap.sculptFilter,
            symmetry: snap.sculptSymmetry,
          }),
        );
      },
    });
  }

  cmds.push(
    {
      id: "tool:sculpt-disarm",
      title: "Exit Sculpt / Return to Select",
      category: "Tool",
      keywords: "sculpt exit stop disarm select pointer gizmo",
      run: () => {
        app.setSculpt(null);
        app.setMaskPaint(false);
        app.disarmWave6Tools();
        app.setActiveTab("select");
      },
    },
    {
      id: "tool:gizmo-translate",
      title: "Transform: Move Gizmo",
      category: "Tool",
      keywords: "move translate gizmo transform position",
      shortcut: "T",
      run: () => {
        app.setSculpt(null);
        app.setMaskPaint(false);
        app.disarmWave6Tools();
        ctx.onSetMode("translate");
        app.setActiveTab("select");
      },
    },
    {
      id: "tool:gizmo-rotate",
      title: "Transform: Rotate Gizmo",
      category: "Tool",
      keywords: "rotate spin angle gizmo transform",
      shortcut: "R",
      run: () => {
        app.setSculpt(null);
        app.setMaskPaint(false);
        app.disarmWave6Tools();
        ctx.onSetMode("rotate");
        app.setActiveTab("select");
      },
    },
    {
      id: "tool:gizmo-scale",
      title: "Transform: Scale Gizmo",
      category: "Tool",
      keywords: "scale resize gizmo transform",
      shortcut: "S",
      run: () => {
        app.setSculpt(null);
        app.setMaskPaint(false);
        app.disarmWave6Tools();
        ctx.onSetMode("scale");
        app.setActiveTab("select");
      },
    },
    {
      id: "tool:select-box",
      title: "Select: Box Face Selection",
      category: "Tool",
      keywords: "select face marquee box frustum extract split delete",
      run: () => {
        app.setSculpt(null);
        app.setMaskPaint(false);
        app.updateWave6((prev) => ({
          ...prev,
          select: { ...prev.select, activeTool: "box" },
          deform: { ...prev.deform, activeTool: null },
          printPrep: { ...prev.printPrep, activeTool: null },
        }));
        app.setActiveTab("select");
      },
    },
    {
      id: "tool:select-lasso",
      title: "Select: Lasso Face Selection",
      category: "Tool",
      keywords: "select face lasso freehand polygon extract split",
      run: () => {
        app.setSculpt(null);
        app.setMaskPaint(false);
        app.updateWave6((prev) => ({
          ...prev,
          select: { ...prev.select, activeTool: "lasso" },
          deform: { ...prev.deform, activeTool: null },
          printPrep: { ...prev.printPrep, activeTool: null },
        }));
        app.setActiveTab("select");
      },
    },
    {
      id: "tool:select-brush",
      title: "Select: Brush Face Selection",
      category: "Tool",
      keywords: "select face brush paint sphere grow shrink",
      run: () => {
        app.setSculpt(null);
        app.setMaskPaint(false);
        app.updateWave6((prev) => ({
          ...prev,
          select: { ...prev.select, activeTool: "brush" },
          deform: { ...prev.deform, activeTool: null },
          printPrep: { ...prev.printPrep, activeTool: null },
        }));
        app.setActiveTab("select");
      },
    },
    {
      id: "tool:select-criteria",
      title: "Select: Faces by Criteria",
      category: "Tool",
      keywords:
        "select face criteria area facing camera overhang self intersecting degenerate sliver orientation inverted boundary sharp dihedral contact touching components",
      run: () => {
        app.setActiveTab("select");
      },
    },
    {
      id: "tool:deform-ffd",
      title: "Deform: FFD Lattice",
      category: "Tool",
      keywords: "deform ffd freeform lattice cage control points",
      run: () => {
        app.updateWave6((prev) => ({
          ...prev,
          deform: { ...prev.deform, algorithm: "ffd" },
        }));
        app.setActiveTab("sculpt");
      },
    },
    {
      id: "tool:deform-arap",
      title: "Deform: ARAP Surface",
      category: "Tool",
      keywords: "deform arap rigid anchor handle protect",
      run: () => {
        app.updateWave6((prev) => ({
          ...prev,
          deform: { ...prev.deform, algorithm: "arap" },
        }));
        app.setActiveTab("sculpt");
      },
    },
    {
      id: "tool:deform-kelvinlet",
      title: "Deform: Kelvinlets",
      category: "Tool",
      keywords: "deform kelvinlet grab pinch twist scale elastic",
      run: () => {
        app.updateWave6((prev) => ({
          ...prev,
          deform: { ...prev.deform, algorithm: "kelvinlet" },
        }));
        app.setActiveTab("sculpt");
      },
    },
    {
      id: "tool:deform-xpbd",
      title: "Deform: XPBD Elastic / Cloth",
      category: "Tool",
      keywords: "deform xpbd cloth elastic self collision gravity",
      run: () => {
        app.updateWave6((prev) => ({
          ...prev,
          deform: { ...prev.deform, algorithm: "xpbd" },
        }));
        app.setActiveTab("sculpt");
      },
    },
    {
      id: "tool:print-drain-hole",
      title: "Print Prep: Punch Drain Hole",
      category: "Tool",
      keywords: "hollow drain hole resin plug chamfer print prep",
      run: () => {
        app.setSculpt(null);
        app.setMaskPaint(false);
        app.updateWave6((prev) => ({
          ...prev,
          select: { ...prev.select, activeTool: null },
          deform: { ...prev.deform, activeTool: null },
          printPrep: { ...prev.printPrep, activeTool: "drainHole" },
        }));
        app.setActiveTab("edit");
      },
    },
    {
      id: "tool:print-place-stamp",
      title: "Print Prep: Surface Stamp (Place Mesh)",
      category: "Tool",
      keywords: "place mesh stamp surface align peg boss normal",
      run: () => {
        app.setSculpt(null);
        app.setMaskPaint(false);
        app.updateWave6((prev) => ({
          ...prev,
          select: { ...prev.select, activeTool: null },
          deform: { ...prev.deform, activeTool: null },
          printPrep: { ...prev.printPrep, activeTool: "placeStamp" },
        }));
        app.setActiveTab("edit");
      },
    },
    {
      id: "tool:points-panel",
      title: "Points: Point Cloud & Surface Reconstruction",
      category: "Tool",
      keywords: "points cloud xyz ply pcd pts sor ror voxel normals poisson bpa reconstruct",
      run: () => {
        app.setActiveTab("align");
      },
    },
    {
      id: "tool:collision-rules",
      title: "Collision: Inter-Object Response Rules",
      category: "Tool",
      keywords: "collision response rigid cut soft deform passthrough rule drag",
      run: () => {
        app.setActiveTab("align");
      },
    },
    {
      id: "tool:inspect-primitive-fit",
      title: "Inspect: Fit Primitive & Align Datum",
      category: "Tool",
      keywords: "inspect fit primitive plane sphere cylinder cone datum align",
      run: () => {
        app.setActiveTab("inspect");
      },
    },
    {
      id: "tool:inspect-contours",
      title: "Inspect: Section Contours & SVG/DXF",
      category: "Tool",
      keywords: "inspect contours slice svg dxf perimeter area",
      run: () => {
        app.setActiveTab("inspect");
      },
    },
    {
      id: "tool:inspect-maps",
      title: "Inspect: Deviation, Overhang & Quality Maps",
      category: "Tool",
      keywords: "inspect map deviation tolerance reference overhang support build direction critical angle triangle quality aspect ratio minimum angle sliver",
      run: () => {
        app.setActiveTab("inspect");
      },
    },
    {
      id: "tool:inspect-ruler",
      title: "Inspect: Measure Ruler",
      category: "Tool",
      keywords: "inspect ruler measure distance two points",
      run: () => {
        app.setActiveTab("inspect");
        app.setInspectTool("ruler");
      },
    },
    {
      id: "tool:inspect-section",
      title: "Inspect: Section Cut Plane",
      category: "Tool",
      keywords: "inspect section cut slice cross",
      run: () => {
        app.setActiveTab("inspect");
        app.setInspectTool("section");
      },
    },
    {
      id: "action:reset-view",
      title: "Reset Camera View",
      category: "Action",
      keywords: "camera reset view fit home frame all",
      run: ctx.onResetView,
    },
    {
      id: "action:toggle-ortho",
      title: ctx.ortho ? "Camera: Switch to Perspective" : "Camera: Switch to Orthographic",
      category: "Camera",
      keywords: "camera projection orthographic perspective persp ortho",
      run: () => {
        app.toggleCameraProjection();
      },
    },
    {
      id: "action:toggle-units",
      title: `Units: Switch to ${ctx.units === "mm" ? "Inches (in)" : "Millimetres (mm)"}`,
      category: "Action",
      keywords: "units mm in millimeters inches scale",
      run: () => app.setUnits(ctx.units === "mm" ? "in" : "mm"),
    },
    {
      id: "action:undo",
      title: "Undo Last Action",
      category: "Action",
      keywords: "undo revert history",
      shortcut: "Ctrl+Z",
      disabled: !ctx.canUndo,
      run: ctx.onUndo,
    },
    {
      id: "action:redo",
      title: "Redo Action",
      category: "Action",
      keywords: "redo repeat history",
      shortcut: "Ctrl+Y",
      disabled: !ctx.canRedo,
      run: ctx.onRedo,
    },
    {
      id: "action:help",
      title: "Open Keyboard Shortcuts & Help",
      category: "Action",
      keywords: "help shortcuts controls guide",
      shortcut: "?",
      run: ctx.onOpenHelp,
    },
  );

  // The placement helpers press the transform panel's buttons (#256).
  for (const c of PLACEMENT_COMMANDS) {
    cmds.push({
      id: `placement:${c.action}`,
      title: c.label,
      category: "Tool",
      keywords: c.keywords,
      disabled: selection === null,
      run: () => {
        app.setActiveTab("select");
        triggerClick(`[data-placement-action="${c.action}"]`);
      },
    });
  }

  for (const item of items) {
    cmds.push({
      id: `object:${item.id}`,
      title: `Select Object: ${item.name}`,
      category: "Object",
      keywords: `object mesh ${item.name} ${item.volumeId !== null ? "ct volume" : ""} ${item.locked ? "locked" : ""}`,
      run: () => {
        ctx.onSelectItem(item.mesh);
        if (item.volumeId !== null) {
          app.setActiveTab("volume");
        }
      },
    });
    cmds.push({
      id: `lock:${item.id}`,
      title: `${item.locked ? "Unlock" : "Lock"} Object: ${item.name}`,
      category: "Object",
      keywords: `lock unlock protect freeze ${item.name}`,
      run: () => app.setMeshLocked(item.mesh, !item.locked),
    });
  }

  return cmds;
}
