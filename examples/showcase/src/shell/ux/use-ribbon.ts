// The workspace ribbon's state and what its buttons do (#228). Picking a
// gizmo mode disarms sculpt, mask paint and the Wave 6 tools and lands on
// Select; a face selection tool toggles and disarms the other Wave 6 tools;
// a deformation algorithm is chosen and its inspector section focused. The
// shelf's toggles (shading, inspect tool, healer overlays, units) flip the
// state the ribbon last rendered, and the option tables it maps over live
// here too.
import {
  type AppState,
  type DiagnosticShadingMode,
  type HealerOverlayKind,
  type InspectTool,
  useApp,
} from "../../app/state/app-state";
import type { GizmoMode } from "../../viewport/ui/Viewport";
import { focusElement } from "./panel-triggers";

/** The face selection tools the Select shelf arms. */
export const FACE_SELECT_TOOLS = [
  ["box", "Box Select"],
  ["lasso", "Lasso"],
  ["brush", "Brush"],
  ["shell", "Shell"],
] as const;

type FaceSelectTool = (typeof FACE_SELECT_TOOLS)[number][0];

/** The free-form deformation algorithms the Sculpt shelf chooses between. */
export const DEFORM_ALGORITHMS = [
  ["ffd", "FFD"],
  ["arap", "ARAP"],
  ["kelvinlet", "Kelvinlet"],
  ["xpbd", "XPBD"],
] as const;

type DeformAlgorithm = (typeof DEFORM_ALGORITHMS)[number][0];

/** The inspect tools the Inspect shelf toggles. */
export const INSPECT_TOOLS: readonly [InspectTool, string][] = [
  ["ruler", "Ruler"],
  ["query", "Vertex / Thickness Query"],
  ["section", "Cross-Section"],
  ["clip", "Clip Box"],
];

/** The ribbon's slice of the app state, and its button handlers. */
export function useRibbon(app: AppState, onSetMode: (mode: GizmoMode) => void) {
  const activeTab = useApp(app, (s) => s.activeTab);
  const units = useApp(app, (s) => s.units);
  const shadingMode = useApp(app, (s) => s.shadingMode);
  const ortho = useApp(app, (s) => s.cameraProjection === "orthographic");
  const inspectTool = useApp(app, (s) => s.inspectTool);
  const collisionMode = useApp(app, (s) => s.collisionMode);
  const sculpt = useApp(app, (s) => s.sculpt);
  const maskPaint = useApp(app, (s) => s.maskPaint);
  const healerOverlays = useApp(app, (s) => s.healerOverlays);
  const faceSelectTool = useApp(app, (s) => s.wave6.select.activeTool);
  const deformAlgo = useApp(app, (s) => s.wave6.deform.algorithm);
  const deformTool = useApp(app, (s) => s.wave6.deform.activeTool);

  const pickMode = (next: GizmoMode) => {
    if (sculpt !== null) app.setSculpt(null);
    if (maskPaint) app.setMaskPaint(false);
    app.disarmWave6Tools();
    onSetMode(next);
    app.setActiveTab("select");
  };

  const toggleFaceSelectTool = (tool: FaceSelectTool) => {
    if (sculpt !== null) app.setSculpt(null);
    if (maskPaint) app.setMaskPaint(false);
    app.updateWave6((prev) => ({
      ...prev,
      select: { ...prev.select, activeTool: prev.select.activeTool === tool ? null : tool },
      deform: { ...prev.deform, activeTool: null },
      printPrep: { ...prev.printPrep, activeTool: null },
    }));
  };

  const chooseDeformAlgorithm = (alg: DeformAlgorithm) => {
    app.updateWave6((prev) => ({
      ...prev,
      deform: { ...prev.deform, algorithm: alg },
    }));
    focusElement('[data-inspector-slot="deform"]');
  };

  /** Turns `mode` on, or back to smooth shading when it is already on. */
  const toggleShading = (mode: DiagnosticShadingMode) =>
    app.setShadingMode(shadingMode === mode ? "smooth" : mode);

  const toggleInspectTool = (tool: InspectTool) => app.setInspectTool(inspectTool === tool ? null : tool);

  const toggleHealerOverlay = (kind: HealerOverlayKind) => app.setHealerOverlay(kind, !healerOverlays[kind]);

  const toggleUnits = () => app.setUnits(units === "mm" ? "in" : "mm");

  return {
    activeTab,
    units,
    shadingMode,
    ortho,
    inspectTool,
    collisionMode,
    healerOverlays,
    faceSelectTool,
    deformAlgo,
    deformTool,
    pickMode,
    toggleFaceSelectTool,
    chooseDeformAlgorithm,
    toggleShading,
    toggleInspectTool,
    toggleHealerOverlay,
    toggleUnits,
  };
}
