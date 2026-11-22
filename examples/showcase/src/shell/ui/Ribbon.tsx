/**
 * MeshInspector-style 2-tier Workspace Ribbon (#228):
 * - Tier 1 (`.toolbar`): Quick-Access header + 8 Workspace Ribbon Tabs.
 * - Tier 2 (`.ribbon-shelf`): Contextual tool actions for the active tab, plus
 *   the always-mounted `<SculptToolbox>` (CSS-hidden outside `Sculpt & Deform`
 *   so E2E controls and sliders stay live and queryable).
 */
import type { DyntopoMode, MeshHandle } from "fast-mesh-wasm";
import { useScene } from "fast-mesh-wasm/react";
import {
  type AppState,
  type InspectTool,
  useApp,
  type WorkbenchTab,
} from "../../app/state/app-state";
import { applyDyntopoRoi, DYNTOPO_MODE_OPTIONS } from "../../features/sculpt/ux/sculpt";
import { DIAGNOSTIC_SHADING_OPTIONS } from "../../features/view/ui/shading";
import { SculptToolbox } from "../../features/sculpt/ui/Toolbox";
import type { GizmoMode } from "../../viewport/ui/Viewport";
import { formatLength } from "../../shared/ux/units";

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

export interface WorkbenchRibbonProps {
  app: AppState;
  mode: GizmoMode;
  canUndo: boolean;
  canRedo: boolean;
  wireframe: boolean;
  selection: MeshHandle | null;
  hasItems: boolean;
  hasVolume: boolean;
  onSetMode: (mode: GizmoMode) => void;
  onResetView: () => void;
  onFocusSelection: () => void;
  onClearSelection: () => void;
  onUndo: () => void;
  onRedo: () => void;
  onOpenHelp: () => void;
}

function triggerClick(selector: string): void {
  const el = document.querySelector<HTMLElement>(selector);
  el?.click();
}

function focusElement(selector: string): void {
  const el = document.querySelector<HTMLElement>(selector);
  if (!el) return;
  el.scrollIntoView({ block: "nearest" });
  el.focus();
}

export function WorkbenchRibbon(props: WorkbenchRibbonProps) {
  const {
    app,
    mode,
    canUndo,
    canRedo,
    wireframe,
    selection,
    hasItems,
    hasVolume,
    onSetMode,
    onResetView,
    onFocusSelection,
    onClearSelection,
    onUndo,
    onRedo,
    onOpenHelp,
  } = props;

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

  const toggleFaceSelectTool = (tool: "box" | "lasso" | "brush" | "shell") => {
    if (sculpt !== null) app.setSculpt(null);
    if (maskPaint) app.setMaskPaint(false);
    app.updateWave6((prev) => ({
      ...prev,
      select: { ...prev.select, activeTool: prev.select.activeTool === tool ? null : tool },
      deform: { ...prev.deform, activeTool: null },
      printPrep: { ...prev.printPrep, activeTool: null },
    }));
  };

  return (
    <header className="workbench-ribbon" aria-label="Workspace ribbon">
      {/* Tier 1: Quick-Access Header + Workspace Ribbon Tabs */}
      <div className="toolbar" role="toolbar" aria-label="Viewport tools" data-tour="gizmo">
        <div className="group">
          <button
            type="button"
            data-action="ribbon-open"
            onClick={() => document.getElementById("file-input")?.click()}
            title="Open mesh or DICOM file"
          >
            Open
          </button>
          <button
            type="button"
            aria-pressed={mode === "translate"}
            onClick={() => pickMode("translate")}
            title="Move (T)"
          >
            Move <kbd>T</kbd>
          </button>
          <button
            type="button"
            aria-pressed={mode === "rotate"}
            onClick={() => pickMode("rotate")}
            title="Rotate (R)"
          >
            Rotate <kbd>R</kbd>
          </button>
          <button
            type="button"
            data-action="mode-scale"
            aria-pressed={mode === "scale"}
            onClick={() => pickMode("scale")}
            title="Scale (S)"
          >
            Scale <kbd>S</kbd>
          </button>
          <button type="button" data-action="view-reset" onClick={onResetView} title="Reset the camera">
            Reset view
          </button>
        </div>

        <div className="group">
          <button type="button" data-action="undo" disabled={!canUndo} onClick={onUndo} title="Undo (Ctrl+Z)">
            Undo
          </button>
          <button type="button" data-action="redo" disabled={!canRedo} onClick={onRedo} title="Redo (Ctrl+Shift+Z)">
            Redo
          </button>
          <button
            type="button"
            data-action="wire"
            aria-pressed={wireframe}
            onClick={() => app.setWireframe(!wireframe)}
            title="Draw every mesh wireframed (W)"
          >
            Wire
          </button>
          <button
            type="button"
            data-action="ribbon-units"
            onClick={() => app.setUnits(units === "mm" ? "in" : "mm")}
            title="Switch display length unit (mm / in)"
          >
            {units}
          </button>
          <button
            type="button"
            data-action="command-palette"
            onClick={() => app.setCommandPaletteOpen(true)}
            title="Command Search Palette (Ctrl+K or /)"
          >
            Search <kbd>Ctrl+K</kbd>
          </button>
          <button type="button" data-action="help" onClick={onOpenHelp} title="Keyboard shortcuts (?)">
            Help <kbd>?</kbd>
          </button>
        </div>

        <nav className="group ribbon-tabs" role="tablist" aria-label="Workspace tabs">
          {WORKBENCH_TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              role="tab"
              data-ribbon-tab={tab.id}
              aria-selected={activeTab === tab.id}
              title={tab.title}
              onClick={() => app.setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Tier 2: Contextual Tool Shelf */}
      <div className="ribbon-shelf" data-ribbon-shelf={activeTab} aria-label={`${workbenchTabLabel(activeTab)} tools`}>
        {/* Always mounted so sculpt brush/mask/radius DOM hooks stay live even on other tabs. */}
        <div
          className={activeTab === "sculpt" ? "ribbon-sculpt-wrap" : "ribbon-sculpt-wrap ribbon-hidden"}
          onClickCapture={() => app.setActiveTab("sculpt")}
        >
          <SculptToolbox app={app} />
        </div>
        {activeTab === "sculpt" ? (
          <div className="ribbon-strip ribbon-strip-compact" data-ribbon-deform-strip>
            <div className="group" aria-label="Free-form deformation">
              {(
                [
                  ["ffd", "FFD"],
                  ["arap", "ARAP"],
                  ["kelvinlet", "Kelvinlet"],
                  ["xpbd", "XPBD"],
                ] as const
              ).map(([alg, label]) => (
                <button
                  key={alg}
                  type="button"
                  data-ribbon-deform-algo={alg}
                  aria-pressed={deformAlgo === alg}
                  onClick={() => {
                    app.updateWave6((prev) => ({
                      ...prev,
                      deform: { ...prev.deform, algorithm: alg },
                    }));
                    focusElement('[data-inspector-slot="deform"]');
                  }}
                >
                  {label}
                </button>
              ))}
              {deformTool !== null ? (
                <button
                  type="button"
                  data-ribbon-deform-disarm
                  onClick={() => app.disarmWave6Tools()}
                >
                  Disarm Deform
                </button>
              ) : null}
            </div>
          </div>
        ) : null}

        {activeTab === "home" ? (
          <div className="ribbon-strip">
            <div className="group">
              <button type="button" onClick={() => document.getElementById("file-input")?.click()}>
                Load File
              </button>
              <button type="button" onClick={() => triggerClick('[data-action="demo-meshes"]')}>
                Load Sample Stage
              </button>
              <button type="button" onClick={onResetView}>
                Reset View
              </button>
              <button type="button" disabled={selection === null} onClick={onFocusSelection}>
                Focus Selection
              </button>
              <button type="button" onClick={() => app.setCommandPaletteOpen(true)}>
                Command Search (Ctrl+K)
              </button>
            </div>
            <div className="group ribbon-shading-strip" aria-label="Shading">
              {DIAGNOSTIC_SHADING_OPTIONS.map((opt) => (
                <button
                  key={opt.mode}
                  type="button"
                  data-ribbon-shading={opt.mode}
                  aria-pressed={shadingMode === opt.mode}
                  title={opt.title}
                  onClick={() => app.setShadingMode(opt.mode)}
                >
                  {opt.shortLabel}
                </button>
              ))}
            </div>
          </div>
        ) : null}

        {activeTab === "select" ? (
          <div className="ribbon-strip">
            <div className="group">
              <button type="button" aria-pressed={mode === "translate"} onClick={() => pickMode("translate")}>
                Move (T)
              </button>
              <button type="button" aria-pressed={mode === "rotate"} onClick={() => pickMode("rotate")}>
                Rotate (R)
              </button>
              <button type="button" aria-pressed={mode === "scale"} onClick={() => pickMode("scale")}>
                Scale (S)
              </button>
              <button type="button" disabled={selection === null} onClick={onFocusSelection}>
                Focus (F)
              </button>
              <button type="button" disabled={selection === null} onClick={onClearSelection}>
                Deselect
              </button>
              <button
                type="button"
                aria-pressed={ortho}
                onClick={() => app.toggleCameraProjection()}
              >
                {ortho ? "Orthographic" : "Perspective"}
              </button>
            </div>
            <div className="group" aria-label="Face selection">
              {(
                [
                  ["box", "Box Select"],
                  ["lasso", "Lasso"],
                  ["brush", "Brush"],
                  ["shell", "Shell"],
                ] as const
              ).map(([toolId, label]) => (
                <button
                  key={toolId}
                  type="button"
                  data-ribbon-select-tool={toolId}
                  aria-pressed={faceSelectTool === toolId}
                  onClick={() => toggleFaceSelectTool(toolId)}
                >
                  {label}
                </button>
              ))}
              <button
                type="button"
                data-ribbon-select-clear
                disabled={!hasItems}
                onClick={() => triggerClick('[data-select-action="clear"]')}
              >
                Clear Faces
              </button>
            </div>
            <div className="group ribbon-shading-strip" aria-label="Diagnostic shading">
              {DIAGNOSTIC_SHADING_OPTIONS.map((opt) => (
                <button
                  key={opt.mode}
                  type="button"
                  data-ribbon-shading={opt.mode}
                  aria-pressed={shadingMode === opt.mode}
                  title={opt.title}
                  onClick={() => app.setShadingMode(opt.mode)}
                >
                  {opt.shortLabel}
                </button>
              ))}
            </div>
          </div>
        ) : null}

        {activeTab === "repair" ? (
          <div className="ribbon-strip">
            <div className="group">
              <button type="button" disabled={!hasItems} onClick={() => triggerClick('[data-repair-op="analyze"]')}>
                Analyze Mesh
              </button>
              <button type="button" disabled={!hasItems} onClick={() => triggerClick('[data-repair-op="autoRepair"]')}>
                Auto Heal
              </button>
              <button
                type="button"
                disabled={!hasItems}
                onClick={() => triggerClick('[data-repair-op="selectiveHeal"]')}
              >
                Selective Heal
              </button>
              <button
                type="button"
                aria-pressed={healerOverlays.selfIntersections}
                onClick={() => app.setHealerOverlay("selfIntersections", !healerOverlays.selfIntersections)}
              >
                Self-Intersect Overlay
              </button>
              <button
                type="button"
                aria-pressed={healerOverlays.slivers}
                onClick={() => app.setHealerOverlay("slivers", !healerOverlays.slivers)}
              >
                Sliver Overlay
              </button>
              <button
                type="button"
                aria-pressed={healerOverlays.inverted}
                onClick={() => app.setHealerOverlay("inverted", !healerOverlays.inverted)}
              >
                Inverted Overlay
              </button>
            </div>
          </div>
        ) : null}

        {activeTab === "edit" ? (
          <div className="ribbon-strip">
            <div className="group">
              <button type="button" onClick={() => focusElement("[data-modeling-boolean-run]")}>
                Boolean CSG
              </button>
              <button type="button" onClick={() => focusElement("[data-modeling-cut-run]")}>
                Plane Cut
              </button>
              <button type="button" onClick={() => focusElement("[data-modeling-offset-run]")}>
                Offset &amp; Shell
              </button>
              <button type="button" onClick={() => focusElement("[data-modeling-simplify-run]")}>
                Simplify
              </button>
              <button type="button" onClick={() => focusElement("[data-modeling-subdivide-run]")}>
                Subdivide
              </button>
              <button type="button" data-ribbon-focus="hollow" onClick={() => focusElement("[data-modeling-hollow-section]")}>
                Hollow
              </button>
              <button type="button" data-ribbon-focus="drain" onClick={() => focusElement("[data-modeling-drain-section]")}>
                Drain Hole
              </button>
              <button type="button" data-ribbon-focus="place" onClick={() => focusElement("[data-modeling-place-section]")}>
                Surface Stamp
              </button>
              <button
                type="button"
                aria-pressed={shadingMode === "wireframeShaded"}
                onClick={() =>
                  app.setShadingMode(shadingMode === "wireframeShaded" ? "smooth" : "wireframeShaded")
                }
              >
                Wire-on-Shaded
              </button>
            </div>
          </div>
        ) : null}

        {activeTab === "align" ? (
          <div className="ribbon-strip">
            <div className="group">
              <button
                type="button"
                aria-pressed={collisionMode === "pair"}
                onClick={() => app.setCollisionMode("pair")}
              >
                Pair Clearance
              </button>
              <button
                type="button"
                aria-pressed={collisionMode === "group"}
                onClick={() => app.setCollisionMode("group")}
              >
                Contact Group
              </button>
              <button type="button" onClick={() => focusElement("[data-align-pick]")}>
                Landmark Align
              </button>
              <button type="button" data-ribbon-focus="points" onClick={() => focusElement("[data-points-panel]")}>
                Point Cloud
              </button>
              <button
                type="button"
                data-ribbon-focus="collision-rules"
                onClick={() => focusElement("[data-collision-rules]")}
              >
                Collision Rules
              </button>
            </div>
          </div>
        ) : null}

        {activeTab === "volume" ? (
          <div className="ribbon-strip">
            <div className="group">
              <button
                type="button"
                disabled={!hasVolume}
                onClick={() => focusElement("[data-panel='volume']")}
              >
                {hasVolume ? "CT / DICOM Active" : "No Volume Selected"}
              </button>
              <button type="button" onClick={() => triggerClick('[data-action="dicom-demo-head"]')}>
                Load Head CT
              </button>
              <button type="button" aria-pressed={ortho} onClick={() => app.toggleCameraProjection()}>
                {ortho ? "Orthographic" : "Perspective"}
              </button>
              <button type="button" onClick={() => app.requestCameraPreset("top")}>
                Axial Top
              </button>
            </div>
          </div>
        ) : null}

        {activeTab === "inspect" ? (
          <div className="ribbon-strip">
            <div className="group">
              {(
                [
                  ["ruler", "Ruler"],
                  ["query", "Vertex / Thickness Query"],
                  ["section", "Cross-Section"],
                  ["clip", "Clip Box"],
                ] as readonly [InspectTool, string][]
              ).map(([toolId, label]) => (
                <button
                  key={toolId}
                  type="button"
                  aria-pressed={inspectTool === toolId}
                  onClick={() => app.setInspectTool(inspectTool === toolId ? null : toolId)}
                >
                  {label}
                </button>
              ))}
              <button type="button" data-ribbon-focus="fit" onClick={() => focusElement("[data-inspect-primitive-fit]")}>
                Fit Primitive
              </button>
              <button type="button" data-ribbon-focus="contours" onClick={() => focusElement("[data-inspect-contours]")}>
                Contours
              </button>
              <button
                type="button"
                aria-pressed={shadingMode === "backfaceWarning"}
                onClick={() =>
                  app.setShadingMode(shadingMode === "backfaceWarning" ? "smooth" : "backfaceWarning")
                }
              >
                Backface Check
              </button>
            </div>
          </div>
        ) : null}
      </div>
    </header>
  );
}

export function InspectorHeader(props: { app: AppState; activeTab: WorkbenchTab }) {
  const { app, activeTab } = props;
  return (
    <div className="inspector-header" data-inspector-header={activeTab}>
      <span className="inspector-tab-title">{workbenchTabLabel(activeTab)}</span>
      <div className="inspector-header-actions">
        {activeTab !== "home" ? (
          <button type="button" data-action="inspector-home" onClick={() => app.setActiveTab("home")}>
            Home
          </button>
        ) : null}
        <button
          type="button"
          data-action="inspector-search"
          title="Open command search palette (Ctrl+K)"
          onClick={() => app.setCommandPaletteOpen(true)}
        >
          Search
        </button>
      </div>
    </div>
  );
}

export function HomeOverviewCard(props: { app: AppState; itemCount: number; onResetView: () => void }) {
  const { app, itemCount, onResetView } = props;
  return (
    <section className="panel home-overview-panel" aria-label="Workspace Home" data-panel="home-overview">
      <header className="panel-head">
        <h2>Workbench</h2>
        <span className="muted">{itemCount} object{itemCount === 1 ? "" : "s"}</span>
      </header>
      <p className="muted">
        Choose a ribbon tab above or press <kbd>Ctrl+K</kbd> to jump directly to any repair, sculpt, edit, align, volume,
        or inspection tool.
      </p>
      <div className="home-quick-grid">
        <button type="button" onClick={() => app.setActiveTab("repair")}>
          Mesh Repair
        </button>
        <button type="button" onClick={() => app.setActiveTab("sculpt")}>
          Sculpt &amp; Deform
        </button>
        <button type="button" onClick={() => app.setActiveTab("edit")}>
          Mesh Edit
        </button>
        <button type="button" onClick={() => app.setActiveTab("inspect")}>
          Inspect
        </button>
        <button type="button" onClick={onResetView}>
          Reset View
        </button>
      </div>
    </section>
  );
}

export function SculptInspectorCard(props: { app: AppState; selection: MeshHandle | null }) {
  const { app, selection } = props;
  const store = useScene();
  const sculpt = useApp(app, (s) => s.sculpt);
  const maskPaint = useApp(app, (s) => s.maskPaint);
  const mask = useApp(app, (s) => s.sculptMask);
  const brushPick = useApp(app, (s) => s.brushPick);
  const pinToActive = useApp(app, (s) => s.sculptPinToActive);
  const liveRemesh = useApp(app, (s) => s.sculptLiveRemesh);
  const remesh = useApp(app, (s) => s.sculptRemesh);
  const dyntopo = useApp(app, (s) => s.sculptDyntopo);
  const dyntopoMode = useApp(app, (s) => s.sculptDyntopoMode);
  const dyntopoDetail = useApp(app, (s) => s.sculptDyntopoDetail);
  const symmetry = useApp(app, (s) => s.sculptSymmetry);
  const shadingMode = useApp(app, (s) => s.shadingMode);
  const units = useApp(app, (s) => s.units);
  const items = useApp(app, (s) => s.items);
  const targetMesh = sculpt?.mesh ?? mask?.mesh ?? selection;
  const targetItem = targetMesh === null ? null : (items.find((i) => i.mesh === targetMesh) ?? null);
  const targetRadius = targetItem?.radius ?? 1;

  return (
    <section className="panel sculpt-inspector-panel" aria-label="Sculpt Parameters" data-panel="sculpt-inspector">
      <header className="panel-head">
        <h2>Sculpt &amp; Deform</h2>
        <span className="muted">{sculpt ? sculpt.tool : maskPaint ? "Mask paint" : "Ready"}</span>
      </header>
      <p className="muted">
        {pinToActive
          ? targetItem
            ? `Pinned to ${targetItem.name}.`
            : "Pin is enabled: select a visible unlocked mesh."
          : "Hover any visible unlocked mesh to sculpt; Alt+Click selects/pins a mesh."}
      </p>
      <div className="op">
        <h3>Targeting &amp; Remeshing</h3>
        <label className="check">
          <input
            type="checkbox"
            checked={pinToActive}
            onChange={(e) => app.setSculptPinToActive(e.currentTarget.checked)}
          />
          Pin to active mesh
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={liveRemesh}
            onChange={(e) => app.setSculptLiveRemesh(e.currentTarget.checked)}
          />
          Auto-split stretched edges (live)
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={remesh}
            onChange={(e) => app.setSculptRemesh(e.currentTarget.checked)}
          />
          Remesh stroke region on release
        </label>
      </div>
      <div className="op">
        <h3>Dynamic Topology (Dyntopo)</h3>
        <label className="check">
          <input
            type="checkbox"
            data-action="sculpt-inspector-dyntopo"
            checked={dyntopo}
            onChange={(e) => app.setSculptDyntopo(e.currentTarget.checked)}
          />
          Enable Dynamic Topology
        </label>
        <label className="field">
          Mode
          <select
            data-action="sculpt-inspector-dyntopo-mode"
            value={dyntopoMode}
            onChange={(e) => app.setSculptDyntopoMode(e.currentTarget.value as DyntopoMode)}
          >
            {DYNTOPO_MODE_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </label>
        <label className="slider">
          Detail{" "}
          <output>
            {dyntopoMode === "brush-relative"
              ? `${Math.round(dyntopoDetail * 100)}%`
              : formatLength(dyntopoDetail * targetRadius, units)}
          </output>
          <input
            type="range"
            data-action="sculpt-inspector-dyntopo-detail"
            min={0.04}
            max={0.8}
            step={0.01}
            value={Math.min(0.8, Math.max(0.04, dyntopoDetail))}
            onChange={(e) => app.setSculptDyntopoDetail(Number(e.currentTarget.value))}
          />
        </label>
        <button
          type="button"
          data-action="sculpt-inspector-dyntopo-apply"
          disabled={targetItem === null}
          onClick={() => {
            if (targetItem === null) {
              return;
            }
            void applyDyntopoRoi({
              scene: store.scene,
              store,
              target: targetItem,
              brushPick,
              mask,
              mode: dyntopoMode,
              detail: dyntopoDetail,
              brushRadius: sculpt?.radius ?? targetItem.radius * 0.25,
            });
          }}
        >
          Subdivide / Clean Dyntopo ROI
        </button>
      </div>
      <div className="op">
        <h3>Mirror Symmetry</h3>
        <div className="modes" role="group" aria-label="Sculpt mirror symmetry">
          {(["x", "y", "z"] as const).map((axis) => (
            <button
              key={axis}
              type="button"
              aria-pressed={symmetry[axis]}
              onClick={() => app.setSculptSymmetry({ [axis]: !symmetry[axis] })}
            >
              {axis.toUpperCase()}
            </button>
          ))}
        </div>
      </div>
      <div className="op">
        <h3>Surface Shading</h3>
        <div className="modes">
          {(["smooth", "matcap", "wireframeShaded", "backfaceWarning"] as const).map((mode) => {
            const opt = DIAGNOSTIC_SHADING_OPTIONS.find((o) => o.mode === mode)!;
            return (
              <button
                key={mode}
                type="button"
                aria-pressed={shadingMode === mode}
                onClick={() => app.setShadingMode(mode)}
              >
                {opt.shortLabel}
              </button>
            );
          })}
        </div>
      </div>
      <div className="op-row">
        <button
          type="button"
          onClick={() => {
            app.setSculpt(null);
            app.setMaskPaint(false);
          }}
        >
          Disarm Tool
        </button>
        <button
          type="button"
          onClick={() => {
            app.setSculpt(null);
            app.setMaskPaint(false);
            app.setActiveTab("select");
          }}
        >
          Exit to Select
        </button>
      </div>
    </section>
  );
}

export function VolumeFallbackCard(props: { app: AppState }) {
  const { app } = props;
  return (
    <section className="panel volume-fallback-panel" aria-label="Volume Workbench" data-panel="volume-fallback">
      <header className="panel-head">
        <h2>Voxel / CT Volume</h2>
      </header>
      <p className="muted">Load or select a DICOM/CT volume to edit window/level, transfer presets, and MPR slices.</p>
      <div className="op-row">
        <button type="button" onClick={() => triggerClick('[data-action="dicom-demo-head"]')}>
          Load Head CT
        </button>
        <button type="button" onClick={() => app.setActiveTab("home")}>
          Back to Home
        </button>
      </div>
    </section>
  );
}
