/**
 * The 2-tier Workspace Ribbon (#228):
 * - Tier 1 (`.toolbar`): Quick-Access header + 8 Workspace Ribbon Tabs.
 * - Tier 2 (`.ribbon-shelf`): Contextual tool actions for the active tab, plus
 *   the always-mounted `<SculptToolbox>` (CSS-hidden outside `Sculpt & Deform`
 *   so E2E controls and sliders stay live and queryable).
 */
import type { DyntopoMode, MeshHandle } from "fast-mesh-wasm";
import type { AppState, WorkbenchTab } from "../../app/state/app-state";
import { ARRANGE_SCOPE, CLONE_COPIES, toggleBuildPlate } from "../../features/build-plate/ux/build-plate-commands";
import { createDialogOf } from "../../features/home/ux/create-dialog-state";
import { textDialogOf } from "../../features/mesh-edit/ux/text-dialog-state";
import { SURFACE_CLEANUP_COMMANDS } from "../../features/mesh-repair/ux/surface-cleanup";
import { DYNTOPO_MODE_OPTIONS } from "../../features/sculpt/ux/sculpt";
import { DIAGNOSTIC_SHADING_OPTIONS } from "../../features/view/ui/shading";
import { PLACEMENT_COMMANDS } from "../../features/transform/ux/placement-picks";
import { SculptToolbox } from "../../features/sculpt/ui/Toolbox";
import type { GizmoMode } from "../../viewport/ui/Viewport";
import { focusElement, openFilePicker, triggerClick } from "../ux/panel-triggers";
import { DEFORM_ALGORITHMS, FACE_SELECT_TOOLS, INSPECT_TOOLS, useRibbon } from "../ux/use-ribbon";
import { SCULPT_SHADING_MODES, useSculptInspector } from "../ux/use-sculpt-inspector";
import { WORKBENCH_TABS, workbenchTabLabel } from "../ux/workbench-tabs";

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

  const {
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
    plateShown,
    pickMode,
    toggleFaceSelectTool,
    chooseDeformAlgorithm,
    toggleShading,
    toggleInspectTool,
    toggleHealerOverlay,
    toggleUnits,
  } = useRibbon(app, onSetMode);

  return (
    <header className="workbench-ribbon" aria-label="Workspace ribbon">
      {/* Tier 1: Quick-Access Header + Workspace Ribbon Tabs */}
      <div className="toolbar" role="toolbar" aria-label="Viewport tools" data-tour="gizmo">
        <div className="group">
          <button
            type="button"
            data-action="ribbon-open"
            onClick={openFilePicker}
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
            onClick={toggleUnits}
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
              {DEFORM_ALGORITHMS.map(([alg, label]) => (
                <button
                  key={alg}
                  type="button"
                  data-ribbon-deform-algo={alg}
                  aria-pressed={deformAlgo === alg}
                  onClick={() => chooseDeformAlgorithm(alg)}
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
              <button type="button" onClick={openFilePicker}>
                Load File
              </button>
              <button type="button" onClick={() => triggerClick('[data-action="demo-meshes"]')}>
                Load Sample Stage
              </button>
              <button
                type="button"
                data-action="create-primitive"
                title="Create a box, cylinder, sphere, torus or other shape"
                onClick={() => createDialogOf(app).open()}
              >
                Create Shape…
              </button>
              <button
                type="button"
                data-action="create-text"
                title="Create solid text, or emboss or engrave it on a surface"
                onClick={() => textDialogOf(app).open()}
              >
                Create Text…
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
            <div className="group" aria-label="Build plate">
              <button
                type="button"
                data-action="build-plate"
                aria-pressed={plateShown}
                title="Show or hide the build plate"
                onClick={() => toggleBuildPlate(app)}
              >
                Build Plate
              </button>
              <button type="button" data-ribbon-focus="arrange" onClick={() => focusElement(ARRANGE_SCOPE)}>
                Arrange…
              </button>
              <button type="button" data-ribbon-focus="multi-clone" onClick={() => focusElement(CLONE_COPIES)}>
                Multi-Clone…
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
              {FACE_SELECT_TOOLS.map(([toolId, label]) => (
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
            <div className="group" aria-label="Placement">
              {PLACEMENT_COMMANDS.filter((c) => c.shelf).map((c) => (
                <button
                  key={c.action}
                  type="button"
                  data-ribbon-placement={c.action}
                  disabled={selection === null}
                  onClick={() => triggerClick(`[data-placement-action="${c.action}"]`)}
                >
                  {c.short}
                </button>
              ))}
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
              {SURFACE_CLEANUP_COMMANDS.map((c) => (
                <button
                  key={c.op}
                  type="button"
                  disabled={!hasItems}
                  onClick={() => triggerClick(`[data-repair-op="${c.op}"]`)}
                >
                  {c.short}
                </button>
              ))}
              <button type="button" disabled={!hasItems} onClick={() => focusElement("[data-wrap-gap]")}>
                Wrap
              </button>
              <button
                type="button"
                aria-pressed={healerOverlays.selfIntersections}
                onClick={() => toggleHealerOverlay("selfIntersections")}
              >
                Self-Intersect Overlay
              </button>
              <button
                type="button"
                aria-pressed={healerOverlays.slivers}
                onClick={() => toggleHealerOverlay("slivers")}
              >
                Sliver Overlay
              </button>
              <button
                type="button"
                aria-pressed={healerOverlays.inverted}
                onClick={() => toggleHealerOverlay("inverted")}
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
              <button
                type="button"
                data-ribbon-focus="batch-boolean"
                onClick={() => focusElement("[data-batch-boolean-section]")}
              >
                Multi Boolean
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
              <button
                type="button"
                data-ribbon-focus="isotropic-remesh"
                onClick={() => focusElement("[data-isotropic-section]")}
              >
                Isotropic Remesh
              </button>
              <button
                type="button"
                data-ribbon-focus="extrude-faces"
                onClick={() => focusElement("[data-extrude-faces-distance]")}
              >
                Extrude Faces
              </button>
              <button
                type="button"
                data-ribbon-focus="extrude-base"
                onClick={() => focusElement("[data-extrude-base-run]")}
              >
                Extrude Base
              </button>
              <button type="button" data-ribbon-focus="hull" onClick={() => focusElement("[data-modeling-hull-run]")}>
                Convex Hull
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
              <button type="button" data-ribbon-focus="polylines" onClick={() => focusElement("[data-panel='polylines']")}>
                Polylines
              </button>
              <button
                type="button"
                data-action="text-tool"
                title="Solid text, or text embossed or engraved on a surface"
                onClick={() => textDialogOf(app).open()}
              >
                Text…
              </button>
              <button
                type="button"
                aria-pressed={shadingMode === "wireframeShaded"}
                onClick={() => toggleShading("wireframeShaded")}
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
              <button
                type="button"
                data-ribbon-focus="coarse-icp"
                onClick={() => focusElement('[data-align-action="multistage-run"]')}
              >
                Coarse + ICP
              </button>
              <button
                type="button"
                data-ribbon-focus="symmetry-plane"
                onClick={() => focusElement('[data-align-action="symmetry-detect"]')}
              >
                Symmetry Plane
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
              {INSPECT_TOOLS.map(([toolId, label]) => (
                <button
                  key={toolId}
                  type="button"
                  aria-pressed={inspectTool === toolId}
                  onClick={() => toggleInspectTool(toolId)}
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
              <button type="button" data-ribbon-focus="maps" onClick={() => focusElement("[data-inspect-maps]")}>
                Maps
              </button>
              <button type="button" data-ribbon-focus="geodesic" onClick={() => focusElement("[data-inspect-geodesic]")}>
                Geodesic
              </button>
              <button type="button" data-ribbon-focus="polylines" onClick={() => focusElement("[data-panel='polylines']")}>
                Polylines
              </button>
              <button
                type="button"
                aria-pressed={shadingMode === "backfaceWarning"}
                onClick={() => toggleShading("backfaceWarning")}
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
  const {
    sculpt,
    maskPaint,
    pinToActive,
    liveRemesh,
    remesh,
    dyntopo,
    dyntopoMode,
    dyntopoDetail,
    detailLabel,
    symmetry,
    shadingMode,
    targetItem,
    applyDyntopo,
    disarm,
    exitToSelect,
  } = useSculptInspector(app, selection);

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
          <output>{detailLabel}</output>
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
          onClick={applyDyntopo}
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
          {SCULPT_SHADING_MODES.map((mode) => {
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
        <button type="button" onClick={disarm}>
          Disarm Tool
        </button>
        <button type="button" onClick={exitToSelect}>
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
