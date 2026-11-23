// The app shell: the mesh list on the left, the viewport in the centre, the
// Repair and Collision panels on the right and the
// status bar along the bottom. Files dropped anywhere on the page load.
// What it does (loading, shortcuts, the drop target, the sidebars, the
// tab auto-switching) lives in `ux/`; this file lays the shell out.
import type { MeshHandle, Scene } from "fast-mesh-wasm";
import { useScene, useSelection } from "fast-mesh-wasm/react";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { Suspense, lazy, useCallback, useState, type RefObject } from "react";
import type { Loader } from "../io/loader";
import { BuildPlatePanel } from "../features/build-plate/ui/BuildPlatePanel";
import { CollisionPanel } from "../features/collision/ui/CollisionPanel";
import { CommandPalette } from "../shell/ui/CommandPalette";
import { CreatePrimitiveDialog } from "../features/home/ui/CreatePrimitiveDialog";
import { TextDialog } from "../features/mesh-edit/ui/TextDialog";
import { DeformPanel } from "../features/sculpt/ui/DeformPanel";
import { ExportPanel } from "../features/home/ui/ExportPanel";
import { GroupsBar } from "../features/collision/ui/GroupsBar";
import { HelpOverlay, TourCard } from "../shell/ui/Help";
import { HoverTip } from "../features/collision/ui/HoverTip";
import { AlignPanel } from "../features/transform/ui/AlignPanel";
import { InspectPanel } from "../features/inspect/ui/InspectPanel";
import { MeshList } from "../features/home/ui/MeshList";
import { ObjectInfoBox } from "../features/inspect/ui/ObjectInfoBox";
import { ModelingPanel } from "../features/mesh-edit/ui/ModelingPanel";
import { PointsPanel } from "../features/points/ui/PointsPanel";
import { PolylinePanel } from "../features/polylines/ui/PolylinePanel";
import { ProjectPanel } from "../features/home/ui/ProjectPanel";
import { RepairPanel } from "../features/mesh-repair/ui/RepairPanel";
import {
  HomeOverviewCard,
  InspectorHeader,
  SculptInspectorCard,
  VolumeFallbackCard,
  WorkbenchRibbon,
} from "../shell/ui/Ribbon";
import { SelectPanel } from "../features/select/ui/SelectPanel";
import { TransformPanel } from "../features/transform/ui/TransformPanel";
import { StatusBar } from "../shell/ui/StatusBar";
import { useHelpAndTour } from "../shell/ux/use-help-tour";
import { tabMatchesSlot } from "../shell/ux/workbench-tabs";
import { type AppState, useApp, type WorkbenchTab } from "./state/app-state";
import { useCameraRequests } from "./ux/use-camera-requests";
import { sidebarResizeHandlers, useDisplaySettings } from "./ux/use-display-settings";
import { useFileDrop } from "./ux/use-file-drop";
import { useKeyboardShortcuts } from "./ux/use-keyboard-shortcuts";
import { useSceneLoading } from "./ux/use-scene-loading";
import { useUndoRedo } from "./ux/use-undo-redo";
import { useWorkspaceAutoSwitch } from "./ux/use-workspace-auto-switch";
import { coarsePointer, viewportHint } from "./ux/viewport-hint";
import { FaceSelectionMarquee } from "../features/select/ui/FaceSelectionLayer";
import { ViewportNavigationHud } from "../features/view/ui/ViewCube";
import { selectMesh } from "../features/select/ux/selection";
import { VolumeOverlay } from "../features/ct/ui/VolumeOverlay";
import { type GizmoMode, Viewport } from "../viewport/ui/Viewport";

/** The engine behind the store, for the status bar. */
export interface Engine {
  mode: "worker" | "local";
  /** The store's scene, for the queries the store does not wrap (analyze, distances, contacts). */
  scene: Scene | WorkerScene;
  /** The engine's threads (as of the Worker's latest start). */
  threads(): number;
  /** Whether the page is cross-origin isolated (needed for threads). */
  isolated: boolean;
}

/** The volume view loads on demand: vtk.js stays out of the main chunk. */
const VolumePanel = lazy(() => import("../features/ct/ui/VolumePanel").then((m) => ({ default: m.VolumePanel })));

/** A request to point the camera at a mesh; `n` makes a repeated request new. */
export interface FocusRequest {
  /** `null` with a `box` that is no mesh's. */
  mesh: MeshHandle | null;
  n: number;
  /** Look from this direction (default: the camera's current one). */
  dir?: readonly [number, number, number];
  /** Frame this world box instead of the mesh (the boot landing frames the volume). */
  box?: { center: readonly [number, number, number]; radius: number };
}

export function App(props: { engine: Engine; app: AppState; loader: Loader }) {
  const { engine, app, loader } = props;
  const store = useScene();
  const items = useApp(app, (s) => s.items);
  const edgeSoftness = useApp(app, (s) => s.edgeSoftness);
  const wireframe = useApp(app, (s) => s.wireframe);
  const grouped = useApp(app, (s) => s.collisionMode === "group");
  const members = useApp(app, (s) => s.group.members);
  const [selection, select] = useSelection();
  const { canUndo, canRedo, undo, redo } = useUndoRedo();
  const [mode, setMode] = useState<GizmoMode>("translate");
  const { focus, focusOn, resetView, requestResetView } = useCameraRequests();
  const activeTab = useApp(app, (s) => s.activeTab);
  const ortho = useApp(app, (s) => s.cameraProjection === "orthographic");
  const units = useApp(app, (s) => s.units);
  const selectedItem = selection === null ? null : (items.find((i) => i.mesh === selection) ?? null);
  const selectedIsVolume = selectedItem !== null && selectedItem.volumeId !== null;
  const hasVolume = items.some((i) => i.volumeId !== null);
  const { shellRef, shellStyle, leftVisible, rightVisible } = useDisplaySettings(app);

  const brushPick = useApp(app, (s) => s.brushPick);
  const picking = brushPick !== null && brushPick.point === null ? brushPick.mesh : null;
  const sculpt = useApp(app, (s) => s.sculpt);
  const sculptRemesh = useApp(app, (s) => s.sculptRemesh);
  const sculptLiveRemesh = useApp(app, (s) => s.sculptLiveRemesh);
  const maskPaint = useApp(app, (s) => s.maskPaint);
  const wave6SelectTool = useApp(app, (s) => s.wave6.select.activeTool);
  const wave6DeformTool = useApp(app, (s) => s.wave6.deform.activeTool);
  const wave6PrintTool = useApp(app, (s) => s.wave6.printPrep.activeTool);
  const placementPick = useApp(app, (s) => s.placementTool.pick);
  const landmarkPick = useApp(app, (s) => s.alignTool.landmarks?.picking === true);
  const surfacePick = useApp(app, (s) => s.surfacePath.armed);
  const onPickPoint = useCallback(
    (mesh: MeshHandle, point: readonly [number, number, number]) =>
      app.setBrushPick({ mesh, point: [point[0], point[1], point[2]] }),
    [app],
  );

  const { load, cancelLoad, loadPhantom, loadSample, loadDemo } = useSceneLoading({
    app,
    scene: engine.scene,
    loader,
    select,
    focusOn,
  });
  const { helpOpen, setHelpOpen, openHelp, closeHelp, tourStep, nextTour, backTour, endTour, startTourFromHelp } =
    useHelpAndTour();
  useWorkspaceAutoSwitch(app, { selection, selectedIsVolume, sculpt, maskPaint });
  useKeyboardShortcuts({
    app,
    select,
    selection,
    undo,
    redo,
    focusOn,
    setMode,
    helpOpen,
    setHelpOpen,
    tourStep,
    endTour,
  });
  const dropping = useFileDrop(load);

  const activateTabForSlot = (targetTab: WorkbenchTab) => {
    app.setActiveTab(targetTab);
  };
  return (
    <div ref={shellRef} className="app" style={shellStyle}>
      <WorkbenchRibbon
        app={app}
        mode={mode}
        canUndo={canUndo}
        canRedo={canRedo}
        wireframe={wireframe}
        selection={selection}
        hasItems={items.length > 0}
        hasVolume={hasVolume}
        onSetMode={setMode}
        onResetView={requestResetView}
        onFocusSelection={() => {
          if (selection !== null) focusOn(selection);
        }}
        onClearSelection={() => selectMesh(app, select, selection, null, false)}
        onUndo={undo}
        onRedo={redo}
        onOpenHelp={openHelp}
      />
      <aside className="sidebar sidebar-left" style={leftVisible ? undefined : { display: "none" }}>
        <SidebarHandle side="left" shell={shellRef} app={app} />
        <MeshList
          app={app}
          onLoad={load}
          onCancelLoad={cancelLoad}
          onLoadPhantom={loadPhantom}
          onLoadSample={loadSample}
          onLoadDemo={loadDemo}
          onFocus={focusOn}
          onFocusBox={(box) => focusOn(null, undefined, box)}
        />
        <GroupsBar app={app} />
        <ProjectPanel app={app} />
        <ObjectInfoBox app={app} scene={engine.scene} />
      </aside>
      <main className="viewport">
        <Viewport
          app={app}
          items={items}
          mode={mode}
          focus={focus}
          resetView={resetView}
          edgeSoftness={edgeSoftness}
          hoverable={grouped}
          scene={engine.scene}
          picking={picking}
          onPickPoint={onPickPoint}
          sculpt={sculpt}
          sculptRemesh={sculptRemesh}
          sculptLiveRemesh={sculptLiveRemesh}
          maskPaint={maskPaint}
          wireframe={wireframe}
          loader={loader}
        />
        <FaceSelectionMarquee />
        {grouped ? <HoverTip app={app} scene={engine.scene} items={items} members={members} /> : null}
        <VolumeOverlay app={app} />
        <ViewportNavigationHud app={app} />
        <CreatePrimitiveDialog app={app} />
        <TextDialog app={app} />
        <p className="hint">
          {viewportHint({
            sculpting: sculpt !== null,
            maskPaint,
            selectTool: wave6SelectTool,
            deformTool: wave6DeformTool,
            printTool: wave6PrintTool,
            picking: picking !== null,
            placementPick: placementPick !== null,
            landmarkPick,
            surfacePick,
            selected: selection !== null,
            coarse: coarsePointer,
          })}
        </p>
        {items.length === 0 ? <p className="empty">Drop STL, OBJ, PLY or DICOM series here</p> : null}
      </main>
      <aside className="sidebar sidebar-right" style={rightVisible ? undefined : { display: "none" }}>
        <SidebarHandle side="right" shell={shellRef} app={app} />
        <InspectorHeader app={app} activeTab={activeTab} />
        <div
          className={tabMatchesSlot(activeTab, "home") ? "inspector-slot" : "inspector-slot inspector-hidden"}
          data-inspector-slot="home"
        >
          <HomeOverviewCard app={app} itemCount={items.length} onResetView={requestResetView} />
        </div>
        <div
          className={tabMatchesSlot(activeTab, "buildPlate") ? "inspector-slot" : "inspector-slot inspector-hidden"}
          data-inspector-slot="build-plate"
          onClickCapture={() => activateTabForSlot("home")}
        >
          <BuildPlatePanel app={app} />
        </div>
        <div
          className={tabMatchesSlot(activeTab, "transform") ? "inspector-slot" : "inspector-slot inspector-hidden"}
          data-inspector-slot="transform"
          onClickCapture={() => activateTabForSlot("select")}
        >
          <TransformPanel app={app} mode={mode} onMode={setMode} />
        </div>
        <div
          className={tabMatchesSlot(activeTab, "select") ? "inspector-slot" : "inspector-slot inspector-hidden"}
          data-inspector-slot="select"
          onClickCapture={() => activateTabForSlot("select")}
        >
          <SelectPanel app={app} store={store} />
        </div>
        <div
          className={tabMatchesSlot(activeTab, "sculpt") ? "inspector-slot" : "inspector-slot inspector-hidden"}
          data-inspector-slot="sculpt"
          onClickCapture={() => activateTabForSlot("sculpt")}
        >
          <SculptInspectorCard app={app} selection={selection} />
        </div>
        <div
          className={tabMatchesSlot(activeTab, "deform") ? "inspector-slot" : "inspector-slot inspector-hidden"}
          data-inspector-slot="deform"
          onClickCapture={() => activateTabForSlot("sculpt")}
        >
          <DeformPanel app={app} store={store} />
        </div>
        <div
          className={tabMatchesSlot(activeTab, "modeling") ? "inspector-slot" : "inspector-slot inspector-hidden"}
          data-inspector-slot="modeling"
          onClickCapture={() => activateTabForSlot("edit")}
        >
          <ModelingPanel app={app} />
        </div>
        <div
          className={tabMatchesSlot(activeTab, "repair") ? "inspector-slot" : "inspector-slot inspector-hidden"}
          data-inspector-slot="repair"
          onClickCapture={() => activateTabForSlot("repair")}
        >
          <RepairPanel app={app} scene={engine.scene} />
        </div>
        <div
          className={tabMatchesSlot(activeTab, "points") ? "inspector-slot" : "inspector-slot inspector-hidden"}
          data-inspector-slot="points"
          onClickCapture={() => activateTabForSlot("align")}
        >
          <PointsPanel app={app} store={store} />
        </div>
        <div
          className={tabMatchesSlot(activeTab, "align") ? "inspector-slot" : "inspector-slot inspector-hidden"}
          data-inspector-slot="align"
          onClickCapture={() => activateTabForSlot("align")}
        >
          <AlignPanel app={app} />
        </div>
        <div
          className={tabMatchesSlot(activeTab, "collision") ? "inspector-slot" : "inspector-slot inspector-hidden"}
          data-inspector-slot="collision"
          onClickCapture={() => activateTabForSlot("align")}
        >
          <CollisionPanel app={app} scene={engine.scene} />
        </div>
        <div
          className={tabMatchesSlot(activeTab, "inspect") ? "inspector-slot" : "inspector-slot inspector-hidden"}
          data-inspector-slot="inspect"
          onClickCapture={() => activateTabForSlot("inspect")}
        >
          <InspectPanel app={app} scene={engine.scene} />
        </div>
        <div
          className={tabMatchesSlot(activeTab, "polylines") ? "inspector-slot" : "inspector-slot inspector-hidden"}
          data-inspector-slot="polylines"
        >
          <PolylinePanel app={app} />
        </div>
        <div
          className={tabMatchesSlot(activeTab, "export") ? "inspector-slot" : "inspector-slot inspector-hidden"}
          data-inspector-slot="export"
          onClickCapture={() => activateTabForSlot("home")}
        >
          <ExportPanel app={app} scene={engine.scene} />
        </div>
        <div
          className={tabMatchesSlot(activeTab, "volume") ? "inspector-slot" : "inspector-slot inspector-hidden"}
          data-inspector-slot="volume"
          onClickCapture={() => activateTabForSlot("volume")}
        >
          {!selectedIsVolume ? <VolumeFallbackCard app={app} /> : null}
          <Suspense fallback={null}>
            <VolumePanel app={app} loader={loader} />
          </Suspense>
        </div>
      </aside>
      <StatusBar engine={engine} app={app} />
      <CommandPalette
        app={app}
        items={items}
        selection={selection}
        canUndo={canUndo}
        canRedo={canRedo}
        ortho={ortho}
        units={units}
        onSelectItem={(mesh) => {
          selectMesh(app, select, selection, mesh, false);
          focusOn(mesh);
        }}
        onSetMode={setMode}
        onResetView={requestResetView}
        onUndo={undo}
        onRedo={redo}
        onOpenHelp={openHelp}
      />
      {helpOpen ? (
        <HelpOverlay onClose={closeHelp} onStartTour={startTourFromHelp} />
      ) : null}
      {tourStep !== null ? (
        <TourCard index={tourStep} onNext={nextTour} onBack={backTour} onDone={endTour} />
      ) : null}
      {dropping ? <div className="drop-overlay">Drop STL, OBJ, PLY or DICOM files to load them</div> : null}
    </div>
  );
}

/** A sidebar's edge: drag to resize it (persisted), double-click for the default width. */
function SidebarHandle(props: { side: "left" | "right"; shell: RefObject<HTMLDivElement | null>; app: AppState }) {
  const { side, shell, app } = props;
  const resize = sidebarResizeHandlers(side, shell, app);
  return (
    <div
      className={`resize-handle resize-${side}`}
      role="separator"
      aria-orientation="vertical"
      aria-label={side === "left" ? "Resize the left panel" : "Resize the right panel"}
      title="Drag to resize · double-click to reset"
      onDoubleClick={resize.onDoubleClick}
      onPointerDown={resize.onPointerDown}
      onPointerMove={resize.onPointerMove}
    />
  );
}
