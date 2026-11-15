// The app shell: the mesh list on the left, the viewport in the centre, the
// Repair and Collision panels on the right and the
// status bar along the bottom. Files dropped anywhere on the page load.
import type { MeshHandle, Scene } from "fast-mesh-wasm";
import { useHistory, useSelection } from "fast-mesh-wasm/react";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import {
  Suspense,
  lazy,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type RefObject,
} from "react";
import { parseBigFileParams } from "./load/big-file";
import type { Loader } from "./load/loader";
import { CollisionPanel } from "./panels/CollisionPanel";
import { CommandPalette } from "./panels/CommandPalette";
import { ExportPanel } from "./panels/ExportPanel";
import { GroupsBar } from "./panels/GroupsBar";
import { HelpOverlay, TourCard } from "./panels/Help";
import { HoverTip } from "./panels/HoverTip";
import { AlignPanel } from "./panels/AlignPanel";
import { InspectPanel } from "./panels/InspectPanel";
import { MeshList } from "./panels/MeshList";
import { ModelingPanel } from "./panels/ModelingPanel";
import { ProjectPanel } from "./panels/ProjectPanel";
import { RepairPanel } from "./panels/RepairPanel";
import {
  HomeOverviewCard,
  InspectorHeader,
  SculptInspectorCard,
  tabMatchesSlot,
  VolumeFallbackCard,
  WorkbenchRibbon,
} from "./panels/Ribbon";
import { TransformPanel } from "./panels/TransformPanel";
import { DEFAULT_SIDEBAR_WIDTH } from "./panels/settings";
import { StatusBar } from "./panels/StatusBar";
import { markTourSeen, resolveTourStart, tourBack, tourNext, tourSeen } from "./panels/tour";
import { type AppState, useApp, type WorkbenchTab } from "./scene/app-state";
import { ViewportNavigationHud } from "./viewport/ViewCube";
import { selectMesh } from "./scene/selection";
import { afterStroke } from "./viewport/view";
import { VolumeOverlay } from "./viewport/VolumeOverlay";
import { type GizmoMode, Viewport } from "./viewport/Viewport";

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
const VolumePanel = lazy(() => import("./panels/VolumePanel").then((m) => ({ default: m.VolumePanel })));

/** A request to point the camera at a mesh; `n` makes a repeated request new. */
export interface FocusRequest {
  mesh: MeshHandle;
  n: number;
  /** Look from this direction (default: the camera's current one). */
  dir?: readonly [number, number, number];
  /** Frame this world box instead of the mesh (the boot landing frames the volume). */
  box?: { center: readonly [number, number, number]; radius: number };
}

/** A finger is the main pointer here: the hints speak of tapping and pinching. */
const coarsePointer = typeof matchMedia === "function" && matchMedia("(pointer: coarse)").matches;

export function App(props: { engine: Engine; app: AppState; loader: Loader }) {
  const { engine, app, loader } = props;
  const items = useApp(app, (s) => s.items);
  const edgeSoftness = useApp(app, (s) => s.edgeSoftness);
  const wireframe = useApp(app, (s) => s.wireframe);
  const grouped = useApp(app, (s) => s.collisionMode === "group");
  const members = useApp(app, (s) => s.group.members);
  const [selection, select] = useSelection();
  const { canUndo, canRedo, undo: undoNow, redo: redoNow } = useHistory();
  const undo = useCallback(() => afterStroke(undoNow), [undoNow]);
  const redo = useCallback(() => afterStroke(redoNow), [redoNow]);
  const [mode, setMode] = useState<GizmoMode>("translate");
  const [focus, setFocus] = useState<FocusRequest | null>(null);
  const [resetView, setResetView] = useState(0);
  const [dropping, setDropping] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [tourStep, setTourStep] = useState<number | null>(null);
  const activeTab = useApp(app, (s) => s.activeTab);
  const ortho = useApp(app, (s) => s.cameraProjection === "orthographic");
  const units = useApp(app, (s) => s.units);
  const inspectTool = useApp(app, (s) => s.inspectTool);
  const collisionMode = useApp(app, (s) => s.collisionMode);
  const selectedItem = selection === null ? null : (items.find((i) => i.mesh === selection) ?? null);
  const selectedIsVolume = selectedItem !== null && selectedItem.volumeId !== null;
  const hasVolume = items.some((i) => i.volumeId !== null);
  // The persisted display settings: the theme (as `data-theme` on the
  // root, for the light overrides) and the sidebars (shown, widths).
  const theme = useApp(app, (s) => s.theme);
  const leftVisible = useApp(app, (s) => s.leftVisible);
  const rightVisible = useApp(app, (s) => s.rightVisible);
  const leftWidth = useApp(app, (s) => s.leftWidth);
  const rightWidth = useApp(app, (s) => s.rightWidth);
  const shellRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  // Ending the tour records it, so first-run only runs once.
  const endTour = useCallback(() => {
    markTourSeen();
    setTourStep(null);
  }, []);
  const nextTour = useCallback(() => {
    if (tourStep === null) {
      return;
    }
    const next = tourNext(tourStep);
    if (next === null) {
      markTourSeen();
    }
    setTourStep(next);
  }, [tourStep]);
  const backTour = useCallback(() => {
    setTourStep((s) => (s === null ? s : tourBack(s)));
  }, []);

  const focusOn = useCallback(
    (
      mesh: MeshHandle,
      dir?: readonly [number, number, number],
      box?: { center: readonly [number, number, number]; radius: number },
    ) => setFocus((f) => ({ mesh, n: (f?.n ?? 0) + 1, dir, box })),
    [],
  );
  const brushPick = useApp(app, (s) => s.brushPick);
  const picking = brushPick !== null && brushPick.point === null ? brushPick.mesh : null;
  const sculpt = useApp(app, (s) => s.sculpt);
  const sculptRemesh = useApp(app, (s) => s.sculptRemesh);
  const sculptLiveRemesh = useApp(app, (s) => s.sculptLiveRemesh);
  const maskPaint = useApp(app, (s) => s.maskPaint);
  const onPickPoint = useCallback(
    (mesh: MeshHandle, point: readonly [number, number, number]) =>
      app.setBrushPick({ mesh, point: [point[0], point[1], point[2]] }),
    [app],
  );

  // `?big` loads the ~1M-triangle demo variants (see load/big-file.ts).
  const demoBig = useMemo(() => {
    try {
      return parseBigFileParams(location.search).demoBig;
    } catch {
      return false;
    }
  }, []);
  const cancelLoad = useCallback((id: number) => loader.cancelJob(id), [loader]);

  const load = useCallback(
    (files: Iterable<File>) => {
      void loader.loadFiles(files).then((added) => {
        // Point the camera at what was just loaded: a file can be anywhere, at any scale.
        const last = added.at(-1);
        if (last !== undefined) {
          focusOn(last.mesh);
        }
      });
    },
    [loader, focusOn],
  );

  // ---- Boot: the CT head, selected and framed ----
  // Once per page (StrictMode runs effects twice in development): the
  // medical landing view (the sample suggests its Vessels preset).
  // Everything else loads from its button.
  const booted = useRef(false);
  useEffect(() => {
    if (booted.current) {
      return;
    }
    booted.current = true;
    void loader.loadSample("headsq").then((added) => {
      const last = added.at(-1);
      if (last === undefined || last.volumeId === null) {
        return;
      }
      selectMesh(app, select, null, last.mesh, false);
      // A three-quarter view of the volume box (the quads span it, past the
      // mesh): the cutaway and its slice quads read best at an angle.
      loader
        .fetchVolume(last.volumeId)
        .then((payload) => {
          const ex = payload.columns * payload.spacing[0];
          const ey = payload.rows * payload.spacing[1];
          const ez = payload.slices * payload.spacing[2];
          const m = engine.scene.matrix4(last.mesh);
          const lx = payload.origin[0] + ex / 2 - last.center[0];
          const ly = payload.origin[1] + ey / 2 - last.center[1];
          const lz = payload.origin[2] + ez / 2 - last.center[2];
          focusOn(
            last.mesh,
            [1, 0.55, 1.15],
            {
              center: [
                m[0] * lx + m[4] * ly + m[8] * lz + m[12],
                m[1] * lx + m[5] * ly + m[9] * lz + m[13],
                m[2] * lx + m[6] * ly + m[10] * lz + m[14],
              ],
              radius: Math.hypot(ex, ey, ez) / 2,
            },
          );
        })
        .catch(() => {
          focusOn(last.mesh);
        });
    });
  }, [app, engine, loader, focusOn, select]);

  // ---- First run: the tour, unless seen, forced off, or headless ----
  // `?tour=1` forces it on (even headless); `?tour=0` / `?notour` and
  // headless Chrome (`showcase-check.mjs`) keep it off, so the scripted
  // session never sees a card over the canvas.
  const onboarded = useRef(false);
  useEffect(() => {
    if (onboarded.current) {
      return;
    }
    onboarded.current = true;
    const start = resolveTourStart(location.search, navigator.userAgent, navigator.webdriver, tourSeen());
    if (start !== "off") {
      setTourStep(0);
    }
  }, []);

  const prevSelectedVolumeRef = useRef<boolean>(false);
  const prevSculptActiveRef = useRef<boolean>(false);
  const prevInspectToolRef = useRef(inspectTool);
  const prevCollisionModeRef = useRef(collisionMode);

  // Contextual workspace auto-switching (#228) on meaningful trigger transitions.
  useEffect(() => {
    const wasVol = prevSelectedVolumeRef.current;
    prevSelectedVolumeRef.current = selectedIsVolume;
    if (!wasVol && selectedIsVolume) {
      app.setActiveTab("volume");
    } else if (wasVol && !selectedIsVolume && app.getSnapshot().activeTab === "volume") {
      app.setActiveTab(selection === null ? "home" : "select");
    }
  }, [app, selectedIsVolume, selection]);

  useEffect(() => {
    const nowActive = sculpt !== null || maskPaint;
    const wasActive = prevSculptActiveRef.current;
    prevSculptActiveRef.current = nowActive;
    if (!wasActive && nowActive) {
      app.setActiveTab("sculpt");
    }
  }, [app, sculpt, maskPaint]);

  useEffect(() => {
    const was = prevInspectToolRef.current;
    prevInspectToolRef.current = inspectTool;
    if (was === null && inspectTool !== null) {
      app.setActiveTab("inspect");
    }
  }, [app, inspectTool]);

  useEffect(() => {
    const was = prevCollisionModeRef.current;
    prevCollisionModeRef.current = collisionMode;
    if (was !== collisionMode) {
      app.setActiveTab("align");
    }
  }, [app, collisionMode]);

  // ---- Keyboard: undo/redo, command palette, gizmo mode, focus, deselect, help ----
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      if ((e.ctrlKey || e.metaKey) && key === "k") {
        e.preventDefault();
        app.setCommandPaletteOpen(!app.getSnapshot().commandPaletteOpen);
        return;
      }
      // Typing in a field (the opacity slider, say) is not a shortcut.
      const target = e.target as HTMLElement | null;
      if (target !== null && (target.isContentEditable || /^(INPUT|SELECT|TEXTAREA)$/.test(target.tagName))) {
        return;
      }
      if ((e.ctrlKey || e.metaKey) && key === "z") {
        e.preventDefault();
        if (e.shiftKey) {
          redo();
        } else {
          undo();
        }
      } else if ((e.ctrlKey || e.metaKey) && key === "y") {
        e.preventDefault();
        redo();
      } else if (e.ctrlKey || e.metaKey || e.altKey) {
        // Other modified keys are the browser's.
      } else if (key === "/") {
        e.preventDefault();
        app.setCommandPaletteOpen(true);
      } else if (key === "?") {
        setHelpOpen((v) => !v);
      } else if (key === "escape") {
        const snap = app.getSnapshot();
        if (snap.commandPaletteOpen) {
          app.setCommandPaletteOpen(false);
        } else if (helpOpen) {
          setHelpOpen(false);
        } else if (tourStep !== null) {
          endTour();
        } else {
          // Sculpt first (it owns the pointer), then the mask paint, then a
          // pick (it never selects), then the selection.
          if (snap.sculpt !== null) {
            app.setSculpt(null);
          } else if (snap.maskPaint) {
            app.setMaskPaint(false);
          } else if (snap.brushPick !== null) {
            app.setBrushPick(null);
          } else {
            selectMesh(app, select, selection, null, false);
          }
        }
      } else if (key === "t") {
        setMode("translate");
      } else if (key === "r") {
        setMode("rotate");
      } else if (key === "s") {
        setMode("scale");
      } else if (key === "f" && selection !== null) {
        focusOn(selection);
      } else if (key === "w") {
        app.setWireframe(!app.getSnapshot().wireframe);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [undo, redo, select, selection, focusOn, app, helpOpen, tourStep, endTour]);

  // ---- Drop files anywhere ----
  useEffect(() => {
    const hasFiles = (e: DragEvent) => e.dataTransfer?.types.includes("Files") === true;
    const onDragOver = (e: DragEvent) => {
      if (hasFiles(e)) {
        e.preventDefault();
        setDropping(true);
      }
    };
    const onDragLeave = (e: DragEvent) => {
      // Leaving the window, not moving between elements.
      if (e.relatedTarget === null) {
        setDropping(false);
      }
    };
    const onDrop = (e: DragEvent) => {
      if (!hasFiles(e)) {
        return;
      }
      e.preventDefault();
      setDropping(false);
      load(e.dataTransfer?.files ?? []);
    };
    window.addEventListener("dragover", onDragOver);
    window.addEventListener("dragleave", onDragLeave);
    window.addEventListener("drop", onDrop);
    return () => {
      window.removeEventListener("dragover", onDragOver);
      window.removeEventListener("dragleave", onDragLeave);
      window.removeEventListener("drop", onDrop);
    };
  }, [load]);

  // A hidden sidebar keeps its panels mounted (their effects, like the
  // volume view's linked MPR state, stay alive): it just takes no room.
  const shellStyle = {
    "--left-w": leftVisible ? `${leftWidth}px` : "0px",
    "--right-w": rightVisible ? `${rightWidth}px` : "0px",
  } as CSSProperties;
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
        onResetView={() => setResetView((n) => n + 1)}
        onFocusSelection={() => {
          if (selection !== null) focusOn(selection);
        }}
        onClearSelection={() => selectMesh(app, select, selection, null, false)}
        onUndo={undo}
        onRedo={redo}
        onOpenHelp={() => setHelpOpen(true)}
      />
      <aside className="sidebar sidebar-left" style={leftVisible ? undefined : { display: "none" }}>
        <SidebarHandle side="left" shell={shellRef} app={app} />
        <MeshList
          app={app}
          onLoad={load}
          onCancelLoad={cancelLoad}
          onLoadPhantom={(phantom) => {
            void loader.loadDicomDemo(phantom).then((added) => {
              const last = added.at(-1);
              if (last !== undefined) {
                focusOn(last.mesh);
              }
            });
          }}
          onLoadSample={(sample) => {
            void loader.loadSample(sample).then((added) => {
              const last = added.at(-1);
              if (last !== undefined) {
                focusOn(last.mesh);
              }
            });
          }}
          onLoadDemo={() => {
            // No focus: the meshes join the current view (Reset view restores the overview).
            void loader.loadDemo(demoBig);
          }}
          onFocus={focusOn}
        />
        <GroupsBar app={app} />
        <ProjectPanel app={app} />
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
        {grouped ? <HoverTip app={app} scene={engine.scene} items={items} members={members} /> : null}
        <VolumeOverlay app={app} />
        <ViewportNavigationHud app={app} />
        <p className="hint">
          {sculpt !== null
            ? "Hover any visible mesh to see brush · drag to sculpt · Esc disarms"
            : maskPaint
              ? "Hover any visible mesh · drag to paint mask · Esc disarms"
            : picking !== null
              ? "Click the mesh to set the brush center · Esc cancels"
              : selection === null
                ? coarsePointer
                  ? "Tap a mesh to select it · drag to rotate · pinch to zoom · two fingers to pan"
                  : "Click a mesh to select it · drag to rotate · scroll to zoom · Ctrl+K search · drop files to load"
                : coarsePointer
                  ? "Drag the gizmo to move · tap empty space to deselect"
                  : "Drag the gizmo to move · T/R/S move/rotate/scale · F focus · Ctrl+K search · Esc deselect"}
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
          <HomeOverviewCard app={app} itemCount={items.length} onResetView={() => setResetView((n) => n + 1)} />
        </div>
        <div
          className={tabMatchesSlot(activeTab, "transform") ? "inspector-slot" : "inspector-slot inspector-hidden"}
          data-inspector-slot="transform"
          onClickCapture={() => activateTabForSlot("select")}
        >
          <TransformPanel app={app} mode={mode} onMode={setMode} />
        </div>
        <div
          className={tabMatchesSlot(activeTab, "sculpt") ? "inspector-slot" : "inspector-slot inspector-hidden"}
          data-inspector-slot="sculpt"
          onClickCapture={() => activateTabForSlot("sculpt")}
        >
          <SculptInspectorCard app={app} selection={selection} />
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
        onResetView={() => setResetView((n) => n + 1)}
        onUndo={undo}
        onRedo={redo}
        onOpenHelp={() => setHelpOpen(true)}
      />
      {helpOpen ? (
        <HelpOverlay
          onClose={() => setHelpOpen(false)}
          onStartTour={() => {
            setHelpOpen(false);
            setTourStep(0);
          }}
        />
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
  const setWidth = side === "left" ? (w: number) => app.setLeftWidth(w) : (w: number) => app.setRightWidth(w);
  return (
    <div
      className={`resize-handle resize-${side}`}
      role="separator"
      aria-orientation="vertical"
      aria-label={side === "left" ? "Resize the left panel" : "Resize the right panel"}
      title="Drag to resize · double-click to reset"
      onDoubleClick={() => setWidth(DEFAULT_SIDEBAR_WIDTH)}
      onPointerDown={(e) => e.currentTarget.setPointerCapture(e.pointerId)}
      onPointerMove={(e) => {
        if (e.buttons !== 1 || !e.isPrimary) {
          return;
        }
        const rect = shell.current?.getBoundingClientRect();
        if (rect === undefined) {
          return;
        }
        setWidth(side === "left" ? e.clientX - rect.left : rect.right - e.clientX);
      }}
    />
  );
}
