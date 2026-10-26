// The app shell: the mesh list on the left, the viewport in the centre, the
// Repair and Collision panels on the right and the
// status bar along the bottom. Files dropped anywhere on the page load.
import type { MeshHandle, Scene } from "fast-mesh-wasm";
import { useHistory, useSelection } from "fast-mesh-wasm/react";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useCallback, useEffect, useState } from "react";
import type { Loader } from "./load/loader";
import { CollisionPanel } from "./panels/CollisionPanel";
import { GroupsBar } from "./panels/GroupsBar";
import { HoverTip } from "./panels/HoverTip";
import { MeshList } from "./panels/MeshList";
import { RepairPanel } from "./panels/RepairPanel";
import { StatusBar } from "./panels/StatusBar";
import { type AppState, useApp } from "./scene/app-state";
import { selectMesh } from "./scene/selection";
import { SculptToolbox } from "./viewport/Toolbox";
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

/** A request to point the camera at a mesh; `n` makes a repeated request new. */
export interface FocusRequest {
  mesh: MeshHandle;
  n: number;
}

export function App(props: { engine: Engine; app: AppState; loader: Loader }) {
  const { engine, app, loader } = props;
  const items = useApp(app, (s) => s.items);
  const edgeSoftness = useApp(app, (s) => s.edgeSoftness);
  const wireframe = useApp(app, (s) => s.wireframe);
  const grouped = useApp(app, (s) => s.collisionMode === "group");
  const members = useApp(app, (s) => s.group.members);
  const [selection, select] = useSelection();
  const { canUndo, canRedo, undo, redo } = useHistory();
  const [mode, setMode] = useState<GizmoMode>("translate");
  const [focus, setFocus] = useState<FocusRequest | null>(null);
  const [dropping, setDropping] = useState(false);

  const focusOn = useCallback((mesh: MeshHandle) => setFocus((f) => ({ mesh, n: (f?.n ?? 0) + 1 })), []);
  const brushPick = useApp(app, (s) => s.brushPick);
  const picking = brushPick !== null && brushPick.point === null ? brushPick.mesh : null;
  const sculpt = useApp(app, (s) => s.sculpt);
  const sculptRemesh = useApp(app, (s) => s.sculptRemesh);
  const sculptLiveRemesh = useApp(app, (s) => s.sculptLiveRemesh);
  const onPickPoint = useCallback(
    (mesh: MeshHandle, point: readonly [number, number, number]) =>
      app.setBrushPick({ mesh, point: [point[0], point[1], point[2]] }),
    [app],
  );

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

  // ---- Keyboard: undo/redo, gizmo mode, focus, deselect ----
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Typing in a field (the opacity slider, say) is not a shortcut.
      const target = e.target as HTMLElement | null;
      if (target !== null && (target.isContentEditable || /^(INPUT|SELECT|TEXTAREA)$/.test(target.tagName))) {
        return;
      }
      const key = e.key.toLowerCase();
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
      } else if (key === "escape") {
        // Sculpt first (it owns the pointer), then a pick (it never selects), then the selection.
        const snap = app.getSnapshot();
        if (snap.sculpt !== null) {
          app.setSculpt(null);
        } else if (snap.brushPick !== null) {
          app.setBrushPick(null);
        } else {
          selectMesh(app, select, selection, null, false);
        }
      } else if (key === "t") {
        setMode("translate");
      } else if (key === "r") {
        setMode("rotate");
      } else if (key === "f" && selection !== null) {
        focusOn(selection);
      } else if (key === "w") {
        app.setWireframe(!app.getSnapshot().wireframe);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [undo, redo, select, selection, focusOn, app]);

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

  return (
    <div className="app">
      <aside className="sidebar sidebar-left">
        <MeshList app={app} onLoad={load} onFocus={focusOn} />
        <GroupsBar app={app} />
      </aside>
      <main className="viewport">
        <Viewport
          app={app}
          items={items}
          mode={mode}
          focus={focus}
          edgeSoftness={edgeSoftness}
          hoverable={grouped}
          scene={engine.scene}
          picking={picking}
          onPickPoint={onPickPoint}
          sculpt={sculpt}
          sculptRemesh={sculptRemesh}
          sculptLiveRemesh={sculptLiveRemesh}
          wireframe={wireframe}
        />
        <SculptToolbox app={app} />
        {grouped ? <HoverTip scene={engine.scene} items={items} members={members} /> : null}
        <div className="toolbar" role="toolbar" aria-label="Viewport tools">
          <div className="group">
            <button type="button" aria-pressed={mode === "translate"} onClick={() => setMode("translate")} title="Move (T)">
              Move <kbd>T</kbd>
            </button>
            <button type="button" aria-pressed={mode === "rotate"} onClick={() => setMode("rotate")} title="Rotate (R)">
              Rotate <kbd>R</kbd>
            </button>
          </div>
          <div className="group">
            <button type="button" data-action="undo" disabled={!canUndo} onClick={undo} title="Undo (Ctrl+Z)">
              Undo
            </button>
            <button type="button" data-action="redo" disabled={!canRedo} onClick={redo} title="Redo (Ctrl+Shift+Z)">
              Redo
            </button>
          </div>
          <div className="group">
            <button
              type="button"
              data-action="wire"
              aria-pressed={wireframe}
              onClick={() => app.setWireframe(!wireframe)}
              title="Draw every mesh wireframed (W)"
            >
              Wire
            </button>
          </div>
        </div>
        <p className="hint">
          {sculpt !== null
            ? "Hover shows the brush ball · press-and-hold to sculpt · drag to grab · Esc disarms"
            : picking !== null
              ? "Click the mesh to set the brush center · Esc cancels"
              : selection === null
                ? "Click a mesh to select it · drag to orbit · scroll to zoom · drop files to load"
                : "Drag the gizmo to move · T/R move or rotate · F focus · Esc deselect · drop files to load"}
        </p>
        {items.length === 0 ? <p className="empty">Drop STL, OBJ or PLY files here</p> : null}
      </main>
      <aside className="sidebar sidebar-right">
        <RepairPanel app={app} scene={engine.scene} />
        <CollisionPanel app={app} scene={engine.scene} />
      </aside>
      <StatusBar engine={engine} app={app} />
      {dropping ? <div className="drop-overlay">Drop STL, OBJ or PLY files to load them</div> : null}
    </div>
  );
}
