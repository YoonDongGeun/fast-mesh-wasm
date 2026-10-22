// The app shell: the mesh list on the left, the viewport in the centre, the
// analysis panels on the right (Repair and Collision come next) and the
// status bar along the bottom. Files dropped anywhere on the page load.
import type { MeshHandle } from "fast-mesh-wasm";
import { useHistory, useSelection } from "fast-mesh-wasm/react";
import { useCallback, useEffect, useState } from "react";
import type { Loader } from "./load/loader";
import { MeshList } from "./panels/MeshList";
import { StatusBar } from "./panels/StatusBar";
import { type AppState, useApp } from "./scene/app-state";
import { type GizmoMode, Viewport } from "./viewport/Viewport";

/** The engine behind the store, for the status bar. */
export interface Engine {
  mode: "worker" | "local";
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
  const [selection, select] = useSelection();
  const { canUndo, canRedo, undo, redo } = useHistory();
  const [mode, setMode] = useState<GizmoMode>("translate");
  const [focus, setFocus] = useState<FocusRequest | null>(null);
  const [dropping, setDropping] = useState(false);

  const focusOn = useCallback((mesh: MeshHandle) => setFocus((f) => ({ mesh, n: (f?.n ?? 0) + 1 })), []);

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
        select(null);
      } else if (key === "t") {
        setMode("translate");
      } else if (key === "r") {
        setMode("rotate");
      } else if (key === "f" && selection !== null) {
        focusOn(selection);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [undo, redo, select, selection, focusOn]);

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
      </aside>
      <main className="viewport">
        <Viewport items={items} mode={mode} focus={focus} />
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
        </div>
        <p className="hint">
          {selection === null
            ? "Click a mesh to select it · drag to orbit · scroll to zoom"
            : "Drag the gizmo to move · T/R move or rotate · F focus · Esc deselect"}
        </p>
        {items.length === 0 ? <p className="empty">Drop STL, OBJ or PLY files here</p> : null}
      </main>
      <aside className="sidebar sidebar-right">
        <section className="panel placeholder" aria-label="Repair">
          <h2>Repair</h2>
          <p className="muted">
            Analyze the selected mesh (holes, orientation, components, non-manifold and degenerate elements) and run
            undoable repairs.
          </p>
          <p className="muted small">Coming soon.</p>
        </section>
        <section className="panel placeholder" aria-label="Collision">
          <h2>Collision</h2>
          <p className="muted">
            Pick two meshes to see their contact heatmap: penetration depth in reds, contact in white, the gap in
            green to blue.
          </p>
          <p className="muted small">Coming soon.</p>
        </section>
      </aside>
      <StatusBar engine={engine} app={app} />
      {dropping ? <div className="drop-overlay">Drop STL, OBJ or PLY files to load them</div> : null}
    </div>
  );
}
