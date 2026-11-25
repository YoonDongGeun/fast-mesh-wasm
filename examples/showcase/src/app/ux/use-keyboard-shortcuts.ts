// The shell's keyboard shortcuts, on the window: Ctrl+K (Cmd+K) toggles
// command search anywhere; outside text fields Ctrl+Z / Ctrl+Shift+Z /
// Ctrl+Y undo and redo, `/` opens command search, `?` toggles help, T/R/S
// pick the gizmo mode, F frames the selection, W wireframes every mesh, 1 / 3 / 7
// (Shift: the opposite face) turn the view to a face preset, and Esc closes or
// disarms the topmost thing. Help's SHORTCUTS lists them.
import type { MeshHandle } from "fast-mesh-wasm";
import { type Dispatch, type SetStateAction, useEffect } from "react";
import { selectMesh } from "../../features/select/ux/selection";
import { cameraPresetForKey } from "../../features/view/ux/navigation-hud";
import type { GizmoMode } from "../../viewport/ui/Viewport";
import type { AppState } from "../state/app-state";

export function useKeyboardShortcuts(opts: {
  app: AppState;
  select: (mesh: MeshHandle | null) => void;
  selection: MeshHandle | null;
  undo: () => void;
  redo: () => void;
  focusOn: (mesh: MeshHandle) => void;
  setMode: Dispatch<SetStateAction<GizmoMode>>;
  helpOpen: boolean;
  setHelpOpen: Dispatch<SetStateAction<boolean>>;
  tourStep: number | null;
  endTour: () => void;
}): void {
  const { app, select, selection, undo, redo, focusOn, setMode, helpOpen, setHelpOpen, tourStep, endTour } = opts;
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
      } else if (cameraPresetForKey(e.code, e.shiftKey) !== null) {
        app.requestCameraPreset(cameraPresetForKey(e.code, e.shiftKey)!);
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
          // Sculpt first (it owns the pointer), then mask paint, then active Wave 6 tool,
          // then brush pick (never selects), then mesh selection.
          const hasWave6Tool =
            snap.wave6.select.activeTool !== null ||
            snap.wave6.deform.activeTool !== null ||
            snap.wave6.printPrep.activeTool !== null;
          if (snap.sculpt !== null) {
            app.setSculpt(null);
          } else if (snap.maskPaint) {
            app.setMaskPaint(false);
          } else if (hasWave6Tool) {
            app.disarmWave6Tools();
          } else if (snap.brushPick !== null) {
            app.setBrushPick(null);
          } else if (snap.placementTool.pick !== null) {
            app.setPlacementTool({ pick: null });
          } else {
            selectMesh(app, select, selection, null, false);
          }
        }
      } else if (key === "t") {
        app.disarmWave6Tools();
        setMode("translate");
      } else if (key === "r") {
        app.disarmWave6Tools();
        setMode("rotate");
      } else if (key === "s") {
        app.disarmWave6Tools();
        setMode("scale");
      } else if (key === "f" && selection !== null) {
        focusOn(selection);
      } else if (key === "w") {
        app.setWireframe(!app.getSnapshot().wireframe);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [undo, redo, select, selection, focusOn, app, helpOpen, tourStep, endTour, setMode, setHelpOpen]);
}
