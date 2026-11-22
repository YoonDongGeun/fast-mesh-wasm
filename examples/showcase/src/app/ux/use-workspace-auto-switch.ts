// Contextual workspace auto-switching (#228): the ribbon follows what the
// user starts, on the transition only, so a tab picked by hand afterwards
// stays. Selecting a volume opens Voxel / CT, and leaving it while that tab
// shows returns to Select (Home with nothing selected); arming a sculpt
// brush or mask paint opens Sculpt; arming an inspect tool opens Inspect;
// a new collision mode opens Points & Align.
import type { MeshHandle } from "fast-mesh-wasm";
import { useEffect, useRef } from "react";
import { type AppSnapshot, type AppState, useApp } from "../state/app-state";

export function useWorkspaceAutoSwitch(
  app: AppState,
  trigger: {
    selection: MeshHandle | null;
    selectedIsVolume: boolean;
    sculpt: AppSnapshot["sculpt"];
    maskPaint: boolean;
  },
): void {
  const { selection, selectedIsVolume, sculpt, maskPaint } = trigger;
  const inspectTool = useApp(app, (s) => s.inspectTool);
  const collisionMode = useApp(app, (s) => s.collisionMode);
  const prevSelectedVolumeRef = useRef<boolean>(false);
  const prevSculptActiveRef = useRef<boolean>(false);
  const prevInspectToolRef = useRef(inspectTool);
  const prevCollisionModeRef = useRef(collisionMode);

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
}
