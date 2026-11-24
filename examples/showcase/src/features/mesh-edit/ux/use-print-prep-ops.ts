// The Modeling panel's Print Prep sections (#236): the Hollow Mesh fields
// and run, the drain hole and surface stamp parameters (kept in the app's
// wave6 state, which the viewport tool reads too), Punch at Center, and
// arming the two viewport tools. Each run leaves one report line.
import type { HollowBackend, MeshHandle, SceneLike, Vec3 } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { useState } from "react";
import { type AppState, type Item, useApp } from "../../../app/state/app-state";
import {
  describeDrainResult,
  describeHollowResult,
  executeDrainHoleAtHit,
  getPlaceAlignToNormal,
  hollowOptionsFrom,
  runHollowMeshOp,
  setPlaceAlignToNormal,
} from "./print-prep-ops";
import type { ModelingReport } from "./use-modeling-ops";

export function usePrintPrepOps(args: {
  app: AppState;
  store: SceneStore;
  /** The surface point Punch at Center punches at (the viewport layer's center raycast). */
  findCenterHit: (scene: SceneLike, mesh: MeshHandle) => Promise<{ point: Vec3; normal: Vec3 } | null>;
}) {
  const { app, store, findCenterHit } = args;
  const printPrep = useApp(app, (s) => s.wave6.printPrep);
  const [hollowWallMm, setHollowWallMm] = useState(1.5);
  const [hollowBackend, setHollowBackend] = useState<HollowBackend>("bvhSdf");
  const [hollowVoxelSizeText, setHollowVoxelSizeText] = useState("");
  const [hollowMaxVoxels, setHollowMaxVoxels] = useState(524288);
  const [hollowCapInnerShell, setHollowCapInnerShell] = useState(false);
  const [drainSavedDepthMm, setDrainSavedDepthMm] = useState(12.0);
  const [placeAlignToNormal, setPlaceAlignState] = useState(() => getPlaceAlignToNormal(app));
  const [prepBusy, setPrepBusy] = useState<string | null>(null);
  const [prepReport, setPrepReport] = useState<ModelingReport | null>(null);

  /** A depth of 0 punches through the wall. */
  const drainThrough = printPrep.drainParams.depth <= 0;

  const updatePrintPrep = (patch: Partial<typeof printPrep>) => {
    const current = app.getSnapshot().wave6.printPrep;
    app.updateWave6({ printPrep: { ...current, ...patch } });
  };

  const updateDrainParams = (patch: Partial<typeof printPrep.drainParams>) => {
    const current = app.getSnapshot().wave6.printPrep;
    app.updateWave6({
      printPrep: { ...current, drainParams: { ...current.drainParams, ...patch } },
    });
  };

  const updatePlaceParams = (patch: Partial<typeof printPrep.placeParams>) => {
    const current = app.getSnapshot().wave6.printPrep;
    app.updateWave6({
      printPrep: { ...current, placeParams: { ...current.placeParams, ...patch } },
    });
  };

  const runHollow = async (target: Item) => {
    setPrepBusy("Hollow Mesh");
    try {
      const options = hollowOptionsFrom({
        wallMm: hollowWallMm,
        backend: hollowBackend,
        voxelSizeText: hollowVoxelSizeText,
        maxVoxels: hollowMaxVoxels,
        capInnerShell: hollowCapInnerShell,
      });
      const res = await runHollowMeshOp({ store, mesh: target.mesh, options });
      setPrepReport({
        label: "Hollow Mesh",
        text: describeHollowResult(res),
        error: false,
      });
    } catch (err) {
      setPrepReport({
        label: "Hollow Mesh",
        text: err instanceof Error ? err.message : String(err),
        error: true,
      });
    } finally {
      setPrepBusy(null);
    }
  };

  const punchAtCenter = async (target: Item) => {
    setPrepBusy("Drain Hole");
    try {
      const hit = await findCenterHit(store.scene, target.mesh);
      if (hit === null) {
        throw new Error("Could not find a surface point on the active mesh.");
      }
      const res = await executeDrainHoleAtHit({
        store,
        app,
        target: target.mesh,
        targetName: target.name,
        hitPoint: hit.point,
        hitNormal: hit.normal,
        drainParams: printPrep.drainParams,
        through: drainThrough,
      });
      setPrepReport({
        label: "Drain Hole",
        text: describeDrainResult(res),
        error: false,
      });
    } catch (err) {
      setPrepReport({
        label: "Drain Hole",
        text: err instanceof Error ? err.message : String(err),
        error: true,
      });
    } finally {
      setPrepBusy(null);
    }
  };

  /** Arms `tool` in the viewport, or disarms it when it is already armed. */
  const toggleTool = (tool: "drainHole" | "placeStamp") => {
    const next = printPrep.activeTool === tool ? null : tool;
    updatePrintPrep({ activeTool: next });
    void store.commitFrame();
  };

  /** A drain depth (at least 0.2), remembered for when Through wall is unticked. */
  const setDrainDepth = (mm: number) => {
    const next = Math.max(0.2, mm);
    setDrainSavedDepthMm(next);
    updateDrainParams({ depth: next });
  };

  /** Through wall sets the depth to 0, keeping the last depth to restore (at least 0.5) when unticked. */
  const setDrainThrough = (through: boolean) => {
    if (through) {
      if (printPrep.drainParams.depth > 0) {
        setDrainSavedDepthMm(printPrep.drainParams.depth);
      }
      updateDrainParams({ depth: 0 });
    } else {
      updateDrainParams({ depth: Math.max(0.5, drainSavedDepthMm) });
    }
  };

  /** The Scene Stamp pick: a mesh id makes it the stamp, empty goes back to a built-in preset. */
  const pickSceneStamp = (value: string) => {
    if (value === "") {
      updatePlaceParams({
        stampMeshId: null,
        ...(printPrep.placeParams.stampSource === "sceneMesh"
          ? { stampSource: "peg" }
          : {}),
      });
    } else {
      updatePlaceParams({
        stampSource: "sceneMesh",
        stampMeshId: Number(value),
      });
    }
  };

  const setAlignToNormal = (align: boolean) => {
    setPlaceAlignState(align);
    setPlaceAlignToNormal(app, align);
  };

  return {
    printPrep,
    drainThrough,
    hollowWallMm,
    setHollowWallMm,
    hollowBackend,
    setHollowBackend,
    hollowVoxelSizeText,
    setHollowVoxelSizeText,
    hollowMaxVoxels,
    setHollowMaxVoxels,
    hollowCapInnerShell,
    setHollowCapInnerShell,
    drainSavedDepthMm,
    placeAlignToNormal,
    prepBusy,
    prepReport,
    updateDrainParams,
    updatePlaceParams,
    runHollow,
    punchAtCenter,
    toggleTool,
    setDrainDepth,
    setDrainThrough,
    pickSceneStamp,
    setAlignToNormal,
  };
}

/** The drain taper field: 0 to 45 degrees; an empty or invalid entry reads as 0. */
export function taperInput(text: string): number {
  return Math.max(0, Math.min(45, Number(text) || 0));
}

/** The stamp scale field: at least 0.05; an empty or invalid entry reads as 1. */
export function stampScaleInput(text: string): number {
  return Math.max(0.05, Number(text) || 1);
}

/** The stamp spin field in degrees; an empty or invalid entry reads as 0. */
export function spinInput(text: string): number {
  return Number(text) || 0;
}
