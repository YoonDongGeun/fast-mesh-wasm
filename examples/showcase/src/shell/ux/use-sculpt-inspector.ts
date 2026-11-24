// The Sculpt & Deform inspector card's state and actions: the mesh it
// targets (the armed brush's, else the mask's, else the selection), the
// dyntopo detail readout, the dyntopo pass over the stroke region, and
// disarming the brush and mask paint.
import type { DyntopoMode, MeshHandle } from "fast-mesh-wasm";
import { useScene } from "fast-mesh-wasm/react";
import { type AppState, useApp } from "../../app/state/app-state";
import { applyDyntopoRoi } from "../../features/sculpt/ux/sculpt";
import { formatLength, type LengthUnit } from "../../shared/ux/units";

/** The shading modes the card offers while sculpting. */
export const SCULPT_SHADING_MODES = ["smooth", "matcap", "wireframeShaded", "backfaceWarning"] as const;

/**
 * The dyntopo detail readout: a percentage in brush-relative mode (the
 * target edge length follows the brush radius), else the length it sets on
 * a mesh of `meshRadius`.
 */
export function dyntopoDetailLabel(mode: DyntopoMode, detail: number, meshRadius: number, units: LengthUnit): string {
  return mode === "brush-relative" ? `${Math.round(detail * 100)}%` : formatLength(detail * meshRadius, units);
}

export function useSculptInspector(app: AppState, selection: MeshHandle | null) {
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

  /** Subdivides and cleans the dyntopo region on the target (no-op without one). */
  const applyDyntopo = () => {
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
  };

  const disarm = () => {
    app.setSculpt(null);
    app.setMaskPaint(false);
  };

  const exitToSelect = () => {
    app.setSculpt(null);
    app.setMaskPaint(false);
    app.setActiveTab("select");
  };

  return {
    sculpt,
    maskPaint,
    pinToActive,
    liveRemesh,
    remesh,
    dyntopo,
    dyntopoMode,
    dyntopoDetail,
    detailLabel: dyntopoDetailLabel(dyntopoMode, dyntopoDetail, targetRadius, units),
    symmetry,
    shadingMode,
    targetItem,
    applyDyntopo,
    disarm,
    exitToSelect,
  };
}
