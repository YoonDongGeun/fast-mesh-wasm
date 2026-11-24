// #234 primitive fitting and datum alignment in the inspect panel: the
// primitive kinds and datum axes it offers, the fitted primitive's readout
// lines, and the fit and align calls through `scene.inspect`.
import type { FittedPrimitive, FittedPrimitiveReport, PrimitiveKindInput, Vec3 } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { useState } from "react";
import type { AppState, Item } from "../../../app/state/app-state";
import type { Wave6MetrologyState } from "../../../app/state/wave6-state";
import { formatLength, formatPoint, type LengthUnit } from "../../../shared/ux/units";

export type DatumAxisKey = "posZ" | "negZ" | "posY" | "posX";

export const PRIMITIVE_KINDS: PrimitiveKindInput[] = ["auto", "plane", "sphere", "cylinder", "cone"];

export const DATUM_AXIS_OPTIONS: { id: DatumAxisKey; label: string; vec: Vec3 }[] = [
  { id: "posZ", label: "+Z", vec: [0, 0, 1] },
  { id: "negZ", label: "-Z", vec: [0, 0, -1] },
  { id: "posY", label: "+Y", vec: [0, 1, 0] },
  { id: "posX", label: "+X", vec: [1, 0, 0] },
];

/** A direction with three decimals and no unit, for normals and axes. */
export function formatVec3Unitless(v: readonly [number, number, number]): string {
  return `(${v[0].toFixed(3)}, ${v[1].toFixed(3)}, ${v[2].toFixed(3)})`;
}

/** The fitted primitive's geometry as readout lines: points and lengths in `unit`, directions unitless. */
export function summarizePrimitiveGeometry(primitive: FittedPrimitive, unit: LengthUnit): string[] {
  switch (primitive.kind) {
    case "plane":
      return [
        `Origin: ${formatPoint(primitive.origin, unit)}`,
        `Normal: ${formatVec3Unitless(primitive.normal)}`,
      ];
    case "sphere":
      return [
        `Center: ${formatPoint(primitive.center, unit)}`,
        `Radius: ${formatLength(primitive.radius, unit)}`,
      ];
    case "cylinder":
      return [
        `Origin: ${formatPoint(primitive.axisOrigin, unit)}`,
        `Axis: ${formatVec3Unitless(primitive.axisDir)}`,
        `Radius: ${formatLength(primitive.radius, unit)}`,
        `Height: ${formatLength(primitive.height, unit)}`,
      ];
    case "cone": {
      const deg = ((primitive.halfAngleRad * 180) / Math.PI).toFixed(2);
      return [
        `Apex: ${formatPoint(primitive.apex, unit)}`,
        `Axis: ${formatVec3Unitless(primitive.axisDir)}`,
        `Half-angle: ${deg}°`,
        `Height: ${formatLength(primitive.height, unit)}`,
      ];
    }
  }
}

/**
 * The primitive fit controls' state and handlers for `targetItem`. A fit
 * stores its report in the app's metrology state (and shows its overlay);
 * an align rigidly moves the mesh so its fitted datum lies on the chosen
 * axis through the origin, in one undo step.
 */
export function usePrimitiveFit(
  app: AppState,
  store: SceneStore,
  targetItem: Item | null,
  metrology: Wave6MetrologyState,
) {
  const [kind, setKind] = useState<PrimitiveKindInput>("auto");
  const [targetAxis, setTargetAxis] = useState<DatumAxisKey>("posZ");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const inspectDomain = store?.scene?.inspect;
  const canFit = typeof inspectDomain?.fitPrimitive === "function";
  const canAlign = typeof inspectDomain?.alignToFittedPrimitive === "function";
  const fitted: FittedPrimitiveReport | null = metrology.fittedPrimitive;

  const onFit = async () => {
    if (targetItem === null || !inspectDomain?.fitPrimitive) {
      return;
    }
    setBusy(true);
    setError(null);
    setStatus(null);
    try {
      const report = await Promise.resolve(inspectDomain.fitPrimitive(targetItem.mesh, { kind }));
      app.updateWave6((prev) => ({
        ...prev,
        metrology: {
          ...prev.metrology,
          fittedPrimitive: report,
          showPrimitiveOverlay: true,
        },
      }));
      setStatus(`Fitted ${report.primitive.kind} (${report.inlierCount} inliers)`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const onAlign = async () => {
    if (targetItem === null || fitted === null || !inspectDomain?.alignToFittedPrimitive) {
      return;
    }
    const axisOpt = DATUM_AXIS_OPTIONS.find((a) => a.id === targetAxis) ?? DATUM_AXIS_OPTIONS[0];
    setBusy(true);
    setError(null);
    try {
      await Promise.resolve(
        inspectDomain.alignToFittedPrimitive(targetItem.mesh, fitted.primitive, axisOpt.vec, [0, 0, 0]),
      );
      store.commitFrame();
      setStatus(`Aligned ${targetItem.name} datum to ${axisOpt.label}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  /** Drops the fitted primitive and the status lines. */
  const onClear = () => {
    app.updateWave6((prev) => ({
      ...prev,
      metrology: { ...prev.metrology, fittedPrimitive: null },
    }));
    setStatus(null);
    setError(null);
  };

  const setShowOverlay = (checked: boolean) => {
    app.updateWave6((prev) => ({
      ...prev,
      metrology: { ...prev.metrology, showPrimitiveOverlay: checked },
    }));
  };

  return {
    kind,
    setKind,
    targetAxis,
    setTargetAxis,
    busy,
    status,
    error,
    canFit,
    canAlign,
    fitted,
    onFit,
    onAlign,
    onClear,
    setShowOverlay,
  };
}
