// #234 section contour slicing and export in the inspect panel: the slicing
// plane (an axis or a custom normal, offset from the mesh's world center),
// one slice or an evenly stepped stack through `scene.inspect`, the totals
// it reads out, and the SVG and DXF downloads.
import type { ContourSlice, Vec3 } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { useState } from "react";
import type { AppState, Item } from "../../../app/state/app-state";
import type { Wave6MetrologyState } from "../../../app/state/wave6-state";
import { ensureWave6Wasm } from "../../../engine/wave6-wasm-lazy";
import { type ContourLike, contoursToDxf, downloadReportText, reportFileName } from "../../../io/report";
import { AXIS_NORMAL, type Axis } from "./section-clip";

export type ContourPlaneAxis = Axis | "custom";

/** The last export: its format, text and file name, for the preview. */
export interface ContourExport {
  format: "svg" | "dxf";
  text: string;
  fileName: string;
}

function worldCenter(scene: SceneStore["scene"], item: Item | null): [number, number, number] {
  if (item === null) {
    return [0, 0, 0];
  }
  try {
    const m = scene.matrix4(item.mesh);
    return [
      m[0] * item.center[0] + m[4] * item.center[1] + m[8] * item.center[2] + m[12],
      m[1] * item.center[0] + m[5] * item.center[1] + m[9] * item.center[2] + m[13],
      m[2] * item.center[0] + m[6] * item.center[1] + m[10] * item.center[2] + m[14],
    ];
  } catch {
    return [0, 0, 0];
  }
}

/** The unit vector along `v`, or +Z when `v` is zero or not finite. */
export function normalizeNormalVec(v: readonly [number, number, number]): [number, number, number] {
  const len = Math.hypot(v[0], v[1], v[2]);
  if (!Number.isFinite(len) || len < 1e-8) {
    return [0, 0, 1];
  }
  return [v[0] / len, v[1] / len, v[2] / len];
}

function safeTriggerDownload(text: string, fileName: string, mime: string): void {
  if (typeof document === "undefined" || typeof URL === "undefined" || typeof URL.createObjectURL !== "function") {
    return;
  }
  downloadReportText(text, fileName, mime);
}

/** Converts multi-slice Wave 6 contours into minimal R12 DXF entities. */
export function exportContourSlicesDxf(slices: readonly ContourSlice[]): string {
  const origin: Vec3 = slices[0]?.planeOrigin ?? [0, 0, 0];
  const normal: Vec3 = slices[0]?.planeNormal ?? [0, 0, 1];
  const contours: ContourLike[] = [];
  for (const slice of slices) {
    for (const loop of slice.loops) {
      contours.push({ points: loop.points3d, closed: loop.closed });
    }
  }
  return contoursToDxf(contours, origin, normal);
}

/** The slices' loop count, perimeter and area summed (all zero without slices). */
export function contourSliceTotals(slices: readonly ContourSlice[] | null): {
  loops: number;
  perimeter: number;
  area: number;
} {
  const totalLoops = slices === null ? 0 : slices.reduce((sum, s) => sum + s.loops.length, 0);
  const totalPerimeter = slices === null ? 0 : slices.reduce((sum, s) => sum + s.totalPerimeter, 0);
  const totalArea = slices === null ? 0 : slices.reduce((sum, s) => sum + s.totalArea, 0);
  return { loops: totalLoops, perimeter: totalPerimeter, area: totalArea };
}

/**
 * The contour slice controls' state and handlers for `targetItem`. Slices
 * land in the app's metrology state (and show their overlay); an export
 * downloads the file and keeps its text for the preview.
 */
export function useContourSlices(
  app: AppState,
  store: SceneStore,
  targetItem: Item | null,
  metrology: Wave6MetrologyState,
) {
  const [planeAxis, setPlaneAxis] = useState<ContourPlaneAxis>("z");
  const [customNormal, setCustomNormal] = useState<[number, number, number]>([0, 0, 1]);
  const [offsetMm, setOffsetMm] = useState(0);
  const [stackCount, setStackCount] = useState(1);
  const [stepMm, setStepMm] = useState(1);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [exportState, setExportState] = useState<ContourExport | null>(null);

  const inspectDomain = store?.scene?.inspect;
  const canSlice =
    typeof inspectDomain?.sliceContours === "function" && typeof inspectDomain?.sliceContourStack === "function";
  const slices = metrology.contourSlices;

  const activeNormal = (): [number, number, number] =>
    planeAxis === "custom" ? normalizeNormalVec(customNormal) : AXIS_NORMAL[planeAxis];

  const onSlice = async () => {
    if (targetItem === null || !inspectDomain?.sliceContours || !inspectDomain?.sliceContourStack) {
      return;
    }
    setBusy(true);
    setError(null);
    setExportState(null);
    try {
      const n = activeNormal();
      const center = worldCenter(store.scene, targetItem);
      const origin: [number, number, number] = [
        center[0] + n[0] * offsetMm,
        center[1] + n[1] * offsetMm,
        center[2] + n[2] * offsetMm,
      ];
      const count = Math.max(1, Math.round(stackCount));
      const nextSlices =
        count > 1
          ? await Promise.resolve(inspectDomain.sliceContourStack(targetItem.mesh, origin, n, Math.max(0.001, stepMm), count))
          : [await Promise.resolve(inspectDomain.sliceContours(targetItem.mesh, origin, n))];
      app.updateWave6((prev) => ({
        ...prev,
        metrology: {
          ...prev.metrology,
          contourSlices: nextSlices,
          showContoursOverlay: true,
        },
      }));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const onExport = async (format: "svg" | "dxf") => {
    if (slices === null || slices.length === 0) {
      return;
    }
    setError(null);
    try {
      const text =
        format === "svg" ? (await ensureWave6Wasm()).sliceContoursExport(slices, "svg") : exportContourSlicesDxf(slices);
      const stem = `${targetItem?.name ?? "contours"}-section`;
      const fileName = reportFileName(stem, format);
      const mime = format === "svg" ? "image/svg+xml" : "application/dxf";
      safeTriggerDownload(text, fileName, mime);
      setExportState({ format, text, fileName });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  /** Drops the slices, the export preview and the error. */
  const onClear = () => {
    app.updateWave6((prev) => ({
      ...prev,
      metrology: { ...prev.metrology, contourSlices: null },
    }));
    setExportState(null);
    setError(null);
  };

  const setShowOverlay = (checked: boolean) => {
    app.updateWave6((prev) => ({
      ...prev,
      metrology: { ...prev.metrology, showContoursOverlay: checked },
    }));
  };

  /** Sets one custom normal component from a typed number, ignoring non-finite input. */
  const setNormalComponent = (idx: number, v: number) => {
    if (!Number.isFinite(v)) {
      return;
    }
    const next: [number, number, number] = [...customNormal] as [number, number, number];
    next[idx] = v;
    setCustomNormal(next);
  };

  /** Sets the stack count from a typed number, rounded into 1..100. */
  const setStackCountInput = (n: number) => {
    if (Number.isFinite(n)) {
      setStackCount(Math.max(1, Math.min(100, Math.round(n))));
    }
  };

  /** Sets the stack step (mm, at least 0.001). */
  const setStep = (mm: number) => setStepMm(Math.max(0.001, mm));

  return {
    planeAxis,
    setPlaneAxis,
    customNormal,
    setNormalComponent,
    offsetMm,
    setOffsetMm,
    stackCount,
    setStackCount: setStackCountInput,
    stepMm,
    setStep,
    busy,
    error,
    exportState,
    canSlice,
    slices,
    totals: contourSliceTotals(slices),
    onSlice,
    onExport,
    onClear,
    setShowOverlay,
  };
}
