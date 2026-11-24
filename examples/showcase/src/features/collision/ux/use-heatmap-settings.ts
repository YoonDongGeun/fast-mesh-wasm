// The collision panel's heatmap settings: the sliders' scale from the meshes
// coloured, their values from the app state clamped to contactHeatmap's
// limits, and the heatmap both modes colour with.
import type { MeshHandle } from "fast-mesh-wasm";
import { contactHeatmap, type DistanceColoring } from "fast-mesh-wasm/store";
import { useMemo } from "react";
import { type AppState, type HeatmapParams, type Item, useApp } from "../../../app/state/app-state";
import { heatmapLimits, meshScale } from "./collision-math";

export type Heatmap = Omit<DistanceColoring, "distanceTo">;

/**
 * The heatmap settings for the meshes `shown`. The values are in mm, whatever
 * the display unit: changing that moves no value, so no colouring (only the
 * sliders convert). `set` writes a slider's value with the others as clamped.
 */
export function useHeatmapSettings(
  app: AppState,
  items: readonly Item[],
  shown: readonly (MeshHandle | null)[],
): {
  scale: number;
  step: number;
  depth: number;
  gap: number;
  tolerance: number;
  toleranceMax: number;
  fast: boolean;
  softness: number;
  heat: Heatmap;
  set: (patch: Partial<HeatmapParams>) => void;
} {
  const scale = meshScale(items, shown);
  // `null` until a slider moves: the defaults then follow the scale.
  // In the app state (persisted: last-used), not local: the status bar's
  // reset restores them with the rest of the display settings.
  const params = useApp(app, (s) => s.heatmapParams);
  const fast = useApp(app, (s) => s.heatmapFast);
  const softness = useApp(app, (s) => s.edgeSoftness);
  const { step, depth, gap, tolerance, toleranceMax } = heatmapLimits(params, scale);
  const set = (patch: Partial<HeatmapParams>) => app.setHeatmapParams({ depth, gap, tolerance, ...patch });

  const heat = useMemo(
    () => contactHeatmap({ depth, gap, tolerance, exactDepth: !fast }),
    [depth, gap, tolerance, fast],
  );
  return { scale, step, depth, gap, tolerance, toleranceMax, fast, softness, heat, set };
}
