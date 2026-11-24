// The linked volume-view state (`AppState.volumeMpr`): seeded when a
// volume's panel opens and dropped when it closes, patched by the panel's
// tabs, the viewport overlay and the slice views, so the scene overlay
// stays linked too. A patch for a volume no longer linked is dropped.
import { type Dispatch, type SetStateAction, useEffect, useMemo } from "react";
import type { VolumePayload } from "../../../io/protocol";
import { type AppState, type VolumeMpr, useApp } from "../../../app/state/app-state";
import { displayValues } from "./volume-data";
import { dataRange } from "./volume-lut";
import { presetWindowLevel, WINDOW_PRESETS } from "./volume-presets";
import type { SliceAxis } from "./volume-slice";

/** Patches the linked state while it is `volumeId`'s. */
export function patchVolumeMpr(app: AppState, volumeId: string, patch: Partial<Omit<VolumeMpr, "volumeId">>): void {
  const cur = app.getSnapshot().volumeMpr;
  if (cur !== null && cur.volumeId === volumeId) {
    app.setVolumeMpr({ ...cur, ...patch });
  }
}

/**
 * The panel's linked state for `payload`: seeded with the middle slices, the
 * first window, `initialPreset` and the 3D tab, and dropped on leaving.
 * `live` is the state while it is this volume's (`null` before the seed
 * lands); `display` and `range` are the volume's display values and their
 * range, which the window and level resolve against.
 */
export function useVolumeMpr(
  app: AppState,
  payload: VolumePayload,
  volumeId: string,
  initialPreset: number,
): {
  live: VolumeMpr | null;
  tab: VolumeMpr["tab"];
  display: Float32Array;
  range: [number, number];
  windowName: string;
  colorWindow: number;
  colorLevel: number;
  setTab: (t: VolumeMpr["tab"]) => void;
  setPreset: (i: number) => void;
  setWindow: (i: number) => void;
  setInScene: (on: boolean) => void;
  setSlices: Dispatch<SetStateAction<Record<SliceAxis, number>>>;
} {
  const mpr = useApp(app, (s) => s.volumeMpr);
  const tab = mpr !== null && mpr.volumeId === volumeId ? mpr.tab : "volume";
  useEffect(() => {
    app.setVolumeMpr({
      volumeId,
      slices: {
        axial: Math.floor((payload.slices - 1) / 2),
        coronal: Math.floor((payload.rows - 1) / 2),
        sagittal: Math.floor((payload.columns - 1) / 2),
      },
      window: 0,
      inScene: true,
      preset: initialPreset,
      tab: "volume",
    });
    return () => {
      if (app.getSnapshot().volumeMpr?.volumeId === volumeId) {
        app.setVolumeMpr(null);
      }
    };
  }, [app, payload, volumeId, initialPreset]);
  const display = useMemo(() => displayValues(payload.values, payload.pseudoHu), [payload]);
  const range = useMemo(() => dataRange(display), [display]);
  const live = mpr !== null && mpr.volumeId === volumeId ? mpr : null;
  const preset = WINDOW_PRESETS[live?.window ?? 0];
  const { window: colorWindow, level: colorLevel } = presetWindowLevel(preset, range);
  const setSlices: Dispatch<SetStateAction<Record<SliceAxis, number>>> = (update) => {
    const cur = app.getSnapshot().volumeMpr;
    if (cur === null || cur.volumeId !== volumeId) {
      return;
    }
    app.setVolumeMpr({ ...cur, slices: typeof update === "function" ? update(cur.slices) : update });
  };
  return {
    live,
    tab,
    display,
    range,
    windowName: preset.name,
    colorWindow,
    colorLevel,
    setTab: (t) => patchVolumeMpr(app, volumeId, { tab: t }),
    setPreset: (i) => patchVolumeMpr(app, volumeId, { preset: i }),
    setWindow: (i) => patchVolumeMpr(app, volumeId, { window: i }),
    setInScene: (on) => patchVolumeMpr(app, volumeId, { inScene: on }),
    setSlices,
  };
}
