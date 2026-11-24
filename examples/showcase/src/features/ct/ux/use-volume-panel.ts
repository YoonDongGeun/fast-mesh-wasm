// The volume panel's behaviour around the linked state: fetching the
// retained volume of the selected mesh (the loader keeps each series'
// volume), bringing the tabs into view, picking another loaded volume, the
// slice views' readiness for the test hook, and click/drag navigation on a
// slice view.
import { useSelection } from "fast-mesh-wasm/react";
import { type PointerEvent, type RefObject, useEffect, useState } from "react";
import type { Loader } from "../../../io/loader";
import type { VolumePayload } from "../../../io/protocol";
import { type AppState, type Item, useApp } from "../../../app/state/app-state";
import { inspect } from "../../../app/state/readouts";
import { selectMesh } from "../../select/ux/selection";
import { VOLUME_PRESETS, WINDOW_PRESETS } from "./volume-presets";
import { pointerSlice, type SliceAxis, type VolumeDims } from "./volume-slice";

/**
 * `volumeId`'s retained volume, fetched once (`null` while it loads), or
 * `failed` once the loader no longer has it. Resets the test hook's volume
 * and slice probes to this volume, and clears them on leaving.
 */
export function usePanelVolume(loader: Loader, volumeId: string): { payload: VolumePayload | null; failed: boolean } {
  const [payload, setPayload] = useState<VolumePayload | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let alive = true;
    inspect.volume = { id: volumeId, preset: VOLUME_PRESETS[0].name, mip: false, planes: 0, ready: false };
    inspect.slices = { id: volumeId, window: WINDOW_PRESETS[0].name, axial: 0, coronal: 0, sagittal: 0, ready: false };
    loader
      .fetchVolume(volumeId)
      .then((got: VolumePayload) => {
        if (alive) {
          setPayload(got);
        }
      })
      .catch(() => {
        if (alive) {
          setFailed(true);
        }
      });
    return () => {
      alive = false;
      if (inspect.volume.id === volumeId) {
        inspect.volume = { id: null, preset: "", mip: false, planes: 0, ready: false };
      }
      if (inspect.slices.id === volumeId) {
        inspect.slices = { id: null, window: "", axial: 0, coronal: 0, sagittal: 0, ready: false };
      }
    };
  }, [loader, volumeId]);

  return { payload, failed };
}

/**
 * Selecting a volume mesh brings its tab buttons into view at once,
 * retried briefly: the panels above can still grow (repair analysis).
 */
export function useRevealTabs(tabsRef: RefObject<HTMLElement | null>, volumeId: string): void {
  useEffect(() => {
    let ticks = 0;
    const timer = setInterval(() => {
      ticks += 1;
      const tabs = tabsRef.current;
      const pane = tabs?.closest(".sidebar") ?? null;
      if (tabs === null || pane === null || ticks > 8) {
        clearInterval(timer);
        return;
      }
      const r = tabs.getBoundingClientRect();
      const p = pane.getBoundingClientRect();
      if (r.top < p.top || r.bottom > p.bottom) {
        tabs.scrollIntoView({ block: "nearest" });
      } else {
        clearInterval(timer);
      }
    }, 250);
    return () => clearInterval(timer);
  }, [volumeId]);
}

/** The loaded meshes with a volume, and choosing one (selects its mesh, alone). */
export function useVolumeSources(app: AppState): { sources: Item[]; chooseSource: (volumeId: string) => void } {
  const [selection, select] = useSelection();
  const sources = useApp(app, (s) => s.items).filter((i) => i.volumeId !== null);
  const chooseSource = (volumeId: string): void => {
    const item = sources.find((i) => i.volumeId === volumeId);
    if (item !== undefined) {
      selectMesh(app, select, selection, item.mesh, false);
    }
  };
  return { sources, chooseSource };
}

/**
 * Mirrors the slice views to the test hook's `inspect.slices`: their window,
 * positions and whether all three have drawn. `markReady` records a view's
 * first drawing.
 */
export function useSliceViewsProbe(
  volumeId: string,
  windowName: string,
  slices: Record<SliceAxis, number>,
): (axis: SliceAxis) => void {
  const [ready, setReady] = useState<Record<SliceAxis, boolean>>({ axial: false, coronal: false, sagittal: false });

  useEffect(() => {
    inspect.slices = {
      id: volumeId,
      window: windowName,
      axial: slices.axial,
      coronal: slices.coronal,
      sagittal: slices.sagittal,
      ready: ready.axial && ready.coronal && ready.sagittal,
    };
  }, [volumeId, windowName, slices, ready]);

  return (axis) => setReady((r) => ({ ...r, [axis]: true }));
}

/**
 * Click and drag on the `axis` slice view: the pointer picks the other two
 * axes' slices (`pointerSlice`), captured from the press, followed while the
 * primary button is held.
 */
export function sliceNavigation(
  canvasRef: RefObject<HTMLCanvasElement | null>,
  axis: SliceAxis,
  dims: VolumeDims,
  onNavigate: (part: Partial<Record<SliceAxis, number>>) => void,
): {
  onPointerDown: (e: PointerEvent<HTMLCanvasElement>) => void;
  onPointerMove: (e: PointerEvent<HTMLCanvasElement>) => void;
} {
  const navigate = (clientX: number, clientY: number): void => {
    const canvas = canvasRef.current;
    if (canvas === null) {
      return;
    }
    const rect = canvas.getBoundingClientRect();
    onNavigate(
      pointerSlice(
        axis,
        { columns: dims.columns, rows: dims.rows, slices: dims.slices },
        { width: canvas.width, height: canvas.height },
        { width: rect.width, height: rect.height },
        clientX - rect.left,
        clientY - rect.top,
      ),
    );
  };
  return {
    onPointerDown: (e) => {
      e.currentTarget.setPointerCapture(e.pointerId);
      navigate(e.clientX, e.clientY);
    },
    onPointerMove: (e) => {
      if (e.buttons & 1) {
        navigate(e.clientX, e.clientY);
      }
    },
  };
}
