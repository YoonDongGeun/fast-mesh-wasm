// The feature chunks (#285): what the page does not need for its first frame,
// one chunk per feature (`features/<area>/ui/<area>-chunk.ts`), and when they load.
// The panels and layers they bring are in app/ui/feature-parts.tsx.
//
// Once the page is up and idle every chunk loads, so a tab opens with its
// panel in place, and every panel is mounted as before (hidden ones too).
// A tab's chunks load at once when it is hovered, focused or opened.
import { useEffect } from "react";
import { LazyChunk } from "../../shared/ux/lazy-chunk";
import { type AppState, useApp, type WorkbenchTab } from "../state/app-state";

export const featureChunks = {
  buildPlate: new LazyChunk("build plate", () => import("../../features/build-plate/ui/build-plate-chunk")),
  collision: new LazyChunk("collision", () => import("../../features/collision/ui/collision-chunk")),
  ct: new LazyChunk("volume", () => import("../../features/ct/ui/ct-chunk")),
  home: new LazyChunk("home", () => import("../../features/home/ui/home-chunk")),
  inspect: new LazyChunk("inspect", () => import("../../features/inspect/ui/inspect-chunk")),
  meshEdit: new LazyChunk("mesh edit", () => import("../../features/mesh-edit/ui/mesh-edit-chunk")),
  meshRepair: new LazyChunk("mesh repair", () => import("../../features/mesh-repair/ui/mesh-repair-chunk")),
  points: new LazyChunk("points", () => import("../../features/points/ui/points-chunk")),
  polylines: new LazyChunk("polylines", () => import("../../features/polylines/ui/polylines-chunk")),
  sculpt: new LazyChunk("sculpt", () => import("../../features/sculpt/ui/sculpt-chunk")),
  select: new LazyChunk("select", () => import("../../features/select/ui/select-chunk")),
  transform: new LazyChunk("transform", () => import("../../features/transform/ui/transform-chunk")),
};

/** The chunks of each tab's panels and of the tools its shelf arms. */
const TAB_CHUNKS: Readonly<Record<WorkbenchTab, readonly LazyChunk[]>> = {
  home: [featureChunks.buildPlate, featureChunks.home],
  select: [featureChunks.select, featureChunks.transform],
  repair: [featureChunks.meshRepair],
  edit: [featureChunks.meshEdit, featureChunks.polylines],
  sculpt: [featureChunks.sculpt],
  align: [featureChunks.points, featureChunks.transform, featureChunks.collision],
  volume: [featureChunks.ct],
  inspect: [featureChunks.inspect, featureChunks.polylines, featureChunks.points],
};

/** Loads `tab`'s chunks now (failed ones again). */
export function preloadTab(tab: WorkbenchTab): void {
  for (const chunk of TAB_CHUNKS[tab]) {
    void chunk.load();
  }
}

function loadAll(): void {
  for (const chunk of Object.values<LazyChunk>(featureChunks)) {
    void chunk.load();
  }
}

/** The longest the chunks wait for an idle moment, in milliseconds. */
const IDLE_TIMEOUT_MS = 1000;

/** Runs `run` once the page has drawn a frame and is idle (or has waited long enough). Returns the function that cancels it. */
function whenIdle(run: () => void): () => void {
  let cancel = () => {};
  const frame = requestAnimationFrame(() => {
    if (typeof requestIdleCallback === "function") {
      const id = requestIdleCallback(run, { timeout: IDLE_TIMEOUT_MS });
      cancel = () => cancelIdleCallback(id);
    } else {
      // Safari has no requestIdleCallback.
      const id = setTimeout(run, 200);
      cancel = () => clearTimeout(id);
    }
  });
  cancel = () => cancelAnimationFrame(frame);
  return () => cancel();
}

/**
 * Loads the open tab's chunks at once and every other one when the page is
 * idle; failed ones again when the network is back.
 */
export function useFeatureChunks(app: AppState): void {
  const tab = useApp(app, (s) => s.activeTab);
  useEffect(() => preloadTab(tab), [tab]);
  useEffect(() => whenIdle(loadAll), []);
  useEffect(() => {
    window.addEventListener("online", loadAll);
    return () => window.removeEventListener("online", loadAll);
  }, []);
}
