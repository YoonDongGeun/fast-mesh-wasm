// A floating volume mini-toolbar: while a volume mesh is selected, its
// 3D preset, 3D/Slices tab and scene overlay sit over the viewport,
// steering the same linked state as the Volume panel.
import { useSelection } from "fast-mesh-wasm/react";
import { VOLUME_PRESETS } from "../panels/volume-presets";
import { type AppState, useApp } from "../scene/app-state";
import { matchVolume } from "./volume-box";

export function VolumeOverlay(props: { app: AppState }) {
  const { app } = props;
  const [selection] = useSelection();
  const items = useApp(app, (s) => s.items);
  const mpr = useApp(app, (s) => s.volumeMpr);
  const found = matchVolume(items, selection, mpr);
  if (found === null) {
    return null;
  }
  const { volumeId, mpr: live } = found;
  const set = (patch: { preset?: number; tab?: "volume" | "slices"; inScene?: boolean }): void => {
    const cur = app.getSnapshot().volumeMpr;
    if (cur !== null && cur.volumeId === volumeId) {
      app.setVolumeMpr({ ...cur, ...patch });
    }
  };
  return (
    <div className="volume-overlay" role="toolbar" aria-label="Volume quick controls">
      <label className="slider" title="Which 3D preset renders the volume">
        preset{" "}
        <select data-action="overlay-volume-preset" value={live.preset} onChange={(e) => set({ preset: Number(e.currentTarget.value) })}>
          {VOLUME_PRESETS.map((p, i) => (
            <option key={p.name} value={i}>
              {p.name}
            </option>
          ))}
        </select>
      </label>
      <div className="group" role="tablist" aria-label="Volume views">
        <button
          type="button"
          role="tab"
          aria-selected={live.tab === "volume"}
          data-action="overlay-volume-tab-3d"
          onClick={() => set({ tab: "volume" })}
        >
          3D
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={live.tab === "slices"}
          data-action="overlay-volume-tab-slices"
          onClick={() => set({ tab: "slices" })}
        >
          Slices
        </button>
      </div>
      <label className="check" title="Show the slices in the main scene too">
        <input
          type="checkbox"
          data-action="overlay-volume-scene"
          checked={live.inScene}
          onChange={(e) => set({ inScene: e.currentTarget.checked })}
        />
        Scene
      </label>
    </div>
  );
}
