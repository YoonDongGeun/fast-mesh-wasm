// The export panel's Height Map section (#267): which meshes, the view they
// are seen from, the resolution and the heights shown, and the export (a
// 16-bit PNG and a JSON with its scale and place) with its progress.
import type { Scene } from "fast-mesh-wasm";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import type { AppState, Item } from "../../../app/state/app-state";
import { LengthInput } from "../../../shared/ui/LengthInput";
import { HEIGHT_MAP_VIEWS, type HeightMapView, MAX_RESOLUTION, MIN_RESOLUTION } from "../ux/height-map-export";
import { type HeightMapSource, useHeightMapExport } from "../ux/use-height-map-export";

export function HeightMapSection(props: { app: AppState; scene: Scene | WorkerScene; picked: Item | null }) {
  const { picked } = props;
  const h = useHeightMapExport(props.app, props.scene, picked);
  const { params, units } = h;
  const busy = h.progress !== null;
  const status = h.problem !== null ? { text: h.problem, problem: true } : h.report;
  return (
    <div className="op" data-height-map-section>
      <h3>Height map</h3>
      <div className="op-row">
        <label className="field">
          meshes
          <select
            data-height-map="source"
            value={h.source}
            disabled={busy}
            onChange={(e) => h.setSource(e.currentTarget.value as HeightMapSource)}
          >
            <option value="mesh">{picked?.name ?? "The mesh above"}</option>
            <option value="visible">All visible meshes</option>
          </select>
        </label>
        <label className="field">
          from
          <select
            data-height-map="view"
            value={params.view}
            disabled={busy}
            onChange={(e) => h.setParams({ view: e.currentTarget.value as HeightMapView, range: null })}
          >
            {HEIGHT_MAP_VIEWS.map(([id, label]) => (
              <option key={id} value={id}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          longer side (px)
          <input
            type="number"
            data-height-map="resolution"
            min={MIN_RESOLUTION}
            max={MAX_RESOLUTION}
            step={1}
            value={params.resolution}
            disabled={busy}
            onChange={(e) => h.setParams({ resolution: Number(e.currentTarget.value) })}
          />
        </label>
      </div>
      <div className="op-row">
        <label className="check">
          <input
            type="checkbox"
            data-height-map="custom-range"
            checked={params.range !== null}
            disabled={busy}
            onChange={(e) => void h.setCustomRange(e.currentTarget.checked)}
          />
          Own height range
        </label>
        {params.range === null ? null : (
          <>
            <label className="field">
              lowest ({units})
              <LengthInput
                data-height-map="low"
                title="The coordinate along the view that is height 0; surfaces below it are not seen"
                mm={params.range[0]}
                unit={units}
                disabled={busy}
                onMm={(mm) => h.setParams({ range: [mm, params.range?.[1] ?? mm] })}
              />
            </label>
            <label className="field">
              highest ({units})
              <LengthInput
                data-height-map="high"
                title="The coordinate along the view that is white; surfaces above it are not seen"
                mm={params.range[1]}
                unit={units}
                disabled={busy}
                onMm={(mm) => h.setParams({ range: [params.range?.[0] ?? mm, mm] })}
              />
            </label>
          </>
        )}
      </div>
      <div className="op-row">
        <button
          type="button"
          data-action="export-height-map"
          disabled={!h.ready || busy}
          onClick={() => void h.run()}
          title="Download an orthographic depth image as a 16-bit PNG, with its scale and origin as JSON"
        >
          {h.progress === null ? "Export height map" : `Sampling… ${Math.round(100 * h.progress)} %`}
        </button>
        {busy ? (
          <button type="button" data-action="cancel-height-map" onClick={h.cancel}>
            Cancel
          </button>
        ) : null}
      </div>
      {status === null ? null : (
        <p className={status.problem ? "error small" : "muted small"} data-height-map-status role="status">
          {status.text}
        </p>
      )}
    </div>
  );
}
