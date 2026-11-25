// The geodesic distance and surface path sections (#252): click sources on
// the selected mesh to colour it by surface distance (fast or accurate, with
// optional iso-lines), or click points to draw a path along the surface
// with its length; a closed loop trims the mesh (keeping either side) or
// splits it into two meshes. Behaviour is in `ux/use-surface-paths.ts` and
// `ux/use-geodesic-map.ts`, the math and wording in `ux/surface-paths.ts`
// and `ux/geodesic-map.ts`.
import type { MeshHandle } from "fast-mesh-wasm";
import type { AppState, Item } from "../../../app/state/app-state";
import { LengthInput } from "../../../shared/ui/LengthInput";
import type { LengthUnit } from "../../../shared/ux/units";
import { geodesicLegend, geodesicReadoutLines } from "../ux/geodesic-map";
import { GEODESIC_METHODS, pathReadout, slowMethodNote } from "../ux/surface-paths";
import { useGeodesicMap } from "../ux/use-geodesic-map";
import { useSurfacePaths } from "../ux/use-surface-paths";
import { FieldLegend } from "./FieldLegend";

export function SurfacePathControls(props: {
  app: AppState;
  items: readonly Item[];
  selection: MeshHandle | null;
  unit: LengthUnit;
  /** Turns the surface field off before the geodesic map takes the overlay. */
  onTakeOverlay: () => void;
}) {
  const { app, items, selection, unit, onTakeOverlay } = props;
  const paths = useSurfacePaths(app, items, selection);
  const map = useGeodesicMap(app, items, onTakeOverlay);
  const { target, armed, method, sources, path, tracing, error, busy, status, locked } = paths;
  const report = map.report;
  const legend = report === null ? null : geodesicLegend(report, unit);
  const readout = pathReadout(path, unit);
  const closed = path?.closed ?? false;
  const slow = slowMethodNote(method, paths.item?.triangles ?? 0);

  return (
    <>
      <div className="op surface-field-panel geodesic-panel" data-inspect-geodesic tabIndex={-1}>
        <h3>Geodesic Distance</h3>
        <div className="modes" role="group" aria-label="Distance method">
          {GEODESIC_METHODS.map((m) => (
            <button
              key={m.method}
              type="button"
              data-geodesic-method={m.method}
              aria-pressed={method === m.method}
              title={m.title}
              onClick={() => paths.setMethod(m.method)}
            >
              {m.label}
            </button>
          ))}
        </div>
        {slow !== null ? (
          <p className="muted small" data-geodesic-slow>
            {slow}
          </p>
        ) : null}
        <div className="op-row">
          <button
            type="button"
            data-action="geodesic-pick"
            aria-pressed={armed === "sources"}
            disabled={target === null}
            title="Click points on the selected mesh: it is coloured by the surface distance to the nearest one"
            onClick={() => paths.arm("sources")}
          >
            {armed === "sources" ? "Picking sources" : "Pick sources"}
          </button>
          <button
            type="button"
            data-action="geodesic-undo"
            disabled={sources.length === 0}
            title="Remove the latest source"
            onClick={paths.undoSource}
          >
            Undo source
          </button>
          <button
            type="button"
            data-action="geodesic-clear"
            disabled={sources.length === 0}
            title="Remove every source and the map"
            onClick={paths.clearSources}
          >
            Clear
          </button>
          {sources.length > 0 && (report === null || map.stale) && !map.busy ? (
            <button type="button" data-action="geodesic-show" title="Colour the mesh by this map again" onClick={map.show}>
              Show map
            </button>
          ) : null}
        </div>
        <div className="op-row">
          <label className="check" title="Lines of equal surface distance">
            <input
              type="checkbox"
              data-geodesic-iso
              checked={map.iso.on}
              onChange={(e) => map.setIso({ on: e.currentTarget.checked })}
            />
            Iso-lines
          </label>
          <label className="field">
            every ({unit})
            <LengthInput
              aria-label="Iso-line spacing"
              data-geodesic-iso-spacing
              min={0}
              mm={map.spacing}
              unit={unit}
              disabled={!map.iso.on}
              onMm={(mm) => map.setIso({ spacing: mm > 0 ? mm : null })}
            />
          </label>
        </div>
        {armed === "sources" ? (
          <p className="muted small">Click the selected mesh to add a source · drag to turn the view · Esc stops.</p>
        ) : null}
        {target === null && sources.length === 0 ? <p className="muted small">Select a mesh to measure on.</p> : null}
        {map.busy ? (
          <div className="op-row">
            <p className="muted busy">Computing surface distances...</p>
            <button
              type="button"
              data-action="geodesic-cancel"
              title="Stop computing this map; the sources stay"
              onClick={map.cancel}
            >
              Cancel
            </button>
          </div>
        ) : null}
        {report !== null && legend !== null ? (
          <>
            <FieldLegend map={legend.map} lo={legend.lo} hi={legend.hi} band={legend.band} />
            <div className="readout" data-geodesic-readout>
              {geodesicReadoutLines(report, unit).map((line) => (
                <p key={line}>{line}</p>
              ))}
            </div>
          </>
        ) : null}
        {map.error !== null ? <p className="report error">{map.error}</p> : null}
      </div>

      <div className="op surface-field-panel geodesic-panel" data-inspect-path tabIndex={-1}>
        <h3>Surface Path</h3>
        <div className="op-row">
          <button
            type="button"
            data-action="path-pick"
            aria-pressed={armed === "path"}
            disabled={target === null}
            title="Click points on the selected mesh: each joins the previous one along the surface"
            onClick={() => paths.arm("path")}
          >
            {armed === "path" ? "Drawing path" : "Draw path"}
          </button>
          <button
            type="button"
            data-action="path-undo"
            disabled={path === null}
            title="Remove the latest point (a closed loop reopens)"
            onClick={paths.undoPoint}
          >
            Undo point
          </button>
          <button
            type="button"
            data-action="path-close"
            disabled={!paths.canClose || tracing}
            title="Join the last point back to the first"
            onClick={paths.close}
          >
            Close loop
          </button>
          <button
            type="button"
            data-action="path-clear"
            disabled={path === null}
            title="Remove the path"
            onClick={paths.clearPath}
          >
            Clear
          </button>
        </div>
        {armed === "path" ? (
          <p className="muted small">
            Click points on the selected mesh · click the first point again to close the loop · Esc stops.
          </p>
        ) : null}
        <p className="muted small">The path follows the distance method above.</p>
        {readout !== null ? (
          <p className="readout" data-path-readout>
            {readout}
          </p>
        ) : null}
        {tracing ? (
          <div className="op-row">
            <p className="muted busy">Tracing the path...</p>
            <button
              type="button"
              data-action="path-cancel"
              title="Stop tracing this leg; the path stays"
              onClick={paths.cancelTracing}
            >
              Cancel
            </button>
          </div>
        ) : null}

        <div className="op-row">
          <button
            type="button"
            data-action="path-trim-inside"
            disabled={!paths.canCut}
            title="Delete the outside of the loop"
            onClick={() => void paths.trim("inside")}
          >
            Keep inside
          </button>
          <button
            type="button"
            data-action="path-trim-outside"
            disabled={!paths.canCut}
            title="Delete the inside of the loop"
            onClick={() => void paths.trim("outside")}
          >
            Keep outside
          </button>
          <button
            type="button"
            data-action="path-split"
            disabled={!paths.canCut}
            title="Cut along the loop into two meshes: this one keeps the outside, a new one the inside"
            onClick={() => void paths.split()}
          >
            Split
          </button>
          <button
            type="button"
            data-action="path-keep"
            disabled={path === null || path.legs.length === 0}
            title="Keep the path, open or closed, as a polyline object of the scene"
            onClick={paths.keep}
          >
            Keep as object
          </button>
        </div>
        {closed ? (
          <p className="muted small">
            Inside is the side away from the open edges, or on a closed mesh the smaller side. Parts the loop is not on
            stay as they are. Each cut is one undo step.
          </p>
        ) : path !== null ? (
          <p className="muted small">Close the loop to trim or split along it.</p>
        ) : null}
        {locked && closed ? <p className="muted small">The mesh is locked: unlock it to cut.</p> : null}
        {busy ? <p className="muted busy">Cutting along the loop...</p> : null}
        {status !== null ? (
          <p className="report" data-path-status>
            {status}
          </p>
        ) : null}
        {error !== null ? (
          <p className="report error" data-path-error>
            {error}
          </p>
        ) : null}
      </div>
    </>
  );
}
