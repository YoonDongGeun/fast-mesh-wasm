// The polyline panel (#268), on the Mesh Edit and Inspect tabs: the selected
// polylines' name and colour, and the tools that work on them. What the
// tools do is in `ux/use-polyline-ops.ts` and `ux/polyline-actions.ts`.
import type { AppState } from "../../../app/state/app-state";
import { LengthInput } from "../../../shared/ui/LengthInput";
import type { ExtrudeSide } from "../ux/extrude-curves";
import type { OffsetCorners } from "../ux/offset";
import { EXPORT_PLANES, type ExportPlane } from "../ux/polyline-export";
import { cssColor, parseColor } from "../ux/polyline-list";
import { TUBE_SIDES } from "../ux/tube";
import { type PolylineLengths, usePolylineOps } from "../ux/use-polyline-ops";
import "./polylines.css";

export function PolylinePanel(props: { app: AppState }) {
  const ops = usePolylineOps(props.app);
  const { polylines, active, unit } = ops;
  const none = active.length === 0;

  const length = (key: keyof PolylineLengths, label: string, options: { min?: number; title?: string } = {}) => (
    <label className="field" title={options.title}>
      {label} ({unit})
      <LengthInput
        aria-label={`Polyline ${label}`}
        data-polyline-length-input={key}
        min={options.min}
        mm={ops.lengths[key]}
        unit={unit}
        disabled={none}
        onMm={(mm) => ops.setLength(key, mm)}
      />
    </label>
  );
  const button = (action: string, label: string, run: () => void, title: string, disabled = none) => (
    <button type="button" data-polyline-action={action} disabled={disabled} title={title} onClick={run}>
      {label}
    </button>
  );

  return (
    <section className="panel polyline-panel" aria-label="Polylines" data-panel="polylines" tabIndex={-1}>
      <header className="panel-head">
        <h2>Polylines</h2>
        <span className="muted">
          {none ? `${polylines.length} kept` : `${active.length} of ${polylines.length} selected`}
        </span>
      </header>
      {polylines.length === 0 ? (
        <p className="muted small">
          Keep section contours or a surface path as objects (Inspect) to simplify, offset and export them, or to make
          tubes and solids of them.
        </p>
      ) : (
        <>
          {none ? (
            <p className="muted small">Select polylines in the mesh list (shift-click adds).</p>
          ) : (
            <div className="readout" data-polyline-readout>
              {active.length === 1 ? (
                <div className="op-row">
                  <input
                    type="text"
                    aria-label="Polyline name"
                    data-polyline-name
                    value={active[0].name}
                    onChange={(e) => ops.setDisplay({ name: e.currentTarget.value })}
                  />
                </div>
              ) : null}
              {ops.readout.map((line) => (
                <p key={line}>{line}</p>
              ))}
            </div>
          )}
          <div className="op-row">
            <label className="field">
              colour
              <input
                type="color"
                aria-label="Polyline colour"
                data-polyline-color
                disabled={none}
                value={cssColor(active[0]?.color ?? 0xffffff)}
                onChange={(e) => {
                  const color = parseColor(e.currentTarget.value);
                  if (color !== null) {
                    ops.setDisplay({ color });
                  }
                }}
              />
            </label>
            {button("select-all", "Select all", ops.selectAll, "Select every polyline", false)}
            {button("remove", "Remove", ops.remove, "Remove the selected polylines (one undo step)")}
          </div>

          <div className="op" data-polyline-section="simplify">
            <h3>Simplify</h3>
            <div className="op-row">
              {length("tolerance", "tolerance", { min: 0, title: "How far the curve may move from a point it drops" })}
              {button("simplify", "Simplify", ops.simplify, "Drop the points the curve does without within the tolerance")}
            </div>
          </div>

          <div className="op" data-polyline-section="resample">
            <h3>Smooth &amp; Resample</h3>
            <div className="op-row">
              {length("spacing", "spacing", { min: 0, title: "The distance between neighbouring points along the curve" })}
              {length("smoothing", "smooth under", {
                min: 0,
                title: "Detail shorter than this is smoothed away; 0 keeps every point on the curve",
              })}
              {button("resample", "Resample", ops.resample, "Evenly spaced points of a spline through the (smoothed) curve")}
            </div>
          </div>

          <div className="op" data-polyline-section="offset">
            <h3>Offset</h3>
            <div className="op-row">
              {length("offset", "distance", {
                title: "Outward (an open curve: to its left, as its plane is faced); negative goes the other way",
              })}
              <label className="field">
                corners
                <select
                  aria-label="Offset corners"
                  data-polyline-corners
                  value={ops.corners}
                  disabled={none}
                  onChange={(e) => ops.setCorners(e.currentTarget.value as OffsetCorners)}
                >
                  <option value="round">Round</option>
                  <option value="sharp">Sharp</option>
                </select>
              </label>
              {button("offset", "Offset", ops.offset, "Add the curves at this distance from a planar curve")}
            </div>
          </div>

          <div className="op" data-polyline-section="tube">
            <h3>Tube</h3>
            <div className="op-row">
              {length("radius", "radius", { min: 0 })}
              <label className="field">
                sides
                <input
                  type="number"
                  aria-label="Tube sides"
                  data-polyline-sides
                  min={TUBE_SIDES.min}
                  max={TUBE_SIDES.max}
                  step={1}
                  value={ops.sides}
                  disabled={none}
                  onChange={(e) => ops.setSides(Math.round(Number(e.currentTarget.value)))}
                />
              </label>
              {button("tube", "Make Tube", ops.tube, "Sweep a circle along each selected polyline into a closed mesh")}
            </div>
          </div>

          <div className="op" data-polyline-section="extrude">
            <h3>Extrude</h3>
            <div className="op-row">
              {length("depth", "depth", { title: "Along the plane's normal; negative goes the other way" })}
              <label className="field">
                side
                <select
                  aria-label="Extrude side"
                  data-polyline-side
                  value={ops.side}
                  disabled={none}
                  onChange={(e) => ops.setSide(e.currentTarget.value as ExtrudeSide)}
                >
                  <option value="one">One way</option>
                  <option value="both">Both ways</option>
                </select>
              </label>
              {button(
                "extrude",
                "Extrude",
                ops.extrude,
                "A solid of the closed planar polylines selected: one inside another is a hole",
              )}
            </div>
          </div>

          <div className="op" data-polyline-section="export">
            <h3>Export</h3>
            <div className="op-row">
              <label className="field">
                plane
                <select
                  aria-label="Export plane"
                  data-polyline-plane
                  value={ops.plane}
                  onChange={(e) => ops.setPlane(e.currentTarget.value as ExportPlane)}
                >
                  {EXPORT_PLANES.map((p) => (
                    <option key={p.plane} value={p.plane}>
                      {p.label}
                    </option>
                  ))}
                </select>
              </label>
              {button("export-svg", "SVG", () => ops.exportAs("svg"), "Download as SVG, in millimetres", false)}
              {button("export-dxf", "DXF", () => ops.exportAs("dxf"), "Download as DXF, in millimetres", false)}
            </div>
            {none ? <p className="muted small">With none selected, every visible polyline is exported.</p> : null}
          </div>
        </>
      )}
      {ops.report !== null ? (
        <p className={ops.report.error ? "report error" : "report"} data-polyline-report role="status">
          {ops.report.text}
        </p>
      ) : null}
    </section>
  );
}
