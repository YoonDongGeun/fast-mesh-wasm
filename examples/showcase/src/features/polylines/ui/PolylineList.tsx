// The kept polylines in the mesh list (#268): each one's colour, name,
// length, visibility, focus and removal. A click on a name selects it for
// the polyline tools (shift-click adds). Nothing shows while there are none.
import { useScene } from "fast-mesh-wasm/react";
import { memo, useMemo } from "react";
import { type AppState, useApp } from "../../../app/state/app-state";
import { formatLength, type LengthUnit } from "../../../shared/ux/units";
import { curveLength, curveReadout, curvesSphere, type Polyline } from "../ux/polyline";
import { removePolylines } from "../ux/polyline-actions";
import { clickPolyline, cssColor } from "../ux/polyline-list";
import "./polylines.css";

type FocusBox = (box: { center: readonly [number, number, number]; radius: number }) => void;

export function PolylineList(props: { app: AppState; onFocusBox?: FocusBox }) {
  const { app, onFocusBox } = props;
  const polylines = useApp(app, (s) => s.polylines);
  const selection = useApp(app, (s) => s.polylineSelection);
  const unit = useApp(app, (s) => s.units);
  if (polylines.length === 0) {
    return null;
  }
  const shown = polylines.filter((p) => p.visible).length;
  return (
    <div className="polyline-list" data-polyline-list role="region" aria-label="Polylines">
      <div className="polyline-list-head">
        <p className="muted small">Polylines ({polylines.length})</p>
        <label className="check" title="Show or hide every polyline">
          <input
            type="checkbox"
            data-polyline-show-all
            checked={shown === polylines.length}
            onChange={(e) =>
              app.setPolylineDisplay(
                polylines.map((p) => p.id),
                { visible: e.currentTarget.checked },
              )
            }
          />
          All
        </label>
      </div>
      <ul className="polyline-rows">
        {polylines.map((p) => (
          <PolylineRow
            key={p.id}
            app={app}
            polyline={p}
            selected={selection.includes(p.id)}
            unit={unit}
            onFocusBox={onFocusBox}
          />
        ))}
      </ul>
    </div>
  );
}

/** Memoised: a long list renders again only where something changed. */
const PolylineRow = memo(function PolylineRow(props: {
  app: AppState;
  polyline: Polyline;
  selected: boolean;
  unit: LengthUnit;
  onFocusBox?: FocusBox;
}) {
  const { app, polyline, selected, unit, onFocusBox } = props;
  const store = useScene();
  const length = useMemo(() => curveLength(polyline), [polyline]);
  return (
    <li className={selected ? "polyline-row selected" : "polyline-row"} data-polyline={polyline.name}>
      <div className="mesh-head">
        <label className="check" title="Visible">
          <input
            type="checkbox"
            aria-label={`Show ${polyline.name}`}
            data-polyline-show={polyline.name}
            checked={polyline.visible}
            onChange={(e) => app.setPolylineDisplay([polyline.id], { visible: e.currentTarget.checked })}
          />
        </label>
        <span className="swatch" style={{ background: cssColor(polyline.color) }} aria-hidden="true" />
        <button
          type="button"
          className="mesh-name"
          aria-pressed={selected}
          title={`${polyline.name}: ${curveReadout(polyline, unit)} (click to select, shift-click adds)`}
          onClick={(e) => clickPolyline(app, polyline.id, e.shiftKey)}
        >
          {polyline.name}
        </button>
        <span className="count" data-polyline-length>
          {formatLength(length, unit)}
        </span>
      </div>
      <div className="mesh-controls">
        <span className="tree-node-tag">{polyline.closed ? "Closed" : "Open"}</span>
        <button
          type="button"
          title={`Point the camera at ${polyline.name}`}
          data-polyline-focus={polyline.name}
          onClick={() => {
            const sphere = curvesSphere([polyline]);
            if (sphere !== null) {
              onFocusBox?.(sphere);
            }
          }}
        >
          Focus
        </button>
        <button
          type="button"
          className="danger"
          title={`Remove ${polyline.name}`}
          data-polyline-remove={polyline.name}
          onClick={() => removePolylines(app, store, [polyline.id])}
        >
          Remove
        </button>
      </div>
    </li>
  );
});
