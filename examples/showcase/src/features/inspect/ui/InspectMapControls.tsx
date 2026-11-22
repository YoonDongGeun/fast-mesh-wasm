// The inspect maps section (#251): deviation from a reference mesh, overhangs
// against a build direction, and triangle quality, as a colour map on the
// selected mesh with its legend and readouts. Behaviour is in
// `ux/use-inspect-maps.ts`, the math and wording in `ux/inspect-maps.ts`.
import type { MeshHandle } from "fast-mesh-wasm";
import type { AppState, Item } from "../../../app/state/app-state";
import { LengthInput } from "../../../shared/ui/LengthInput";
import type { LengthUnit } from "../../../shared/ux/units";
import {
  type BuildDirectionSource,
  INSPECT_MAP_KINDS,
  inspectMapLegend,
  inspectMapReadoutLines,
  type QualityMapMetric,
  QUALITY_METRICS,
} from "../ux/inspect-maps";
import { useInspectMaps } from "../ux/use-inspect-maps";
import { FieldLegend } from "./FieldLegend";

/** A typed number, or `null` for an empty or non-finite one (the input keeps its value). */
function typedNumber(text: string): number | null {
  if (text.trim() === "") {
    return null;
  }
  const v = Number(text);
  return Number.isFinite(v) ? v : null;
}

export function InspectMapControls(props: {
  app: AppState;
  items: readonly Item[];
  selection: MeshHandle | null;
  unit: LengthUnit;
  /** Turns the surface field off before a map takes the overlay. */
  onTakeOverlay: () => void;
}) {
  const { app, items, selection, unit, onTakeOverlay } = props;
  const maps = useInspectMaps(app, items, selection, onTakeOverlay);
  const { settings, report, style, busy, error, target, reference } = maps;
  const nameOf = (m: MeshHandle | null) => (m === null ? null : (items.find((i) => i.mesh === m)?.name ?? null));
  const mapped = report?.request.mesh ?? null;
  const refExclude = mapped ?? target;
  const legend = report !== null && style !== null ? inspectMapLegend(report, style, unit) : null;
  const lines =
    report === null
      ? []
      : inspectMapReadoutLines(report, unit, report.kind === "deviation" ? nameOf(report.request.reference) : null);
  const needsReference = settings.kind === "deviation" && reference === null;

  return (
    <div className="op surface-field-panel" data-inspect-maps tabIndex={-1}>
      <h3>Inspect Maps</h3>
      <div className="modes" role="group" aria-label="Inspect map">
        {INSPECT_MAP_KINDS.map((k) => (
          <button
            key={k.kind}
            type="button"
            data-inspect-map-kind={k.kind}
            aria-pressed={settings.kind === k.kind}
            title={k.title}
            disabled={busy}
            onClick={() => maps.update({ kind: k.kind })}
          >
            {k.label}
          </button>
        ))}
      </div>

      {settings.kind === "deviation" ? (
        <>
          <div className="picker">
            <span>Reference</span>
            <select
              aria-label="Deviation reference mesh"
              data-inspect-map-reference
              value={reference === null ? "" : String(items.findIndex((i) => i.mesh === reference))}
              disabled={busy || reference === null}
              onChange={(e) => {
                const picked = items[Number(e.currentTarget.value)]?.mesh;
                if (picked !== undefined) {
                  maps.update({ reference: picked });
                }
              }}
            >
              {reference === null ? <option value="">No other mesh</option> : null}
              {items
                .filter((it) => it.mesh !== refExclude)
                .map((it) => (
                  <option key={it.id} value={String(items.indexOf(it))}>
                    {it.name}
                  </option>
                ))}
            </select>
          </div>
          <div className="op-row">
            <label className="check" title="Negative inside the reference, positive outside">
              <input
                type="checkbox"
                data-inspect-map-signed
                checked={settings.signed}
                disabled={busy}
                onChange={(e) => maps.update({ signed: e.currentTarget.checked })}
              />
              Signed
            </label>
            <label className="check" title="Show deviations within the tolerance in one colour">
              <input
                type="checkbox"
                data-inspect-map-tolerance-on
                checked={settings.toleranceOn}
                onChange={(e) => maps.update({ toleranceOn: e.currentTarget.checked })}
              />
              Tolerance
            </label>
            <label className="field">
              ± ({unit})
              <LengthInput
                aria-label="Deviation tolerance"
                data-inspect-map-tolerance
                min={0}
                mm={settings.tolerance}
                unit={unit}
                disabled={!settings.toleranceOn}
                onMm={(mm) => maps.update({ tolerance: Math.max(0, mm) })}
              />
            </label>
          </div>
        </>
      ) : null}

      {settings.kind === "overhang" ? (
        <div className="op-row">
          <label className="field">
            build
            <select
              aria-label="Build direction"
              data-inspect-map-build
              value={settings.buildFrom}
              disabled={busy}
              onChange={(e) => maps.update({ buildFrom: e.currentTarget.value as BuildDirectionSource })}
            >
              <option value="worldZ">World +Z</option>
              <option value="viewUp">View up</option>
            </select>
          </label>
          <label className="field" title="Faces tilted down past this angle from horizontal need support">
            critical (°)
            <input
              type="number"
              min={0}
              max={90}
              step={1}
              aria-label="Critical overhang angle"
              data-inspect-map-critical
              value={settings.criticalAngleDeg}
              disabled={busy}
              onChange={(e) => {
                const v = typedNumber(e.currentTarget.value);
                if (v !== null) {
                  maps.update({ criticalAngleDeg: Math.min(90, Math.max(0, v)) });
                }
              }}
            />
          </label>
        </div>
      ) : null}

      {settings.kind === "quality" ? (
        <div className="op-row">
          <label className="field">
            metric
            <select
              aria-label="Quality metric"
              data-inspect-map-metric
              value={settings.metric}
              disabled={busy}
              title={QUALITY_METRICS.find((m) => m.metric === settings.metric)?.title}
              onChange={(e) => maps.update({ metric: e.currentTarget.value as QualityMapMetric })}
            >
              {QUALITY_METRICS.map((m) => (
                <option key={m.metric} value={m.metric} title={m.title}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            threshold{settings.metric === "minAngleDeg" ? " (°)" : ""}
            <input
              type="number"
              min={settings.metric === "minAngleDeg" ? 0 : 1}
              max={settings.metric === "minAngleDeg" ? 60 : undefined}
              step="any"
              aria-label="Quality threshold"
              data-inspect-map-threshold
              value={settings.threshold}
              disabled={busy}
              onChange={(e) => {
                const v = typedNumber(e.currentTarget.value);
                if (v !== null && v >= 0) {
                  maps.update({ threshold: v });
                }
              }}
            />
          </label>
        </div>
      ) : null}

      <div className="op-row">
        <button
          type="button"
          data-action="inspect-map-show"
          disabled={busy || target === null || needsReference}
          title="Colour the selected mesh by this map"
          onClick={maps.show}
        >
          {busy ? "Computing..." : "Show map"}
        </button>
        <button
          type="button"
          data-action="inspect-map-clear"
          disabled={report === null && !busy}
          title="Remove the map from its mesh"
          onClick={maps.clear}
        >
          Clear
        </button>
      </div>

      {target === null && report === null ? <p className="muted small">Select a mesh to map.</p> : null}
      {needsReference && target !== null ? (
        <p className="muted small">Load a second mesh to measure the deviation against.</p>
      ) : null}

      {report !== null && legend !== null ? (
        <>
          <p className="muted small" data-inspect-map-on>
            {INSPECT_MAP_KINDS.find((k) => k.kind === report.kind)?.label} on {nameOf(report.request.mesh) ?? "a mesh"}
          </p>
          <FieldLegend map={legend.map} lo={legend.lo} hi={legend.hi} band={legend.band} />
          <div className="readout" data-inspect-map-readout={report.kind}>
            {lines.map((line) => (
              <p key={line}>{line}</p>
            ))}
          </div>
        </>
      ) : null}

      {error !== null ? <p className="report error">{error}</p> : null}
    </div>
  );
}
