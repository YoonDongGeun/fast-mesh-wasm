// Numeric Transform & Scale panel (#195): unit-aware Position (X, Y, Z),
// Euler Rotation (X, Y, Z) in degrees, Scale gizmo mode (S), and undoable
// local-space mesh scaling synced live with selection placement and edits.
import { useScene, useSelection } from "fast-mesh-wasm/react";
import type { GizmoMode } from "../../../viewport/ui/Viewport";
import { type AppState, useApp } from "../../../app/state/app-state";
import { useTransformFields } from "../ux/use-transform-fields";
import { LengthInput } from "../../../shared/ui/LengthInput";

export function TransformPanel(props: {
  app: AppState;
  mode: GizmoMode;
  onMode: (mode: GizmoMode) => void;
}) {
  const { app, mode, onMode } = props;
  const store = useScene();
  const [selection] = useSelection();
  const items = useApp(app, (s) => s.items);
  const units = useApp(app, (s) => s.units);

  const selectedItem = selection === null ? null : (items.find((i) => i.mesh === selection) ?? null);

  const fields = useTransformFields(store, selectedItem);
  const { pos, rotDeg, scaleVal, uniformScale, busy, error, commitScale } = fields;

  return (
    <section className="panel transform-panel" aria-label="Transform" data-panel="transform">
      <header className="panel-head">
        <h2>Transform</h2>
        <button
          type="button"
          data-transform-mode-scale
          aria-pressed={mode === "scale"}
          onClick={() => onMode(mode === "scale" ? "translate" : "scale")}
          title="Scale gizmo (S)"
        >
          Scale (S)
        </button>
      </header>

      {selectedItem === null ? (
        <p className="muted">Select a mesh to inspect or edit position, rotation, and scale.</p>
      ) : (
        <>
          <div className="op">
            <h3>Position ({units})</h3>
            <div className="transform-grid">
              {(["x", "y", "z"] as const).map((axis, idx) => (
                <label key={axis} className="field">
                  {axis.toUpperCase()}
                  <LengthInput
                    aria-label={`Position ${axis.toUpperCase()}`}
                    data-transform-pos={axis}
                    mm={pos[idx]}
                    unit={units}
                    onMm={(mm) => fields.setPosAxis(idx, mm)}
                  />
                </label>
              ))}
            </div>
          </div>

          <div className="op">
            <h3>Rotation (deg)</h3>
            <div className="transform-grid">
              {(["x", "y", "z"] as const).map((axis, idx) => (
                <label key={axis} className="field">
                  {axis.toUpperCase()}
                  <input
                    type="number"
                    step={1}
                    aria-label={`Rotation ${axis.toUpperCase()} degrees`}
                    data-transform-rot={axis}
                    value={rotDeg[idx]}
                    onChange={(e) => fields.setRotAxis(idx, Number(e.currentTarget.value))}
                  />
                </label>
              ))}
            </div>
          </div>

          <div className="op">
            <div className="op-row">
              <h3>Scale (×)</h3>
              <label className="check">
                <input
                  type="checkbox"
                  data-transform-scale-uniform
                  checked={uniformScale}
                  disabled={busy}
                  onChange={(e) => fields.setUniformScale(e.currentTarget.checked)}
                />
                Uniform
              </label>
            </div>
            <div className="transform-grid">
              {(["x", "y", "z"] as const).map((axis, idx) => (
                <label key={axis} className="field">
                  {axis.toUpperCase()}
                  <input
                    type="number"
                    min={0.001}
                    step={0.1}
                    aria-label={`Scale ${axis.toUpperCase()}`}
                    data-transform-scale={axis}
                    value={scaleVal[idx]}
                    disabled={busy}
                    onChange={(e) => fields.editScale(idx, Number(e.currentTarget.value))}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        void commitScale(scaleVal);
                      }
                    }}
                  />
                </label>
              ))}
            </div>
            <div className="op-row">
              <button
                type="button"
                data-transform-scale-apply
                disabled={busy || scaleVal.some((v) => !(v > 0))}
                onClick={() => void commitScale(scaleVal)}
              >
                {busy ? "Scaling..." : "Apply Scale"}
              </button>
              <button
                type="button"
                data-transform-scale-reset
                disabled={busy}
                onClick={() => void commitScale([1, 1, 1])}
              >
                Reset 1×
              </button>
            </div>
          </div>
        </>
      )}

      {error !== null ? <p className="report error">{error}</p> : null}
    </section>
  );
}
