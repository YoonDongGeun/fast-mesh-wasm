// Numeric Transform & Scale panel (#195): unit-aware Position (X, Y, Z),
// Euler Rotation (X, Y, Z) in degrees, Scale gizmo mode (S), and undoable
// local-space mesh scaling synced live with selection placement and edits.
import type { MeshHandle } from "fast-mesh-wasm";
import { useScene, useSelection } from "fast-mesh-wasm/react";
import { useEffect, useRef, useState } from "react";
import type { GizmoMode } from "../viewport/Viewport";
import { type AppState, useApp } from "../scene/app-state";
import {
  applyMeshScale,
  computeLocalExtents,
  placementToValues,
  valuesToPlacement,
} from "../scene/transform-math";
import { LengthInput } from "./LengthInput";

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

  const [pos, setPos] = useState<[number, number, number]>([0, 0, 0]);
  const [rotDeg, setRotDeg] = useState<[number, number, number]>([0, 0, 0]);
  const [scaleVal, setScaleVal] = useState<[number, number, number]>([1, 1, 1]);
  const [uniformScale, setUniformScale] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Baseline extents per mesh handle so displayed scale stays consistent across undo/redo.
  const baselineExtents = useRef(new Map<MeshHandle, [number, number, number]>());

  useEffect(() => {
    if (selectedItem === null) {
      setPos([0, 0, 0]);
      setRotDeg([0, 0, 0]);
      setScaleVal([1, 1, 1]);
      return;
    }
    const mesh = selectedItem.mesh;
    const syncFromScene = () => {
      try {
        const p = store.scene.placement(mesh);
        const vals = placementToValues(p);
        setPos(vals.position);
        setRotDeg([
          Number(vals.rotationDeg[0].toFixed(3)),
          Number(vals.rotationDeg[1].toFixed(3)),
          Number(vals.rotationDeg[2].toFixed(3)),
        ]);

        const v = store.scene.views(mesh);
        const { extents } = computeLocalExtents(v.positions(), v.indices());
        let base = baselineExtents.current.get(mesh);
        if (base === undefined) {
          base = [extents[0] || 1, extents[1] || 1, extents[2] || 1];
          baselineExtents.current.set(mesh, base);
        }
        setScaleVal([
          Number((base[0] > 0 ? extents[0] / base[0] : 1).toFixed(4)),
          Number((base[1] > 0 ? extents[1] / base[1] : 1).toFixed(4)),
          Number((base[2] > 0 ? extents[2] / base[2] : 1).toFixed(4)),
        ]);
      } catch {
        // Mesh may have been removed mid-frame
      }
    };
    syncFromScene();
    const unsubChanges = store.scene.onChanges((changes) => {
      if (changes.some((c) => c.mesh === mesh)) {
        syncFromScene();
      }
    });
    const unsubStore = store.subscribe(syncFromScene);
    return () => {
      unsubChanges();
      unsubStore();
    };
  }, [selectedItem, store]);

  const commitPlacement = (nextPos: [number, number, number], nextRotDeg: [number, number, number]) => {
    if (selectedItem === null) return;
    setError(null);
    store.setPlacement(selectedItem.mesh, valuesToPlacement({ position: nextPos, rotationDeg: nextRotDeg }));
    setPos(nextPos);
    setRotDeg(nextRotDeg);
  };

  const commitScale = async (targetScale: [number, number, number]) => {
    if (selectedItem === null) return;
    if (targetScale[0] <= 0 || targetScale[1] <= 0 || targetScale[2] <= 0) {
      setError("Scale factors must be positive");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const mesh = selectedItem.mesh;
      const v = store.scene.views(mesh);
      const { extents } = computeLocalExtents(v.positions(), v.indices());
      const base = baselineExtents.current.get(mesh) ?? [extents[0] || 1, extents[1] || 1, extents[2] || 1];
      baselineExtents.current.set(mesh, base);

      const cur: [number, number, number] = [
        base[0] > 0 ? extents[0] / base[0] : 1,
        base[1] > 0 ? extents[1] / base[1] : 1,
        base[2] > 0 ? extents[2] / base[2] : 1,
      ];
      const delta: [number, number, number] = [
        cur[0] > 0 ? targetScale[0] / cur[0] : targetScale[0],
        cur[1] > 0 ? targetScale[1] / cur[1] : targetScale[1],
        cur[2] > 0 ? targetScale[2] / cur[2] : targetScale[2],
      ];
      await applyMeshScale(store, mesh, delta);
      setScaleVal(targetScale);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

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
                    onMm={(mm) => {
                      const next: [number, number, number] = [...pos];
                      next[idx] = mm;
                      commitPlacement(next, rotDeg);
                    }}
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
                    onChange={(e) => {
                      const val = Number(e.currentTarget.value);
                      if (!Number.isFinite(val)) return;
                      const next: [number, number, number] = [...rotDeg];
                      next[idx] = val;
                      commitPlacement(pos, next);
                    }}
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
                  onChange={(e) => setUniformScale(e.currentTarget.checked)}
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
                    onChange={(e) => {
                      const val = Number(e.currentTarget.value);
                      if (!Number.isFinite(val)) return;
                      if (uniformScale) {
                        setScaleVal([val, val, val]);
                      } else {
                        const next: [number, number, number] = [...scaleVal];
                        next[idx] = val;
                        setScaleVal(next);
                      }
                    }}
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
