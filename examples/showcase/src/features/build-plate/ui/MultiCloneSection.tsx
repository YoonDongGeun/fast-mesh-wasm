// The Build Plate panel's Multi-Clone section (#266): copies of the selected
// mesh in a grid on the floor or in a line along an axis.
import { useScene, useSelection } from "fast-mesh-wasm/react";
import { type AppState, useApp } from "../../../app/state/app-state";
import { LengthInput } from "../../../shared/ui/LengthInput";
import type { SignedAxis } from "../../transform/ux/placement-math";
import { type CloneLayout, cloneColumns, MAX_COPIES } from "../ux/multi-clone";
import { useMultiClone } from "../ux/use-multi-clone";

const AXES: readonly SignedAxis[] = ["+x", "-x", "+y", "-y", "+z", "-z"];

export function MultiCloneSection(props: { app: AppState }) {
  const { app } = props;
  const store = useScene();
  const [selection] = useSelection();
  const items = useApp(app, (s) => s.items);
  const selectedItem = selection === null ? null : (items.find((i) => i.mesh === selection) ?? null);
  const c = useMultiClone(app, store, selectedItem);
  const { layout } = c;

  return (
    <div className="multi-clone" data-panel="multi-clone">
      <h3>Multi-Clone</h3>
      <div className="build-plate-fields">
        <label className="field">
          <span>Pattern</span>
          <select
            data-multi-clone="pattern"
            value={layout.pattern}
            onChange={(e) => c.set({ pattern: e.currentTarget.value as CloneLayout["pattern"] })}
          >
            <option value="grid">Grid</option>
            <option value="linear">Linear</option>
          </select>
        </label>
        <label className="field" title="New meshes to make; the selected one keeps the first place">
          <span>Copies</span>
          <input
            type="number"
            data-multi-clone="copies"
            min={1}
            max={MAX_COPIES}
            step={1}
            value={layout.copies}
            onChange={(e) => c.set({ copies: Number(e.currentTarget.value) })}
          />
        </label>
        {layout.pattern === "grid" ? (
          <label className="field" title="Places per row along X (clear it for a near-square grid)">
            <span>Columns</span>
            <input
              type="number"
              data-multi-clone="columns"
              min={1}
              step={1}
              value={layout.columns ?? ""}
              placeholder={String(cloneColumns(layout) || 1)}
              onChange={(e) => c.set({ columns: e.currentTarget.value === "" ? null : Number(e.currentTarget.value) })}
            />
          </label>
        ) : (
          <label className="field">
            <span>Along</span>
            <select
              data-multi-clone="axis"
              value={layout.axis}
              onChange={(e) => c.set({ axis: e.currentTarget.value as SignedAxis })}
            >
              {AXES.map((a) => (
                <option key={a} value={a}>
                  {a.toUpperCase()}
                </option>
              ))}
            </select>
          </label>
        )}
        <label className="field" title="The gap between neighbours">
          <span>Spacing ({c.units})</span>
          <LengthInput
            data-multi-clone="spacing"
            min={0}
            mm={layout.spacing}
            unit={c.units}
            onMm={(spacing) => c.set({ spacing })}
          />
        </label>
      </div>
      <div className="op-row">
        <button
          type="button"
          data-multi-clone-action="run"
          disabled={!c.ready}
          title="Add the copies as new meshes (one undo step)"
          onClick={() => void c.run()}
        >
          {c.busy ? "Copying…" : "Make Copies"}
        </button>
      </div>
      {selectedItem === null ? <p className="muted small">Select the mesh to copy.</p> : null}
      {c.problem !== null ? (
        <p className="warn small" data-multi-clone-problem>
          {c.problem}
        </p>
      ) : null}
      {c.report !== null ? (
        <p className={c.report.error ? "report error" : "report"} data-multi-clone-report role="status">
          {c.report.text}
        </p>
      ) : null}
    </div>
  );
}
