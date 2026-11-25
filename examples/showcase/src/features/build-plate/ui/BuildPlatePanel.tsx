// The Home tab's Build Plate panel (#266): the plate's size, corner and up
// axis in the display unit, whether it shows, the spacing and options of
// Arrange, its scope, and its report; then Multi-Clone.
import { useScene } from "fast-mesh-wasm/react";
import type { AppState } from "../../../app/state/app-state";
import { LengthInput } from "../../../shared/ui/LengthInput";
import type { UpAxis } from "../../transform/ux/placement-math";
import { depthAxisName } from "../ux/plate";
import { useBuildPlate } from "../ux/use-build-plate";
import { MultiCloneSection } from "./MultiCloneSection";
import "./build-plate.css";

export function BuildPlatePanel(props: { app: AppState }) {
  const { app } = props;
  const store = useScene();
  const p = useBuildPlate(app, store);
  const { plate, units } = p;
  const depthAxis = depthAxisName(plate.up);
  const corner = `The plate's corner with the lowest X and ${depthAxis}`;

  return (
    <section className="panel build-plate-panel" aria-label="Build plate" data-panel="build-plate">
      <header className="panel-head">
        <h2>Build Plate</h2>
        <label className="check" title="Draw the plate in the viewport">
          <input
            type="checkbox"
            data-build-plate="visible"
            checked={plate.visible}
            onChange={(e) => p.set({ visible: e.currentTarget.checked })}
          />
          Show
        </label>
      </header>
      <div className="build-plate-fields">
        <label className="field">
          <span>Width X ({units})</span>
          <LengthInput data-build-plate="width" min={0} mm={plate.width} unit={units} onMm={(width) => p.set({ width })} />
        </label>
        <label className="field">
          <span>
            Depth {depthAxis} ({units})
          </span>
          <LengthInput data-build-plate="depth" min={0} mm={plate.depth} unit={units} onMm={(depth) => p.set({ depth })} />
        </label>
        <label className="field" title={corner}>
          <span>Origin X ({units})</span>
          <LengthInput
            data-build-plate="origin-x"
            mm={plate.origin[0]}
            unit={units}
            onMm={(mm) => p.set({ origin: [mm, plate.origin[1]] })}
          />
        </label>
        <label className="field" title={corner}>
          <span>
            Origin {depthAxis} ({units})
          </span>
          <LengthInput
            data-build-plate="origin-y"
            mm={plate.origin[1]}
            unit={units}
            onMm={(mm) => p.set({ origin: [plate.origin[0], mm] })}
          />
        </label>
        <label className="field" title="The axis that points away from the plate, which lies where it is 0">
          <span>Up</span>
          <select
            data-build-plate="up"
            aria-label="Plate up axis"
            value={plate.up}
            onChange={(e) => p.set({ up: e.currentTarget.value as UpAxis })}
          >
            <option value="z">+Z</option>
            <option value="y">+Y</option>
          </select>
        </label>
        <label className="field" title="The least gap between two footprints">
          <span>Spacing ({units})</span>
          <LengthInput
            data-build-plate="spacing"
            min={0}
            mm={plate.spacing}
            unit={units}
            onMm={(spacing) => p.set({ spacing })}
          />
        </label>
      </div>
      <label className="check" title="Arrange may turn meshes 90° about the up axis">
        <input
          type="checkbox"
          data-build-plate="turn"
          checked={plate.turn}
          onChange={(e) => p.set({ turn: e.currentTarget.checked })}
        />
        Turn by 90° to fit
      </label>
      <label className="check" title="Arrange keeps off the visible meshes it does not move">
        <input
          type="checkbox"
          data-build-plate="keep-clear"
          checked={plate.keepClear}
          onChange={(e) => p.set({ keepClear: e.currentTarget.checked })}
        />
        Keep clear of other meshes
      </label>
      <div className="op-row">
        <select
          data-build-plate-scope
          aria-label="Meshes to arrange"
          value={p.scope}
          disabled={p.busy}
          onChange={(e) => p.setScope(e.currentTarget.value as "selected" | "all")}
        >
          <option value="selected">Selected meshes ({p.selected})</option>
          <option value="all">All visible meshes ({p.all})</option>
        </select>
        <button
          type="button"
          className="primary"
          data-build-plate-action="arrange"
          disabled={!p.ready}
          title="Pack the meshes' footprints on the plate and rest each on it (one undo step)"
          onClick={() => void p.arrange()}
        >
          {p.busy ? "Arranging…" : "Arrange"}
        </button>
      </div>
      {p.problem !== null ? <p className="warn small">{p.problem}</p> : null}
      {p.report !== null ? (
        <div className={p.report.error ? "report error" : "report"} data-build-plate-report role="status">
          {p.report.text}
          {p.report.left.map((line) => (
            <p key={line} className="warn small" data-build-plate-left>
              {line}
            </p>
          ))}
        </div>
      ) : null}
      <MultiCloneSection app={app} />
    </section>
  );
}
