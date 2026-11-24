// The Modeling panel's lattice section (#265), next to Hollow Mesh: the
// lattice type, cell size, wall thickness, optional shell and what is filled,
// the grid, triangles and memory the settings take, a preview on a coarse
// grid, and the fill of the active mesh as one undo step. What it does lives
// in `ux/use-lattice.ts`.
import type { LatticeRegion } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import type { AppState, Item } from "../../../app/state/app-state";
import { LengthInput } from "../../../shared/ui/LengthInput";
import type { LengthUnit } from "../../../shared/ux/units";
import { describeLatticePlan, LATTICE_KINDS } from "../ux/lattice";
import { useLattice } from "../ux/use-lattice";

export function LatticeSection(props: {
  app: AppState;
  store: SceneStore;
  /** The selected mesh, else the first one. */
  activeItem: Item | null;
  units: LengthUnit;
  /** Whether another operation runs. */
  disabled: boolean;
}) {
  const { activeItem, units } = props;
  const l = useLattice(props);
  const { settings, lengths } = l;
  const disabled = props.disabled || l.busy || activeItem === null;

  return (
    <div className="op" data-lattice-section>
      <h3>Lattice Infill</h3>
      <p className="muted small">
        Fills {activeItem?.name ?? "a closed mesh"}, or the cavity of a hollowed one, with thin walls.
      </p>
      <div className="modes" role="group" aria-label="Lattice type">
        {LATTICE_KINDS.map(([id, label, title]) => (
          <button
            key={id}
            type="button"
            data-lattice-kind={id}
            aria-pressed={settings.kind === id}
            disabled={disabled}
            title={title}
            onClick={() => l.update({ kind: id })}
          >
            {label}
          </button>
        ))}
      </div>
      <div className="op-row">
        <label className="field" title="The edge length of one cell">
          cell ({units})
          <LengthInput
            aria-label="Lattice cell size"
            data-lattice-cell
            min={0}
            mm={lengths.cell}
            unit={units}
            disabled={disabled}
            onMm={(mm) => l.update({ cell: mm > 0 ? mm : null })}
          />
        </label>
        <label className="field" title="At most a third of the cell size">
          wall ({units})
          <LengthInput
            aria-label="Lattice wall thickness"
            data-lattice-wall
            min={0}
            mm={lengths.wall}
            unit={units}
            disabled={disabled}
            onMm={(mm) => l.update({ wall: mm > 0 ? mm : null })}
          />
        </label>
        <button
          type="button"
          data-lattice-auto
          aria-pressed={settings.cell === null && settings.wall === null}
          disabled={disabled}
          title="A sixth of the part per cell, and 15% of the cell per wall"
          onClick={() => l.update({ cell: null, wall: null })}
        >
          Auto
        </button>
      </div>
      <div className="op-row">
        <label className="check" title="Keep the mesh's surface and a solid layer inside it">
          <input
            type="checkbox"
            data-lattice-shell
            checked={settings.shell}
            disabled={disabled}
            onChange={(e) => l.update({ shell: e.currentTarget.checked })}
          />
          Shell
        </label>
        {settings.shell ? (
          <label className="field" title="How thick the solid layer is; in a cavity, on top of its wall">
            thickness ({units})
            <LengthInput
              aria-label="Lattice shell thickness"
              data-lattice-shell-thickness
              min={0}
              mm={lengths.shell}
              unit={units}
              disabled={disabled}
              onMm={(mm) => l.update({ shellThickness: mm > 0 ? mm : null })}
            />
          </label>
        ) : null}
      </div>
      <div className="op-row">
        <label className="field">
          fill
          <select
            aria-label="Lattice region"
            data-lattice-region
            value={settings.region}
            disabled={disabled}
            onChange={(e) => l.update({ region: e.currentTarget.value as LatticeRegion })}
          >
            <option value="auto">Auto (cavity if there is one)</option>
            <option value="inside">Inside</option>
            <option value="cavity">Cavity</option>
          </select>
        </label>
        <label className="field" title="The grid's spacing: at least two voxels to a wall">
          voxel ({units})
          <LengthInput
            aria-label="Lattice voxel size"
            data-lattice-voxel
            min={0}
            mm={l.voxel}
            unit={units}
            disabled={disabled}
            onMm={(mm) => l.update({ voxel: mm > 0 ? mm : null })}
          />
        </label>
        <button
          type="button"
          data-lattice-voxel-auto
          aria-pressed={settings.voxel === null}
          disabled={disabled}
          title="2.5 voxels to a wall"
          onClick={() => l.update({ voxel: null })}
        >
          Auto
        </button>
      </div>
      {l.planError !== null ? (
        <p className={l.planError.error ? "report error" : "muted small"} data-lattice-plan role="status">
          {l.planError.text}
        </p>
      ) : l.plan !== null ? (
        <p className={l.plan.fits ? "muted small" : "report error"} data-lattice-plan>
          {describeLatticePlan(l.plan, units)}
        </p>
      ) : null}
      <div className="op-row">
        <button
          type="button"
          data-lattice-preview
          aria-pressed={l.previewing}
          disabled={disabled}
          title="Show the walls on a coarse grid, in the see-through mesh"
          onClick={() => l.setPreviewing(!l.previewing)}
        >
          Preview
        </button>
        <button
          type="button"
          className="primary"
          data-lattice-run
          disabled={disabled || l.blocked}
          title="Fill the mesh with the lattice (one undo step)"
          onClick={() => void l.run()}
        >
          Fill with Lattice
        </button>
      </div>
      {l.previewBusy ? (
        <p className="busy small" data-lattice-preview-busy>
          Preview: computing…
        </p>
      ) : l.previewLine !== null ? (
        <p className={l.previewLine.error ? "report error" : "muted small"} data-lattice-preview-line role="status">
          {l.previewLine.text}
        </p>
      ) : null}
      {l.busy ? (
        <p className="busy small" data-lattice-busy>
          Lattice: computing… <span className="progress" aria-hidden="true" />
        </p>
      ) : l.report !== null ? (
        <p className={l.report.error ? "report error" : "report"} data-lattice-report role="status">
          {l.report.text}
        </p>
      ) : null}
    </div>
  );
}
