// The Create dialog (#259): shape buttons, the shape's parameters in the
// display unit, name, centre and rotation, and Create. It floats over the
// viewport without a scrim, so the live preview (CreatePreviewLayer) stays
// in view while the fields change. Opened from the Home shelf and the
// command palette; Esc inside it, Cancel or × close it.
import type { PrimitiveShapeKind } from "fast-mesh-wasm";
import type { AppState } from "../../../app/state/app-state";
import { LengthInput } from "../../../shared/ui/LengthInput";
import { useCreateDialog } from "../ux/create-dialog-state";
import { CREATE_SHAPES } from "../ux/create-primitive";
import { useCreatePrimitive } from "../ux/use-create-primitive";
import "./create-dialog.css";

const AXES = ["X", "Y", "Z"] as const;

export function CreatePrimitiveDialog(props: { app: AppState }) {
  const { app } = props;
  const open = useCreateDialog(app, (s) => s.open);
  const session = useCreateDialog(app, (s) => s.session);
  const kind = useCreateDialog(app, (s) => s.kind);
  if (!open) {
    return null;
  }
  // A fresh form per opening.
  return <CreateForm key={session} app={app} initialKind={kind} />;
}

function CreateForm(props: { app: AppState; initialKind: PrimitiveShapeKind }) {
  const f = useCreatePrimitive(props.app, props.initialKind);
  const status = f.problem ?? f.error;
  return (
    <section
      className="create-dialog"
      role="dialog"
      aria-modal="false"
      aria-label="Create shape"
      data-panel="create-primitive"
      onKeyDown={(e) => {
        if (e.key === "Escape") {
          e.stopPropagation();
          f.cancel();
        }
      }}
    >
      <header className="panel-head">
        <h2>Create shape</h2>
        <button type="button" className="icon" aria-label="Close the Create dialog" onClick={f.cancel}>
          ×
        </button>
      </header>
      <div className="create-shapes" role="group" aria-label="Shape">
        {CREATE_SHAPES.map((s) => (
          <button
            key={s.kind}
            type="button"
            data-create-shape={s.kind}
            aria-pressed={s.kind === f.kind}
            onClick={() => f.setKind(s.kind)}
          >
            {s.label}
          </button>
        ))}
      </div>
      <div className="create-fields" aria-label={`${f.spec.label} parameters`}>
        {f.spec.params.map((p) => (
          <label key={`${f.kind}:${p.key}`} className="field">
            <span>
              {p.label}
              {p.type === "length" ? ` (${f.units})` : ""}
            </span>
            {p.type === "length" ? (
              <LengthInput
                data-create-param={p.key}
                mm={f.params[p.key]}
                unit={f.units}
                min={0}
                onMm={(mm) => f.setParam(p.key, mm)}
              />
            ) : (
              <input
                type="number"
                data-create-param={p.key}
                min={p.min}
                max={p.max}
                step={1}
                value={f.params[p.key]}
                onChange={(e) => f.setParam(p.key, Number(e.currentTarget.value))}
              />
            )}
          </label>
        ))}
      </div>
      <div className="create-place">
        <label className="field create-name">
          <span>Name</span>
          <input type="text" data-create-name value={f.name} onChange={(e) => f.setName(e.currentTarget.value)} />
        </label>
        <div className="create-vec" role="group" aria-label={`Centre (${f.units})`}>
          <span>Centre ({f.units})</span>
          {AXES.map((a, k) => (
            <LengthInput
              key={a}
              aria-label={`Centre ${a}`}
              data-create-centre={a.toLowerCase()}
              mm={f.centre[k]}
              unit={f.units}
              onMm={(mm) => f.setCentre(k, mm)}
            />
          ))}
        </div>
        <div className="create-vec" role="group" aria-label="Rotation (degrees)">
          <span>Rotation (°)</span>
          {AXES.map((a, k) => (
            <input
              key={a}
              type="number"
              step="any"
              aria-label={`Rotation ${a}`}
              data-create-rotation={a.toLowerCase()}
              value={f.rotationDeg[k]}
              onChange={(e) => f.setRotation(k, Number(e.currentTarget.value) || 0)}
            />
          ))}
        </div>
      </div>
      <p className={status === null ? "muted small" : "create-problem small"} data-create-status role="status">
        {status ?? `${f.triangles.toLocaleString("en")} triangles${f.pending ? " · updating the preview…" : ""}`}
      </p>
      <div className="op-row">
        <button
          type="button"
          className="primary"
          data-action="create-primitive-confirm"
          disabled={f.problem !== null || f.busy}
          onClick={() => void f.create()}
        >
          {f.busy ? "Creating…" : "Create"}
        </button>
        <button type="button" data-action="create-primitive-reset" onClick={f.resetParams}>
          Reset values
        </button>
        <button type="button" data-action="create-primitive-cancel" onClick={f.cancel}>
          Cancel
        </button>
      </div>
    </section>
  );
}
