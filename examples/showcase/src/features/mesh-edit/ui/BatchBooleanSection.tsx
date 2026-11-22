// The Modeling panel's multi-mesh boolean section (#254): the shift-clicked
// meshes (primary first), union all / primary minus others / intersect all,
// the backend, Keep inputs, and the run. What it does lives in
// `ux/use-batch-boolean.ts`.
import type { BooleanMethod } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import type { AppState, Item } from "../../../app/state/app-state";
import { BATCH_BOOLEAN_METHODS, BATCH_BOOLEAN_OPERATIONS } from "../ux/batch-boolean";
import { useBatchBoolean } from "../ux/use-batch-boolean";

export function BatchBooleanSection(props: { app: AppState; store: SceneStore; items: readonly Item[] }) {
  const b = useBatchBoolean(props);
  const busy = b.busy !== null;

  return (
    <div className="op" data-batch-boolean-section>
      <h3>Multi-mesh Boolean</h3>
      {b.inputs.length === 0 ? (
        <p className="muted">Shift-click meshes to combine; the first one is the primary.</p>
      ) : (
        <ol className="muted" data-batch-boolean-inputs aria-label="Multi-mesh boolean inputs">
          {b.inputs.map((it, idx) => (
            <li key={it.id}>
              {it.name}
              {idx === 0 ? " (primary)" : ""}
            </li>
          ))}
        </ol>
      )}
      <div className="modes" role="group" aria-label="Multi-mesh boolean operation">
        {BATCH_BOOLEAN_OPERATIONS.map(([op, label]) => (
          <button
            key={op}
            type="button"
            data-batch-boolean-op={op}
            aria-pressed={b.op === op}
            disabled={busy}
            onClick={() => b.setOp(op)}
          >
            {label}
          </button>
        ))}
      </div>
      <div className="op-row">
        <label className="field">
          method
          <select
            aria-label="Multi-mesh boolean method"
            data-batch-boolean-method
            value={b.method}
            disabled={busy}
            onChange={(e) => b.setMethod(e.currentTarget.value as BooleanMethod)}
          >
            {BATCH_BOOLEAN_METHODS.map(([method, label]) => (
              <option key={method} value={method}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <label className="check">
          <input
            type="checkbox"
            data-batch-boolean-keep
            checked={b.keepInputs}
            disabled={busy}
            onChange={(e) => b.setKeepInputs(e.currentTarget.checked)}
          />
          Keep inputs
        </label>
      </div>
      {b.blocked !== null && b.inputs.length > 0 ? <p className="muted">{b.blocked}</p> : null}
      <button
        type="button"
        data-batch-boolean-run
        disabled={busy || b.blocked !== null}
        title={b.keepInputs ? "The result is a new mesh" : "The primary takes the result; the others are emptied (one undo step)"}
        onClick={() => void b.run()}
      >
        {b.blocked === null ? `Run on ${b.inputs.length} meshes` : "Run Multi Boolean"}
      </button>
      {b.busy !== null ? <p className="muted busy">{b.busy}...</p> : null}
      {b.report !== null ? (
        <p className={b.report.error ? "report error" : "report"} data-batch-boolean-report>
          <strong>{b.report.label}:</strong> {b.report.text}
        </p>
      ) : null}
    </div>
  );
}
