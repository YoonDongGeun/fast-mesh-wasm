// Face Selection workbench panel (#233): interactive viewport tool selector,
// combine modes, region/brush parameters, selection modifiers (`grow`, `shrink`,
// `invert`, `fillSelectionHoles`, `smoothSelectionBoundary`, `selectSmallFaceIslands`,
// mask conversion), and selection-driven mesh operations (`extract`, `split`, `delete`).
import type { SceneStore } from "fast-mesh-wasm/store";
import type { AppState } from "../../../app/state/app-state";
import {
  ADJACENCY_OPTIONS,
  COMBINE_OPTIONS,
  TOOL_OPTIONS,
  useFaceSelectionActions,
} from "../ux/use-face-selection-actions";

export interface SelectPanelProps {
  app: AppState;
  store: SceneStore;
}

/** Control panel for Wave 6 interactive face selection and face-set mesh editing. */
export function SelectPanel({ app, store }: SelectPanelProps) {
  const {
    select,
    items,
    busy,
    report,
    activeItem,
    liveTriCount,
    selectedCount,
    hasSelection,
    hasRepair,
    canConvertFromMask,
    patchSelect,
    onClear,
    onInvert,
    onGrow,
    onShrink,
    onFillSelectionHoles,
    onSmoothBoundary,
    onSelectIslands,
    onToMask,
    onFromMask,
    onExtract,
    onSplit,
    onDelete,
    onFillMeshHoles,
  } = useFaceSelectionActions(app, store);

  return (
    <section className="panel select-panel" aria-label="Face Selection" data-panel="select">
      <header className="panel-head">
        <h2>Select</h2>
      </header>

      <div
        className="readout"
        data-select-readout
        data-select-count={selectedCount}
        data-select-total={liveTriCount}
      >
        {activeItem
          ? `${selectedCount} / ${liveTriCount} triangles — ${activeItem.name}`
          : items.length === 0
            ? "No mesh loaded"
            : "No active mesh"}
      </div>

      <div className="op">
        <div className="op-row">
          <span>Selection Tool</span>
          <button
            type="button"
            disabled={busy || select.activeTool === null}
            data-select-tool="off"
            data-select-disarm="true"
            aria-pressed={select.activeTool === null}
            onClick={() => patchSelect({ activeTool: null })}
          >
            Off
          </button>
        </div>
        <div className="modes" role="group" aria-label="Selection tool">
          {TOOL_OPTIONS.map((opt) => (
            <button
              key={opt.id}
              type="button"
              disabled={busy}
              data-select-tool={opt.id}
              aria-pressed={select.activeTool === opt.id}
              onClick={() => patchSelect({ activeTool: opt.id })}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      <div className="op">
        <span>Combine Mode</span>
        <div className="modes" role="group" aria-label="Combine mode">
          {COMBINE_OPTIONS.map((opt) => (
            <button
              key={opt.id}
              type="button"
              disabled={busy}
              data-select-combine={opt.id}
              aria-pressed={select.combineMode === opt.id}
              onClick={() => patchSelect({ combineMode: opt.id })}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      <div className="op">
        <label className="check">
          <input
            type="checkbox"
            checked={select.visibleOnly}
            disabled={busy}
            data-select-visible-only
            onChange={(e) => patchSelect({ visibleOnly: e.currentTarget.checked })}
          />
          <span>Visible faces only</span>
        </label>

        <label className="check">
          <input
            type="checkbox"
            checked={select.backfaceCull}
            disabled={busy}
            data-select-backface-cull
            onChange={(e) => patchSelect({ backfaceCull: e.currentTarget.checked })}
          />
          <span>Ignore backfaces</span>
        </label>

        <div className="slider">
          <label htmlFor="select-brush-radius">
            <span>Brush radius</span>
            <span>{select.brushRadius.toFixed(2)}</span>
          </label>
          <input
            id="select-brush-radius"
            type="range"
            min={0.01}
            max={5}
            step={0.01}
            value={select.brushRadius}
            disabled={busy}
            data-select-brush-radius
            onChange={(e) => patchSelect({ brushRadius: Number(e.currentTarget.value) })}
          />
        </div>

        <div className="slider">
          <label htmlFor="select-crease-angle">
            <span>Crease angle</span>
            <span>{Math.round(select.creaseAngleDeg)} deg</span>
          </label>
          <input
            id="select-crease-angle"
            type="range"
            min={1}
            max={180}
            step={1}
            value={select.creaseAngleDeg}
            disabled={busy}
            data-select-crease-angle
            onChange={(e) => patchSelect({ creaseAngleDeg: Number(e.currentTarget.value) })}
          />
        </div>

        <div className="slider">
          <label htmlFor="select-island-max-faces">
            <span>Max island / hole tris</span>
            <span>{Math.round(select.islandMaxFaces)}</span>
          </label>
          <input
            id="select-island-max-faces"
            type="range"
            min={1}
            max={500}
            step={1}
            value={select.islandMaxFaces}
            disabled={busy}
            data-select-island-max-faces
            onChange={(e) => patchSelect({ islandMaxFaces: Number(e.currentTarget.value) })}
          />
        </div>

        <div className="op-row">
          <span>Adjacency</span>
          <div className="modes" role="group" aria-label="Grow/shrink adjacency">
            {ADJACENCY_OPTIONS.map((opt) => (
              <button
                key={opt.id}
                type="button"
                disabled={busy}
                data-select-adjacency={opt.id}
                aria-pressed={select.adjacency === opt.id}
                onClick={() => patchSelect({ adjacency: opt.id })}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {!activeItem ? (
        <p className="muted">Load or select a mesh to edit its face selection.</p>
      ) : (
        <>
          <div className="op">
            <span>Modify Selection</span>
            <div className="op-row">
              <button
                type="button"
                disabled={busy || !hasSelection}
                data-select-action="clear"
                onClick={onClear}
              >
                Clear
              </button>
              <button
                type="button"
                disabled={busy || liveTriCount === 0}
                data-select-action="invert"
                onClick={onInvert}
              >
                Invert
              </button>
              <button
                type="button"
                disabled={busy || !hasRepair || !hasSelection}
                data-select-action="grow"
                onClick={onGrow}
              >
                Grow
              </button>
              <button
                type="button"
                disabled={busy || !hasRepair || !hasSelection}
                data-select-action="shrink"
                onClick={onShrink}
              >
                Shrink
              </button>
            </div>
            <div className="op-row">
              <button
                type="button"
                disabled={busy || !hasRepair || !hasSelection}
                data-select-action="fill-selection-holes"
                onClick={onFillSelectionHoles}
              >
                Fill Holes
              </button>
              <button
                type="button"
                disabled={busy || !hasRepair || !hasSelection}
                data-select-action="smooth-boundary"
                onClick={onSmoothBoundary}
              >
                Smooth
              </button>
              <button
                type="button"
                disabled={busy || !hasRepair || liveTriCount === 0}
                data-select-action="select-islands"
                onClick={onSelectIslands}
              >
                Small Islands
              </button>
            </div>
            <div className="op-row">
              <button
                type="button"
                disabled={busy || !hasRepair || !hasSelection}
                data-select-action="to-mask"
                onClick={onToMask}
              >
                To Mask
              </button>
              <button
                type="button"
                disabled={busy || !hasRepair || !canConvertFromMask}
                data-select-action="from-mask"
                onClick={onFromMask}
              >
                From Mask
              </button>
            </div>
          </div>

          <div className="op">
            <span>Mesh Actions</span>
            <div className="op-row">
              <button
                type="button"
                className="primary"
                disabled={busy || !hasRepair || !hasSelection}
                data-select-action="extract"
                onClick={onExtract}
              >
                Extract
              </button>
              <button
                type="button"
                disabled={busy || !hasRepair || !hasSelection}
                data-select-action="split"
                onClick={onSplit}
              >
                Split
              </button>
              <button
                type="button"
                className="danger"
                disabled={busy || !hasRepair || !hasSelection}
                data-select-action="delete"
                onClick={onDelete}
              >
                Delete
              </button>
              <button
                type="button"
                disabled={busy || !hasRepair || liveTriCount === 0}
                data-select-action="fill-holes"
                onClick={onFillMeshHoles}
              >
                Fill Mesh Holes
              </button>
            </div>
          </div>
        </>
      )}

      {report && (
        <p
          className={report.error ? "report error" : "report"}
          data-select-report
          data-select-error={report.error ? "true" : "false"}
        >
          {report.text}
        </p>
      )}
    </section>
  );
}
