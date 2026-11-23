// The inspect panel (#110): in-viewport inspection tools. The ruler
// (click two vertices, world-space points that survive placement undo),
// click-to-query (the picked vertex's coloring value, through the
// thickness/curvature/distance colorings), a draggable section plane
// (wired to `setSectionPlane` through the store) and an axis-aligned clip
// box for looking inside meshes. Scalar mesh numbers reuse `measure`.
// The surface field and the inspect maps (#251: deviation, overhang,
// triangle quality) colour a mesh through one shared overlay, one at a time.
import type { MeshHandle, PrimitiveKindInput, Scene } from "fast-mesh-wasm";
import { useScene, useSelection } from "fast-mesh-wasm/react";
import type { SceneStore } from "fast-mesh-wasm/store";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { type AppState, type Item, type SurfaceColormap, useApp } from "../../../app/state/app-state";
import { type Wave6MetrologyState } from "../../../app/state/wave6-state";
import { formatCoord, INSPECT_TOOLS, rulerDistance } from "../ux/inspect-tools";
import {
  AXES,
  type PanelSection,
  clipBoxWithEnd,
  clipSliderBounds,
  fitWorldBox,
  sectionActions,
  sectionReadout,
} from "../ux/section-clip";
import { type ContourPlaneAxis, useContourSlices } from "../ux/use-contour-slices";
import { usePublishInspectTools, useQueryReadout } from "../ux/use-inspect-readouts";
import {
  DATUM_AXIS_OPTIONS,
  type DatumAxisKey,
  PRIMITIVE_KINDS,
  summarizePrimitiveGeometry,
  usePrimitiveFit,
} from "../ux/use-primitive-fit";
import { COLORMAPS, SURFACE_FIELD_MODES, useSurfaceField } from "../ux/use-surface-field";
import { AngleReadout } from "./AngleReadout";
import { FieldLegend } from "./FieldLegend";
import { InspectMapControls } from "./InspectMapControls";
import { LengthInput } from "../../../shared/ui/LengthInput";
import { formatArea, formatCoordLength, formatLength, formatPoint, formatVolume, type LengthUnit } from "../../../shared/ux/units";

const INITIAL_METROLOGY: Wave6MetrologyState = {
  fittedPrimitive: null,
  showPrimitiveOverlay: true,
  contourSlices: null,
  showContoursOverlay: true,
  pointCloud: null,
  showPointCloudOverlay: true,
  pointCloudReport: null,
};

export function InspectPanel(props: { app: AppState; scene: Scene | WorkerScene }) {
  const { app, scene } = props;
  const store = useScene();
  const [selection] = useSelection();
  const items = useApp(app, (s) => s.items);
  const tool = useApp(app, (s) => s.inspectTool);
  const ruler = useApp(app, (s) => s.ruler);
  const query = useApp(app, (s) => s.inspectQuery);
  const section = useApp(app, (s) => s.inspectSection);
  const clip = useApp(app, (s) => s.clipBox);
  const units = useApp(app, (s) => s.units);
  const overlayValues = useApp(app, (s) => s.surfaceFieldValues);
  const metrology = useApp(app, (s) => s.wave6?.metrology ?? INITIAL_METROLOGY);
  const item = items.find((i) => i.mesh === selection) ?? null;

  const distance = ruler !== null && ruler.b !== null ? rulerDistance(ruler.a, ruler.b) : null;

  const { queryValue, queryMeasure } = useQueryReadout(store, scene, query, overlayValues);
  usePublishInspectTools({ items, tool, ruler, distance, query, queryValue, section, clip });
  // The surface field and the inspect maps share the overlay: a map turns the field off first.
  const surfaceField = useSurfaceField(app, items, selection);
  const takeOverlayFromField = () => {
    if (surfaceField.sf.mode !== "none") {
      surfaceField.switchMode("none");
    }
  };

  return (
    <section className="panel" aria-label="Inspect">
      <h2>Inspect</h2>
      <div className="modes" role="group" aria-label="Inspection tool">
        {INSPECT_TOOLS.map((t) => (
          <button
            key={t.tool}
            type="button"
            data-action={`inspect-${t.tool}`}
            aria-pressed={tool === t.tool}
            title={t.title}
            onClick={() => app.setInspectTool(tool === t.tool ? null : t.tool)}
          >
            {t.label}
          </button>
        ))}
      </div>
      {tool === "ruler" ? (
        <p className="muted small">Click two vertices in the viewport; Esc clears the selection first.</p>
      ) : null}
      {tool === "query" ? (
        <p className="muted small">Click a vertex in the viewport to read its coloring value.</p>
      ) : null}
      <RulerReadout app={app} ruler={ruler} distance={distance} unit={units} />
      <AngleReadout app={app} tool={tool} />
      <QueryReadout query={query} items={items} value={queryValue} measure={queryMeasure} unit={units} />
      <SectionControls app={app} scene={scene} store={store} items={items} item={item} section={section} active={tool === "section"} unit={units} />
      {tool === "clip" || clip !== null ? <ClipControls app={app} scene={scene} items={items} item={item} clip={clip} unit={units} /> : null}
      <SurfaceFieldControls field={surfaceField} items={items} unit={units} />
      <InspectMapControls
        app={app}
        items={items}
        selection={selection}
        unit={units}
        onTakeOverlay={takeOverlayFromField}
      />
      <PrimitiveFitControls app={app} store={store} items={items} item={item} metrology={metrology} unit={units} />
      <ContourSliceControls app={app} store={store} items={items} item={item} metrology={metrology} unit={units} />
    </section>
  );
}

function SurfaceFieldControls(props: {
  field: ReturnType<typeof useSurfaceField>;
  items: readonly Item[];
  unit: LengthUnit;
}) {
  const { field, items, unit } = props;
  const { sf, summary, busy, error, inspectedMesh, targetMesh, mapMeta } = field;

  return (
    <div className="op surface-field-panel">
      <h3>Surface Field Analysis</h3>
      <div className="modes" role="group" aria-label="Surface field mode">
        {SURFACE_FIELD_MODES.map((m) => (
          <button
            key={m.id}
            type="button"
            data-inspect-field-mode={m.id}
            aria-pressed={sf.mode === m.id}
            title={m.title}
            disabled={busy || (m.id !== "none" && items.length === 0) || (m.id === "distance" && items.length < 2)}
            onClick={() => field.switchMode(m.id)}
          >
            {m.label}
          </button>
        ))}
      </div>

      {sf.mode !== "none" && items.length > 0 ? (
        <>
          <div className="picker">
            <span>On</span>
            <select
              aria-label="Field source mesh"
              data-inspect-field-mesh
              value={inspectedMesh === null ? "" : String(items.findIndex((i) => i.mesh === inspectedMesh))}
              disabled={busy}
              onChange={(e) => field.chooseMesh(Number(e.currentTarget.value))}
            >
              {items.map((it, idx) => (
                <option key={it.id} value={String(idx)}>
                  {it.name}
                </option>
              ))}
            </select>
          </div>

          {sf.mode === "distance" ? (
            <div className="picker">
              <span>To</span>
              <select
                aria-label="Distance target mesh"
                data-inspect-field-target
                value={targetMesh === null ? "" : String(items.findIndex((i) => i.mesh === targetMesh))}
                disabled={busy}
                onChange={(e) => field.chooseTarget(Number(e.currentTarget.value))}
              >
                {items
                  .filter((it) => it.mesh !== inspectedMesh)
                  .map((it) => {
                    const idx = items.indexOf(it);
                    return (
                      <option key={it.id} value={String(idx)}>
                        {it.name}
                      </option>
                    );
                  })}
              </select>
            </div>
          ) : null}

          <div className="op-row">
            <label className="field">
              colormap
              <select
                aria-label="Field colormap"
                data-inspect-field-map
                value={sf.map}
                onChange={(e) => field.updateAppearance({ map: e.currentTarget.value as SurfaceColormap })}
              >
                {COLORMAPS.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.label}
                  </option>
                ))}
              </select>
            </label>
            {(sf.mode === "distance" || sf.mode === "thickness") ? (
              <label className="field">
                limit ({unit})
                <LengthInput
                  aria-label="Field max distance"
                  data-inspect-field-maxdist
                  min={0.01}
                  mm={sf.maxDistance}
                  unit={unit}
                  onMm={(mm) => field.setLimit(mm)}
                />
              </label>
            ) : null}
          </div>

          <div className="op-row">
            <label className="field">
              min
              <input
                type="number"
                step="any"
                aria-label="Field range minimum"
                data-inspect-field-min
                value={Number(sf.min.toFixed(4))}
                onChange={(e) => field.setRangeEnd("min", Number(e.currentTarget.value))}
              />
            </label>
            <label className="field">
              max
              <input
                type="number"
                step="any"
                aria-label="Field range maximum"
                data-inspect-field-max
                value={Number(sf.max.toFixed(4))}
                onChange={(e) => field.setRangeEnd("max", Number(e.currentTarget.value))}
              />
            </label>
            <button
              type="button"
              data-inspect-field-auto
              disabled={summary === null}
              onClick={field.autoRange}
            >
              Auto
            </button>
          </div>

          <FieldLegend map={mapMeta.id} lo={sf.min.toPrecision(3)} hi={sf.max.toPrecision(3)} />

          {summary !== null ? (
            <p className="muted" data-inspect-field-stats>
              min {summary.min.toPrecision(3)} · mean {summary.mean.toPrecision(3)} · max {summary.max.toPrecision(3)}
            </p>
          ) : null}
        </>
      ) : null}

      {busy ? <p className="muted busy">Computing surface field...</p> : null}
      {error !== null ? <p className="report error">{error}</p> : null}
    </div>
  );
}

function RulerReadout(props: {
  app: AppState;
  ruler: { a: [number, number, number]; b: [number, number, number] | null } | null;
  distance: number | null;
  unit: LengthUnit;
}) {
  const { app, ruler, distance, unit } = props;
  if (ruler === null) {
    return null;
  }
  return (
    <p className="readout" data-readout="ruler">
      A {formatPoint(ruler.a, unit)}
      {ruler.b === null ? " · click a second vertex" : ` → B ${formatPoint(ruler.b, unit)}`}{" "}
      {distance === null ? null : (
        <strong>= {formatLength(distance, unit)}</strong>
      )}{" "}
      <button type="button" data-action="ruler-clear" onClick={() => app.setRuler(null)} title="Clear the ruler">
        Clear
      </button>
    </p>
  );
}

function QueryReadout(props: {
  query: { mesh: MeshHandle; slot: number; point: [number, number, number] } | null;
  items: readonly Item[];
  value: number | null;
  measure: { area: number; volume: number | null } | null;
  unit: LengthUnit;
}) {
  const { query, items, value, measure, unit } = props;
  if (query === null) {
    return null;
  }
  const name = items.find((i) => i.mesh === query.mesh)?.name ?? "a mesh";
  return (
    <div className="readout" data-readout="query">
      <p>
        {name} vertex {query.slot} at {formatPoint(query.point, unit)}
      </p>
      <p>
        coloring value: {value === null ? "no coloring" : formatCoordLength(value, unit)}
        {measure === null ? null : (
          <>
            {" · "}area {formatArea(measure.area, unit)}
            {measure.volume === null ? " · open mesh (no volume)" : ` · volume ${formatVolume(measure.volume, unit)}`}
          </>
        )}
      </p>
    </div>
  );
}

function SectionControls(props: {
  app: AppState;
  scene: Scene | WorkerScene;
  store: ReturnType<typeof useScene>;
  items: readonly Item[];
  item: Item | null;
  section: PanelSection | null;
  active: boolean;
  unit: LengthUnit;
}) {
  const { app, scene, store, items, item, section, active, unit } = props;
  if (!active && section === null) {
    return null;
  }
  const readout = sectionReadout(scene, section, item);
  const { center, axis, range, offset } = readout;
  const actions = sectionActions(app, store, items, section, readout);

  return (
    <div className="readout" data-readout="section">
      {section === null ? (
        <p>
          <button type="button" data-action="section-enable" disabled={items.length === 0} onClick={actions.enable} title="Cut the meshes with a plane">
            Cut with a plane
          </button>
        </p>
      ) : (
        <>
          <p className="muted small">
            origin {formatPoint(section.plane.origin, unit)} · normal (
            {section.plane.normal.map(formatCoord).join(", ")}) · drag the handle in the viewport
          </p>
          <div className="modes" role="group" aria-label="Section axis">
            {AXES.map((a) => (
              <button
                key={a}
                type="button"
                data-axis={a}
                aria-pressed={axis === a}
                onClick={() => actions.setAxis(a)}
              >
                {a.toUpperCase()}
              </button>
            ))}
            <button
              type="button"
              data-action="section-flip"
              title="Flip the kept side"
              onClick={actions.flip}
            >
              Flip
            </button>
          </div>
          <label className="slider">
            offset <output>{formatCoordLength(offset, unit)}</output>
            <input
              type="range"
              min={-range}
              max={range}
              step={range / 200}
              value={Math.min(range, Math.max(-range, offset))}
              disabled={center === null}
              onChange={(e) => actions.setOffset(Number(e.currentTarget.value))}
            />
          </label>
          <p>
            <button
              type="button"
              data-action="section-disable"
              onClick={actions.remove}
            >
              Remove plane
            </button>
          </p>
        </>
      )}
    </div>
  );
}

function ClipControls(props: {
  app: AppState;
  scene: Scene | WorkerScene;
  items: readonly Item[];
  item: Item | null;
  clip: { min: [number, number, number]; max: [number, number, number] } | null;
  unit: LengthUnit;
}) {
  const { app, scene, items, item, clip, unit } = props;
  if (clip === null) {
    return (
      <p>
        <button
          type="button"
          data-action="clip-fit"
          disabled={item === null}
          onClick={() => {
            const fit = item === null ? null : fitWorldBox(scene, item);
            if (fit !== null) {
              app.setClipBox(fit);
            }
          }}
          title="Keep the selected mesh's middle half"
        >
          Clip to selection
        </button>
      </p>
    );
  }
  const { lo, hi } = clipSliderBounds(clip);
  void items;
  const set = (axis: number, end: 0 | 1, v: number) => {
    app.setClipBox(clipBoxWithEnd(clip, axis, end, v));
  };
  return (
    <div className="readout" data-readout="clip">
      {AXES.map((a, i) => (
        <div key={a}>
          <label className="slider">
            {a} min <output>{formatCoordLength(clip.min[i], unit)}</output>
            <input
              type="range"
              min={lo[i]}
              max={hi[i]}
              step={(hi[i] - lo[i]) / 200}
              value={clip.min[i]}
              onChange={(e) => set(i, 0, Number(e.currentTarget.value))}
            />
          </label>
          <label className="slider">
            {a} max <output>{formatCoordLength(clip.max[i], unit)}</output>
            <input
              type="range"
              min={lo[i]}
              max={hi[i]}
              step={(hi[i] - lo[i]) / 200}
              value={clip.max[i]}
              onChange={(e) => set(i, 1, Number(e.currentTarget.value))}
            />
          </label>
        </div>
      ))}
      <p>
        <button type="button" data-action="clip-clear" onClick={() => app.setClipBox(null)}>
          Clear box
        </button>
      </p>
    </div>
  );
}

/** #234 Primitive fitting and datum alignment controls inside the Inspect panel. */
function PrimitiveFitControls(props: {
  app: AppState;
  store: SceneStore;
  items: readonly Item[];
  item: Item | null;
  metrology: Wave6MetrologyState;
  unit: LengthUnit;
}) {
  const { app, store, items, item, metrology, unit } = props;
  const targetItem = item ?? items[0] ?? null;
  const fit = usePrimitiveFit(app, store, targetItem, metrology);
  const { kind, targetAxis, busy, status, error, canFit, canAlign, fitted } = fit;

  return (
    <div className="op" data-inspect-primitive-fit>
      <div className="op-head">
        <strong>Primitive Fit &amp; Datum</strong>
        {fitted !== null ? (
          <button
            type="button"
            data-action="inspect-fit-clear"
            onClick={fit.onClear}
          >
            Clear
          </button>
        ) : null}
      </div>

      <p className="muted small">
        {targetItem === null ? "Load a mesh to fit a geometric primitive." : `Target: ${targetItem.name}`}
      </p>

      <div className="op-row">
        <label className="field">
          kind
          <select
            aria-label="Primitive kind"
            data-inspect-fit-kind
            value={kind}
            disabled={busy}
            onChange={(e) => fit.setKind(e.currentTarget.value as PrimitiveKindInput)}
          >
            {PRIMITIVE_KINDS.map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          data-action="inspect-fit-primitive"
          disabled={busy || targetItem === null || !canFit}
          onClick={() => void fit.onFit()}
        >
          {busy ? "Working..." : "Fit"}
        </button>
      </div>

      <label className="check">
        <input
          type="checkbox"
          data-inspect-fit-overlay
          checked={metrology.showPrimitiveOverlay}
          onChange={(e) => fit.setShowOverlay(e.currentTarget.checked)}
        />
        Show datum overlay
      </label>

      {fitted !== null ? (
        <div className="readout" data-inspect-fit-report>
          <p>
            Kind: <strong>{fitted.primitive.kind}</strong> · Inliers: {fitted.inlierCount} (
            {(fitted.inlierRatio * 100).toFixed(1)}%)
          </p>
          <p>
            RMS: {formatLength(fitted.rmsError, unit)} · Max: {formatLength(fitted.maxError, unit)}
          </p>
          {summarizePrimitiveGeometry(fitted.primitive, unit).map((line) => (
            <p key={line}>{line}</p>
          ))}
        </div>
      ) : null}

      <div className="op-row">
        <label className="field">
          target axis
          <select
            aria-label="Datum target axis"
            data-inspect-align-axis
            value={targetAxis}
            disabled={busy || fitted === null}
            onChange={(e) => fit.setTargetAxis(e.currentTarget.value as DatumAxisKey)}
          >
            {DATUM_AXIS_OPTIONS.map((a) => (
              <option key={a.id} value={a.id}>
                {a.label}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          data-action="inspect-align-datum"
          disabled={busy || targetItem === null || fitted === null || !canAlign}
          onClick={() => void fit.onAlign()}
        >
          Align Datum
        </button>
      </div>

      {status !== null ? <p className="muted small">{status}</p> : null}
      {error !== null ? (
        <p className="report error small" role="alert">
          {error}
        </p>
      ) : null}
    </div>
  );
}

/** #234 Section contour slicing and SVG/DXF export inside the Inspect panel. */
function ContourSliceControls(props: {
  app: AppState;
  store: SceneStore;
  items: readonly Item[];
  item: Item | null;
  metrology: Wave6MetrologyState;
  unit: LengthUnit;
}) {
  const { app, store, items, item, metrology, unit } = props;
  const targetItem = item ?? items[0] ?? null;
  const contours = useContourSlices(app, store, targetItem, metrology);
  const { planeAxis, customNormal, offsetMm, stackCount, stepMm, busy, error, exportState, canSlice, slices, totals } =
    contours;

  return (
    <div className="op" data-inspect-contours>
      <div className="op-head">
        <strong>Section Contours</strong>
        {slices !== null ? (
          <button
            type="button"
            data-action="inspect-contours-clear"
            onClick={contours.onClear}
          >
            Clear
          </button>
        ) : null}
      </div>

      <div className="op-row">
        <label className="field">
          plane axis
          <select
            aria-label="Contour plane axis"
            data-inspect-contour-axis
            value={planeAxis}
            disabled={busy}
            onChange={(e) => contours.setPlaneAxis(e.currentTarget.value as ContourPlaneAxis)}
          >
            <option value="z">Z (0, 0, 1)</option>
            <option value="y">Y (0, 1, 0)</option>
            <option value="x">X (1, 0, 0)</option>
            <option value="custom">Custom</option>
          </select>
        </label>
        <label className="field">
          offset ({unit})
          <LengthInput
            aria-label="Contour offset"
            data-inspect-contour-offset
            mm={offsetMm}
            unit={unit}
            onMm={contours.setOffsetMm}
          />
        </label>
      </div>

      {planeAxis === "custom" ? (
        <div className="op-row">
          {(["X", "Y", "Z"] as const).map((label, idx) => (
            <label key={label} className="field">
              n{label.toLowerCase()}
              <input
                type="number"
                step="any"
                aria-label={`Contour normal ${label}`}
                data-inspect-contour-normal={label.toLowerCase()}
                value={customNormal[idx]}
                onChange={(e) => contours.setNormalComponent(idx, Number(e.currentTarget.value))}
              />
            </label>
          ))}
        </div>
      ) : null}

      <div className="op-row">
        <label className="field">
          stack count
          <input
            type="number"
            aria-label="Contour stack count"
            data-inspect-contour-count
            min={1}
            max={100}
            step={1}
            value={stackCount}
            onChange={(e) => contours.setStackCount(Number(e.currentTarget.value))}
          />
        </label>
        <label className="field">
          step ({unit})
          <LengthInput
            aria-label="Contour stack step"
            data-inspect-contour-step
            min={0.001}
            mm={stepMm}
            unit={unit}
            onMm={(mm) => contours.setStep(mm)}
          />
        </label>
        <button
          type="button"
          data-action="inspect-slice-contours"
          disabled={busy || targetItem === null || !canSlice}
          onClick={() => void contours.onSlice()}
        >
          {busy ? "Slicing..." : "Slice Contours"}
        </button>
      </div>

      <label className="check">
        <input
          type="checkbox"
          data-inspect-contours-overlay
          checked={metrology.showContoursOverlay}
          onChange={(e) => contours.setShowOverlay(e.currentTarget.checked)}
        />
        Show contour overlay
      </label>

      {slices !== null ? (
        <>
          <div className="readout" data-inspect-contours-summary>
            <p>
              Slices: {slices.length} · Loops: {totals.loops}
            </p>
            <p>
              Perimeter: {formatLength(totals.perimeter, unit)} · Area: {formatArea(totals.area, unit)}
            </p>
          </div>
          <div className="op-row">
            <button
              type="button"
              data-action="inspect-export-svg"
              disabled={slices.length === 0}
              onClick={() => void contours.onExport("svg")}
            >
              Export SVG
            </button>
            <button
              type="button"
              data-action="inspect-export-dxf"
              disabled={slices.length === 0}
              onClick={() => void contours.onExport("dxf")}
            >
              Export DXF
            </button>
          </div>
        </>
      ) : null}

      {exportState !== null ? (
        <div className="report" data-inspect-contour-export-preview>
          <p className="small muted">
            Exported {exportState.format.toUpperCase()} ({exportState.fileName}, {exportState.text.length} chars)
          </p>
          <pre className="small">{exportState.text.slice(0, 600)}</pre>
        </div>
      ) : null}

      {error !== null ? (
        <p className="report error small" role="alert">
          {error}
        </p>
      ) : null}
    </div>
  );
}
