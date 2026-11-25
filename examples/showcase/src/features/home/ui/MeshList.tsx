// The mesh list: open files (or drop them anywhere), see each load's
// progress, and per mesh its name, triangle count, visibility, opacity,
// selection, focus and removal. Heavy loads warn past `?warn-tris=` and can
// be decimated on load (the #118 simplify op) past `?decimate-tris=`.
import { useMeshDisplay, useScene, useSelection } from "fast-mesh-wasm/react";
import { useMemo, useRef, useState } from "react";
import { formatMeshBytes, formatTriangles } from "../../../io/big-file";
import { MESH_EXTENSIONS } from "../../../io/mesh-extensions";
import { PHANTOM_NAMES, type PhantomKind } from "../../../io/phantom";
import { SAMPLE_NAMES, type SampleKind } from "../../../io/vtkjs";
import { type AppState, type DiagnosticShadingMode, type Item, useApp } from "../../../app/state/app-state";
import type { MirrorAxis } from "../../../engine/workbench-ops";
import { useMeshPreview } from "./mesh-preview";
import { IMPORT_UNITS } from "../../../shared/ux/units";
import { angleDegrees, formatAngle } from "../../inspect/ux/angle-measure";
import { PolylineList } from "../../polylines/ui/PolylineList";
import {
  bigFileParams,
  clickRow,
  formatCount,
  heavyMeshes,
  LOAD_PHASE,
  meshGroupIndex,
  useImportScaleDraft,
  useMeshActions,
  useTriangleCount,
} from "../ux/mesh-list";

export function MeshList(props: {
  app: AppState;
  onLoad: (files: Iterable<File>) => void;
  onCancelLoad: (id: number) => void;
  onLoadPhantom: (phantom: PhantomKind) => void;
  onLoadSample: (sample: SampleKind) => void;
  onLoadDemo: () => void;
  onFocus: (mesh: Item["mesh"]) => void;
  /** Frames a world sphere (a polyline's). */
  onFocusBox?: (box: { center: readonly [number, number, number]; radius: number }) => void;
}) {
  const { app, onLoad, onCancelLoad, onLoadPhantom, onLoadSample, onLoadDemo, onFocus, onFocusBox } = props;
  const store = useScene();
  const [selection] = useSelection();
  const items = useApp(app, (s) => s.items);
  const jobs = useApp(app, (s) => s.jobs);
  const decimateOnLoad = useApp(app, (s) => s.decimateOnLoad);
  const meshGroups = useApp(app, (s) => s.meshGroups);
  const ruler = useApp(app, (s) => s.ruler);
  const inspectSection = useApp(app, (s) => s.inspectSection);
  const clipBox = useApp(app, (s) => s.clipBox);
  const anglePoints = useApp(app, (s) => s.objectInfo.angle);
  const [collapsedGroups, setCollapsedGroups] = useState<Record<number, boolean>>({});
  const selectedItem = selection === null ? null : (items.find((i) => i.mesh === selection) ?? null);
  const { mirrorAxis, setMirrorAxis, mirrorBusy, duplicate, mirror } = useMeshActions(store, app, selectedItem);
  const groupByMesh = useMemo(() => meshGroupIndex(meshGroups), [meshGroups]);
  const volumes = useMemo(() => items.filter((i) => i.volumeId !== null), [items]);
  const input = useRef<HTMLInputElement>(null);
  const big = useMemo(bigFileParams, []);
  const { heavy, bytes: heavyBytes } = heavyMeshes(items, big.warnTriangles);

  return (
    <section className="panel" aria-label="Meshes" data-tour="mesh-list">
      <header className="panel-head">
        <h2>Meshes</h2>
        <button type="button" className="primary" onClick={() => input.current?.click()}>
          Open files…
        </button>
        <input
          ref={input}
          id="file-input"
          type="file"
          multiple
          accept={MESH_EXTENSIONS.join(",")}
          hidden
          onChange={(e) => {
            const files = [...(e.currentTarget.files ?? [])];
            // Cleared, so choosing the same file again loads it again.
            e.currentTarget.value = "";
            if (files.length > 0) {
              onLoad(files);
            }
          }}
        />
      </header>
      <p className="muted small">STL (binary, ASCII), OBJ, PLY (ASCII, binary), DICOM series. Or drop files anywhere.</p>
      <ImportScaleControl app={app} />
      <div className="muted small model-group">
        <span>DICOM demos:</span>
        {(Object.keys(PHANTOM_NAMES) as PhantomKind[]).map((kind) => (
          <ModelButton
            key={kind}
            action={`dicom-demo-${kind}`}
            name={PHANTOM_NAMES[kind]}
            title={`Mesh a procedural ${PHANTOM_NAMES[kind].toLowerCase()} DICOM series`}
            onClick={() => onLoadPhantom(kind)}
          />
        ))}
      </div>
      <div className="muted small model-group">
        <span>Sample CT:</span>
        {(Object.keys(SAMPLE_NAMES) as SampleKind[]).map((kind) => (
          <ModelButton
            key={kind}
            action={`sample-${kind}`}
            name={SAMPLE_NAMES[kind]}
            title={`Mesh the bundled ${SAMPLE_NAMES[kind].toLowerCase()} CT series`}
            onClick={() => onLoadSample(kind)}
          />
        ))}
      </div>
      <div className="muted small model-group">
        <span>Demo meshes:</span>
        <ModelButton
          action="demo-meshes"
          name="A/B/C"
          title="Generate the demo meshes A, B and C"
          onClick={() => onLoadDemo()}
        />
      </div>
      <label
        className="check small"
        title={`Past ${formatTriangles(big.decimateTriangles)} a load is simplified to ${big.decimateTarget.toLocaleString("en")} triangles (the simplify op)`}
      >
        <input
          type="checkbox"
          data-action="decimate-on-load"
          checked={decimateOnLoad}
          onChange={(e) => app.setDecimateOnLoad(e.currentTarget.checked)}
        />
        Decimate big files on load (→ {formatTriangles(big.decimateTarget)})
      </label>
      {heavy.length > 0 ? (
        <p className="warn small" data-big-warning>
          Heavy {heavy.length === 1 ? "mesh" : "meshes"}:{" "}
          {heavy.map((i) => `${i.name} (${formatTriangles(i.triangles)})`).join(", ")} {formatMeshBytes(heavyBytes)} in
          memory.{" "}
          {decimateOnLoad
            ? "Future loads decimate past " + formatTriangles(big.decimateTriangles) + "."
            : "Tick “Decimate big files on load” to simplify future loads."}
        </p>
      ) : null}
      {jobs.length > 0 ? (
        <ul className="jobs" aria-label="Loads">
          {jobs.map((job) => (
            <li key={job.id} className={`job job-${job.phase}`} data-phase={job.phase}>
              <span className="job-name" title={job.name}>
                {job.name}
              </span>
              <span className="job-phase">{job.message === "cancelled" ? "cancelled" : LOAD_PHASE[job.phase]}</span>
              {job.phase !== "done" && job.phase !== "error" ? (
                <button
                  type="button"
                  className="icon"
                  data-action="cancel-load"
                  data-job={job.id}
                  title={`Cancel ${job.name}`}
                  onClick={() => onCancelLoad(job.id)}
                >
                  Cancel
                </button>
              ) : job.phase === "error" ? (
                <button type="button" className="icon" aria-label="Dismiss" onClick={() => app.dropJob(job.id)}>
                  ×
                </button>
              ) : null}
              {typeof job.progress === "number" && job.phase !== "done" ? (
                <span
                  className="progress determinate"
                  role="progressbar"
                  aria-valuenow={Math.round(job.progress * 100)}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-label={`${job.name} progress`}
                  style={{ width: `${Math.round(job.progress * 100)}%` }}
                />
              ) : job.phase !== "done" && job.phase !== "error" ? (
                <span className="progress" aria-hidden="true" />
              ) : null}
              {job.phase === "error" && job.message !== undefined && job.message !== "cancelled" ? (
                <span className="job-message">{job.message}</span>
              ) : null}
              {job.phase === "done" && job.summary !== undefined ? (
                <span className="job-summary">{job.summary}</span>
              ) : null}
            </li>
          ))}
        </ul>
      ) : null}
      <div className="mesh-actions" role="group" aria-label="Selected mesh actions">
        <button
          type="button"
          data-action="mesh-duplicate"
          disabled={selectedItem === null || mirrorBusy}
          title="Clone selected mesh's live geometry, attributes, and placement"
          onClick={duplicate}
        >
          Duplicate
        </button>
        <select
          aria-label="Mirror axis"
          data-action="mirror-axis"
          value={mirrorAxis}
          disabled={selectedItem === null || mirrorBusy}
          onChange={(e) => setMirrorAxis(e.currentTarget.value as MirrorAxis)}
        >
          <option value="x">X axis</option>
          <option value="y">Y axis</option>
          <option value="z">Z axis</option>
        </select>
        <button
          type="button"
          data-action="mesh-mirror"
          disabled={selectedItem === null || mirrorBusy}
          title="Mirror selected mesh across chosen axis and flip triangle winding"
          onClick={mirror}
        >
          Mirror
        </button>
      </div>
      {meshGroups.length > 0 ? (
        <div className="scene-tree-groups" data-scene-tree-groups role="region" aria-label="Scene group folders">
          {meshGroups.map((g) => {
            const collapsed = Boolean(collapsedGroups[g.id]);
            return (
              <button
                key={g.id}
                type="button"
                className="tree-group-toggle"
                data-tree-group={g.id}
                aria-expanded={!collapsed}
                onClick={() =>
                  setCollapsedGroups((prev) => ({
                    ...prev,
                    [g.id]: !collapsed,
                  }))
                }
              >
                <span aria-hidden="true">{collapsed ? "▸" : "▾"}</span> {g.name} ({g.members.length})
              </button>
            );
          })}
        </div>
      ) : null}
      <ul className="meshes">
        {items.map((item) => {
          const group = groupByMesh.get(item.mesh) ?? null;
          const collapsed = group !== null && Boolean(collapsedGroups[group.id]);
          return (
            <MeshRow
              key={item.id}
              app={app}
              item={item}
              groupName={group?.name ?? null}
              collapsed={collapsed}
              onFocus={onFocus}
            />
          );
        })}
      </ul>
      {items.length === 0 && jobs.length === 0 ? <p className="muted">No meshes. Open or drop a file.</p> : null}
      <PolylineList app={app} onFocusBox={onFocusBox} />
      {volumes.length > 0 || ruler !== null || anglePoints.length > 0 || inspectSection !== null || clipBox !== null ? (
        <div className="scene-tree-extras" data-scene-tree-extras aria-label="Volumes and annotations">
          <p className="muted small">Volumes &amp; annotations</p>
          <ul className="tree-extras-list">
            {volumes.map((v) => (
              <li key={`vol-${v.id}`} className="tree-extra-row" data-tree-volume={v.name}>
                <span className="tree-node-tag">CT</span>
                <span className="tree-extra-name">{v.name}</span>
                <button
                  type="button"
                  onClick={() => {
                    onFocus(v.mesh);
                    app.setActiveTab("volume");
                  }}
                >
                  Open CT
                </button>
              </li>
            ))}
            {inspectSection !== null ? (
              <li className="tree-extra-row" data-tree-artifact="section">
                <span className="tree-node-tag">Section</span>
                <span className="tree-extra-name">
                  Normal ({inspectSection.plane.normal.map((n) => n.toFixed(1)).join(", ")})
                </span>
                <button type="button" onClick={() => app.setActiveTab("inspect")}>
                  Inspect
                </button>
              </li>
            ) : null}
            {ruler !== null ? (
              <li className="tree-extra-row" data-tree-artifact="ruler">
                <span className="tree-node-tag">Ruler</span>
                <span className="tree-extra-name">
                  {ruler.b !== null
                    ? `${Math.hypot(ruler.b[0] - ruler.a[0], ruler.b[1] - ruler.a[1], ruler.b[2] - ruler.a[2]).toFixed(2)} mm`
                    : "1 point placed"}
                </span>
                <button type="button" onClick={() => app.setRuler(null)}>
                  Clear
                </button>
              </li>
            ) : null}
            {anglePoints.length > 0 ? (
              <li className="tree-extra-row" data-tree-artifact="angle">
                <span className="tree-node-tag">Angle</span>
                <span className="tree-extra-name">
                  {anglePoints.length === 3
                    ? formatAngle(angleDegrees(anglePoints[0], anglePoints[1], anglePoints[2]) ?? 0)
                    : `${anglePoints.length} of 3 points placed`}
                </span>
                <button type="button" onClick={() => app.setObjectInfo({ angle: [] })}>
                  Clear
                </button>
              </li>
            ) : null}
            {clipBox !== null ? (
              <li className="tree-extra-row" data-tree-artifact="clipBox">
                <span className="tree-node-tag">Clip Box</span>
                <span className="tree-extra-name">Active bounds</span>
                <button type="button" onClick={() => app.setClipBox(null)}>
                  Clear
                </button>
              </li>
            ) : null}
          </ul>
        </div>
      ) : null}
    </section>
  );
}

/**
 * A built-in model's button: its committed picture (public/previews, from
 * scripts/showcase-previews.mjs, named after `action`) and name. Showing it
 * loads no model.
 */
function ModelButton(props: { action: string; name: string; title: string; onClick: () => void }) {
  const { action, name, title, onClick } = props;
  return (
    <button type="button" className="link model-card" data-action={action} title={title} onClick={onClick}>
      <img
        src={`${import.meta.env.BASE_URL}previews/${action}.webp`}
        alt={`Preview of ${name}`}
        width={48}
        height={48}
        decoding="async"
      />
      {name}
    </button>
  );
}

/**
 * What files loaded from here on are assumed to be in: a unit, or any factor
 * to millimetres. Multiplies the geometry once, on load (files and drops; a
 * DICOM series states its own spacing). Meshes already loaded stay as they are.
 */
function ImportScaleControl(props: { app: AppState }) {
  const { app } = props;
  const { scale, draft, invalid, chooseUnits, typeFactor, endEdit } = useImportScaleDraft(app);
  return (
    <div className="import-scale small">
      <label className="field" title="The unit the next files' numbers are in">
        Assume files in
        <select
          data-control="import-units"
          value={scale.units ?? "custom"}
          onChange={(e) => chooseUnits(e.currentTarget.value)}
        >
          {IMPORT_UNITS.map((u) => (
            <option key={u} value={u}>
              {u}
            </option>
          ))}
          {scale.units === null ? <option value="custom">custom</option> : null}
        </select>
      </label>
      <label className="field" title="Millimetres per file unit: the next files' numbers are multiplied by this once, on load">
        ×
        <input
          type="number"
          data-control="import-scale"
          min={0}
          step="any"
          aria-invalid={invalid}
          value={draft ?? scale.factor}
          onChange={(e) => typeFactor(e.currentTarget.value)}
          onBlur={endEdit}
        />
      </label>
    </div>
  );
}

function MeshRow(props: {
  app: AppState;
  item: Item;
  groupName: string | null;
  collapsed: boolean;
  onFocus: (mesh: Item["mesh"]) => void;
}) {
  const { app, item, groupName, collapsed, onFocus } = props;
  const store = useScene();
  const [selection, select] = useSelection();
  const [state, setDisplay] = useMeshDisplay(item.mesh);
  const triangles = useTriangleCount(store, item);
  const selected = selection === item.mesh;
  const checks = useApp(app, (s) => s.checks);
  const checked = checks.includes(item.mesh);
  const multi = useApp(app, (s) => s.multi);
  const secondary = !selected && multi.includes(item.mesh);
  const swatch = `#${item.color.toString(16).padStart(6, "0")}`;
  const preview = useMeshPreview(store, item);
  const rowClasses = [
    "mesh",
    selected ? "selected" : secondary ? "secondary" : "",
    groupName !== null ? "tree-child" : "",
    item.locked ? "locked" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <li className={rowClasses} data-mesh={item.name} hidden={collapsed} style={collapsed ? { display: "none" } : undefined}>
      <div className="mesh-head">
        <label className="check" title={`${item.name}: tick for grouping`}>
          <input
            type="checkbox"
            data-check={item.name}
            checked={checked}
            onChange={(e) => app.toggleCheck(item.mesh, e.currentTarget.checked)}
          />
        </label>
        <span className="swatch" style={{ background: swatch }} aria-hidden="true" />
        {preview.url !== null ? (
          <img className="mesh-preview" src={preview.url} alt={`Preview of ${item.name}`} width={36} height={36} />
        ) : preview.empty ? (
          <span className="mesh-preview empty" role="img" aria-label={`${item.name} has nothing to draw`} data-preview-empty>
            empty
          </span>
        ) : (
          <span className="mesh-preview" aria-hidden="true" />
        )}
        <button
          type="button"
          className="mesh-name"
          aria-pressed={selected}
          title={`${item.name}: select (shift-click adds)`}
          onClick={(e) => clickRow(app, select, selection, item.mesh, multi, e.shiftKey)}
        >
          {item.name}
        </button>
        {groupName !== null ? (
          <span className="tree-group-badge" title={`Member of ${groupName}`}>
            {groupName}
          </span>
        ) : null}
        {item.volumeId !== null ? (
          <span className="tree-vol-badge" title="CT / DICOM volume">
            CT
          </span>
        ) : null}
        {item.importScale.factor === 1 ? null : (
          <span
            className="count"
            data-import-scale={item.importScale.factor}
            title={`Loaded as ${item.importScale.units ?? "a custom unit"}: its numbers were multiplied by ${item.importScale.factor} to millimetres, once`}
          >
            ×{item.importScale.factor}
          </span>
        )}
        <span className="count" title="triangles">
          {formatCount(triangles)}
        </span>
      </div>
      <div className="mesh-controls">
        <label className="check" title="Visible">
          <input
            type="checkbox"
            checked={state.desired.visible}
            onChange={(e) => setDisplay({ visible: e.currentTarget.checked })}
          />
          Show
        </label>
        <input
          type="range"
          min={0.1}
          max={1}
          step={0.05}
          value={state.desired.opacity}
          aria-label={`${item.name} opacity`}
          title={`Opacity ${Math.round(state.desired.opacity * 100)}%`}
          onChange={(e) => setDisplay({ opacity: Number(e.currentTarget.value) })}
        />
        <button type="button" onClick={() => onFocus(item.mesh)} title="Point the camera at this mesh">
          Focus
        </button>
        <button
          type="button"
          className="tree-lock-btn"
          data-lock={item.name}
          aria-pressed={item.locked}
          title={item.locked ? "Unlock mesh (allow moving & sculpting)" : "Lock mesh (block moving & sculpting)"}
          onClick={() => app.setMeshLocked(item.mesh, !item.locked)}
        >
          {item.locked ? "Locked" : "Lock"}
        </button>
        <select
          className="tree-shading-pill"
          data-mesh-shading={item.name}
          aria-label={`${item.name} shading override`}
          title="Per-object viewport shading override"
          value={item.shadingOverride ?? ""}
          onChange={(e) => {
            const v = e.currentTarget.value;
            app.setMeshShadingOverride(item.mesh, v === "" ? null : (v as DiagnosticShadingMode));
          }}
        >
          <option value="">Shading: Auto</option>
          <option value="smooth">Smooth</option>
          <option value="flat">Flat</option>
          <option value="autoCrease">Auto-Crease</option>
          <option value="wireframeShaded">Wire + Shaded</option>
          <option value="matcap">MatCap</option>
          <option value="backfaceWarning">Backface Warn</option>
        </select>
        <button type="button" className="danger" onClick={() => app.removeMesh(store, item)} title="Remove the mesh">
          Remove
        </button>
      </div>
    </li>
  );
}
