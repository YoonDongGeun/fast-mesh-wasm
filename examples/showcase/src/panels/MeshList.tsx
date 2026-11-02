// The mesh list: open files (or drop them anywhere), see each load's
// progress, and per mesh its name, triangle count, visibility, opacity,
// selection, focus and removal. Heavy loads warn past `?warn-tris=` and can
// be decimated on load (the #118 simplify op) past `?decimate-tris=`.
import { useMeshDisplay, useScene, useSelection } from "fast-mesh-wasm/react";
import type { SceneStore } from "fast-mesh-wasm/store";
import { useEffect, useMemo, useRef, useState } from "react";
import { BIG_FILE_DEFAULTS, estimateMeshBytes, formatMeshBytes, formatTriangles, parseBigFileParams } from "../load/big-file";
import { MESH_EXTENSIONS } from "../load/parse";
import { PHANTOM_NAMES, type PhantomKind } from "../load/phantom";
import { SAMPLE_NAMES, type SampleKind } from "../load/vtkjs";
import { type AppState, type Item, type Job, useApp } from "../scene/app-state";
import { selectMesh } from "../scene/selection";

const PHASE: Record<Job["phase"], string> = {
  reading: "reading…",
  parsing: "parsing…",
  adding: "adding…",
  done: "loaded",
  error: "failed",
};

/** Reads the large-file thresholds from the URL (tolerant; see big-file.ts). */
function bigFileParams(): { warnTriangles: number; decimateTriangles: number; decimateTarget: number } {
  try {
    const p = parseBigFileParams(location.search);
    return { warnTriangles: p.warnTriangles, decimateTriangles: p.decimateTriangles, decimateTarget: p.decimateTarget };
  } catch {
    return {
      warnTriangles: BIG_FILE_DEFAULTS.warnTriangles,
      decimateTriangles: BIG_FILE_DEFAULTS.decimateTriangles,
      decimateTarget: BIG_FILE_DEFAULTS.decimateTarget,
    };
  }
}

export function MeshList(props: {
  app: AppState;
  onLoad: (files: Iterable<File>) => void;
  onCancelLoad: (id: number) => void;
  onLoadPhantom: (phantom: PhantomKind) => void;
  onLoadSample: (sample: SampleKind) => void;
  onLoadDemo: () => void;
  onFocus: (mesh: Item["mesh"]) => void;
}) {
  const { app, onLoad, onCancelLoad, onLoadPhantom, onLoadSample, onLoadDemo, onFocus } = props;
  const items = useApp(app, (s) => s.items);
  const jobs = useApp(app, (s) => s.jobs);
  const decimateOnLoad = useApp(app, (s) => s.decimateOnLoad);
  const input = useRef<HTMLInputElement>(null);
  const big = useMemo(bigFileParams, []);
  const heavy = items.filter((i) => i.triangles >= big.warnTriangles);
  const heavyBytes = heavy.reduce((n, i) => n + estimateMeshBytes(i.triangles), 0);

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
      <p className="muted small">
        DICOM demos:{" "}
        {(Object.keys(PHANTOM_NAMES) as PhantomKind[]).map((kind) => (
          <button
            key={kind}
            type="button"
            className="link"
            data-action={`dicom-demo-${kind}`}
            title={`Mesh a procedural ${PHANTOM_NAMES[kind].toLowerCase()} DICOM series`}
            onClick={() => onLoadPhantom(kind)}
          >
            {PHANTOM_NAMES[kind]}
          </button>
        ))}
      </p>
      <p className="muted small">
        Sample CT:{" "}
        {(Object.keys(SAMPLE_NAMES) as SampleKind[]).map((kind) => (
          <button
            key={kind}
            type="button"
            className="link"
            data-action={`sample-${kind}`}
            title={`Mesh the bundled ${SAMPLE_NAMES[kind].toLowerCase()} CT series`}
            onClick={() => onLoadSample(kind)}
          >
            {SAMPLE_NAMES[kind]}
          </button>
        ))}
      </p>
      <p className="muted small">
        Demo meshes:{" "}
        <button
          type="button"
          className="link"
          data-action="demo-meshes"
          title="Generate the demo meshes A, B and C"
          onClick={() => onLoadDemo()}
        >
          A/B/C
        </button>
      </p>
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
              <span className="job-phase">{job.message === "cancelled" ? "cancelled" : PHASE[job.phase]}</span>
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
      <ul className="meshes">
        {items.map((item) => (
          <MeshRow key={item.id} app={app} item={item} onFocus={onFocus} />
        ))}
      </ul>
      {items.length === 0 && jobs.length === 0 ? <p className="muted">No meshes. Open or drop a file.</p> : null}
    </section>
  );
}

function MeshRow(props: { app: AppState; item: Item; onFocus: (mesh: Item["mesh"]) => void }) {
  const { app, item, onFocus } = props;
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

  return (
    <li className={selected ? "mesh selected" : secondary ? "mesh secondary" : "mesh"} data-mesh={item.name}>
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
        <button
          type="button"
          className="mesh-name"
          aria-pressed={selected}
          title={`${item.name}: select (shift-click adds)`}
          onClick={(e) => {
            // Plain clicks reset (a lone selection toggles off); shift-clicks toggle.
            const target = e.shiftKey || !selected || multi.length > 1 ? item.mesh : null;
            selectMesh(app, select, selection, target, e.shiftKey && target !== null);
          }}
        >
          {item.name}
        </button>
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
        <button type="button" className="danger" onClick={() => app.removeMesh(store, item)} title="Remove the mesh">
          Remove
        </button>
      </div>
    </li>
  );
}

/** "12,288" or "1.05 M". */
function formatCount(n: number): string {
  return n >= 1_000_000 ? `${(n / 1_000_000).toFixed(2)} M` : n.toLocaleString("en");
}

/**
 * The mesh's live triangle count: as loaded, then recounted from the scene's
 * index view whenever its triangles change (repairs, their undo). A removed
 * triangle reads `0, 0, 0` and is not counted.
 */
function useTriangleCount(store: SceneStore, item: Item): number {
  const [count, setCount] = useState(item.triangles);
  useEffect(
    () =>
      store.scene.onChanges((changes) => {
        for (const c of changes) {
          if (c.mesh !== item.mesh || c.removed || c.added) {
            continue;
          }
          if (c.indices === "all" || c.indices.length > 0) {
            const idx = store.scene.views(item.mesh).indices();
            let n = 0;
            for (let t = 0; t < idx.length; t += 3) {
              if (idx[t] !== 0 || idx[t + 1] !== 0 || idx[t + 2] !== 0) {
                n++;
              }
            }
            setCount(n);
          }
        }
      }),
    [store, item.mesh],
  );
  return count;
}
