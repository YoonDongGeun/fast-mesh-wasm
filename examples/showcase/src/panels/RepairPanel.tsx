// The repair panel for the selected mesh: its analysis (holes and their
// sizes, orientation, components, non-manifold and degenerate elements),
// refreshed whenever its triangles change (a repair, its undo), and the
// library's repair ops, each one undo step through `store.repair`.
// Analysis is read-only: the app calls it on its own scene.
import type { MeshHandle, RepairOp, RepairReport, Scene } from "fast-mesh-wasm";
import { useScene, useSelection } from "fast-mesh-wasm/react";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useCallback, useEffect, useRef, useState } from "react";
import { type AppState, useApp } from "../scene/app-state";
import { formatDistance } from "./collision-math";
import { inspect } from "./inspect";
import { type AnalysisSummary, describeReport, plural, summarize } from "./repair-summary";

/** The most entries the analysis lists (counts stay exact). */
const MAX_LISTED = 64;
/** "Any size" for Fill holes (an empty field): the op needs a bound (or a list of loops). */
const ANY_SIZE = 1e9;
/** The Fill holes size limit to start with: small and medium holes, not an open mesh's outer boundary. */
const DEFAULT_MAX_EDGES = 64;
/** How long after an op its own triangle change may still arrive (it has none when nothing changed). */
const OWN_CHANGE_MS = 1000;
/** The holes listed by size. */
const HOLES_SHOWN = 6;

export function RepairPanel(props: { app: AppState; scene: Scene | WorkerScene }) {
  const { app, scene } = props;
  const store = useScene();
  const [selection] = useSelection();
  const items = useApp(app, (s) => s.items);
  const item = items.find((i) => i.mesh === selection) ?? null;
  const mesh = item?.mesh ?? null;

  const [analysis, setAnalysis] = useState<{ mesh: MeshHandle; summary: AnalysisSummary } | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [report, setReport] = useState<{ mesh: MeshHandle; line: string; failed: boolean } | null>(null);
  const [weldTolerance, setWeldTolerance] = useState(1e-5);
  const [minTriangles, setMinTriangles] = useState(500);
  const [maxEdges, setMaxEdges] = useState(DEFAULT_MAX_EDGES);
  const [fair, setFair] = useState(true);
  const [center, setCenter] = useState<readonly [number, number, number]>([0, 0, 0]);
  const [brushRadius, setBrushRadius] = useState(1);
  const [strength, setStrength] = useState(1);
  const [falloff, setFalloff] = useState<"linear" | "smooth">("smooth");
  const [cycles, setCycles] = useState(3);
  const [targetLength, setTargetLength] = useState("");
  const [maxRounds, setMaxRounds] = useState("");
  // The brush follows the mesh: its bounding center and a quarter of its
  // half-diagonal. Strength and the rest are preferences and persist.
  useEffect(() => {
    if (item !== null) {
      setCenter(item.center);
      setBrushRadius(item.radius / 4);
    }
  }, [item]);
  const seq = useRef(0);
  // An op's own triangle change keeps its report; any other (undo, redo)
  // makes the report stale and clears it.
  const own = useRef<{ pending: boolean; timer?: ReturnType<typeof setTimeout> }>({ pending: false });
  const weldRef = useRef(weldTolerance);
  weldRef.current = weldTolerance;

  const analyze = useCallback(
    async (target: MeshHandle) => {
      const n = ++seq.current;
      setAnalyzing(true);
      try {
        const a = await scene.analyze(target, { weldTolerance: weldRef.current, maxListed: MAX_LISTED });
        if (n === seq.current) {
          const summary = summarize(a);
          setAnalysis({ mesh: target, summary });
          const name = app.getSnapshot().items.find((i) => i.mesh === target)?.name ?? "";
          inspect.repair.analysis = { mesh: name, summary };
          inspect.repair.analyses++;
        }
      } catch {
        // Removed meanwhile: the selection follows.
      } finally {
        if (n === seq.current) {
          setAnalyzing(false);
        }
      }
    },
    [scene, app],
  );

  // Analyse on selection, and again whenever its triangles change (a
  // repair, undo, redo), coalesced to one analysis per burst of changes.
  useEffect(() => {
    if (mesh === null) {
      seq.current++;
      setAnalysis(null);
      setAnalyzing(false);
      inspect.repair.analysis = null;
      return;
    }
    void analyze(mesh);
    let timer: ReturnType<typeof setTimeout> | undefined;
    const off = store.scene.onChanges((changes) => {
      if (changes.some((c) => c.mesh === mesh && !c.removed && (c.indices === "all" || c.indices.length > 0))) {
        if (own.current.pending) {
          own.current.pending = false;
        } else {
          setReport(null);
          inspect.repair.report = null;
        }
        clearTimeout(timer);
        timer = setTimeout(() => void analyze(mesh), 50);
      }
    });
    return () => {
      off();
      clearTimeout(timer);
    };
  }, [store, mesh, analyze]);

  const run = async (label: string, op: RepairOp) => {
    if (mesh === null || busy !== null) {
      return;
    }
    setBusy(label);
    inspect.repair.busy = label;
    clearTimeout(own.current.timer);
    own.current.pending = true;
    try {
      const r = (await store.repair(mesh, op)) as RepairReport;
      const line = describeReport(op.op, r);
      setReport({ mesh, line, failed: false });
      inspect.repair.report = line;
    } catch (e) {
      const line = `${label} failed: ${e instanceof Error ? e.message : String(e)}`;
      setReport({ mesh, line, failed: true });
      inspect.repair.report = line;
    } finally {
      setBusy(null);
      inspect.repair.busy = null;
      if (own.current.pending) {
        own.current.timer = setTimeout(() => (own.current.pending = false), OWN_CHANGE_MS);
      }
    }
  };

  if (item === null) {
    return (
      <section className="panel" aria-label="Repair">
        <h2>Repair</h2>
        <p className="muted">Select a mesh to analyse and repair it.</p>
      </section>
    );
  }

  const s = analysis?.mesh === mesh ? analysis.summary : null;
  const disabled = busy !== null;
  return (
    <section className="panel" aria-label="Repair" data-repair={item.name}>
      <header className="panel-head">
        <h2>Repair</h2>
        <button type="button" data-repair-op="analyze" onClick={() => void analyze(item.mesh)} disabled={analyzing}>
          Analyze
        </button>
      </header>
      <p className="muted small mesh-title" title={item.name}>
        {item.name}
      </p>
      {s === null ? (
        <p className="muted small">{analyzing ? "Analysing…" : "No analysis."}</p>
      ) : (
        <AnalysisList summary={s} stale={analyzing} />
      )}

      <div className="ops">
        <div className="op">
          <button
            type="button"
            data-repair-op="weld"
            disabled={disabled}
            onClick={() => void run("Weld", { op: "weld", tolerance: weldTolerance })}
          >
            Weld
          </button>
          <label className="field" title="Vertices at most this far apart are merged (mesh units)">
            within
            <input
              type="number"
              min={0}
              step="any"
              value={weldTolerance}
              onChange={(e) => setWeldTolerance(Math.max(0, Number(e.currentTarget.value) || 0))}
            />
          </label>
        </div>
        <div className="op">
          <button
            type="button"
            data-repair-op="removeDegenerate"
            disabled={disabled}
            onClick={() => void run("Remove degenerate", { op: "removeDegenerate" })}
          >
            Remove degenerate
          </button>
        </div>
        <div className="op">
          <button
            type="button"
            data-repair-op="orient"
            disabled={disabled}
            onClick={() => void run("Orient", { op: "orient" })}
          >
            Orient
          </button>
        </div>
        <div className="op">
          <button
            type="button"
            data-repair-op="removeSmallComponents"
            disabled={disabled}
            onClick={() => void run("Remove small components", { op: "removeSmallComponents", minTriangles })}
          >
            Remove small components
          </button>
          <label className="field" title="Components with fewer triangles are removed">
            below
            <input
              type="number"
              min={1}
              step={1}
              value={minTriangles}
              onChange={(e) => setMinTriangles(Math.max(1, Math.round(Number(e.currentTarget.value)) || 1))}
            />
            tris
          </label>
        </div>
        <div className="op">
          <button
            type="button"
            data-repair-op="fillHoles"
            disabled={disabled}
            onClick={() =>
              void run("Fill holes", { op: "fillHoles", fair, maxEdges: maxEdges > 0 ? maxEdges : ANY_SIZE })
            }
          >
            Fill holes
          </button>
          <label
            className="field"
            title="Only holes with at most this many edges. Empty: any size, which also fills an open mesh's outer boundary"
          >
            max
            <input
              type="number"
              min={0}
              step={1}
              placeholder="any"
              value={maxEdges === 0 ? "" : maxEdges}
              onChange={(e) => setMaxEdges(Math.max(0, Math.round(Number(e.currentTarget.value)) || 0))}
            />
            edges
          </label>
          <label className="check" title="Refine and fair the patch so it follows the surrounding curvature">
            <input type="checkbox" checked={fair} onChange={(e) => setFair(e.currentTarget.checked)} />
            Fair (follow curvature)
          </label>
        </div>
        <p className="muted small">Fill holes: an empty max fills any size, including an open mesh's outer boundary.</p>
      </div>

      <div className="ops">
        <div className="op">
          <span className="op-label">Brush</span>
          <label className="field" title="Brush center in local mesh coordinates">
            center
            <input
              type="number"
              step="any"
              aria-label="Brush center x"
              value={center[0]}
              onChange={(e) => setCenter([Number(e.currentTarget.value) || 0, center[1], center[2]])}
            />
            <input
              type="number"
              step="any"
              aria-label="Brush center y"
              value={center[1]}
              onChange={(e) => setCenter([center[0], Number(e.currentTarget.value) || 0, center[2]])}
            />
            <input
              type="number"
              step="any"
              aria-label="Brush center z"
              value={center[2]}
              onChange={(e) => setCenter([center[0], center[1], Number(e.currentTarget.value) || 0])}
            />
          </label>
          <label className="field" title="Brush radius in mesh units">
            radius
            <input
              type="number"
              min={0}
              step="any"
              value={brushRadius}
              onChange={(e) => setBrushRadius(Math.max(0, Number(e.currentTarget.value) || 0))}
            />
          </label>
          <label className="field" title="Displacement scale (negative dents)">
            strength
            <input
              type="number"
              step="any"
              value={strength}
              onChange={(e) => setStrength(Number(e.currentTarget.value) || 0)}
            />
          </label>
          <label className="field" title="How the strength falls off with distance">
            falloff
            <select value={falloff} onChange={(e) => setFalloff(e.currentTarget.value as "linear" | "smooth")}>
              <option value="smooth">smooth</option>
              <option value="linear">linear</option>
            </select>
          </label>
        </div>
        <div className="op">
          <button
            type="button"
            data-repair-op="inflate"
            disabled={disabled}
            onClick={() =>
              void run("Inflate", { op: "inflate", center, radius: brushRadius, falloff, strength })
            }
          >
            Inflate
          </button>
          <button
            type="button"
            data-repair-op="smooth"
            disabled={disabled}
            onClick={() =>
              void run("Smooth", { op: "smooth", center, radius: brushRadius, falloff, strength, cycles })
            }
          >
            Smooth
          </button>
          <label className="field" title="Taubin smoothing cycles">
            cycles
            <input
              type="number"
              min={0}
              step={1}
              value={cycles}
              onChange={(e) => setCycles(Math.max(0, Math.round(Number(e.currentTarget.value)) || 0))}
            />
          </label>
        </div>
        <div className="op">
          <button
            type="button"
            data-repair-op="remesh"
            disabled={disabled}
            onClick={() =>
              void run("Remesh", {
                op: "remesh",
                center,
                radius: brushRadius,
                ...(targetLength === "" ? {} : { targetLength: Number(targetLength) }),
                ...(maxRounds === "" ? {} : { maxRounds: Number(maxRounds) }),
              })
            }
          >
            Remesh
          </button>
          <label className="field" title="Target edge length (empty: the mesh mean)">
            length
            <input
              type="number"
              min={0}
              step="any"
              placeholder="auto"
              value={targetLength}
              onChange={(e) => setTargetLength(e.currentTarget.value)}
            />
          </label>
          <label className="field" title="Split/collapse/flip rounds (empty: default)">
            rounds
            <input
              type="number"
              min={0}
              step={1}
              placeholder="auto"
              value={maxRounds}
              onChange={(e) => setMaxRounds(e.currentTarget.value)}
            />
          </label>
        </div>
        <p className="muted small">Deform: inflate puffs the brush up (negative dents), smooth relaxes it, remesh retriangulates it.</p>
      </div>

      {busy !== null ? (
        <p className="busy small" data-repair-busy={busy}>
          {busy}: computing… <span className="progress" aria-hidden="true" />
        </p>
      ) : report !== null && report.mesh === mesh ? (
        <p className={report.failed ? "report error small" : "report small"} data-repair-report>
          {report.line} <span className="muted">(undo with Ctrl+Z)</span>
        </p>
      ) : null}
    </section>
  );
}

function AnalysisList(props: { summary: AnalysisSummary; stale: boolean }) {
  const { summary: s, stale } = props;
  const bad = (n: number) => (n > 0 ? "warn" : undefined);
  return (
    <dl
      className={stale ? "analysis stale" : "analysis"}
      data-holes={s.holes.length}
      data-components={s.components.length}
    >
      <dt>Holes</dt>
      <dd className={bad(s.holes.length)}>
        {s.holes.length === 0 ? "none" : s.holes.length}
        {s.holes.length > 0 ? (
          <ul className="holes">
            {s.holes.slice(0, HOLES_SHOWN).map((h, k) => (
              <li key={k}>
                {plural(h.edges, "edge")} · perimeter {formatDistance(h.perimeter)}
              </li>
            ))}
            {s.holes.length > HOLES_SHOWN ? <li className="muted">and {s.holes.length - HOLES_SHOWN} more</li> : null}
          </ul>
        ) : null}
        {s.openChains > 0 ? <span className="muted"> ({plural(s.openChains, "open boundary chain")})</span> : null}
      </dd>
      <dt>Orientation</dt>
      <dd className={bad(s.orientationConflicts)}>
        {s.orientationConflicts === 0 ? "consistent" : `${plural(s.orientationConflicts, "conflicting edge")}`}
      </dd>
      <dt>Components</dt>
      <dd className={bad(s.components.length - 1)}>
        {s.components.length}
        <span className="muted">
          {" "}
          (
          {s.components
            .slice(0, 3)
            .map((c) => `${c.triangles.toLocaleString("en")}${c.closed ? "" : " open"}`)
            .join(", ")}
          {s.components.length > 3 ? ", …" : ""} tris)
        </span>
      </dd>
      <dt>Non-manifold</dt>
      <dd className={bad(s.nonManifoldEdges + s.nonManifoldVertices + s.pinchVertices)}>
        {plural(s.nonManifoldEdges, "edge")}, {plural(s.nonManifoldVertices, "vertex", "vertices")}
        {s.pinchVertices > 0 ? `, ${plural(s.pinchVertices, "pinch", "pinches")}` : ""}
      </dd>
      <dt>Degenerate</dt>
      <dd className={bad(s.degenerate + s.duplicate)}>
        {plural(s.degenerate, "triangle")}
        {s.duplicate > 0 ? `, ${plural(s.duplicate, "duplicate")}` : ""}
      </dd>
    </dl>
  );
}
