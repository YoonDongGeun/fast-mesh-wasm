// The repair panel for the selected mesh: its analysis (holes and their
// sizes, orientation, components, non-manifold and degenerate elements),
// refreshed whenever its triangles change (a repair, its undo), and the
// library's repair ops, each one undo step through `store.repair`.
// Analysis is read-only: the app calls it on its own scene.
import type { MeshHandle, RepairOp, RepairReport, Scene } from "fast-mesh-wasm";
import { useScene, useSelection } from "fast-mesh-wasm/react";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useCallback, useEffect, useRef, useState } from "react";
import { type AppState, type Item, useApp } from "../scene/app-state";
import { formatDistance } from "./collision-math";
import { inspect } from "./inspect";
import { type AnalysisSummary, describeJoin, describeReport, plural, summarize } from "./repair-summary";

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
  const brushPick = useApp(app, (s) => s.brushPick);
  const sculpt = useApp(app, (s) => s.sculpt);
  const [sculptOp, setSculptOp] = useState<"inflate" | "smooth">("inflate");
  const sculpting = sculpt !== null && sculpt.mesh === mesh;
  // While armed, the brush inputs drive the sculpt brush live (radius and
  // strength are the ring and the dab: adjustable mid-stroke).
  useEffect(() => {
    if (sculpting && mesh !== null) {
      app.setSculpt({ mesh, op: sculptOp, radius: brushRadius, strength, falloff, cycles });
    }
  }, [app, sculpting, mesh, sculptOp, brushRadius, strength, falloff, cycles]);
  const [bridgeA, setBridgeA] = useState("");
  const [bridgeB, setBridgeB] = useState("");
  const [loopMax, setLoopMax] = useState(128);
  const [joinSource, setJoinSource] = useState<number | "">("");
  const [joinLoopA, setJoinLoopA] = useState("");
  const [joinLoopB, setJoinLoopB] = useState("");
  const [sourceSummary, setSourceSummary] = useState<AnalysisSummary | null>(null);
  // The brush follows the mesh: its bounding center and a quarter of its
  // half-diagonal. Strength and the rest are preferences and persist.
  useEffect(() => {
    if (item !== null) {
      setCenter(item.center);
      setBrushRadius(item.radius / 4);
    }
  }, [item]);
  // A landed pick for this mesh becomes the center (and clears the pick).
  useEffect(() => {
    if (brushPick !== null && brushPick.point !== null && brushPick.mesh === mesh) {
      setCenter(brushPick.point);
      app.setBrushPick(null);
    }
  }, [app, brushPick, mesh]);
  // Loop picks are loop ids: a fresh analysis (or mesh) invalidates them.
  // A sculpt brush belongs to its mesh: selecting away disarms it.
  useEffect(() => {
    setBridgeA("");
    setBridgeB("");
    setJoinLoopA("");
    setJoinSource("");
    setJoinLoopB("");
    setSourceSummary(null);
    if (app.getSnapshot().sculpt !== null) {
      app.setSculpt(null);
    }
  }, [app, mesh]);
  useEffect(() => {
    setBridgeA("");
    setBridgeB("");
    setJoinLoopA("");
  }, [analysis]);
  useEffect(() => {
    setJoinLoopB("");
  }, [sourceSummary]);
  // The join source's holes, fetched when the source changes.
  useEffect(() => {
    if (joinSource === "") {
      setSourceSummary(null);
      return;
    }
    const source = items.find((i) => i.id === joinSource) ?? null;
    if (source === null) {
      setSourceSummary(null);
      return;
    }
    let alive = true;
    void Promise.resolve()
      .then(() => scene.analyze(source.mesh, { maxListed: MAX_LISTED }))
      .then((a) => {
        if (alive) {
          setSourceSummary(summarize(a));
        }
      })
      .catch(() => {
        if (alive) {
          setSourceSummary(null);
        }
      });
    return () => {
      alive = false;
    };
  }, [scene, joinSource, items]);
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

  const runJoin = async (source: Item, loopA: number, loopB: number) => {
    if (mesh === null || busy !== null) {
      return;
    }
    const target = s?.holes[loopA] ?? null;
    const from = sourceSummary?.holes[loopB] ?? null;
    if (target === null || from === null) {
      return;
    }
    const targetName = items.find((i) => i.mesh === mesh)?.name ?? "mesh";
    setBusy("Join");
    inspect.repair.busy = "Join";
    clearTimeout(own.current.timer);
    own.current.pending = true;
    try {
      // Like a removal: release the pair colouring first, in the same tick
      // the engine drops the source.
      const snap = app.getSnapshot();
      if (snap.collisionMode === "pair" && (snap.collision.a === source.mesh || snap.collision.b === source.mesh)) {
        const other = snap.collision.a === source.mesh ? snap.collision.b : snap.collision.a;
        if (other !== null && other !== source.mesh) {
          store.setDisplay(other, { coloring: null });
        }
      }
      const r = await scene.join(mesh, source.mesh, { loopA: target.id, loopB: from.id, maxEdges: loopMax });
      app.dropItem(store, source);
      const line = `${describeJoin(r)} — ${source.name} removed (undo restores ${targetName} only)`;
      setReport({ mesh, line, failed: false });
      inspect.repair.report = line;
    } catch (e) {
      const line = `Join ${source.name} into ${targetName} failed: ${e instanceof Error ? e.message : String(e)}`;
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
          <button
            type="button"
            data-action="pick-center"
            aria-pressed={brushPick !== null && brushPick.point === null && brushPick.mesh === mesh}
            disabled={disabled}
            title="Click the mesh to set the brush center"
            onClick={() =>
              app.setBrushPick(
                brushPick !== null && brushPick.point === null && brushPick.mesh === mesh
                  ? null
                  : { mesh: item.mesh, point: null },
              )
            }
          >
            {brushPick !== null && brushPick.point === null && brushPick.mesh === mesh ? "Picking…" : "Pick"}
          </button>
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
          <label
            className="field"
            title={sculpting ? "Displacement per sculpt dab (negative dents)" : "Displacement scale (negative dents)"}
          >
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
            data-action="sculpt"
            aria-pressed={sculpting}
            disabled={disabled}
            title="Paint the brush on the mesh: hover shows the ring, press-and-hold sculpts (Esc disarms)"
            onClick={() => {
              if (sculpting || mesh === null) {
                app.setSculpt(null);
                return;
              }
              // A click-sized strength would explode per dab: arm with a dab-sized one.
              const dab = strength > 0.5 ? 0.1 : strength;
              if (dab !== strength) {
                setStrength(dab);
              }
              app.setSculpt({ mesh, op: sculptOp, radius: brushRadius, strength: dab, falloff, cycles });
            }}
          >
            {sculpting ? "Sculpting…" : "Sculpt"}
          </button>
          <label className="field" title="What a sculpt dab applies">
            paint
            <select
              value={sculptOp}
              disabled={disabled}
              onChange={(e) => setSculptOp(e.currentTarget.value as "inflate" | "smooth")}
            >
              <option value="inflate">inflate</option>
              <option value="smooth">smooth</option>
            </select>
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

      <div className="ops">
        <div className="op">
          <span className="op-label">Bridge</span>
          <label className="field" title="First loop (holes of this mesh, largest first)">
            loop A
            <select data-bridge-loop="a" value={bridgeA} onChange={(e) => setBridgeA(e.currentTarget.value)}>
              <option value="">—</option>
              {(s?.holes ?? []).map((h, n) => (
                <option key={n} value={n}>
                  #{n + 1} · {h.edges} edges
                </option>
              ))}
            </select>
          </label>
          <label className="field" title="Second loop">
            loop B
            <select data-bridge-loop="b" value={bridgeB} onChange={(e) => setBridgeB(e.currentTarget.value)}>
              <option value="">—</option>
              {(s?.holes ?? []).map((h, n) => (
                <option key={n} value={n}>
                  #{n + 1} · {h.edges} edges
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            data-repair-op="bridgeLoops"
            disabled={disabled || bridgeA === "" || bridgeB === "" || bridgeA === bridgeB}
            title={s === null || s.holes.length < 2 ? "Needs two holes" : "Stitch the two loops with a tube"}
            onClick={() => {
              const a = s?.holes[Number(bridgeA)] ?? null;
              const b = s?.holes[Number(bridgeB)] ?? null;
              if (a !== null && b !== null) {
                void run("Bridge loops", { op: "bridgeLoops", loops: [a.id, b.id], maxEdges: loopMax });
              }
            }}
          >
            Bridge
          </button>
          <label className="field" title="Maximum vertices per loop">
            max
            <input
              type="number"
              min={0}
              step={1}
              value={loopMax}
              onChange={(e) => setLoopMax(Math.max(0, Math.round(Number(e.currentTarget.value)) || 0))}
            />
          </label>
        </div>
        <div className="op">
          <span className="op-label">Join</span>
          <label className="field" title="The mesh copied into this one (and removed)">
            source
            <select
              data-join-source="true"
              value={joinSource}
              onChange={(e) => setJoinSource(e.currentTarget.value === "" ? "" : Number(e.currentTarget.value))}
            >
              <option value="">—</option>
              {items
                .filter((i) => i.mesh !== mesh)
                .map((i) => (
                  <option key={i.id} value={i.id}>
                    {i.name}
                  </option>
                ))}
            </select>
          </label>
          <label className="field" title="This mesh's loop">
            loop A
            <select data-join-loop="a" value={joinLoopA} onChange={(e) => setJoinLoopA(e.currentTarget.value)}>
              <option value="">—</option>
              {(s?.holes ?? []).map((h, n) => (
                <option key={n} value={n}>
                  #{n + 1} · {h.edges} edges
                </option>
              ))}
            </select>
          </label>
          <label className="field" title="The source's loop">
            loop B
            <select data-join-loop="b" value={joinLoopB} onChange={(e) => setJoinLoopB(e.currentTarget.value)}>
              <option value="">—</option>
              {(sourceSummary?.holes ?? []).map((h, n) => (
                <option key={n} value={n}>
                  #{n + 1} · {h.edges} edges
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            data-action="join"
            disabled={disabled || joinSource === "" || joinLoopA === "" || joinLoopB === ""}
            title="Copy the source into this mesh, bridge the loops, remove the source"
            onClick={() => {
              const source = items.find((i) => i.id === joinSource) ?? null;
              if (source !== null) {
                void runJoin(source, Number(joinLoopA), Number(joinLoopB));
              }
            }}
          >
            Join
          </button>
        </div>
        <p className="muted small">Bridge stitches two holes of this mesh; Join merges another mesh into it (the source is removed).</p>
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
