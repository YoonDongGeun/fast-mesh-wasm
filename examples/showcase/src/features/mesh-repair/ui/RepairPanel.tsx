// The repair panel for the selected mesh: its analysis (holes and their
// sizes, orientation, components, non-manifold and degenerate elements),
// refreshed whenever its triangles change (a repair, its undo), the 3D
// overlays drawing those issues on the mesh, and the library's repair ops,
// each one undo step through `store.repair`.
// Analysis is read-only: the app calls it on its own scene.
import type {
  Analysis,
  AutoRepairPreset,
  BridgeStats,
  FillContinuity,
  FillContinuityStats,
  MeshHandle,
  RepairOp,
  RepairReport,
  Scene,
  SelfIntersectionMode,
} from "fast-mesh-wasm";
import { useScene, useSelection } from "fast-mesh-wasm/react";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useCallback, useEffect, useRef, useState } from "react";
import { REPAIR_OVERLAY_KINDS, type AppState, type Item, type RepairOverlayKind, useApp } from "../../../app/state/app-state";
import { nextJoinSource } from "../../select/ux/selection";
import { countHealerOverlays, type HealerOverlayCounts } from "./healer-overlays";
import { REPAIR_OVERLAY_COLORS } from "./repair-overlays";
import { inspect } from "../../../app/state/readouts";
import {
  buildSelectiveHealOps,
  formatHealerDelta,
  MeshHealerSection,
  type SelectiveHealCategory,
} from "./MeshHealerSection";
import {
  AUTO_REPAIR_ANY_SIZE,
  runAutoRepairPipeline,
} from "../ux/auto-repair";
import {
  type AnalysisSummary,
  AUTO_REPAIR_PRESET_LABELS,
  buildPresetAutoRepairRequest,
  describeJoin,
  describeReport,
  formatPresetAutoRepairReport,
  plural,
  SELF_INTERSECTION_MODE_LABELS,
  summarize,
} from "../ux/repair-summary";
import { formatLength, type LengthUnit } from "../../../shared/ux/units";
import { LengthInput } from "../../../shared/ui/LengthInput";

/** The most entries the analysis lists (counts stay exact). */
const MAX_LISTED = 64;
/** "Any size" for Fill holes (an empty field): the op needs a bound (or a list of loops). */
const ANY_SIZE = AUTO_REPAIR_ANY_SIZE;
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
  const units = useApp(app, (s) => s.units);

  const [analysis, setAnalysis] = useState<{ mesh: MeshHandle; summary: AnalysisSummary } | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [report, setReport] = useState<{ mesh: MeshHandle; line: string; failed: boolean } | null>(null);
  const [healerDelta, setHealerDelta] = useState<{ mesh: MeshHandle; text: string } | null>(null);
  const repairAnalysis = useApp(app, (snap) => snap.repairAnalysis);
  const [weldTolerance, setWeldTolerance] = useState(1e-5);
  const [minTriangles, setMinTriangles] = useState(500);
  const [maxEdges, setMaxEdges] = useState(DEFAULT_MAX_EDGES);
  const [fair, setFair] = useState(true);
  const [continuity, setContinuity] = useState<FillContinuity>("curvature");
  const [refinePatch, setRefinePatch] = useState(true);
  const [detectIslands, setDetectIslands] = useState(true);
  const [continuityLoop, setContinuityLoop] = useState("all");
  const [preset, setPreset] = useState<AutoRepairPreset>("general");
  const [siOverride, setSiOverride] = useState<SelfIntersectionMode | "auto">("auto");
  const [healSiMode, setHealSiMode] = useState<SelfIntersectionMode>("deleteAndFill");
  const [siVoxelSize, setSiVoxelSize] = useState("");
  const [checkSelfIntersections, setCheckSelfIntersections] = useState(false);
  const [selfIntersects, setSelfIntersects] = useState<boolean | null>(null);
  const [bridgeA, setBridgeA] = useState("");
  const [bridgeB, setBridgeB] = useState("");
  const [partialA, setPartialA] = useState("");
  const [partialB, setPartialB] = useState("");
  const [arcAStart, setArcAStart] = useState(0);
  const [arcAEnd, setArcAEnd] = useState(3);
  const [arcBStart, setArcBStart] = useState(0);
  const [arcBEnd, setArcBEnd] = useState(3);
  const [partialTwist, setPartialTwist] = useState(0);
  const [partialSubdiv, setPartialSubdiv] = useState(1);
  const [loopMax, setLoopMax] = useState(128);
  const [joinSource, setJoinSource] = useState<number | "">("");
  const [joinLoopA, setJoinLoopA] = useState("");
  const [joinLoopB, setJoinLoopB] = useState("");
  const [sourceSummary, setSourceSummary] = useState<AnalysisSummary | null>(null);
  // The shift-clicked Join queue: every secondary but the target, in click order.
  const multi = useApp(app, (s) => s.multi);
  const queue = multi.filter((m) => m !== mesh).map((m) => items.find((i) => i.mesh === m) ?? null);
  // A multi-selection seeds the Join source (a manual pick stands while it
  // is still queued); each join drops its source and the next seeds itself.
  useEffect(() => {
    const ids: number[] = [];
    for (const m of multi) {
      if (m !== mesh) {
        const found = items.find((i) => i.mesh === m);
        if (found !== undefined) {
          ids.push(found.id);
        }
      }
    }
    setJoinSource((cur) => nextJoinSource(ids, cur));
  }, [mesh, multi, items]);
  // Loop picks are loop ids: a fresh analysis (or mesh) invalidates them.
  // Note: selecting another mesh no longer disarms an active sculpt brush (#228, #229).
  useEffect(() => {
    setBridgeA("");
    setBridgeB("");
    setPartialA("");
    setPartialB("");
    setContinuityLoop("all");
    setJoinLoopA("");
    setJoinSource("");
    setJoinLoopB("");
    setSourceSummary(null);
  }, [mesh]);
  useEffect(() => {
    setBridgeA("");
    setBridgeB("");
    setPartialA("");
    setPartialB("");
    setContinuityLoop("all");
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
  const checkSiRef = useRef(checkSelfIntersections);
  checkSiRef.current = checkSelfIntersections;

  const analyze = useCallback(
    async (target: MeshHandle) => {
      const n = ++seq.current;
      setAnalyzing(true);
      try {
        const a = await scene.analyze(target, {
          weldTolerance: weldRef.current,
          maxListed: MAX_LISTED,
          selfIntersection: checkSiRef.current,
        });
        if (n === seq.current) {
          const summary = summarize(a);
          setAnalysis({ mesh: target, summary });
          setSelfIntersects(a.selfIntersects);
          // The 3D overlay shares the raw analysis (loop vertices, edge and
          // triangle lists the summary drops); a new object refreshes it.
          app.setRepairAnalysis({ mesh: target, analysis: a });
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
      setSelfIntersects(null);
      setAnalyzing(false);
      inspect.repair.analysis = null;
      app.setRepairAnalysis(null);
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
          setHealerDelta(null);
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

  const runBusy = async (label: string, action: () => Promise<string>) => {
    if (mesh === null || busy !== null) {
      return;
    }
    setBusy(label);
    inspect.repair.busy = label;
    clearTimeout(own.current.timer);
    own.current.pending = true;
    try {
      const line = await action();
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

  const run = async (label: string, op: RepairOp) =>
    runBusy(label, async () => {
      const r = (await store.repair(mesh!, op)) as RepairReport;
      return describeReport(op.op, r, units);
    });

  const runWithUndo = async (label: string, work: () => Promise<string>) =>
    runBusy(label, async () => {
      store.beginStep();
      try {
        const line = await work();
        store.endStep();
        void analyze(mesh!);
        return line;
      } catch (err) {
        store.cancelStep();
        throw err;
      }
    });

  const runContinuityFill = async () => {
    if (mesh === null) {
      return;
    }
    const modeLabel = continuity === "flat" ? "G0" : continuity === "tangent" ? "G1" : "G2";
    await runWithUndo(`Fill continuity (${modeLabel})`, async () => {
      const stats: FillContinuityStats =
        continuityLoop === "all"
          ? await scene.fillAllHolesContinuity(mesh, {
              continuity,
              refinePatch,
              detectIslands,
              ...(maxEdges > 0 ? { maxHoleVertices: maxEdges } : {}),
            })
          : await scene.fillHoleContinuity(mesh, Number(continuityLoop), {
              continuity,
              refinePatch,
              detectIslands,
            });
      if (stats.filledTriangles === 0) {
        return `Fill continuity (${modeLabel}): no matching holes filled`;
      }
      const islands =
        stats.islandLoopsMerged > 0 ? `, ${plural(stats.islandLoopsMerged, "island")} merged` : "";
      return `Fill continuity (${modeLabel}): +${plural(stats.filledTriangles, "triangle")}, +${plural(stats.filledVertices, "vertex", "vertices")}${islands}`;
    });
  };

  const runPartialBridge = async () => {
    if (mesh === null || partialA === "" || partialB === "") {
      return;
    }
    const targetA = holeTargets[Number(partialA)];
    const targetB = holeTargets[Number(partialB)];
    if (targetA === undefined || targetB === undefined) {
      return;
    }
    await runWithUndo("Bridge partial arcs", async () => {
      const stats: BridgeStats = await scene.bridgePartialArcs(
        mesh,
        targetA.vertices ?? [],
        targetB.vertices ?? [],
        {
          loopAIndex: targetA.loopIndex,
          arcARange: [arcAStart, arcAEnd],
          loopBIndex: targetB.loopIndex,
          arcBRange: [arcBStart, arcBEnd],
          twistOffset: partialTwist,
          subdivisions: partialSubdiv,
        },
      );
      return `Arc bridge: +${plural(stats.trianglesAdded, "triangle")}, +${plural(stats.verticesAdded, "vertex", "vertices")} (arcs ${stats.arcAVertices}x${stats.arcBVertices}, ${plural(stats.resultingLoops, "loop")} left)`;
    });
  };

  // One "Auto repair" click: the library's weld -> remove-degenerate -> orient
  // -> remove-small-components -> fill holes as ONE repair call (one grouped
  // undo step, all-or-nothing), using this panel's live settings, with a
  // per-stage report and a before/after summary. The triangle-change effect
  const snapshotHealerCounts = (targetMesh: MeshHandle, raw: Analysis): HealerOverlayCounts | null => {
    try {
      const views = scene.views(targetMesh);
      return countHealerOverlays(raw, views.positions(), views.indices());
    } catch {
      return null;
    }
  };

  // Run the repair pipeline as a single undo step and record the before/after delta.
  const runAuto = async () => {
    if (mesh === null || busy !== null) {
      return;
    }
    setBusy("Auto repair");
    inspect.repair.busy = "Auto repair";
    clearTimeout(own.current.timer);
    own.current.pending = true;
    try {
      const settings = { weldTolerance, minTriangles, maxEdges, fair };
      if (preset === "general" && siOverride === "auto") {
        let beforeExtra: HealerOverlayCounts | null = null;
        let afterExtra: HealerOverlayCounts | null = null;
        try {
          const rawBefore = await scene.analyze(mesh, { weldTolerance, maxListed: MAX_LISTED });
          beforeExtra = snapshotHealerCounts(mesh, rawBefore);
        } catch {
          beforeExtra = null;
        }
        const result = await runAutoRepairPipeline(
          {
            analyze: async () => summarize(await scene.analyze(mesh, { weldTolerance, maxListed: MAX_LISTED })),
            repair: (request) => store.repair(mesh, request),
          },
          settings,
        );
        try {
          const rawAfter = await scene.analyze(mesh, { weldTolerance, maxListed: MAX_LISTED });
          afterExtra = snapshotHealerCounts(mesh, rawAfter);
        } catch {
          afterExtra = null;
        }
        setHealerDelta({
          mesh,
          text: formatHealerDelta(result.before, result.after, beforeExtra, afterExtra),
        });
        setReport({ mesh, line: result.line, failed: false });
        inspect.repair.report = result.line;
      } else {
        const inspectSi = checkSelfIntersections || preset === "print3d" || siOverride !== "auto";
        const aBefore = await scene.analyze(mesh, {
          weldTolerance,
          maxListed: MAX_LISTED,
          selfIntersection: inspectSi,
        });
        const beforeExtra = snapshotHealerCounts(mesh, aBefore);
        const request = buildPresetAutoRepairRequest({
          preset,
          selfIntersectionMode: siOverride === "auto" ? undefined : siOverride,
          ...settings,
          ...(siVoxelSize !== "" && Number(siVoxelSize) > 0 ? { voxelSize: Number(siVoxelSize) } : {}),
        });
        const r = await store.repair(mesh, request);
        const aAfter = await scene.analyze(mesh, {
          weldTolerance,
          maxListed: MAX_LISTED,
          selfIntersection: inspectSi,
        });
        const afterExtra = snapshotHealerCounts(mesh, aAfter);
        const beforeSummary = summarize(aBefore);
        const afterSummary = summarize(aAfter);
        const formatted = formatPresetAutoRepairReport(
          preset,
          r,
          beforeSummary,
          afterSummary,
          aBefore.selfIntersects,
          aAfter.selfIntersects,
          units,
        );
        setSelfIntersects(aAfter.selfIntersects);
        setHealerDelta({
          mesh,
          text: formatHealerDelta(beforeSummary, afterSummary, beforeExtra, afterExtra),
        });
        setReport({ mesh, line: formatted.line, failed: false });
        inspect.repair.report = formatted.line;
      }
      void analyze(mesh);
    } catch (e) {
      const line = `Auto repair failed: ${e instanceof Error ? e.message : String(e)}`;
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

  const runSelective = async (cats: Record<SelectiveHealCategory, boolean>) => {
    if (mesh === null || busy !== null) {
      return;
    }
    const ops = buildSelectiveHealOps(cats, {
      weldTolerance,
      minTriangles,
      maxEdges,
      fair,
      anySize: ANY_SIZE,
      selfIntersectionMode: healSiMode,
      siVoxelSize,
    });
    if (ops.length === 0) return;
    setBusy("Selective heal");
    inspect.repair.busy = "Selective heal";
    clearTimeout(own.current.timer);
    own.current.pending = true;
    try {
      const inspectSi = checkSelfIntersections || cats.resolveSelfIntersections;
      const aBefore = await scene.analyze(mesh, {
        weldTolerance,
        maxListed: MAX_LISTED,
        selfIntersection: inspectSi,
      });
      const beforeExtra = snapshotHealerCounts(mesh, aBefore);
      const reports = (await store.repair(mesh, ops)) as RepairReport[];
      const aAfter = await scene.analyze(mesh, {
        weldTolerance,
        maxListed: MAX_LISTED,
        selfIntersection: inspectSi,
      });
      const afterExtra = snapshotHealerCounts(mesh, aAfter);
      const beforeSummary = summarize(aBefore);
      const afterSummary = summarize(aAfter);
      const delta = formatHealerDelta(beforeSummary, afterSummary, beforeExtra, afterExtra);
      const stageLines = ops.map((op, i) => describeReport(op.op, reports[i]!, units));
      const line = `Selective heal (${plural(ops.length, "stage")}, one undo step): ${delta}. ${stageLines.join("; ")}`;
      setSelfIntersects(aAfter.selfIntersects);
      setHealerDelta({ mesh, text: delta });
      setReport({ mesh, line, failed: false });
      inspect.repair.report = line;
      void analyze(mesh);
    } catch (e) {
      const line = `Selective heal failed: ${e instanceof Error ? e.message : String(e)}`;
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
      // The queue advances: the next secondary seeds itself.
      setJoinSource("");
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
      <section className="panel" aria-label="Repair" data-tour="repair">
        <h2>Repair</h2>
        <p className="muted">Select a mesh to analyse and repair it.</p>
      </section>
    );
  }

  const s = analysis?.mesh === mesh ? analysis.summary : null;
  const rawLoops = repairAnalysis?.mesh === mesh ? repairAnalysis.analysis.loops : [];
  const holeTargets = (s?.holes ?? []).map((h, displayIndex) => {
    const rawIdx = rawLoops.findIndex(
      (l) => l.closed && l.id.minVertex === h.id.minVertex && l.id.len === h.id.len,
    );
    const loopIndex = rawIdx >= 0 ? rawIdx : displayIndex;
    return { ...h, displayIndex, loopIndex, vertices: rawLoops[loopIndex]?.vertices };
  });
  const disabled = busy !== null;
  return (
    <section className="panel" aria-label="Repair" data-repair={item.name} data-tour="repair">
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
        <>
          <AnalysisList summary={s} selfIntersects={selfIntersects} stale={analyzing} unit={units} />
          <label className="check" title="Include triangle self-intersection detection in live analysis">
            <input
              type="checkbox"
              data-check-self-intersections
              checked={checkSelfIntersections}
              onChange={(e) => {
                const on = e.currentTarget.checked;
                setCheckSelfIntersections(on);
                checkSiRef.current = on;
                if (mesh !== null) {
                  void analyze(mesh);
                }
              }}
            />
            Check self-intersections
          </label>
          <OverlayToggles app={app} summary={s} />
          {mesh !== null ? (
            <MeshHealerSection
              app={app}
              scene={scene}
              mesh={mesh}
              rawAnalysis={repairAnalysis?.mesh === mesh ? repairAnalysis.analysis : null}
              summary={s}
              disabled={disabled}
              deltaLine={healerDelta?.mesh === mesh ? healerDelta.text : null}
              onRunSelective={runSelective}
            />
          ) : null}
        </>
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
          <label className="field" title="Vertices at most this far apart are merged">
            within ({units})
            <LengthInput min={0} mm={weldTolerance} unit={units} onMm={(mm) => setWeldTolerance(Math.max(0, mm))} />
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
        <div className="op" data-continuity-fill="true">
          <button
            type="button"
            data-repair-op="fillContinuity"
            disabled={disabled}
            title="Fill holes using selectable G0/G1/G2 boundary continuity (#210)"
            onClick={() => void runContinuityFill()}
          >
            Fill continuity
          </button>
          <label className="field" title="Boundary seam continuity across filled patch">
            continuity
            <select
              data-continuity-mode="true"
              value={continuity}
              onChange={(e) => setContinuity(e.currentTarget.value as FillContinuity)}
            >
              <option value="flat">Flat (G0)</option>
              <option value="tangent">Tangent (G1)</option>
              <option value="curvature">Curvature (G2)</option>
            </select>
          </label>
          <label className="field" title="Fill all eligible holes or one specific boundary loop">
            target
            <select
              data-continuity-target="true"
              value={continuityLoop}
              onChange={(e) => setContinuityLoop(e.currentTarget.value)}
            >
              <option value="all">All holes</option>
              {holeTargets.map((h) => (
                <option key={h.displayIndex} value={String(h.loopIndex)}>
                  #{h.displayIndex + 1} · {h.edges} edges
                </option>
              ))}
            </select>
          </label>
          <label className="check" title="Subdivide and smooth interior patch triangles">
            <input
              type="checkbox"
              data-continuity-refine="true"
              checked={refinePatch}
              onChange={(e) => setRefinePatch(e.currentTarget.checked)}
            />
            Refine Patch
          </label>
          <label className="check" title="Detect and merge enclosed nested island loops inside holes">
            <input
              type="checkbox"
              data-continuity-islands="true"
              checked={detectIslands}
              onChange={(e) => setDetectIslands(e.currentTarget.checked)}
            />
            Detect Islands
          </label>
        </div>
        <p className="muted small">Fill holes: an empty max fills any size, including an open mesh's outer boundary.</p>
        <div className="op">
          <button
            type="button"
            data-repair-op="makeManifold"
            disabled={disabled}
            onClick={() => void run("Make manifold", { op: "makeManifold" })}
          >
            Make manifold
          </button>
          <button
            type="button"
            data-repair-op="cleanupSlivers"
            disabled={disabled}
            onClick={() => void run("Cleanup slivers", { op: "cleanupSlivers" })}
          >
            Cleanup slivers
          </button>
        </div>
        <div className="op">
          <button
            type="button"
            data-repair-op="resolveSelfIntersections"
            disabled={disabled}
            onClick={() =>
              void run("Resolve self-intersections", {
                op: "resolveSelfIntersections",
                mode: healSiMode,
                ...(healSiMode === "voxelHeal" && siVoxelSize !== "" && Number(siVoxelSize) > 0
                  ? { voxelSize: Number(siVoxelSize) }
                  : {}),
              })
            }
          >
            Resolve self-intersections
          </button>
          <label className="field" title="Self-intersection resolution mode">
            mode
            <select
              data-si-mode
              value={healSiMode}
              onChange={(e) => setHealSiMode(e.currentTarget.value as SelfIntersectionMode)}
            >
              {(Object.keys(SELF_INTERSECTION_MODE_LABELS) as SelfIntersectionMode[]).map((mode) => (
                <option key={mode} value={mode}>
                  {SELF_INTERSECTION_MODE_LABELS[mode]}
                </option>
              ))}
            </select>
          </label>
          {healSiMode === "voxelHeal" ? (
            <label className="field" title="Voxel spacing (empty = automatic)">
              voxel ({units})
              <input
                type="number"
                min={0}
                step="any"
                placeholder="auto"
                data-si-voxel
                value={siVoxelSize}
                onChange={(e) => setSiVoxelSize(e.currentTarget.value)}
              />
            </label>
          ) : null}
        </div>
      </div>

      <div className="ops">
        <div className="op">
          <label className="field" title="Domain repair preset selecting pipeline stages and tolerances">
            preset
            <select
              data-repair-preset
              value={preset}
              onChange={(e) => setPreset(e.currentTarget.value as AutoRepairPreset)}
            >
              {(Object.keys(AUTO_REPAIR_PRESET_LABELS) as AutoRepairPreset[]).map((p) => (
                <option key={p} value={p}>
                  {AUTO_REPAIR_PRESET_LABELS[p]}
                </option>
              ))}
            </select>
          </label>
          <label className="field" title="Override self-intersection healing stage for Auto repair">
            intersections
            <select
              data-repair-si-mode
              value={siOverride}
              onChange={(e) => setSiOverride(e.currentTarget.value as SelfIntersectionMode | "auto")}
            >
              <option value="auto">Preset default</option>
              {(Object.keys(SELF_INTERSECTION_MODE_LABELS) as SelfIntersectionMode[]).map((mode) => (
                <option key={mode} value={mode}>
                  {SELF_INTERSECTION_MODE_LABELS[mode]}
                </option>
              ))}
            </select>
          </label>
          <button type="button" data-repair-op="autoRepair" disabled={disabled} onClick={() => void runAuto()}>
            Auto repair
          </button>
          <p className="muted small">
            {preset === "general" && siOverride === "auto"
              ? "Order: weld → remove degenerate → orient → remove small components → fill holes. Uses the tolerance, “below”, “max edges” and “fair” settings above; one grouped undo step (Ctrl+Z). Reports one line per stage with before → after counts."
              : `Preset ${AUTO_REPAIR_PRESET_LABELS[preset]}: tailors healing stages and tolerances (${preset === "print3d" ? "watertight solid with self-intersection healing + volumetric fallback" : preset === "scan3d" ? "non-manifold healing, outlier pruning, slivers & smooth hole filling" : preset === "cadImport" ? "duplicate & non-manifold cleanup keeping sharp CAD boundaries open" : "standard 5-stage cleanup"}) in one undo step (Ctrl+Z).`}
          </p>
        </div>
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
        <div className="op" data-partial-bridge="true">
          <span className="op-label">Arc bridge</span>
          <label className="field" title="First boundary loop">
            loop A
            <select
              data-partial-bridge-loop="a"
              value={partialA}
              onChange={(e) => setPartialA(e.currentTarget.value)}
            >
              <option value="">—</option>
              {holeTargets.map((h) => (
                <option key={h.displayIndex} value={h.displayIndex}>
                  #{h.displayIndex + 1} · {h.edges} edges
                </option>
              ))}
            </select>
          </label>
          <label className="field" title="Start..end vertex indices along loop A">
            arc A
            <input
              type="number"
              data-partial-arc="a-start"
              min={0}
              step={1}
              value={arcAStart}
              onChange={(e) => setArcAStart(Math.max(0, Math.round(Number(e.currentTarget.value)) || 0))}
            />
            <span>-</span>
            <input
              type="number"
              data-partial-arc="a-end"
              min={0}
              step={1}
              value={arcAEnd}
              onChange={(e) => setArcAEnd(Math.max(0, Math.round(Number(e.currentTarget.value)) || 0))}
            />
          </label>
          <label className="field" title="Second boundary loop (may match loop A to split a hole)">
            loop B
            <select
              data-partial-bridge-loop="b"
              value={partialB}
              onChange={(e) => setPartialB(e.currentTarget.value)}
            >
              <option value="">—</option>
              {holeTargets.map((h) => (
                <option key={h.displayIndex} value={h.displayIndex}>
                  #{h.displayIndex + 1} · {h.edges} edges
                </option>
              ))}
            </select>
          </label>
          <label className="field" title="Start..end vertex indices along loop B">
            arc B
            <input
              type="number"
              data-partial-arc="b-start"
              min={0}
              step={1}
              value={arcBStart}
              onChange={(e) => setArcBStart(Math.max(0, Math.round(Number(e.currentTarget.value)) || 0))}
            />
            <span>-</span>
            <input
              type="number"
              data-partial-arc="b-end"
              min={0}
              step={1}
              value={arcBEnd}
              onChange={(e) => setArcBEnd(Math.max(0, Math.round(Number(e.currentTarget.value)) || 0))}
            />
          </label>
          <label className="field" title="Cyclic offset along arc B">
            twist
            <input
              type="number"
              data-partial-twist="true"
              step={1}
              value={partialTwist}
              onChange={(e) => setPartialTwist(Math.round(Number(e.currentTarget.value)) || 0)}
            />
          </label>
          <label className="field" title="Interior bridge subdivisions">
            subdiv
            <input
              type="number"
              data-partial-subdiv="true"
              min={0}
              step={1}
              value={partialSubdiv}
              onChange={(e) => setPartialSubdiv(Math.max(0, Math.round(Number(e.currentTarget.value)) || 0))}
            />
          </label>
          <button
            type="button"
            data-repair-op="bridgePartialArcs"
            disabled={disabled || partialA === "" || partialB === ""}
            title="Bridge two partial boundary arcs with undo (#210)"
            onClick={() => void runPartialBridge()}
          >
            Bridge arc
          </button>
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
        {queue.some((q) => q !== null) ? (
          <p className="muted small" data-join-queue>
            Shift-selected join into {item.name}, in order:{" "}
            {queue
              .filter((q) => q !== null)
              .map((q) => q.name)
              .join(", then ")}
            .
          </p>
        ) : null}
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

function AnalysisList(props: {
  summary: AnalysisSummary;
  selfIntersects: boolean | null;
  stale: boolean;
  unit: LengthUnit;
}) {
  const { summary: s, selfIntersects, stale, unit } = props;
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
                {plural(h.edges, "edge")} · perimeter {formatLength(h.perimeter, unit)}
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
      <dt>Self-intersections</dt>
      <dd className={selfIntersects ? "warn" : undefined}>
        {selfIntersects === null ? "not checked" : selfIntersects ? "detected" : "none"}
      </dd>
    </dl>
  );
}

/** The overlay toggles: which analysed issues draw on the mesh in 3D. */
const OVERLAY_LABELS: Record<RepairOverlayKind, string> = {
  holes: "Holes",
  nonManifold: "Non-manifold",
  degenerate: "Degenerate",
  components: "Small components",
};

function OverlayToggles(props: { app: AppState; summary: AnalysisSummary }) {
  const { app, summary: s } = props;
  const kinds = useApp(app, (snap) => snap.repairOverlays);
  const counts: Record<RepairOverlayKind, number> = {
    holes: s.holes.length,
    nonManifold: s.nonManifoldEdges,
    degenerate: s.degenerate + s.duplicate,
    components: Math.max(0, s.components.length - 1),
  };
  return (
    <fieldset className="overlays" data-overlays>
      <legend className="muted small">Show in 3D</legend>
      {REPAIR_OVERLAY_KINDS.map((kind) => (
        <label
          key={kind}
          className="check"
          title={`${OVERLAY_LABELS[kind]}: ${counts[kind] === 0 ? "none" : plural(counts[kind], "issue")} on the mesh`}
        >
          <input
            type="checkbox"
            data-overlay={kind}
            checked={kinds[kind]}
            onChange={(e) => app.setRepairOverlay(kind, e.currentTarget.checked)}
          />
          <span
            aria-hidden="true"
            style={{
              display: "inline-block",
              width: "0.7em",
              height: "0.7em",
              borderRadius: "2px",
              background: `#${REPAIR_OVERLAY_COLORS[kind].toString(16).padStart(6, "0")}`,
            }}
          />{" "}
          {OVERLAY_LABELS[kind]} ({counts[kind]})
        </label>
      ))}
    </fieldset>
  );
}
