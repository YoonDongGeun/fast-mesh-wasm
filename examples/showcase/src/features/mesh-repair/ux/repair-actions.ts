// The repair panel's actions, each one undo step: the per-op buttons
// through `store.repair`, the continuity fill and arc bridge as scene calls
// wrapped in one store step, Auto repair (the library's default pipeline,
// or a preset) and Selective heal with their before/after summaries, and
// Join, which copies another mesh in, bridges a loop pair and removes the
// source. The runner shows each report and tells the op's own triangle
// change from an undo's.
import type {
  BridgeStats,
  FillContinuity,
  FillContinuityStats,
  MeshHandle,
  RepairOp,
  RepairReport,
  Scene,
} from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import type { AppState, Item } from "../../../app/state/app-state";
import type { LengthUnit } from "../../../shared/ux/units";
import { runAutoRepairPipeline } from "./auto-repair";
import type { HealerOverlayCounts } from "./healer-defects";
import { buildSelectiveHealOps, formatHealerDelta, healerCountsOf, type SelectiveHealCategory } from "./mesh-healer";
import {
  type AnalysisSummary,
  buildPresetAutoRepairRequest,
  describeJoin,
  describeReport,
  formatPresetAutoRepairReport,
  plural,
  summarize,
} from "./repair-summary";
import { type LiveRepairAnalysis, MAX_LISTED } from "./use-repair-analysis";
import type { HoleTarget, RepairLoopPicks } from "./use-repair-loop-picks";
import { ANY_SIZE, type RepairOptions, voxelSizeField } from "./use-repair-options";
import type { RepairRunner } from "./use-repair-runner";

/** The boundary continuity grade of a fill: G0 flat, G1 tangent, G2 curvature. */
export function continuityGrade(continuity: FillContinuity): string {
  return continuity === "flat" ? "G0" : continuity === "tangent" ? "G1" : "G2";
}

/** The panel's actions for this render (they read its current options, picks and busy state). */
export function repairActions(ctx: {
  app: AppState;
  scene: Scene | WorkerScene;
  store: SceneStore;
  mesh: MeshHandle | null;
  items: readonly Item[];
  units: LengthUnit;
  options: RepairOptions;
  runner: RepairRunner;
  live: LiveRepairAnalysis;
  picks: RepairLoopPicks;
  /** The live analysis summary of `mesh` (`null` while there is none). */
  summary: AnalysisSummary | null;
  holeTargets: readonly HoleTarget[];
}) {
  const { app, scene, store, mesh, items, units, options, runner, live, picks, summary, holeTargets } = ctx;
  const { busy, runBusy, runGuarded, showReport, setHealerDelta } = runner;
  const { analyze, setSelfIntersects, checkSelfIntersections } = live;

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

  const fillContinuity = async () => {
    if (mesh === null) {
      return;
    }
    const { continuity, refinePatch, detectIslands, maxEdges } = options;
    const modeLabel = continuityGrade(continuity);
    await runWithUndo(`Fill continuity (${modeLabel})`, async () => {
      const stats: FillContinuityStats =
        picks.continuityLoop === "all"
          ? await scene.fillAllHolesContinuity(mesh, {
              continuity,
              refinePatch,
              detectIslands,
              ...(maxEdges > 0 ? { maxHoleVertices: maxEdges } : {}),
            })
          : await scene.fillHoleContinuity(mesh, Number(picks.continuityLoop), {
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

  const bridgePartialArcs = async () => {
    if (mesh === null || picks.partialA === "" || picks.partialB === "") {
      return;
    }
    const targetA = holeTargets[Number(picks.partialA)];
    const targetB = holeTargets[Number(picks.partialB)];
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
          arcARange: [options.arcAStart, options.arcAEnd],
          loopBIndex: targetB.loopIndex,
          arcBRange: [options.arcBStart, options.arcBEnd],
          twistOffset: options.partialTwist,
          subdivisions: options.partialSubdiv,
        },
      );
      return `Arc bridge: +${plural(stats.trianglesAdded, "triangle")}, +${plural(stats.verticesAdded, "vertex", "vertices")} (arcs ${stats.arcAVertices}x${stats.arcBVertices}, ${plural(stats.resultingLoops, "loop")} left)`;
    });
  };

  // One "Auto repair" click: the library's weld -> remove-degenerate -> orient
  // -> remove-small-components -> fill holes as ONE repair call (one grouped
  // undo step, all-or-nothing), using this panel's live settings, with a
  // per-stage report and a before/after summary; a preset (or an
  // intersection override) sends the preset request instead.
  const autoRepair = () =>
    runGuarded(
      "Auto repair",
      async (target) => {
        const { weldTolerance, minTriangles, maxEdges, fair, preset, siOverride, siVoxelSize } = options;
        const settings = { weldTolerance, minTriangles, maxEdges, fair };
        if (preset === "general" && siOverride === "auto") {
          let beforeExtra: HealerOverlayCounts | null = null;
          let afterExtra: HealerOverlayCounts | null = null;
          try {
            const rawBefore = await scene.analyze(target, { weldTolerance, maxListed: MAX_LISTED });
            beforeExtra = healerCountsOf(scene, target, rawBefore);
          } catch {
            beforeExtra = null;
          }
          const result = await runAutoRepairPipeline(
            {
              analyze: async () => summarize(await scene.analyze(target, { weldTolerance, maxListed: MAX_LISTED })),
              repair: (request) => store.repair(target, request),
            },
            settings,
          );
          try {
            const rawAfter = await scene.analyze(target, { weldTolerance, maxListed: MAX_LISTED });
            afterExtra = healerCountsOf(scene, target, rawAfter);
          } catch {
            afterExtra = null;
          }
          setHealerDelta({
            mesh: target,
            text: formatHealerDelta(result.before, result.after, beforeExtra, afterExtra),
          });
          showReport(target, result.line, false);
        } else {
          const inspectSi = checkSelfIntersections || preset === "print3d" || siOverride !== "auto";
          const aBefore = await scene.analyze(target, {
            weldTolerance,
            maxListed: MAX_LISTED,
            selfIntersection: inspectSi,
          });
          const beforeExtra = healerCountsOf(scene, target, aBefore);
          const request = buildPresetAutoRepairRequest({
            preset,
            selfIntersectionMode: siOverride === "auto" ? undefined : siOverride,
            ...settings,
            ...voxelSizeField(siVoxelSize),
          });
          const r = await store.repair(target, request);
          const aAfter = await scene.analyze(target, {
            weldTolerance,
            maxListed: MAX_LISTED,
            selfIntersection: inspectSi,
          });
          const afterExtra = healerCountsOf(scene, target, aAfter);
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
            mesh: target,
            text: formatHealerDelta(beforeSummary, afterSummary, beforeExtra, afterExtra),
          });
          showReport(target, formatted.line, false);
        }
        void analyze(target);
      },
      (message) => `Auto repair failed: ${message}`,
    );

  /** Runs only the checked healing stages, in one undo step, with the before/after delta. */
  const selectiveHeal = async (cats: Record<SelectiveHealCategory, boolean>) => {
    if (mesh === null || busy !== null) {
      return;
    }
    const ops = buildSelectiveHealOps(cats, {
      weldTolerance: options.weldTolerance,
      minTriangles: options.minTriangles,
      maxEdges: options.maxEdges,
      fair: options.fair,
      anySize: ANY_SIZE,
      selfIntersectionMode: options.healSiMode,
      siVoxelSize: options.siVoxelSize,
    });
    if (ops.length === 0) return;
    await runGuarded(
      "Selective heal",
      async (target) => {
        const { weldTolerance } = options;
        const inspectSi = checkSelfIntersections || cats.resolveSelfIntersections;
        const aBefore = await scene.analyze(target, {
          weldTolerance,
          maxListed: MAX_LISTED,
          selfIntersection: inspectSi,
        });
        const beforeExtra = healerCountsOf(scene, target, aBefore);
        const reports = (await store.repair(target, ops)) as RepairReport[];
        const aAfter = await scene.analyze(target, {
          weldTolerance,
          maxListed: MAX_LISTED,
          selfIntersection: inspectSi,
        });
        const afterExtra = healerCountsOf(scene, target, aAfter);
        const beforeSummary = summarize(aBefore);
        const afterSummary = summarize(aAfter);
        const delta = formatHealerDelta(beforeSummary, afterSummary, beforeExtra, afterExtra);
        const stageLines = ops.map((op, i) => describeReport(op.op, reports[i]!, units));
        const line = `Selective heal (${plural(ops.length, "stage")}, one undo step): ${delta}. ${stageLines.join("; ")}`;
        setSelfIntersects(aAfter.selfIntersects);
        setHealerDelta({ mesh: target, text: delta });
        showReport(target, line, false);
        void analyze(target);
      },
      (message) => `Selective heal failed: ${message}`,
    );
  };

  const runJoin = async (source: Item, loopA: number, loopB: number) => {
    if (mesh === null || busy !== null) {
      return;
    }
    const target = summary?.holes[loopA] ?? null;
    const from = picks.sourceSummary?.holes[loopB] ?? null;
    if (target === null || from === null) {
      return;
    }
    const targetName = items.find((i) => i.mesh === mesh)?.name ?? "mesh";
    await runGuarded(
      "Join",
      async (into) => {
        // Like a removal: release the pair colouring first, in the same tick
        // the engine drops the source.
        const snap = app.getSnapshot();
        if (snap.collisionMode === "pair" && (snap.collision.a === source.mesh || snap.collision.b === source.mesh)) {
          const other = snap.collision.a === source.mesh ? snap.collision.b : snap.collision.a;
          if (other !== null && other !== source.mesh) {
            store.setDisplay(other, { coloring: null });
          }
        }
        const r = await scene.join(into, source.mesh, { loopA: target.id, loopB: from.id, maxEdges: options.loopMax });
        app.dropItem(store, source);
        // The queue advances: the next secondary seeds itself.
        picks.setJoinSource("");
        const line = `${describeJoin(r)} — ${source.name} removed (undo restores ${targetName} only)`;
        showReport(into, line, false);
      },
      (message) => `Join ${source.name} into ${targetName} failed: ${message}`,
    );
  };

  return {
    weld: () => run("Weld", { op: "weld", tolerance: options.weldTolerance }),
    removeDegenerate: () => run("Remove degenerate", { op: "removeDegenerate" }),
    orient: () => run("Orient", { op: "orient" }),
    removeSmallComponents: () =>
      run("Remove small components", { op: "removeSmallComponents", minTriangles: options.minTriangles }),
    fillHoles: () =>
      run("Fill holes", {
        op: "fillHoles",
        fair: options.fair,
        maxEdges: options.maxEdges > 0 ? options.maxEdges : ANY_SIZE,
      }),
    fillContinuity,
    makeManifold: () => run("Make manifold", { op: "makeManifold" }),
    cleanupSlivers: () => run("Cleanup slivers", { op: "cleanupSlivers" }),
    resolveSelfIntersections: () =>
      run("Resolve self-intersections", {
        op: "resolveSelfIntersections",
        mode: options.healSiMode,
        ...(options.healSiMode === "voxelHeal" ? voxelSizeField(options.siVoxelSize) : {}),
      }),
    autoRepair,
    selectiveHeal,
    /** Stitches the two picked holes of this mesh with a tube. */
    bridgeLoops: () => {
      const a = summary?.holes[Number(picks.bridgeA)] ?? null;
      const b = summary?.holes[Number(picks.bridgeB)] ?? null;
      if (a !== null && b !== null) {
        void run("Bridge loops", { op: "bridgeLoops", loops: [a.id, b.id], maxEdges: options.loopMax });
      }
    },
    bridgePartialArcs,
    /** Copies the picked source into this mesh, bridges the picked loops and removes the source. */
    join: () => {
      const source = items.find((i) => i.id === picks.joinSource) ?? null;
      if (source !== null) {
        void runJoin(source, Number(picks.joinLoopA), Number(picks.joinLoopB));
      }
    },
  };
}
