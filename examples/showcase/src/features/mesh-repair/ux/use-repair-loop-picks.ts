// The repair panel's loop picks: the two Bridge loops, the two Arc bridge
// loops, the Fill continuity target, and Join's source mesh with a loop on
// each side, plus the source's holes and the shift-clicked Join queue.
import type { BoundaryLoop, MeshHandle, Scene } from "fast-mesh-wasm";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useEffect, useState } from "react";
import { type AppState, type Item, useApp } from "../../../app/state/app-state";
import { nextJoinSource } from "../../select/ux/selection";
import { type AnalysisSummary, summarize } from "./repair-summary";
import { MAX_LISTED } from "./use-repair-analysis";

/** A hole of the summary with the raw analysis loop it came from (its index and vertices). */
export type HoleTarget = AnalysisSummary["holes"][number] & {
  /** Its place in the summary's list (largest first). */
  displayIndex: number;
  /** Its index among the raw analysis loops (the summary's place when not found). */
  loopIndex: number;
  /** The raw loop's vertices, when the raw analysis has that loop. */
  vertices: Uint32Array | undefined;
};

/** Matches each summary hole to its raw analysis loop (same smallest vertex and length). */
export function holeTargetsOf(summary: AnalysisSummary | null, rawLoops: readonly BoundaryLoop[]): HoleTarget[] {
  return (summary?.holes ?? []).map((h, displayIndex) => {
    const rawIdx = rawLoops.findIndex(
      (l) => l.closed && l.id.minVertex === h.id.minVertex && l.id.len === h.id.len,
    );
    const loopIndex = rawIdx >= 0 ? rawIdx : displayIndex;
    return { ...h, displayIndex, loopIndex, vertices: rawLoops[loopIndex]?.vertices };
  });
}

export function useRepairLoopPicks(args: {
  app: AppState;
  scene: Scene | WorkerScene;
  mesh: MeshHandle | null;
  items: readonly Item[];
  /** The live analysis object: a new one clears this mesh's loop picks. */
  analysis: unknown;
}) {
  const { app, scene, mesh, items, analysis } = args;
  const [bridgeA, setBridgeA] = useState("");
  const [bridgeB, setBridgeB] = useState("");
  const [partialA, setPartialA] = useState("");
  const [partialB, setPartialB] = useState("");
  const [continuityLoop, setContinuityLoop] = useState("all");
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

  return {
    bridgeA,
    setBridgeA,
    bridgeB,
    setBridgeB,
    partialA,
    setPartialA,
    partialB,
    setPartialB,
    continuityLoop,
    setContinuityLoop,
    joinSource,
    setJoinSource,
    joinLoopA,
    setJoinLoopA,
    joinLoopB,
    setJoinLoopB,
    sourceSummary,
    queue,
  };
}

export type RepairLoopPicks = ReturnType<typeof useRepairLoopPicks>;
