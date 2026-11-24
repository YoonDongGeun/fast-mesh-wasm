// How the repair panel runs one op at a time: the busy label, the report
// line it leaves (or the failure line), the Mesh Healer's before/after
// delta, and which triangle changes are the op's own. An op's own triangle
// change keeps its report; any other (undo, redo) makes the report stale
// and clears it. The test hook reads the same through `inspect.repair`.
import type { MeshHandle } from "fast-mesh-wasm";
import { useCallback, useRef, useState } from "react";
import { inspect } from "../../../app/state/readouts";

/** How long after an op its own triangle change may still arrive (it has none when nothing changed). */
const OWN_CHANGE_MS = 1000;

/** The last op's report line, for the mesh it ran on. */
export interface RepairReportLine {
  mesh: MeshHandle;
  line: string;
  failed: boolean;
}

export function useRepairRunner(mesh: MeshHandle | null) {
  const [busy, setBusy] = useState<string | null>(null);
  const [report, setReport] = useState<RepairReportLine | null>(null);
  const [healerDelta, setHealerDelta] = useState<{ mesh: MeshHandle; text: string } | null>(null);
  const own = useRef<{ pending: boolean; timer?: ReturnType<typeof setTimeout> }>({ pending: false });

  /** A triangle change of the panel's mesh arrived: the op's own keeps its report, any other clears it. */
  const noteTriangleChange = useCallback(() => {
    if (own.current.pending) {
      own.current.pending = false;
    } else {
      setReport(null);
      setHealerDelta(null);
      inspect.repair.report = null;
    }
  }, []);

  /** Shows `line` as the report for `target`. */
  const showReport = (target: MeshHandle, line: string, failed: boolean) => {
    setReport({ mesh: target, line, failed });
    inspect.repair.report = line;
  };

  /**
   * Runs `body` on the panel's mesh as the one busy op (nothing while
   * another runs, or with no mesh). `body` shows its own report; a throw
   * shows `failure(message)` instead.
   */
  const runGuarded = async (
    label: string,
    body: (target: MeshHandle) => Promise<void>,
    failure: (message: string) => string,
  ) => {
    if (mesh === null || busy !== null) {
      return;
    }
    setBusy(label);
    inspect.repair.busy = label;
    clearTimeout(own.current.timer);
    own.current.pending = true;
    try {
      await body(mesh);
    } catch (e) {
      showReport(mesh, failure(e instanceof Error ? e.message : String(e)), true);
    } finally {
      setBusy(null);
      inspect.repair.busy = null;
      if (own.current.pending) {
        own.current.timer = setTimeout(() => (own.current.pending = false), OWN_CHANGE_MS);
      }
    }
  };

  /** Runs `action` as the busy op and shows the line it returns (or `<label> failed: ...`). */
  const runBusy = (label: string, action: () => Promise<string>) =>
    runGuarded(
      label,
      async (target) => {
        const line = await action();
        showReport(target, line, false);
      },
      (message) => `${label} failed: ${message}`,
    );

  return { busy, report, healerDelta, setHealerDelta, noteTriangleChange, showReport, runGuarded, runBusy };
}

export type RepairRunner = ReturnType<typeof useRepairRunner>;
