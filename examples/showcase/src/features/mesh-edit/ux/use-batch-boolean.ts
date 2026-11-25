// The multi-mesh boolean section's behaviour (#254): the inputs from the
// multi-selection (primary first), the operation, backend and Keep inputs
// fields, and the run through `engine/workbench-ops.ts` with a one-line
// report.
import type { BooleanMethod, BooleanOp } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { useSelection } from "fast-mesh-wasm/react";
import { useState } from "react";
import { type AppState, type Item, useApp } from "../../../app/state/app-state";
import { runBatchBoolean } from "../../../engine/workbench-ops";
import { batchBlocked, batchInputs, describeBatchBoolean } from "./batch-boolean";
import type { ModelingReport } from "./use-modeling-ops";

export function useBatchBoolean(args: { app: AppState; store: SceneStore; items: readonly Item[] }) {
  const { app, store, items } = args;
  const multi = useApp(app, (s) => s.multi);
  const [primary] = useSelection();
  const [op, setOp] = useState<BooleanOp>("union");
  const [method, setMethod] = useState<BooleanMethod>("manifold");
  const [keepInputs, setKeepInputs] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [report, setReport] = useState<ModelingReport | null>(null);

  const inputs = batchInputs(multi, primary, items);
  const blocked = batchBlocked(inputs.length);

  const run = async () => {
    if (blocked !== null || busy !== null) {
      return;
    }
    const label = "Multi-mesh boolean";
    const runInputs = inputs;
    setBusy(label);
    setReport(null);
    try {
      const r = await runBatchBoolean({ store, app, inputs: runInputs, op, keepInputs, method });
      setReport({
        label,
        text: describeBatchBoolean({ op, inputs: runInputs.length, name: r.name, triangles: r.triangles, keptInputs: keepInputs }),
      });
    } catch (e) {
      setReport({ label, text: e instanceof Error ? e.message : String(e), error: true });
    } finally {
      setBusy(null);
    }
  };

  return {
    inputs,
    blocked,
    op,
    setOp,
    method,
    setMethod,
    keepInputs,
    setKeepInputs,
    busy,
    report,
    run,
  };
}
