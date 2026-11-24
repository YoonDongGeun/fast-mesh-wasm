// The export panel's Height Map section (#267): its settings, the export with
// its progress and Cancel, and the two downloads (a 16-bit PNG and its JSON).
import type { Scene } from "fast-mesh-wasm";
import { useScene } from "fast-mesh-wasm/react";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useEffect, useRef, useState } from "react";
import { type AppState, type Item, useApp } from "../../../app/state/app-state";
import { downloadBytes, downloadReportText } from "../../../io/report";
import { view as viewport } from "../../../viewport/ux/view";
import {
  CancelledError,
  DEFAULT_HEIGHT_MAP,
  describeFrame,
  exportHeightMap,
  frameOptions,
  frameRange,
  type HeightMapParams,
  heightMapProblem,
  viewAxes,
} from "./height-map-export";

/** Which meshes the map shows: the panel's mesh, or every visible one. */
export type HeightMapSource = "mesh" | "visible";

function message(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

export function useHeightMapExport(app: AppState, scene: Scene | WorkerScene, picked: Item | null) {
  const store = useScene();
  const units = useApp(app, (s) => s.units);
  const items = useApp(app, (s) => s.items);
  const [params, setParams] = useState<HeightMapParams>(DEFAULT_HEIGHT_MAP);
  const [source, setSource] = useState<HeightMapSource>("mesh");
  const [progress, setProgress] = useState<number | null>(null);
  const [report, setReport] = useState<{ text: string; problem: boolean } | null>(null);
  const abort = useRef<AbortController | null>(null);
  useEffect(() => () => abort.current?.abort(), []);

  const problem = heightMapProblem(params);

  const meshes = (): readonly Item[] => {
    if (source === "mesh") {
      return picked === null ? [] : [picked];
    }
    const { meshes: states } = store.getSnapshot();
    return items.filter((i) => states.get(i.mesh)?.desired.visible ?? true);
  };

  const axes = () => {
    const a = viewAxes(params.view, viewport.camera);
    if (a === null) {
      throw new Error("There is no camera to look from yet.");
    }
    return a;
  };

  /** Switches between the meshes' own range and a typed one, which starts as theirs. */
  const setCustomRange = async (custom: boolean) => {
    if (!custom) {
      setParams((p) => ({ ...p, range: null }));
      return;
    }
    let range: [number, number] = [0, 1];
    try {
      const list = meshes().map((i) => i.mesh);
      if (list.length > 0) {
        const frame = await scene.heightMapFrame(list, { ...axes(), resolution: 2 });
        range = frameRange(frame);
      }
      setReport(null);
    } catch (e) {
      setReport({ text: message(e), problem: true });
    }
    setParams((p) => ({ ...p, range }));
  };

  const run = async () => {
    if (problem !== null || progress !== null) {
      return;
    }
    const list = meshes();
    if (list.length === 0) {
      setReport({ text: "There is no mesh to show.", problem: true });
      return;
    }
    const control = new AbortController();
    abort.current = control;
    setProgress(0);
    setReport(null);
    try {
      const out = await exportHeightMap(scene, list, frameOptions(params, axes()), {
        onProgress: setProgress,
        signal: control.signal,
      });
      if (control.signal.aborted) {
        throw new CancelledError();
      }
      downloadBytes(out.png.bytes, out.png.name, "image/png");
      downloadReportText(out.json.text, out.json.name, "application/json");
      const empty = out.seen === 0 ? " · no surface in view" : "";
      setReport({ text: `${describeFrame(out.frame, units)}${empty}`, problem: out.seen === 0 });
    } catch (e) {
      if (!(e instanceof CancelledError)) {
        setReport({ text: `Could not export the height map: ${message(e)}`, problem: true });
      }
    } finally {
      abort.current = null;
      setProgress(null);
    }
  };

  return {
    units,
    params,
    setParams: (patch: Partial<HeightMapParams>) => setParams((p) => ({ ...p, ...patch })),
    source,
    setSource,
    setCustomRange,
    problem,
    /** The share of rows sampled while an export runs, else `null`. */
    progress,
    report,
    ready: problem === null && (source === "visible" ? items.length > 0 : picked !== null),
    run,
    cancel: () => abort.current?.abort(),
  };
}
