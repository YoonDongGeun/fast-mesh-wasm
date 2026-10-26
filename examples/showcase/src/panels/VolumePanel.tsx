// The volume view: our own direct volume rendering (a three.js
// raycaster) of the selected mesh's source DICOM volume, with HU-tuned
// presets and a MIP toggle — plus axial/coronal/sagittal slice views on
// 2D canvas with window presets, like a PACS viewer. The loader retains
// each series' volume; this panel fetches it on selection.
import { useSelection } from "fast-mesh-wasm/react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Loader } from "../load/loader";
import type { VolumePayload } from "../load/protocol";
import { type AppState, useApp } from "../scene/app-state";
import { inspect } from "./inspect";
import { displayValues } from "./volume-data";
import { dataRange, normalizeVolume, presetLut } from "./volume-lut";
import { VOLUME_PRESETS, WINDOW_PRESETS, type WindowValue } from "./volume-presets";
import { extractSlice, windowToGray, type SliceAxis } from "./volume-slice";
import { VolumeView3D } from "./volume-three";

const SLICE_MAX: Record<SliceAxis, (p: VolumePayload) => number> = {
  axial: (p) => p.slices - 1,
  coronal: (p) => p.rows - 1,
  sagittal: (p) => p.columns - 1,
};

const SLICE_AXES: SliceAxis[] = ["axial", "coronal", "sagittal"];

export function VolumePanel(props: { app: AppState; loader: Loader }) {
  const { app, loader } = props;
  const [selection] = useSelection();
  const items = useApp(app, (s) => s.items);
  const item = items.find((i) => i.mesh === selection) ?? null;
  if (item === null || item.volumeId === null) {
    return null;
  }
  return <VolumeWork key={item.volumeId} loader={loader} volumeId={item.volumeId} name={item.name} />;
}

function VolumeWork(props: { loader: Loader; volumeId: string; name: string }) {
  const { loader, volumeId, name } = props;
  const [payload, setPayload] = useState<VolumePayload | null>(null);
  const [failed, setFailed] = useState(false);
  const [tab, setTab] = useState<"volume" | "slices">("volume");

  useEffect(() => {
    let alive = true;
    inspect.volume = { id: volumeId, preset: VOLUME_PRESETS[0].name, mip: false, ready: false };
    inspect.slices = { id: volumeId, window: WINDOW_PRESETS[0].name, axial: 0, coronal: 0, sagittal: 0, ready: false };
    loader
      .fetchVolume(volumeId)
      .then((got: VolumePayload) => {
        if (alive) {
          setPayload(got);
        }
      })
      .catch(() => {
        if (alive) {
          setFailed(true);
        }
      });
    return () => {
      alive = false;
      if (inspect.volume.id === volumeId) {
        inspect.volume = { id: null, preset: "", mip: false, ready: false };
      }
      if (inspect.slices.id === volumeId) {
        inspect.slices = { id: null, window: "", axial: 0, coronal: 0, sagittal: 0, ready: false };
      }
    };
  }, [loader, volumeId]);

  if (failed) {
    return (
      <section className="panel" aria-label="Volume">
        <header className="panel-head">
          <h2>Volume</h2>
          <span className="muted small">{name}</span>
        </header>
        <p className="muted">The volume is no longer retained.</p>
      </section>
    );
  }
  if (payload === null) {
    return (
      <section className="panel" aria-label="Volume">
        <header className="panel-head">
          <h2>Volume</h2>
          <span className="muted small">{name}</span>
        </header>
        <p className="muted">Loading the volume…</p>
      </section>
    );
  }
  return (
    <section className="panel" aria-label="Volume">
      <header className="panel-head">
        <h2>Volume</h2>
        <span className="muted small">{name}</span>
      </header>
      <div className="row" role="tablist" aria-label="Volume views">
        <button type="button" role="tab" aria-selected={tab === "volume"} data-action="volume-tab-3d" onClick={() => setTab("volume")}>
          3D
        </button>
        <button type="button" role="tab" aria-selected={tab === "slices"} data-action="volume-tab-slices" onClick={() => setTab("slices")}>
          Slices
        </button>
      </div>
      {tab === "volume" ? (
        <VolumeCanvas payload={payload} volumeId={volumeId} />
      ) : (
        <SliceWork payload={payload} volumeId={volumeId} />
      )}
    </section>
  );
}

function VolumeCanvas(props: { payload: VolumePayload; volumeId: string }) {
  const { payload, volumeId } = props;
  const divRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const viewRef = useRef<VolumeView3D | null>(null);
  const [preset, setPreset] = useState(0);
  const [mip, setMip] = useState(false);
  const [failed, setFailed] = useState(false);
  const prepared = useMemo(() => {
    const display = displayValues(payload.values, payload.pseudoHu);
    const [min, max] = dataRange(display);
    return { data: normalizeVolume(display, min, max), min, max };
  }, [payload]);

  useEffect(() => {
    const div = divRef.current;
    const canvas = canvasRef.current;
    if (div === null || canvas === null) {
      return;
    }
    let view: VolumeView3D | null = null;
    try {
      view = new VolumeView3D(
        canvas,
        prepared.data,
        { columns: payload.columns, rows: payload.rows, slices: payload.slices, spacing: payload.spacing },
        presetLut(VOLUME_PRESETS[preset], prepared.min, prepared.max),
      );
    } catch {
      setFailed(true);
      return;
    }
    view.setMip(mip);
    view.setSize(div.clientWidth, div.clientHeight);
    const resize = new ResizeObserver(() => view?.setSize(div.clientWidth, div.clientHeight));
    resize.observe(div);
    viewRef.current = view;
    inspect.volume = { id: volumeId, preset: VOLUME_PRESETS[preset].name, mip, ready: true };
    return () => {
      resize.disconnect();
      viewRef.current = null;
      view?.dispose();
      if (inspect.volume.id === volumeId) {
        inspect.volume = { id: volumeId, preset: VOLUME_PRESETS[preset].name, mip, ready: false };
      }
    };
    // The view belongs to the payload; preset/mip steer it below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [payload, volumeId]);

  useEffect(() => {
    const view = viewRef.current;
    if (view !== null) {
      view.setLut(presetLut(VOLUME_PRESETS[preset], prepared.min, prepared.max));
      view.setMip(mip);
      inspect.volume = { id: volumeId, preset: VOLUME_PRESETS[preset].name, mip, ready: true };
    }
  }, [preset, mip, prepared, volumeId]);

  return (
    <>
      <div className="row">
        <label className="slider">
          preset{" "}
          <select data-action="volume-preset" value={preset} onChange={(e) => setPreset(Number(e.currentTarget.value))}>
            {VOLUME_PRESETS.map((p, i) => (
              <option key={p.name} value={i}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
        <label className="check" title="Maximum intensity projection (the classic angio look)">
          <input
            type="checkbox"
            data-action="volume-mip"
            checked={mip}
            onChange={(e) => setMip(e.currentTarget.checked)}
          />
          MIP
        </label>
      </div>
      {failed ? <p className="muted">WebGL is unavailable: the volume cannot render here.</p> : null}
      <div ref={divRef} className="volume-canvas" data-volume={volumeId}>
        <canvas ref={canvasRef} />
      </div>
    </>
  );
}

function SliceWork(props: { payload: VolumePayload; volumeId: string }) {
  const { payload, volumeId } = props;
  const display = useMemo(() => displayValues(payload.values, payload.pseudoHu), [payload]);
  const [window, setWindow] = useState(0);
  const [slices, setSlices] = useState<Record<SliceAxis, number>>(() => ({
    axial: Math.floor((payload.slices - 1) / 2),
    coronal: Math.floor((payload.rows - 1) / 2),
    sagittal: Math.floor((payload.columns - 1) / 2),
  }));
  const [ready, setReady] = useState<Record<SliceAxis, boolean>>({ axial: false, coronal: false, sagittal: false });
  const range = useMemo(() => dataRange(display), [display]);
  const preset = WINDOW_PRESETS[window];
  const resolve = (v: WindowValue, auto: number): number => (v === "auto" ? auto : v);
  const colorWindow = resolve(preset.window, range[1] - range[0]);
  const colorLevel = resolve(preset.level, (range[0] + range[1]) / 2);

  useEffect(() => {
    inspect.slices = {
      id: volumeId,
      window: preset.name,
      axial: slices.axial,
      coronal: slices.coronal,
      sagittal: slices.sagittal,
      ready: ready.axial && ready.coronal && ready.sagittal,
    };
  }, [volumeId, preset.name, slices, ready]);

  return (
    <>
      <div className="row">
        <label className="slider">
          window{" "}
          <select data-action="volume-window" value={window} onChange={(e) => setWindow(Number(e.currentTarget.value))}>
            {WINDOW_PRESETS.map((p, i) => (
              <option key={p.name} value={i}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
      </div>
      {SLICE_AXES.map((axis) => (
        <div key={axis} className="slice-row">
          <div className="row">
            <span className="slice-label">{axis}</span>
            <label className="slider">
              slice{" "}
              <input
                type="range"
                data-action={`slice-${axis}`}
                min={0}
                max={SLICE_MAX[axis](payload)}
                value={slices[axis]}
                onChange={(e) => {
                  // Read the event now: the updater runs after dispatch,
                  // when currentTarget is null.
                  const v = Number(e.currentTarget.value);
                  setSlices((s) => ({ ...s, [axis]: v }));
                }}
              />
              <span className="muted small">
                {slices[axis]}/{SLICE_MAX[axis](payload)}
              </span>
            </label>
          </div>
          <SliceView
            display={display}
            payload={payload}
            axis={axis}
            slice={slices[axis]}
            colorWindow={colorWindow}
            colorLevel={colorLevel}
            onReady={() => setReady((r) => ({ ...r, [axis]: true }))}
          />
        </div>
      ))}
    </>
  );
}

function SliceView(props: {
  display: Float32Array;
  payload: VolumePayload;
  axis: SliceAxis;
  slice: number;
  colorWindow: number;
  colorLevel: number;
  onReady: () => void;
}) {
  const { display, payload, axis, slice, colorWindow, colorLevel, onReady } = props;
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas === null) {
      return;
    }
    const image = extractSlice(display, payload.columns, payload.rows, payload.slices, axis, slice);
    canvas.width = image.width;
    canvas.height = image.height;
    const ctx = canvas.getContext("2d");
    ctx?.putImageData(new ImageData(windowToGray(image.data, colorWindow, colorLevel), image.width, image.height), 0, 0);
    onReady();
  }, [display, payload, axis, slice, colorWindow, colorLevel, onReady]);

  return (
    <div className="slice-canvas" data-slice={axis}>
      <canvas ref={canvasRef} />
    </div>
  );
}
