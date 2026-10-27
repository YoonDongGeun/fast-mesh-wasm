// The volume view: our own direct volume rendering (a three.js
// raycaster) of the selected mesh's source DICOM volume, with HU-tuned
// presets, a MIP toggle, and the slice planes overlaid in 3D — plus
// linked axial/coronal/sagittal slice views on 2D canvas with window
// presets, crosshairs, and click/drag navigation, like a PACS viewer.
// The loader retains each series' volume; this panel fetches it on
// selection.
import { useSelection } from "fast-mesh-wasm/react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Loader } from "../load/loader";
import type { VolumePayload } from "../load/protocol";
import { type AppState, useApp } from "../scene/app-state";
import { inspect } from "./inspect";
import { displayValues } from "./volume-data";
import { dataRange, normalizeVolume, presetLut } from "./volume-lut";
import { VOLUME_PRESETS, WINDOW_PRESETS, type WindowValue } from "./volume-presets";
import { AXIS_COLOR, extractSlice, pointerSlice, slicePlaneImages, windowToGray, type SliceAxis } from "./volume-slice";
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
  return <VolumeWork key={item.volumeId} app={app} loader={loader} volumeId={item.volumeId} name={item.name} />;
}

function VolumeWork(props: { app: AppState; loader: Loader; volumeId: string; name: string }) {
  const { app, loader, volumeId, name } = props;
  const [payload, setPayload] = useState<VolumePayload | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let alive = true;
    inspect.volume = { id: volumeId, preset: VOLUME_PRESETS[0].name, mip: false, planes: 0, ready: false };
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
        inspect.volume = { id: null, preset: "", mip: false, planes: 0, ready: false };
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
  return <VolumeTabs app={app} payload={payload} volumeId={volumeId} name={name} />;
}

/** The tabs share the MPR state through the app, so the scene overlay stays linked too. */
function VolumeTabs(props: { app: AppState; payload: VolumePayload; volumeId: string; name: string }) {
  const { app, payload, volumeId, name } = props;
  const [tab, setTab] = useState<"volume" | "slices">("volume");
  const mpr = useApp(app, (s) => s.volumeMpr);
  useEffect(() => {
    app.setVolumeMpr({
      volumeId,
      slices: {
        axial: Math.floor((payload.slices - 1) / 2),
        coronal: Math.floor((payload.rows - 1) / 2),
        sagittal: Math.floor((payload.columns - 1) / 2),
      },
      window: 0,
      inScene: true,
    });
    return () => {
      if (app.getSnapshot().volumeMpr?.volumeId === volumeId) {
        app.setVolumeMpr(null);
      }
    };
  }, [app, payload, volumeId]);
  const display = useMemo(() => displayValues(payload.values, payload.pseudoHu), [payload]);
  const range = useMemo(() => dataRange(display), [display]);
  const live = mpr !== null && mpr.volumeId === volumeId ? mpr : null;
  const preset = WINDOW_PRESETS[live?.window ?? 0];
  const resolve = (v: WindowValue, auto: number): number => (v === "auto" ? auto : v);
  const colorWindow = resolve(preset.window, range[1] - range[0]);
  const colorLevel = resolve(preset.level, (range[0] + range[1]) / 2);
  const setSlices: React.Dispatch<React.SetStateAction<Record<SliceAxis, number>>> = (update) => {
    const cur = app.getSnapshot().volumeMpr;
    if (cur === null || cur.volumeId !== volumeId) {
      return;
    }
    app.setVolumeMpr({ ...cur, slices: typeof update === "function" ? update(cur.slices) : update });
  };
  const setWindowIdx = (i: number): void => {
    const cur = app.getSnapshot().volumeMpr;
    if (cur !== null && cur.volumeId === volumeId) {
      app.setVolumeMpr({ ...cur, window: i });
    }
  };
  const setInScene = (on: boolean): void => {
    const cur = app.getSnapshot().volumeMpr;
    if (cur !== null && cur.volumeId === volumeId) {
      app.setVolumeMpr({ ...cur, inScene: on });
    }
  };

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
        <label className="check" title="Show the slices in the main scene too">
          <input
            type="checkbox"
            data-action="volume-scene"
            checked={live?.inScene ?? true}
            onChange={(e) => setInScene(e.currentTarget.checked)}
          />
          Scene
        </label>
      </div>
      {live === null ? null : tab === "volume" ? (
        <VolumeCanvas
          payload={payload}
          volumeId={volumeId}
          display={display}
          range={range}
          slices={live.slices}
          colorWindow={colorWindow}
          colorLevel={colorLevel}
        />
      ) : (
        <SliceWork
          payload={payload}
          volumeId={volumeId}
          display={display}
          slices={live.slices}
          setSlices={setSlices}
          windowIdx={live.window}
          setWindowIdx={setWindowIdx}
          windowName={preset.name}
          colorWindow={colorWindow}
          colorLevel={colorLevel}
        />
      )}
    </section>
  );
}

function VolumeCanvas(props: {
  payload: VolumePayload;
  volumeId: string;
  display: Float32Array;
  range: [number, number];
  slices: Record<SliceAxis, number>;
  colorWindow: number;
  colorLevel: number;
}) {
  const { payload, volumeId, display, range, slices, colorWindow, colorLevel } = props;
  const divRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const viewRef = useRef<VolumeView3D | null>(null);
  const [preset, setPreset] = useState(0);
  const [mip, setMip] = useState(false);
  const [planesOn, setPlanesOn] = useState(true);
  const [failed, setFailed] = useState(false);
  const prepared = useMemo(() => normalizeVolume(display, range[0], range[1]), [display, range]);

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
        prepared,
        { columns: payload.columns, rows: payload.rows, slices: payload.slices, spacing: payload.spacing },
        presetLut(VOLUME_PRESETS[preset], range[0], range[1]),
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
    inspect.volume = { id: volumeId, preset: VOLUME_PRESETS[preset].name, mip, planes: 0, ready: true };
    return () => {
      resize.disconnect();
      viewRef.current = null;
      view?.dispose();
      if (inspect.volume.id === volumeId) {
        inspect.volume = { id: volumeId, preset: VOLUME_PRESETS[preset].name, mip, planes: 0, ready: false };
      }
    };
    // The view belongs to the payload; preset/mip/planes steer it below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [payload, volumeId]);

  useEffect(() => {
    const view = viewRef.current;
    if (view !== null) {
      view.setLut(presetLut(VOLUME_PRESETS[preset], range[0], range[1]));
      view.setMip(mip);
      inspect.volume = { id: volumeId, preset: VOLUME_PRESETS[preset].name, mip, planes: planesOn ? 3 : 0, ready: true };
    }
  }, [preset, mip, planesOn, prepared, range, volumeId]);

  useEffect(() => {
    const view = viewRef.current;
    if (view === null) {
      return;
    }
    if (!planesOn) {
      view.setPlanes(null);
      inspect.volume = { id: volumeId, preset: VOLUME_PRESETS[preset].name, mip, planes: 0, ready: true };
      return;
    }
    view.setPlanes(
      slicePlaneImages(
        display,
        { columns: payload.columns, rows: payload.rows, slices: payload.slices },
        slices,
        colorWindow,
        colorLevel,
      ),
    );
    inspect.volume = { id: volumeId, preset: VOLUME_PRESETS[preset].name, mip, planes: 3, ready: true };
  }, [slices, colorWindow, colorLevel, planesOn, display, payload, preset, mip, volumeId]);

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
        <label className="check" title="Show the slice positions as planes in 3D">
          <input
            type="checkbox"
            data-action="volume-planes"
            checked={planesOn}
            onChange={(e) => setPlanesOn(e.currentTarget.checked)}
          />
          Planes
        </label>
      </div>
      {failed ? <p className="muted">WebGL is unavailable: the volume cannot render here.</p> : null}
      <div ref={divRef} className="volume-canvas" data-volume={volumeId}>
        <canvas ref={canvasRef} />
      </div>
    </>
  );
}

function SliceWork(props: {
  payload: VolumePayload;
  volumeId: string;
  display: Float32Array;
  slices: Record<SliceAxis, number>;
  setSlices: React.Dispatch<React.SetStateAction<Record<SliceAxis, number>>>;
  windowIdx: number;
  setWindowIdx: (i: number) => void;
  windowName: string;
  colorWindow: number;
  colorLevel: number;
}) {
  const { payload, volumeId, display, slices, setSlices, windowIdx, setWindowIdx, windowName, colorWindow, colorLevel } = props;
  const [ready, setReady] = useState<Record<SliceAxis, boolean>>({ axial: false, coronal: false, sagittal: false });

  useEffect(() => {
    inspect.slices = {
      id: volumeId,
      window: windowName,
      axial: slices.axial,
      coronal: slices.coronal,
      sagittal: slices.sagittal,
      ready: ready.axial && ready.coronal && ready.sagittal,
    };
  }, [volumeId, windowName, slices, ready]);

  return (
    <>
      <div className="row">
        <label className="slider">
          window{" "}
          <select data-action="volume-window" value={windowIdx} onChange={(e) => setWindowIdx(Number(e.currentTarget.value))}>
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
            slices={slices}
            colorWindow={colorWindow}
            colorLevel={colorLevel}
            onNavigate={(part) => setSlices((s) => ({ ...s, ...part }))}
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
  slices: Record<SliceAxis, number>;
  colorWindow: number;
  colorLevel: number;
  onNavigate: (part: Partial<Record<SliceAxis, number>>) => void;
  onReady: () => void;
}) {
  const { display, payload, axis, slices, colorWindow, colorLevel, onNavigate, onReady } = props;
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas === null) {
      return;
    }
    const image = extractSlice(display, payload.columns, payload.rows, payload.slices, axis, slices[axis]);
    canvas.width = image.width;
    canvas.height = image.height;
    const ctx = canvas.getContext("2d");
    if (ctx === null) {
      return;
    }
    ctx.putImageData(new ImageData(windowToGray(image.data, colorWindow, colorLevel), image.width, image.height), 0, 0);
    // The crosshair: the other two slice positions, colored by axis.
    const lines: Array<{ vertical: boolean; at: number; color: string }> =
      axis === "axial"
        ? [
            { vertical: true, at: slices.sagittal, color: AXIS_COLOR.sagittal },
            { vertical: false, at: payload.rows - 1 - slices.coronal, color: AXIS_COLOR.coronal },
          ]
        : axis === "coronal"
          ? [
              { vertical: true, at: slices.sagittal, color: AXIS_COLOR.sagittal },
              { vertical: false, at: payload.slices - 1 - slices.axial, color: AXIS_COLOR.axial },
            ]
          : [
              { vertical: true, at: slices.coronal, color: AXIS_COLOR.coronal },
              { vertical: false, at: payload.slices - 1 - slices.axial, color: AXIS_COLOR.axial },
            ];
    for (const line of lines) {
      ctx.fillStyle = line.color;
      if (line.vertical) {
        ctx.fillRect(line.at, 0, 1, image.height);
      } else {
        ctx.fillRect(0, line.at, image.width, 1);
      }
    }
    onReady();
  }, [display, payload, axis, slices, colorWindow, colorLevel, onReady]);

  const navigate = (clientX: number, clientY: number): void => {
    const canvas = canvasRef.current;
    if (canvas === null) {
      return;
    }
    const rect = canvas.getBoundingClientRect();
    onNavigate(
      pointerSlice(
        axis,
        { columns: payload.columns, rows: payload.rows, slices: payload.slices },
        { width: canvas.width, height: canvas.height },
        { width: rect.width, height: rect.height },
        clientX - rect.left,
        clientY - rect.top,
      ),
    );
  };

  return (
    <div className="slice-canvas" data-slice={axis}>
      <canvas
        ref={canvasRef}
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          navigate(e.clientX, e.clientY);
        }}
        onPointerMove={(e) => {
          if (e.buttons & 1) {
            navigate(e.clientX, e.clientY);
          }
        }}
      />
    </div>
  );
}
