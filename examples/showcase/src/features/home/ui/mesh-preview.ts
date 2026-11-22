// The small picture of a mesh in the mesh list: a software z-buffer over the
// render views, on the main thread in slices of a few milliseconds at idle
// (no GPU readback, no copy of the buffers to a Worker). It uses every
// triangle, so nothing thins out at 1M: a pass costs tens of milliseconds, cut
// across frames. When the slices run is `ux/preview-schedule`'s.
import type { MeshHandle } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { useEffect, useState } from "react";
import type { Item } from "../../../app/state/app-state";
import {
  PREVIEW_FIRST_MS,
  PREVIEW_QUIET_MS,
  type PreviewJob,
  PreviewQueue,
  type PreviewRun,
  turnChanged,
} from "../ux/preview-schedule";

/** The picture's side in pixels: sharp at 48 CSS pixels on a 2x screen. */
export const PREVIEW_SIZE = 96;
/** Samples per pixel side, for smooth silhouettes. */
const SUPERSAMPLE = 2;
/** Triangles between looks at the clock in the bounds pass; work units in the raster pass (a pixel tested is one). */
const CHUNK = 16_384;
/** A triangle's own work in the raster pass, in pixels tested. */
const TRIANGLE_WORK = 4;

/** The fixed view: a three-quarter look, y up. */
const YAW = (-35 * Math.PI) / 180;
const PITCH = (20 * Math.PI) / 180;
const VIEW = [
  Math.cos(YAW),
  0,
  Math.sin(YAW),
  Math.sin(PITCH) * Math.sin(YAW),
  Math.cos(PITCH),
  -Math.sin(PITCH) * Math.cos(YAW),
  -Math.cos(PITCH) * Math.sin(YAW),
  Math.sin(PITCH),
  Math.cos(PITCH) * Math.cos(YAW),
];
/** A light from the upper left, in view space (unit length). */
const LIGHT = [-0.37, 0.56, 0.74];

/** One mesh of a picture. */
export interface PreviewSource {
  /** Fresh views: a kept view can go stale between slices. It may throw once the mesh is gone. */
  views(): { positions(): Float32Array; indices(): Uint32Array };
  /** `0xrrggbb`. */
  color: number;
  /** A column-major 4x4 placement; none draws the mesh in its own coordinates. */
  matrix?: ArrayLike<number>;
}

/** A local-to-view transform: 3 rows of 4. */
function viewOf(matrix: ArrayLike<number> | undefined): Float64Array {
  const m = matrix ?? [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const out = new Float64Array(12);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 4; c++) {
      out[4 * r + c] = VIEW[3 * r] * m[4 * c] + VIEW[3 * r + 1] * m[4 * c + 1] + VIEW[3 * r + 2] * m[4 * c + 2];
    }
  }
  return out;
}

/**
 * One picture being drawn: {@link step} until it returns true, then
 * {@link image}. Two passes over the triangles (the framing, then the z-buffer)
 * that any slicing leaves bit-identical. Removed triangles (`0, 0, 0`) and dead
 * vertex slots never count; a triangle with a non-finite corner is skipped.
 */
export class PreviewRender {
  /** Triangles drawn (when done). */
  triangles = 0;
  readonly #sources: { source: PreviewSource; view: Float64Array }[];
  readonly #side = PREVIEW_SIZE * SUPERSAMPLE;
  #pass: "bounds" | "raster" | "done" = "bounds";
  #at = 0;
  #tri = 0;
  #minX = Infinity;
  #minY = Infinity;
  #maxX = -Infinity;
  #maxY = -Infinity;
  #scale = 0;
  #offX = 0;
  #offY = 0;
  #z: Float32Array | null = null;
  #rgb: Uint32Array | null = null;
  #covered = false;

  constructor(sources: PreviewSource[]) {
    this.#sources = sources.map((source) => ({ source, view: viewOf(source.matrix) }));
  }

  /** Works for about `budgetMs` (at least one chunk); true once the picture is done. */
  step(budgetMs: number, now: () => number = () => performance.now()): boolean {
    const end = budgetMs === Infinity ? Infinity : now() + budgetMs;
    for (;;) {
      if (this.#pass === "done") {
        return true;
      }
      if (this.#at >= this.#sources.length) {
        this.#nextPass();
        continue;
      }
      const { source, view } = this.#sources[this.#at];
      const views = source.views();
      const positions = views.positions();
      const indices = views.indices();
      for (;;) {
        if (this.#tri * 3 >= indices.length) {
          this.#at++;
          this.#tri = 0;
          break;
        }
        if (this.#pass === "bounds") {
          this.#boundsChunk(positions, indices, view);
        } else {
          this.#rasterChunk(positions, indices, view, source.color);
        }
        if (end !== Infinity && now() >= end) {
          return false;
        }
      }
    }
  }

  /** The finished picture as RGBA (straight alpha), `PREVIEW_SIZE` square; `null` when nothing shows. */
  image(): Uint8ClampedArray | null {
    const rgb = this.#rgb;
    if (rgb === null || !this.#covered) {
      return null;
    }
    const out = new Uint8ClampedArray(4 * PREVIEW_SIZE * PREVIEW_SIZE);
    const n = SUPERSAMPLE * SUPERSAMPLE;
    for (let y = 0; y < PREVIEW_SIZE; y++) {
      for (let x = 0; x < PREVIEW_SIZE; x++) {
        let r = 0;
        let g = 0;
        let b = 0;
        let covered = 0;
        for (let sy = 0; sy < SUPERSAMPLE; sy++) {
          for (let sx = 0; sx < SUPERSAMPLE; sx++) {
            const c = rgb[(y * SUPERSAMPLE + sy) * this.#side + x * SUPERSAMPLE + sx];
            if (c !== 0) {
              r += (c >>> 16) & 0xff;
              g += (c >>> 8) & 0xff;
              b += c & 0xff;
              covered++;
            }
          }
        }
        if (covered > 0) {
          const o = 4 * (y * PREVIEW_SIZE + x);
          out[o] = r / covered;
          out[o + 1] = g / covered;
          out[o + 2] = b / covered;
          out[o + 3] = (255 * covered) / n;
        }
      }
    }
    return out;
  }

  #nextPass(): void {
    this.#at = 0;
    this.#tri = 0;
    if (this.#pass === "raster") {
      this.#pass = "done";
      this.#z = null;
      return;
    }
    const w = this.#maxX - this.#minX;
    const h = this.#maxY - this.#minY;
    const extent = Math.max(w, h);
    if (!(extent > 0) || !Number.isFinite(extent)) {
      this.#pass = "done"; // nothing drawable
      return;
    }
    // A small margin: a silhouette touching the edge reads as cropped.
    this.#scale = (this.#side * 0.94) / extent;
    this.#offX = this.#side / 2 - this.#scale * (this.#minX + w / 2);
    this.#offY = this.#side / 2 + this.#scale * (this.#minY + h / 2);
    this.#z = new Float32Array(this.#side * this.#side).fill(-Infinity);
    this.#rgb = new Uint32Array(this.#side * this.#side);
    this.#pass = "raster";
  }

  #boundsChunk(p: Float32Array, idx: Uint32Array, v: Float64Array): void {
    const last = Math.min(idx.length / 3, this.#tri + CHUNK);
    let { minX, minY, maxX, maxY } = { minX: this.#minX, minY: this.#minY, maxX: this.#maxX, maxY: this.#maxY };
    for (let t = this.#tri; t < last; t++) {
      const a = 3 * idx[3 * t];
      const b = 3 * idx[3 * t + 1];
      const c = 3 * idx[3 * t + 2];
      if (a === b && b === c) {
        continue; // removed
      }
      const ax = v[0] * p[a] + v[1] * p[a + 1] + v[2] * p[a + 2] + v[3];
      const ay = v[4] * p[a] + v[5] * p[a + 1] + v[6] * p[a + 2] + v[7];
      const bx = v[0] * p[b] + v[1] * p[b + 1] + v[2] * p[b + 2] + v[3];
      const by = v[4] * p[b] + v[5] * p[b + 1] + v[6] * p[b + 2] + v[7];
      const cx = v[0] * p[c] + v[1] * p[c + 1] + v[2] * p[c + 2] + v[3];
      const cy = v[4] * p[c] + v[5] * p[c + 1] + v[6] * p[c + 2] + v[7];
      // NaN or an infinity anywhere makes this sum not finite (x - x is 0 only for finite x).
      if (ax - ax + ay - ay + bx - bx + by - by + cx - cx + cy - cy !== 0) {
        continue;
      }
      minX = Math.min(minX, ax, bx, cx);
      maxX = Math.max(maxX, ax, bx, cx);
      minY = Math.min(minY, ay, by, cy);
      maxY = Math.max(maxY, ay, by, cy);
    }
    this.#minX = minX;
    this.#minY = minY;
    this.#maxX = maxX;
    this.#maxY = maxY;
    this.#tri = last;
  }

  #rasterChunk(p: Float32Array, idx: Uint32Array, v: Float64Array, color: number): void {
    const zb = this.#z as Float32Array;
    const rgb = this.#rgb as Uint32Array;
    const side = this.#side;
    const scale = this.#scale;
    const offX = this.#offX;
    const offY = this.#offY;
    const slots = idx.length / 3;
    let work = 0;
    let t = this.#tri;
    for (; t < slots && work < CHUNK; t++) {
      work += TRIANGLE_WORK;
      const a = 3 * idx[3 * t];
      const b = 3 * idx[3 * t + 1];
      const c = 3 * idx[3 * t + 2];
      if (a === b && b === c) {
        continue;
      }
      const ax = v[0] * p[a] + v[1] * p[a + 1] + v[2] * p[a + 2] + v[3];
      const ay = v[4] * p[a] + v[5] * p[a + 1] + v[6] * p[a + 2] + v[7];
      const az = v[8] * p[a] + v[9] * p[a + 1] + v[10] * p[a + 2] + v[11];
      const bx = v[0] * p[b] + v[1] * p[b + 1] + v[2] * p[b + 2] + v[3];
      const by = v[4] * p[b] + v[5] * p[b + 1] + v[6] * p[b + 2] + v[7];
      const bz = v[8] * p[b] + v[9] * p[b + 1] + v[10] * p[b + 2] + v[11];
      const cx = v[0] * p[c] + v[1] * p[c + 1] + v[2] * p[c + 2] + v[3];
      const cy = v[4] * p[c] + v[5] * p[c + 1] + v[6] * p[c + 2] + v[7];
      const cz = v[8] * p[c] + v[9] * p[c + 1] + v[10] * p[c + 2] + v[11];
      if (ax - ax + ay - ay + az - az + bx - bx + by - by + bz - bz + cx - cx + cy - cy + cz - cz !== 0) {
        continue;
      }
      this.triangles++;
      // Screen coordinates: y down.
      const sax = offX + scale * ax;
      const say = offY - scale * ay;
      const sbx = offX + scale * bx;
      const sby = offY - scale * by;
      const scx = offX + scale * cx;
      const scy = offY - scale * cy;
      // Pixel centres sit at .5; most triangles of a dense mesh hold none and end here.
      const x0 = Math.max(0, Math.ceil(Math.min(sax, sbx, scx) - 0.5));
      const x1 = Math.min(side - 1, Math.floor(Math.max(sax, sbx, scx) - 0.5));
      if (x0 > x1) {
        continue;
      }
      const y0 = Math.max(0, Math.ceil(Math.min(say, sby, scy) - 0.5));
      const y1 = Math.min(side - 1, Math.floor(Math.max(say, sby, scy) - 0.5));
      if (y0 > y1) {
        continue;
      }
      const area = (sbx - sax) * (scy - say) - (sby - say) * (scx - sax);
      if (area === 0) {
        continue;
      }
      const inv = 1 / area;
      work += (x1 - x0 + 1) * (y1 - y0 + 1);
      let shaded = 0; // computed at the triangle's first visible sample
      for (let y = y0; y <= y1; y++) {
        const py = y + 0.5;
        for (let x = x0; x <= x1; x++) {
          const px = x + 0.5;
          const w0 = ((sbx - px) * (scy - py) - (sby - py) * (scx - px)) * inv;
          const w1 = ((scx - px) * (say - py) - (scy - py) * (sax - px)) * inv;
          const w2 = 1 - w0 - w1;
          if (w0 < 0 || w1 < 0 || w2 < 0) {
            continue;
          }
          const z = w0 * az + w1 * bz + w2 * cz;
          const at = y * side + x;
          if (z > zb[at]) {
            zb[at] = z;
            if (shaded === 0) {
              shaded = shade(ax, ay, az, bx, by, bz, cx, cy, cz, color);
            }
            rgb[at] = shaded;
            this.#covered = true;
          }
        }
      }
    }
    this.#tri = t;
  }
}

/** A face's colour (never 0, which marks an empty sample): both sides lit, so an open mesh reads too. */
function shade(ax: number, ay: number, az: number, bx: number, by: number, bz: number, cx: number, cy: number, cz: number, color: number): number {
  const ux = bx - ax;
  const uy = by - ay;
  const uz = bz - az;
  const vx = cx - ax;
  const vy = cy - ay;
  const vz = cz - az;
  let nx = uy * vz - uz * vy;
  let ny = uz * vx - ux * vz;
  let nz = ux * vy - uy * vx;
  const len = Math.hypot(nx, ny, nz) || 1;
  const facing = nz < 0 ? -1 / len : 1 / len;
  nx *= facing;
  ny *= facing;
  nz *= facing;
  const diffuse = Math.max(0, nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2]);
  const light = 0.3 + 0.75 * diffuse + 0.1 * nz;
  const ch = (shift: number) => Math.min(255, Math.round(((color >>> shift) & 0xff) * light));
  return (0xff000000 | (ch(16) << 16) | (ch(8) << 8) | ch(0)) >>> 0;
}

const queue = new PreviewQueue<MeshHandle, PreviewRender>();
const arms = new Map<MeshHandle, (quietMs: number) => void>();

/** Redraws a mesh's picture now, as an edit would after its quiet time. For measuring. */
export function redrawMeshPreview(mesh: MeshHandle): void {
  arms.get(mesh)?.(0);
}

/** Per mesh: pictures started and finished, and what the last one cost. Read by the browser check. */
export interface MeshPreviewStats {
  starts: number;
  renders: number;
  triangles: number;
  last: PreviewRun | null;
  /** A redraw is due or under way, or its picture is not on the page yet. A start an edit overtook never counts as a render. */
  pending: boolean;
}
const stats = new Map<MeshHandle, MeshPreviewStats>();

/** The stats of a mesh's picture (zeros before its first). */
export function meshPreviewStats(mesh: MeshHandle): MeshPreviewStats {
  return stats.get(mesh) ?? { starts: 0, renders: 0, triangles: 0, last: null, pending: true };
}

/** A canvas holding the picture, for encoding it. */
export function previewCanvas(pixels: Uint8ClampedArray): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = PREVIEW_SIZE;
  canvas.height = PREVIEW_SIZE;
  const image = new ImageData(new Uint8ClampedArray(pixels), PREVIEW_SIZE, PREVIEW_SIZE);
  (canvas.getContext("2d") as CanvasRenderingContext2D).putImageData(image, 0, 0);
  return canvas;
}

/** Colours of a built-in model's picture: fixed, unlike the list's palette, so a picture does not depend on load order. */
const MODEL_COLORS = [0xd9735f, 0x6ea8d9, 0x8fc46e];

/**
 * The picture of a built-in model: its meshes where they are placed. What
 * `scripts/showcase-previews.mjs` commits and the browser check compares with.
 */
export function drawModelPreview(scene: SceneStore["scene"], meshes: MeshHandle[]): HTMLCanvasElement | null {
  const render = new PreviewRender(
    meshes.map((mesh, i) => ({
      views: () => scene.views(mesh),
      color: meshes.length === 1 ? MODEL_COLORS[1] : MODEL_COLORS[i % MODEL_COLORS.length],
      matrix: scene.matrix4(mesh),
    })),
  );
  render.step(Infinity);
  const pixels = render.image();
  return pixels === null ? null : previewCanvas(pixels);
}

/** What a row's picture draws: the mesh as placed now, in its own colour. */
function rowSource(scene: SceneStore["scene"], mesh: MeshHandle, color: number): PreviewSource {
  return { views: () => scene.views(mesh), color, matrix: scene.matrix4(mesh) };
}

/** A row's picture drawn at once, to compare with the one in the list. */
export function drawRowPreview(scene: SceneStore["scene"], mesh: MeshHandle, color: number): HTMLCanvasElement | null {
  const render = new PreviewRender([rowSource(scene, mesh, color)]);
  render.step(Infinity);
  const pixels = render.image();
  return pixels === null ? null : previewCanvas(pixels);
}

/** A row's picture: its URL, or none yet, or (`empty`) a mesh with nothing left to draw. */
export interface MeshPreview {
  url: string | null;
  empty: boolean;
}

/**
 * The URL of a loaded mesh's picture (`null` until there is one), redrawn from
 * its render views after an edit or a turn. An `<img>` by design: the app and its checks
 * take the page's first canvas for the viewport, and the list comes first.
 */
export function useMeshPreview(store: SceneStore, item: Item): MeshPreview {
  const [preview, setPreview] = useState<MeshPreview>({ url: null, empty: false });
  const { mesh, color } = item;
  useEffect(() => {
    let live = true;
    let shown: string | null = null;
    let armed = 0;
    let drawn = 0;
    // A load lists its mesh before its buffers arrive: empty only counts once they have.
    let arrived = false;
    let matrix: ArrayLike<number> | null = null;
    const settle = (at: number) => {
      if (at === armed) {
        stats.set(mesh, { ...meshPreviewStats(mesh), pending: false });
      }
    };
    const job: PreviewJob<PreviewRender> = {
      start: () => {
        const s = meshPreviewStats(mesh);
        stats.set(mesh, { ...s, starts: s.starts + 1 });
        const source = rowSource(store.scene, mesh, color);
        matrix = source.matrix as ArrayLike<number>;
        return new PreviewRender([source]);
      },
      done: (render, run) => {
        const s = meshPreviewStats(mesh);
        stats.set(mesh, { ...s, renders: s.renders + 1, triangles: render.triangles, last: run });
        const at = armed;
        const turn = ++drawn; // a slower encode of an older picture must not land over this one
        const pixels = render.image();
        if (pixels === null) {
          if (arrived) {
            if (shown !== null) {
              URL.revokeObjectURL(shown);
              shown = null;
            }
            setPreview({ url: null, empty: true });
          }
          settle(at);
          return;
        }
        arrived = true;
        previewCanvas(pixels).toBlob((blob) => {
          if (!live || blob === null || turn !== drawn) {
            return;
          }
          if (shown !== null) {
            URL.revokeObjectURL(shown);
          }
          shown = URL.createObjectURL(blob);
          setPreview({ url: shown, empty: false });
          settle(at);
        });
      },
    };
    const arm = (quietMs: number) => {
      armed++;
      stats.set(mesh, { ...meshPreviewStats(mesh), pending: true });
      queue.invalidate(mesh, job, quietMs);
    };
    arms.set(mesh, arm);
    arm(PREVIEW_FIRST_MS);
    const off = store.scene.onChanges((changes) => {
      for (const c of changes) {
        if (c.mesh !== mesh || c.removed) {
          continue;
        }
        if (c.added || c.positions === "all" || c.indices === "all" || c.positions.length > 0 || c.indices.length > 0) {
          arrived = true;
          arm(PREVIEW_QUIET_MS);
        } else if (c.placementChanged && matrix !== null && turnChanged(matrix, store.scene.matrix4(mesh))) {
          arm(PREVIEW_QUIET_MS);
        }
      }
    });
    return () => {
      live = false;
      off();
      queue.cancel(mesh);
      if (shown !== null) {
        URL.revokeObjectURL(shown);
      }
      stats.delete(mesh);
      arms.delete(mesh);
    };
  }, [store, mesh, color]);
  return preview;
}
