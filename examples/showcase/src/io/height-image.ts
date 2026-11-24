// Height maps as images, both ways (#267): heights to the codes of a 16-bit
// PNG with the JSON that says where they are, and an image to a grid of
// heights. Pure apart from the browser's own image decoder; the page runs it
// in a Worker (`height-image-client.ts`).
//
// A code is a height above the frame's zero plane: `heightRange * code /
// 65535`, so 0 is both height 0 and "no surface", and a relief of the image
// stands on its zero plane there.
import type { HeightMapFrame } from "fast-mesh-wasm";
import { canDecodePng, decodePng, encodePng16, isPng, type PngImage, readPngHeader } from "./png";

/** The largest code of a 16-bit image. */
export const MAX_CODE = 65535;

/** The most pixels an image may have: 8192 x 8192. */
export const MAX_IMAGE_PIXELS = 2 ** 26;

/** Files the image import takes, for a file input's `accept`. */
export const IMAGE_ACCEPT = ".png,.jpg,.jpeg,.webp,.bmp,.gif,.avif,image/*";

/** Whether a file is an image, by its type or else its extension. */
export function isImageFile(file: { name: string; type: string }): boolean {
  return file.type.startsWith("image/") || /\.(?:png|jpe?g|webp|bmp|gif|avif)$/i.test(file.name);
}

// ---- Heights to an image ----

/** The codes of heights in `[0, range]`; NaN (no surface) is 0. */
export function heightCodes(heights: Float32Array, range: number): Uint16Array {
  const codes = new Uint16Array(heights.length);
  if (!(range > 0)) {
    return codes;
  }
  const scale = MAX_CODE / range;
  for (let i = 0; i < heights.length; i++) {
    const c = Math.round(heights[i] * scale);
    // NaN fails both comparisons and leaves 0.
    if (c > 0) {
      codes[i] = c < MAX_CODE ? c : MAX_CODE;
    }
  }
  return codes;
}

/** What the JSON next to a height map says: its scale, and where it is in the world. */
export interface HeightMapFile {
  type: "height-map";
  version: 1;
  /** The image's file name. */
  image: string;
  /** Pixels. */
  width: number;
  height: number;
  /** The unit of every length here. */
  units: "mm";
  /** The distance between the centres of neighbouring pixels. */
  pixelSize: number;
  /** The height of code 65535; a code's height is `heightRange * code / 65535`. */
  heightRange: number;
  /** One code's height. */
  heightStep: number;
  /** The code of pixels that see no surface (it is also height 0). */
  noSurface: 0;
  /** The centre of the top left pixel, at height 0. */
  origin: [number, number, number];
  /** Unit vectors: the image's right and up, and the direction heights grow in. */
  right: [number, number, number];
  up: [number, number, number];
  normal: [number, number, number];
  /** The meshes it shows, by name. */
  meshes: string[];
}

/** The JSON of a height map of `meshes` in `frame`, saved as `image`. */
export function heightMapFile(frame: HeightMapFrame, image: string, meshes: readonly string[]): HeightMapFile {
  return {
    type: "height-map",
    version: 1,
    image,
    width: frame.width,
    height: frame.height,
    units: "mm",
    pixelSize: frame.pixelSize,
    heightRange: frame.range,
    heightStep: frame.range / MAX_CODE,
    noSurface: 0,
    origin: [...frame.origin],
    right: [...frame.right],
    up: [...frame.up],
    normal: [...frame.normal],
    meshes: [...meshes],
  };
}

function isVec3(v: unknown): v is [number, number, number] {
  return Array.isArray(v) && v.length === 3 && v.every((c) => typeof c === "number" && Number.isFinite(c));
}

/** The frame a height map's JSON describes, or `null` when the text is not one. */
export function parseHeightMapFile(text: string): HeightMapFrame | null {
  let doc: Partial<HeightMapFile> | null;
  try {
    doc = JSON.parse(text) as Partial<HeightMapFile> | null;
  } catch {
    return null;
  }
  if (typeof doc !== "object" || doc === null || doc.type !== "height-map") {
    return null;
  }
  const { width, height, pixelSize, heightRange: range, origin, right, up } = doc;
  const count = (n: unknown): n is number => typeof n === "number" && Number.isInteger(n) && n > 0;
  if (!count(width) || !count(height) || !isVec3(origin) || !isVec3(right) || !isVec3(up)) {
    return null;
  }
  if (!(typeof pixelSize === "number" && pixelSize > 0 && typeof range === "number" && range >= 0)) {
    return null;
  }
  if (!Number.isFinite(pixelSize) || !Number.isFinite(range)) {
    return null;
  }
  const normal: [number, number, number] = [
    right[1] * up[2] - right[2] * up[1],
    right[2] * up[0] - right[0] * up[2],
    right[0] * up[1] - right[1] * up[0],
  ];
  return { width, height, origin, right, up, normal, pixelSize, range };
}

// ---- An image to heights ----

/** An image as heights: row by row from the top left, 0 black to 1 white. */
export interface HeightImage {
  width: number;
  height: number;
  /** Bits per channel the values came from: 8 or 16. */
  depth: number;
  values: Float32Array;
}

/** Rec. 709 luma weights, for images in colour. */
const LUMA: readonly number[] = [0.2126, 0.7152, 0.0722];

/** The brightness of `count` pixels of `channels` samples each, from 0 to 1; transparent pixels fade to 0. */
function luminance(samples: ArrayLike<number>, count: number, channels: number, max: number): Float32Array {
  const out = new Float32Array(count);
  const alpha = channels === 2 || channels === 4;
  for (let i = 0; i < count; i++) {
    const at = i * channels;
    let v = samples[at];
    if (channels >= 3) {
      v = LUMA[0] * v + LUMA[1] * samples[at + 1] + LUMA[2] * samples[at + 2];
    }
    if (alpha) {
      v *= samples[at + channels - 1] / max;
    }
    out[i] = v / max;
  }
  return out;
}

/** A decoded PNG as heights. */
export function pngHeights(png: PngImage): HeightImage {
  const { width, height, channels, depth, samples } = png;
  const max = depth === 16 ? MAX_CODE : 255;
  return { width, height, depth, values: luminance(samples, width * height, channels, max) };
}

function checkSize(width: number, height: number): void {
  if (width * height > MAX_IMAGE_PIXELS) {
    throw new Error(`The image has ${(width * height).toLocaleString("en")} pixels: at most ${MAX_IMAGE_PIXELS.toLocaleString("en")}.`);
  }
  if (width < 2 || height < 2) {
    throw new Error("The image needs at least 2 pixels on each side.");
  }
}

/** Any image the browser decodes, at 8 bits a channel. */
async function browserHeights(blob: Blob): Promise<HeightImage> {
  let bitmap: ImageBitmap;
  try {
    // The stored values: neither a colour profile nor alpha applied.
    bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  } catch {
    throw new Error("The file is not an image this browser can read.");
  }
  try {
    const { width, height } = bitmap;
    checkSize(width, height);
    let canvas: OffscreenCanvas | HTMLCanvasElement;
    if (typeof OffscreenCanvas === "undefined") {
      canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
    } else {
      canvas = new OffscreenCanvas(width, height);
    }
    const ctx = canvas.getContext("2d", { willReadFrequently: true }) as OffscreenCanvasRenderingContext2D | null;
    if (ctx === null) {
      throw new Error("The image could not be read: no canvas.");
    }
    ctx.drawImage(bitmap, 0, 0);
    const { data } = ctx.getImageData(0, 0, width, height);
    return { width, height, depth: 8, values: luminance(data, width * height, 4, 255) };
  } finally {
    bitmap.close();
  }
}

/**
 * Reads an image file as heights: a PNG with whole 8 or 16-bit channels by
 * `png.ts`, so that 16 bits stay 16, anything else by the browser.
 *
 * @throws with a message for the user when it cannot be read.
 */
export async function readHeightImage(blob: Blob): Promise<HeightImage> {
  const head = new Uint8Array(await blob.slice(0, 64).arrayBuffer());
  if (isPng(head)) {
    const header = readPngHeader(head);
    if (header !== null && canDecodePng(header)) {
      checkSize(header.width, header.height);
      return pngHeights(await decodePng(new Uint8Array(await blob.arrayBuffer())));
    }
  }
  return browserHeights(blob);
}

/**
 * The size of the grid that samples an image with at most `samples` along its
 * longer side, keeping its proportions (and at least 2 on each side).
 */
export function gridSize(width: number, height: number, samples: number): [number, number] {
  const longer = Math.max(width, height);
  const k = Math.min(1, (Math.max(2, Math.round(samples)) - 1) / (longer - 1));
  const side = (n: number) => Math.max(2, Math.round((n - 1) * k) + 1);
  return [side(width), side(height)];
}

/** For each of `n` samples across `size` pixels: the first pixel it takes, and the weights from there. */
function weights(size: number, n: number): { first: number; w: number[] }[] {
  const step = (size - 1) / (n - 1);
  const out: { first: number; w: number[] }[] = [];
  for (let i = 0; i < n; i++) {
    const x = i * step;
    if (step <= 1) {
      // Between pixels (or on one): interpolate.
      const first = Math.min(Math.floor(x), size - 1);
      const f = x - first;
      out.push(f > 1e-9 && first + 1 < size ? { first, w: [1 - f, f] } : { first, w: [1] });
      continue;
    }
    // Fewer samples than pixels: the mean over what lies around the sample.
    const lo = Math.max(x - 0.5 * step, -0.5);
    const hi = Math.min(x + 0.5 * step, size - 0.5);
    const first = Math.max(0, Math.floor(lo + 0.5));
    const last = Math.min(size - 1, Math.ceil(hi - 0.5));
    const w: number[] = [];
    let sum = 0;
    for (let k = first; k <= last; k++) {
      const part = Math.max(0, Math.min(hi, k + 0.5) - Math.max(lo, k - 0.5));
      w.push(part);
      sum += part;
    }
    out.push({ first, w: w.map((v) => v / sum) });
  }
  return out;
}

/** How a grid's heights come from an image's values. */
export interface GridOptions {
  /** The height of white. */
  scale: number;
  /** Black is high instead. */
  invert: boolean;
}

/**
 * The heights of a `w` by `h` grid over `image`: its corner samples lie on the
 * corner pixels, so the grid spans the same area at every size. At the image's
 * own size the values are its own; at a smaller one each is the mean of the
 * pixels around it.
 */
export function imageGrid(image: HeightImage, w: number, h: number, options: GridOptions): Float32Array {
  const { width, height, values } = image;
  if (!(w >= 2 && h >= 2 && width >= 2 && height >= 2)) {
    throw new RangeError("a grid needs at least 2 samples on each side");
  }
  const { scale, invert } = options;
  const out = new Float32Array(w * h);
  const finish = (v: number) => scale * (invert ? 1 - v : v);
  if (w === width && h === height) {
    for (let i = 0; i < out.length; i++) {
      out[i] = finish(values[i]);
    }
    return out;
  }
  // Rows first, only those the columns take, then columns.
  const across = weights(width, w);
  const down = weights(height, h);
  const lines = new Map<number, Float64Array>();
  const line = (row: number): Float64Array => {
    let l = lines.get(row);
    if (l === undefined) {
      l = new Float64Array(w);
      for (let col = 0; col < w; col++) {
        const { first, w: ws } = across[col];
        let sum = 0;
        for (let k = 0; k < ws.length; k++) {
          sum += ws[k] * values[row * width + first + k];
        }
        l[col] = sum;
      }
      lines.set(row, l);
    }
    return l;
  };
  for (let row = 0; row < h; row++) {
    const { first, w: ws } = down[row];
    const sums = new Float64Array(w);
    for (let k = 0; k < ws.length; k++) {
      const l = line(first + k);
      for (let col = 0; col < w; col++) {
        sums[col] += ws[k] * l[col];
      }
    }
    for (let col = 0; col < w; col++) {
      out[row * w + col] = finish(sums[col]);
    }
    // Rows before the next sample's are done with.
    const next = row + 1 < h ? down[row + 1].first : height;
    for (const key of lines.keys()) {
      if (key < next) {
        lines.delete(key);
      }
    }
  }
  return out;
}

// ---- The jobs the page runs in a Worker ----

/** An opened image: its size and depth, and the id its grids are asked for by. */
export interface HeightImageInfo {
  id: number;
  width: number;
  height: number;
  depth: number;
}

/** Runs the jobs: in the Worker, or on the page where there is none. */
export class HeightImageEngine {
  readonly #images = new Map<number, HeightImage>();
  #next = 1;

  /** Heights as a 16-bit PNG's bytes. */
  encode(heights: Float32Array, width: number, height: number, range: number): Promise<Uint8Array> {
    return encodePng16(heightCodes(heights, range), width, height);
  }

  /** Reads an image and keeps it until {@link close}. */
  async open(blob: Blob): Promise<HeightImageInfo> {
    const image = await readHeightImage(blob);
    const id = this.#next++;
    this.#images.set(id, image);
    return { id, width: image.width, height: image.height, depth: image.depth };
  }

  /** The heights of a grid over an open image. */
  grid(id: number, w: number, h: number, options: GridOptions): Float32Array {
    const image = this.#images.get(id);
    if (image === undefined) {
      throw new Error("the image is closed");
    }
    return imageGrid(image, w, h, options);
  }

  close(id: number): void {
    this.#images.delete(id);
  }
}

/** A job of {@link HeightImageEngine}, as the page posts it. */
export type HeightImageRequest = { n: number } & (
  | { op: "encode"; heights: Float32Array; width: number; height: number; range: number }
  | { op: "open"; blob: Blob }
  | { op: "grid"; id: number; w: number; h: number; options: GridOptions }
  | { op: "close"; id: number }
);

/** A job's result, or what it failed with. */
export type HeightImageReply = { n: number } & ({ ok: true; value: unknown } | { ok: false; message: string });
