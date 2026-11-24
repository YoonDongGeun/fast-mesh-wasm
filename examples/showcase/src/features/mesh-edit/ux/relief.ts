// Image to relief (#267): the settings of a relief made from an image, their
// defaults and checks, the grid they sample it with, and what a height map's
// JSON (its scale and place) sets them to.
import type { HeightMapFrame } from "fast-mesh-wasm";
import type { Vec3 } from "../../../io/mesh-data";
import { gridSize, isImageFile } from "../../../io/height-image";
import { formatLength, type LengthUnit } from "../../../shared/ux/units";
import { mat3ToEulerDeg } from "../../transform/ux/transform-math";

/** A relief's settings; lengths in millimetres. */
export interface ReliefParams {
  /** The extent along the image's rows, from the first sample to the last. */
  width: number;
  /** The height of white above height 0 (black). */
  maxHeight: number;
  /** The thickness below height 0. */
  base: number;
  /** Samples along the image's longer side. */
  samples: number;
  /** Black is high instead. */
  invert: boolean;
}

/** An image's size, for the settings that depend on it. */
export interface ImageSize {
  width: number;
  height: number;
}

/** The most triangles a relief is made with, to keep the page responsive. */
export const MAX_RELIEF_TRIANGLES = 10_000_000;

/** The default samples along the longer side: half a million triangles for a square. */
export const DEFAULT_SAMPLES = 512;

/** The most samples along the longer side of the live preview. */
export const PREVIEW_SAMPLES = 160;

/** The triangles of a relief of a `w` by `h` grid, as the engine builds it. */
export function reliefTriangles(w: number, h: number): number {
  return 2 * (w - 1) * (h - 1) + 3 * (2 * (w + h) - 4) - 2;
}

/** The most samples along an image's longer side that stay within {@link MAX_RELIEF_TRIANGLES}. */
export function maxSamples(image: ImageSize): number {
  const longer = Math.max(image.width, image.height);
  let lo = 2;
  let hi = longer;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (reliefTriangles(...gridSize(image.width, image.height, mid)) <= MAX_RELIEF_TRIANGLES) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}

/** The settings a relief of `image` starts with, about `size` across its longer side. */
export function defaultRelief(image: ImageSize, size: number): ReliefParams {
  const longer = Math.max(image.width, image.height);
  return {
    width: (size * (image.width - 1)) / (longer - 1),
    maxHeight: size / 10,
    base: size / 20,
    samples: Math.min(DEFAULT_SAMPLES, longer, maxSamples(image)),
    invert: false,
  };
}

/** Why the settings cannot make a relief of `image`, in words, or `null` when they can. */
export function reliefProblem(p: ReliefParams, image: ImageSize): string | null {
  const positive: readonly (readonly [number, string])[] = [
    [p.width, "The width"],
    [p.maxHeight, "The height"],
    [p.base, "The base thickness"],
  ];
  for (const [v, what] of positive) {
    if (!(Number.isFinite(v) && v > 0)) {
      return `${what} must be above zero.`;
    }
  }
  const longer = Math.max(image.width, image.height);
  if (!Number.isInteger(p.samples) || p.samples < 2 || p.samples > longer) {
    return `The samples must be a whole number from 2 to ${longer}, the image's longer side.`;
  }
  const triangles = reliefTriangles(...gridSize(image.width, image.height, p.samples));
  if (triangles > MAX_RELIEF_TRIANGLES) {
    return `${triangles.toLocaleString("en")} triangles: at most ${MAX_RELIEF_TRIANGLES.toLocaleString("en")} (${maxSamples(image)} samples).`;
  }
  return null;
}

/** The extent across the image's rows that keeps its proportions. */
export function reliefDepth(width: number, image: ImageSize): number {
  return (width * (image.height - 1)) / (image.width - 1);
}

/** The grid a relief samples its image with, and the distance between its samples. */
export function reliefGrid(
  p: ReliefParams,
  image: ImageSize,
  samples = p.samples,
): { w: number; h: number; pixelSize: [number, number] } {
  const [w, h] = gridSize(image.width, image.height, Math.min(samples, p.samples));
  return { w, h, pixelSize: [p.width / (w - 1), reliefDepth(p.width, image) / (h - 1)] };
}

/** What the status line says of a relief: its triangles, samples and size. */
export function describeRelief(p: ReliefParams, image: ImageSize, unit: LengthUnit): string {
  const { w, h } = reliefGrid(p, image);
  const size = [p.width, reliefDepth(p.width, image), p.maxHeight + p.base].map((mm) => formatLength(mm, unit));
  return `${reliefTriangles(w, h).toLocaleString("en")} triangles · ${w} × ${h} samples · ${size.join(" × ")}`;
}

/** Where a relief goes: the centre of its height 0, and its turn from facing `+z`. */
export interface ReliefPlace {
  centre: Vec3;
  /** Euler XYZ degrees. */
  rotationDeg: Vec3;
}

/**
 * What a height map's frame sets for its image: the settings that make the
 * relief its surface, and where it was. `null` when the frame is for an image
 * of another size, or shows no height.
 */
export function reliefOfFrame(
  frame: HeightMapFrame,
  image: ImageSize,
  base: number,
): { params: ReliefParams; place: ReliefPlace } | null {
  if (frame.width !== image.width || frame.height !== image.height || !(frame.range > 0)) {
    return null;
  }
  // As the library's `heightMapPlacement`: the image's axes as columns, at its
  // centre (written out here, so that the page loads no wasm for it).
  const { right, up, normal, origin, pixelSize } = frame;
  const u = 0.5 * (frame.width - 1) * pixelSize;
  const v = -0.5 * (frame.height - 1) * pixelSize;
  const rotation = [0, 1, 2].flatMap((k) => [right[k], up[k], normal[k]]);
  const centre = [0, 1, 2].map((k) => origin[k] + u * right[k] + v * up[k]) as Vec3;
  return {
    params: {
      width: (frame.width - 1) * frame.pixelSize,
      maxHeight: frame.range,
      base,
      samples: maxSamples(image),
      invert: false,
    },
    place: {
      centre,
      rotationDeg: mat3ToEulerDeg(rotation),
    },
  };
}

/** A mesh name from an image's file name. */
export function reliefName(file: string): string {
  return file.replace(/\.[^./\\]+$/, "").trim() || "Relief";
}

/** The files of a drop or a file picker that a relief takes: an image, and the JSON that may come with it. */
export interface ReliefFiles {
  image: File;
  /** A `.json` next to the image: a height map's scale and place, or anything else. */
  json: File | null;
}

/**
 * Splits dropped files into the ones a relief takes (the first image, with the
 * `.json` of its name or else the first one) and the rest, which load as
 * before. Without an image everything is the rest.
 */
export function splitReliefFiles(files: Iterable<File>): { relief: ReliefFiles | null; rest: File[] } {
  const list = [...files];
  const image = list.find(isImageFile);
  if (image === undefined) {
    return { relief: null, rest: list };
  }
  const jsons = list.filter((f) => /\.json$/i.test(f.name));
  const stem = reliefName(image.name);
  const json = jsons.find((f) => reliefName(f.name) === stem) ?? jsons[0] ?? null;
  return {
    relief: { image, json },
    rest: list.filter((f) => f !== json && !isImageFile(f)),
  };
}
