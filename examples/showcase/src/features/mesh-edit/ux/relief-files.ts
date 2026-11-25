// Which dropped files the Image to Relief dialog takes (#267). Apart from
// relief.ts: every drop on the page asks, before the Mesh Edit chunk is here.
import { isImageFile } from "../../../io/image-file";

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
