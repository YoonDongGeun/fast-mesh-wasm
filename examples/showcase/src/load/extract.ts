// The DICOM extraction pipeline: files parsed, series stacked, each
// isosurface meshed (Otsu's threshold unless one is given) and named for
// the mesh list. The loader Worker runs it; failures name their file.
import { parseDicom } from "./dicom";
import { isosurface, otsu } from "./isosurface";
import type { MeshData } from "./mesh-data";
import { checkMesh } from "./mesh-data";
import { stackSeries } from "./series";

export interface ExtractedSeries {
  name: string;
  data: MeshData;
  threshold: number;
  auto: boolean;
  seriesUid: string;
  /** [columns, rows, slices]. */
  dims: [number, number, number];
}

const fmt = (t: number): string => String(Math.round(t * 100) / 100);

/**
 * Extracts every series in `files` (`threshold` null takes Otsu's).
 * `@throws` with the file or series and the reason.
 */
export function extractSeries(
  files: readonly { name: string; buffer: ArrayBuffer }[],
  threshold: number | null,
): ExtractedSeries[] {
  const slices = files.map((f) => parseDicom(f.name, f.buffer));
  return stackSeries(slices).map((v) => {
    const thr = threshold ?? otsu(v.values);
    const data = isosurface(v.values, v, thr, v.inverted);
    const label = v.description !== "" ? v.description : `DICOM ${v.seriesUid.slice(-12)}`;
    if (data.indices.length === 0) {
      throw new Error(`${label}: threshold ${fmt(thr)} extracts no surface (a uniform volume?)`);
    }
    return {
      name: `${label} (iso ${fmt(thr)})`,
      data: checkMesh(data, label),
      threshold: thr,
      auto: threshold === null,
      seriesUid: v.seriesUid,
      dims: [v.columns, v.rows, v.slices],
    };
  });
}
