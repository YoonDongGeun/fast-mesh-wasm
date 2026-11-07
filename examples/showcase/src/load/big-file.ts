// Large-file UX (issue #113): weight thresholds, memory estimates, URL
// overrides for testing, and the post-load summary line.
//
// Pure (no scene, no DOM) and unit-tested: the loader, the worker and the
// mesh list share these so the headless check (`?warn-tris=` etc.) drives
// the same code the UI shows.
//
// Threshold rationale: past ~1M triangles landing a background add alone
// blocks the worker ~20 ms (and ~136 ms at 10M; see #64), so 500k warns and
// 1M offers decimation. The decimate target (200k) matches the #118
// `simplify` example the decimate-on-load step reuses.
export interface BigFileOptions {
  /** At or past this many triangles a mesh counts as heavy (warning). */
  warnTriangles: number;
  /** At or past this many triangles a load offers decimation. */
  decimateTriangles: number;
  /** Decimate-on-load simplifies heavy meshes down to this many triangles. */
  decimateTarget: number;
  /** Whether heavy loads are decimated automatically (the list's checkbox). */
  decimateOnLoad: boolean;
  /** `?big`: the demo button loads the ~1M-triangle A/B variants; `?big=N`: about N triangles each. */
  demoBig: boolean | number;
}

/** Warn past 500k triangles, offer decimation past 1M, down to 200k. */
export const BIG_WARN_TRIANGLES = 500_000;
/** @see BIG_WARN_TRIANGLES */
export const BIG_DECIMATE_TRIANGLES = 1_000_000;
/** @see BIG_WARN_TRIANGLES */
export const BIG_DECIMATE_TARGET = 200_000;

export const BIG_FILE_DEFAULTS: BigFileOptions = {
  warnTriangles: BIG_WARN_TRIANGLES,
  decimateTriangles: BIG_DECIMATE_TRIANGLES,
  decimateTarget: BIG_DECIMATE_TARGET,
  decimateOnLoad: true,
  demoBig: false,
};

/**
 * Reads the large-file options from a query string (`location.search`):
 * `?warn-tris=`, `?decimate-tris=`, `?decimate-target=` (positive integers;
 * garbage falls back to the defaults), `?decimate=0` (off) and `?big` (the
 * big demo). Tolerant: never throws.
 */
export function parseBigFileParams(search: string): BigFileOptions {
  let params: URLSearchParams;
  try {
    params = new URLSearchParams(search);
  } catch {
    return { ...BIG_FILE_DEFAULTS };
  }
  const num = (key: string, fallback: number): number => {
    const raw = params.get(key);
    if (raw === null) {
      return fallback;
    }
    const n = Math.floor(Number(raw));
    return Number.isFinite(n) && n > 0 ? n : fallback;
  };
  return {
    warnTriangles: num("warn-tris", BIG_FILE_DEFAULTS.warnTriangles),
    decimateTriangles: num("decimate-tris", BIG_FILE_DEFAULTS.decimateTriangles),
    decimateTarget: num("decimate-target", BIG_FILE_DEFAULTS.decimateTarget),
    decimateOnLoad: params.get("decimate") === "0" ? false : BIG_FILE_DEFAULTS.decimateOnLoad,
    demoBig: params.has("big") ? num("big", 0) || true : false,
  };
}

/** Whether `triangles` is heavy, and whether it is offered decimation. */
export function bigFilePlan(
  triangles: number,
  opts: Pick<BigFileOptions, "warnTriangles" | "decimateTriangles">,
): { warn: boolean; offerDecimate: boolean } {
  return {
    warn: triangles >= opts.warnTriangles,
    offerDecimate: triangles >= opts.decimateTriangles,
  };
}

/**
 * The decimate-on-load target for `triangles` (`null` when it is already at
 * or under `target`: #118 `simplify` would leave it alone).
 */
export function decimateTargetFor(triangles: number, target: number): number | null {
  return triangles > target ? target : null;
}

/**
 * Rough in-memory bytes for a mesh: positions as f64 (24 B/vertex) plus
 * indices as u32 (12 B/triangle). Unknown vertex counts assume a closed
 * mesh (`tris / 2 + 2` vertices).
 */
export function estimateMeshBytes(triangles: number, vertices?: number): number {
  const v = vertices ?? Math.ceil(Math.max(0, triangles) / 2) + 2;
  return Math.max(0, v) * 24 + Math.max(0, triangles) * 12;
}

/** "12,288" or "1.05 M". */
export function formatTriangles(n: number): string {
  return n >= 1_000_000 ? `${(n / 1_000_000).toFixed(2)} M` : Math.round(n).toLocaleString("en");
}

/** "850 ms" or "2.3 s". */
export function formatLoadMs(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}

/** "≈ 24.0 MB". */
export function formatMeshBytes(bytes: number): string {
  if (bytes >= 1_000_000) {
    return `≈ ${(bytes / 1_000_000).toFixed(1)} MB`;
  }
  return `≈ ${Math.max(1, Math.round(bytes / 1000))} KB`;
}

/** What a finished load reports: its weight, time, and what decimation did. */
export interface LoadSummaryInput {
  name: string;
  triangles: number;
  loadMs: number;
  /** `null` when the load was not decimated. */
  decimated: { from: number; to: number } | null;
}

/**
 * One post-load line, e.g. `big-sphere.stl: 1.00 M tris in 2.3 s
 * (decimated 1.00 M → 200,000)`.
 */
export function summarizeLoad(input: LoadSummaryInput): string {
  const base = `${input.name}: ${formatTriangles(input.triangles)} in ${formatLoadMs(input.loadMs)}`;
  if (input.decimated === null) {
    return base;
  }
  const { from, to } = input.decimated;
  return `${base} (decimated ${formatTriangles(from)} → ${to.toLocaleString("en")})`;
}
