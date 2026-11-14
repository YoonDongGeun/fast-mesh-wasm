// Showcase display settings, persisted in localStorage (issue #144): the
// theme, the sidebars (shown, widths), the display prefs (wireframe,
// boundary softness, repair overlays, collision mode, decimate-on-load,
// sculpt remesh) and the last-used heatmap sliders (the collision colour
// scale). Meshes and projects stay file-based (see load/project.ts), and
// volume views keep their suggested-or-Bone preset: this is prefs only,
// so a fresh profile gets the defaults and the scripted session
// (scripts/showcase-check.mjs) never sees stored state.
//
// Pure helpers (no DOM at import, like tour.ts): storage is touched only
// inside load/save/reset, each guarded the way tour.ts guards its flag,
// so private mode just means "no persistence".
import type { GrabMethod, InflateMethod, SmoothMethod, TwistMethod } from "fast-mesh-wasm";
import type { CollisionMode, HeatmapParams, RepairOverlayKind } from "../scene/app-state";
import type { LengthUnit } from "./units";

export interface SculptMethods {
  inflate: InflateMethod;
  smooth: SmoothMethod;
  grab: GrabMethod;
  twist: TwistMethod;
}

export type SculptFilterKind = "none" | "peaksOnly" | "valleysOnly" | "normalAngle";

export interface SculptFilterSettings {
  kind: SculptFilterKind;
  threshold: number;
  contrast: number;
  maxAngleDeg: number;
}

export interface SculptSymmetrySettings {
  x: boolean;
  y: boolean;
  z: boolean;
}

/** The settings document's storage key: namespaced, with its version. */
export const SETTINGS_KEY = "fm.showcase.settings.v1";

/** The only settings version this build reads; anything else is defaults. */
export const SETTINGS_VERSION = 1;

/** The app theme: the dark default, or the light one. */
export type ShowcaseTheme = "dark" | "light";

/** The default sidebar width, in pixels (see styles.css `.app`). */
export const DEFAULT_SIDEBAR_WIDTH = 290;
/** How narrow/wide a drag may make a sidebar, in pixels. */
export const MIN_SIDEBAR_WIDTH = 200;
export const MAX_SIDEBAR_WIDTH = 560;

export function clampSidebarWidth(width: number): number {
  if (!Number.isFinite(width)) {
    return DEFAULT_SIDEBAR_WIDTH;
  }
  return Math.min(MAX_SIDEBAR_WIDTH, Math.max(MIN_SIDEBAR_WIDTH, width));
}

export interface ShowcaseSettings {
  version: typeof SETTINGS_VERSION;
  theme: ShowcaseTheme;
  leftVisible: boolean;
  rightVisible: boolean;
  leftWidth: number;
  rightWidth: number;
  wireframe: boolean;
  edgeSoftness: number;
  repairOverlays: Record<RepairOverlayKind, boolean>;
  collisionMode: CollisionMode;
  /** The last-used heatmap sliders; `null` follows the meshes' scale. */
  heatmap: HeatmapParams | null;
  heatmapFast: boolean;
  decimateOnLoad: boolean;
  /**
   * The app's `sculptRemesh`. A new key, not the old `sculptRemesh`: that one
   * saved the old default of `true`, which must not keep the remesh on.
   */
  sculptEvenOut: boolean;
  sculptLiveRemesh: boolean;
  maskRadius: number;
  /** Per-brush algorithm choices (#177, #192). */
  sculptMethods: SculptMethods;
  /** Kelvinlet grab Poisson ratio in `[0, 0.5]` (#177). */
  sculptGrabPoisson: number;
  /** Composable surface feature filter (#177, #192). */
  sculptFilter: SculptFilterSettings;
  /** Mirror symmetry axes across local `x = 0`, `y = 0`, `z = 0` planes (#192). */
  sculptSymmetry: SculptSymmetrySettings;
  /** The display unit new sessions and older project files start on; a project file's own wins. */
  defaultUnits: LengthUnit;
}

/** A fresh defaults document (a new object every call: callers may keep it). */
export function defaultSettings(): ShowcaseSettings {
  return {
    version: SETTINGS_VERSION,
    theme: "dark",
    leftVisible: true,
    rightVisible: true,
    leftWidth: DEFAULT_SIDEBAR_WIDTH,
    rightWidth: DEFAULT_SIDEBAR_WIDTH,
    wireframe: false,
    edgeSoftness: 0,
    repairOverlays: { holes: true, nonManifold: true, degenerate: true, components: true },
    collisionMode: "pair",
    heatmap: null,
    heatmapFast: false,
    decimateOnLoad: true,
    sculptEvenOut: false,
    sculptLiveRemesh: true,
    maskRadius: 0.25,
    sculptMethods: {
      inflate: "vertexNormal",
      smooth: "laplacian",
      grab: "sphere",
      twist: "vortex",
    },
    sculptGrabPoisson: 0.4,
    sculptFilter: {
      kind: "none",
      threshold: 0.05,
      contrast: 0.5,
      maxAngleDeg: 45,
    },
    sculptSymmetry: { x: false, y: false, z: false },
    defaultUnits: "mm",
  };
}

/** `localStorage.getItem`, guarded (private mode reads as missing). */
export function readStorage(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

/** `localStorage.setItem`, guarded (private mode just does not persist). */
export function writeStorage(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Private mode, say: the prefs just do not survive the reload.
  }
}

/** `localStorage.removeItem`, guarded like the rest. */
export function removeStorage(key: string): void {
  try {
    localStorage.removeItem(key);
  } catch {
    // Nothing stored, nothing to forget.
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function asBoolean(value: unknown, fallback: boolean): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function asNumber(value: unknown, fallback: number, min = 0): number {
  return typeof value === "number" && Number.isFinite(value) && value >= min ? value : fallback;
}

function asLiteral<T extends string>(value: unknown, allowed: readonly T[], fallback: T): T {
  return typeof value === "string" && (allowed as readonly string[]).includes(value) ? (value as T) : fallback;
}

function parseHeatmap(value: unknown): HeatmapParams | null {
  if (!isRecord(value)) {
    return null;
  }
  const { depth, gap, tolerance } = value;
  if (
    typeof depth !== "number" ||
    !Number.isFinite(depth) ||
    depth < 0 ||
    typeof gap !== "number" ||
    !Number.isFinite(gap) ||
    gap < 0 ||
    typeof tolerance !== "number" ||
    !Number.isFinite(tolerance) ||
    tolerance < 0
  ) {
    return null;
  }
  return { depth, gap, tolerance };
}

function asBoundedNumber(value: unknown, fallback: number, min: number, max: number, minInclusive = true): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return fallback;
  }
  const okMin = minInclusive ? value >= min : value > min;
  return okMin && value <= max ? value : fallback;
}

/**
 * Reads a stored document tolerantly: a wrong version, a corrupt shape or
 * a mistyped field falls back to the defaults (per field, so one bad
 * value never wipes the rest).
 */
export function parseSettings(doc: unknown): ShowcaseSettings {
  const fallback = defaultSettings();
  if (!isRecord(doc) || doc.version !== SETTINGS_VERSION) {
    return fallback;
  }
  const overlays = isRecord(doc.repairOverlays) ? doc.repairOverlays : {};
  const methods = isRecord(doc.sculptMethods) ? doc.sculptMethods : {};
  const filter = isRecord(doc.sculptFilter) ? doc.sculptFilter : {};
  const symmetry = isRecord(doc.sculptSymmetry) ? doc.sculptSymmetry : {};
  return {
    version: SETTINGS_VERSION,
    theme: asLiteral(doc.theme, ["dark", "light"], fallback.theme),
    leftVisible: asBoolean(doc.leftVisible, fallback.leftVisible),
    rightVisible: asBoolean(doc.rightVisible, fallback.rightVisible),
    leftWidth: clampSidebarWidth(asNumber(doc.leftWidth, fallback.leftWidth)),
    rightWidth: clampSidebarWidth(asNumber(doc.rightWidth, fallback.rightWidth)),
    wireframe: asBoolean(doc.wireframe, fallback.wireframe),
    edgeSoftness: asNumber(doc.edgeSoftness, fallback.edgeSoftness),
    repairOverlays: {
      holes: asBoolean(overlays.holes, true),
      nonManifold: asBoolean(overlays.nonManifold, true),
      degenerate: asBoolean(overlays.degenerate, true),
      components: asBoolean(overlays.components, true),
    },
    collisionMode: asLiteral(doc.collisionMode, ["pair", "group"], fallback.collisionMode),
    heatmap: parseHeatmap(doc.heatmap),
    heatmapFast: asBoolean(doc.heatmapFast, fallback.heatmapFast),
    decimateOnLoad: asBoolean(doc.decimateOnLoad, fallback.decimateOnLoad),
    sculptEvenOut: asBoolean(doc.sculptEvenOut, fallback.sculptEvenOut),
    sculptLiveRemesh: asBoolean(doc.sculptLiveRemesh, fallback.sculptLiveRemesh),
    maskRadius: asNumber(doc.maskRadius, fallback.maskRadius, Number.MIN_VALUE),
    sculptMethods: {
      inflate: asLiteral(
        methods.inflate,
        ["vertexNormal", "averageNormal", "strokeNormal"],
        fallback.sculptMethods.inflate,
      ),
      smooth: asLiteral(
        methods.smooth,
        ["taubin", "laplacian", "hcLaplacian", "cotangent"],
        fallback.sculptMethods.smooth,
      ),
      grab: asLiteral(methods.grab, ["sphere", "geodesic", "kelvinlet"], fallback.sculptMethods.grab),
      twist: asLiteral(methods.twist, ["vortex", "kelvinlet"], fallback.sculptMethods.twist),
    },
    sculptGrabPoisson: asBoundedNumber(doc.sculptGrabPoisson, fallback.sculptGrabPoisson, 0, 0.5),
    sculptFilter: {
      kind: asLiteral(filter.kind, ["none", "peaksOnly", "valleysOnly", "normalAngle"], fallback.sculptFilter.kind),
      threshold: asBoundedNumber(filter.threshold, fallback.sculptFilter.threshold, 0, 1),
      contrast: asBoundedNumber(filter.contrast, fallback.sculptFilter.contrast, 0, 1),
      maxAngleDeg: asBoundedNumber(filter.maxAngleDeg, fallback.sculptFilter.maxAngleDeg, 0, 180, false),
    },
    sculptSymmetry: {
      x: asBoolean(symmetry.x, fallback.sculptSymmetry.x),
      y: asBoolean(symmetry.y, fallback.sculptSymmetry.y),
      z: asBoolean(symmetry.z, fallback.sculptSymmetry.z),
    },
    defaultUnits: asLiteral(doc.defaultUnits, ["mm", "in"], fallback.defaultUnits),
  };
}

/** Loads the stored settings, or the defaults (missing, corrupt, foreign). */
export function loadSettings(): ShowcaseSettings {
  const raw = readStorage(SETTINGS_KEY);
  if (raw === null) {
    return defaultSettings();
  }
  try {
    return parseSettings(JSON.parse(raw) as unknown);
  } catch {
    return defaultSettings();
  }
}

/**
 * Stores `patch` over the stored settings (top-level fields wholesale)
 * and returns what is stored now.
 */
export function saveSettings(patch: Partial<ShowcaseSettings>): ShowcaseSettings {
  const next: ShowcaseSettings = { ...loadSettings(), ...patch, version: SETTINGS_VERSION };
  writeStorage(SETTINGS_KEY, JSON.stringify(next));
  return next;
}

/** Forgets the stored settings; the defaults apply from here on. */
export function resetSettings(): ShowcaseSettings {
  removeStorage(SETTINGS_KEY);
  return defaultSettings();
}

/** The app snapshot's persisted slice (structural: AppSnapshot satisfies it). */
export interface PersistedSnapshot {
  theme: ShowcaseTheme;
  leftVisible: boolean;
  rightVisible: boolean;
  leftWidth: number;
  rightWidth: number;
  wireframe: boolean;
  edgeSoftness: number;
  repairOverlays: Record<RepairOverlayKind, boolean>;
  collisionMode: CollisionMode;
  heatmapParams: HeatmapParams | null;
  heatmapFast: boolean;
  decimateOnLoad: boolean;
  sculptRemesh: boolean;
  sculptLiveRemesh: boolean;
  maskRadius: number;
  sculptMethods: SculptMethods;
  sculptGrabPoisson: number;
  sculptFilter: SculptFilterSettings;
  sculptSymmetry: SculptSymmetrySettings;
  defaultUnits: LengthUnit;
}

/** Collects the snapshot's persisted slice into a storable document. */
export function settingsFromSnapshot(snap: PersistedSnapshot): ShowcaseSettings {
  return {
    version: SETTINGS_VERSION,
    theme: snap.theme,
    leftVisible: snap.leftVisible,
    rightVisible: snap.rightVisible,
    leftWidth: snap.leftWidth,
    rightWidth: snap.rightWidth,
    wireframe: snap.wireframe,
    edgeSoftness: snap.edgeSoftness,
    repairOverlays: { ...snap.repairOverlays },
    collisionMode: snap.collisionMode,
    heatmap: snap.heatmapParams === null ? null : { ...snap.heatmapParams },
    heatmapFast: snap.heatmapFast,
    decimateOnLoad: snap.decimateOnLoad,
    sculptEvenOut: snap.sculptRemesh,
    sculptLiveRemesh: snap.sculptLiveRemesh,
    maskRadius: snap.maskRadius,
    sculptMethods: { ...snap.sculptMethods },
    sculptGrabPoisson: snap.sculptGrabPoisson,
    sculptFilter: { ...snap.sculptFilter },
    sculptSymmetry: { ...snap.sculptSymmetry },
    defaultUnits: snap.defaultUnits,
  };
}
