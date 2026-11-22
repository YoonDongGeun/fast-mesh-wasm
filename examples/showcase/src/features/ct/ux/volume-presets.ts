// The volume presets: HU-tuned color/opacity ramps (bone, soft tissue,
// vessels) applied to the vtk.js transfer functions. Air (-1000 HU and
// below) stays invisible in every preset.
export type WindowValue = number | "auto";

export interface WindowPreset {
  name: string;
  window: WindowValue;
  level: WindowValue;
}

/** Window/level for the slice views; full range follows the data min/max. */
export const WINDOW_PRESETS: readonly WindowPreset[] = [
  { name: "Bone", window: 2000, level: 400 },
  { name: "Soft tissue", window: 400, level: 40 },
  { name: "Lung", window: 1600, level: -600 },
  { name: "Full range", window: "auto", level: "auto" },
];

export interface VolumePreset {
  name: string;
  /** [value, r, g, b] color points, ascending. */
  color: [number, number, number, number][];
  /** [value, opacity] points, ascending. */
  opacity: [number, number][];
}

export const VOLUME_PRESETS: VolumePreset[] = [
  {
    name: "Bone",
    color: [
      [-1000, 0, 0, 0],
      [200, 0.55, 0.45, 0.38],
      [500, 0.9, 0.82, 0.72],
      [1200, 1, 0.96, 0.88],
      [2500, 1, 1, 0.95],
    ],
    opacity: [
      [-1000, 0],
      [200, 0],
      [400, 0.18],
      [800, 0.55],
      [1500, 0.85],
      [2500, 0.95],
    ],
  },
  {
    name: "Soft tissue",
    color: [
      [-1000, 0, 0, 0],
      [-150, 0.35, 0.12, 0.12],
      [-20, 0.75, 0.42, 0.35],
      [120, 0.95, 0.8, 0.7],
      [400, 1, 0.95, 0.88],
    ],
    opacity: [
      [-1000, 0],
      [-150, 0],
      [-50, 0.22],
      [80, 0.45],
      [220, 0.65],
      [400, 0.8],
    ],
  },
  {
    name: "Vessels",
    color: [
      [-1000, 0, 0, 0],
      [100, 0.25, 0.02, 0.02],
      [250, 0.9, 0.15, 0.08],
      [450, 1, 0.55, 0.3],
      [800, 1, 0.92, 0.78],
    ],
    opacity: [
      [-1000, 0],
      [100, 0],
      [220, 0.35],
      [350, 0.7],
      [600, 0.95],
    ],
  },
];
