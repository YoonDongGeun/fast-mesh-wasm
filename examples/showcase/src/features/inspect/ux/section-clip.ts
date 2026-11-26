// The section plane's and the clip box's panel side (#110): where the plane
// is anchored, what its axis buttons and offset slider read, the store
// calls behind them, and the clip box's fit and slider edits. Both work in
// world coordinates, through the mesh's placement.
import type { Scene, SectionHandle } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import type { AppState, Item } from "../../../app/state/app-state";
import { worldCenter } from "../../../shared/ux/mesh-center";

export type Axis = "x" | "y" | "z";
export const AXES: Axis[] = ["x", "y", "z"];
export const AXIS_NORMAL: Record<Axis, [number, number, number]> = {
  x: [1, 0, 0],
  y: [0, 1, 0],
  z: [0, 0, 1],
};

/** The section as the panel sees it: the store's handle and its world plane. */
export type PanelSection = {
  handle: unknown;
  plane: { origin: readonly [number, number, number]; normal: readonly [number, number, number] };
};

type SectionPlaneInput = { origin: [number, number, number]; normal: [number, number, number] };

/**
 * The section's anchor: the selected mesh's world center (through the
 * placement, so the offset slider follows gizmo moves).
 */
export function sectionCenter(
  scene: Scene | WorkerScene,
  section: { plane: { origin: readonly [number, number, number] } } | null,
  item: Item | null,
): [number, number, number] | null {
  void section;
  if (item === null) {
    return null;
  }
  try {
    return worldCenter(scene, item.mesh);
  } catch {
    return null; // removed meanwhile
  }
}

/** What the section controls read: the anchor, the normal, its dominant axis, the slider's half range and the plane's offset from the anchor. */
export interface SectionReadout {
  center: [number, number, number] | null;
  normal: readonly [number, number, number];
  axis: Axis;
  range: number;
  offset: number;
}

/**
 * The section controls' readout. Without a section the normal is +Z; the
 * axis is the normal's largest component (ties go to x, then y); the
 * slider spans the selected mesh's radius each way.
 */
export function sectionReadout(
  scene: Scene | WorkerScene,
  section: PanelSection | null,
  item: Item | null,
): SectionReadout {
  const center = sectionCenter(scene, section, item);
  const normal = section?.plane.normal ?? ([0, 0, 1] as readonly [number, number, number]);
  const axis: Axis = Math.abs(normal[0]) >= Math.abs(normal[1]) && Math.abs(normal[0]) >= Math.abs(normal[2]) ? "x" : Math.abs(normal[1]) >= Math.abs(normal[2]) ? "y" : "z";
  const range = Math.max(item?.radius ?? 1, 1e-6);
  const offset =
    section === null || center === null
      ? 0
      : (section.plane.origin[0] - center[0]) * normal[0] +
        (section.plane.origin[1] - center[1]) * normal[1] +
        (section.plane.origin[2] - center[2]) * normal[2];
  return { center, normal, axis, range, offset };
}

/**
 * The store calls behind the section controls: add a +Z plane through the
 * anchor over every mesh, snap to an axis, flip the kept side, slide along
 * the normal, and remove. Each move updates the store and the app state.
 */
export function sectionActions(
  app: AppState,
  store: SceneStore,
  items: readonly Item[],
  section: PanelSection | null,
  readout: SectionReadout,
) {
  const { center, normal } = readout;
  const enable = () => {
    if (items.length === 0) {
      return;
    }
    const c = center ?? [0, 0, 0];
    const plane = { origin: [...c] as [number, number, number], normal: [0, 0, 1] as [number, number, number] };
    const handle = store.addSection({ plane, meshes: items.map((i) => i.mesh), fill: false });
    app.setInspectSection({ handle, plane });
  };
  const move = (plane: SectionPlaneInput) => {
    if (section === null) {
      return;
    }
    const handle = section.handle as SectionHandle;
    try {
      store.setSection(handle, { plane });
    } catch {
      // Removed meanwhile: the gizmo cleans up.
    }
    app.setInspectSection({ handle, plane });
  };
  const setAxis = (a: Axis) => {
    const n = AXIS_NORMAL[a];
    const c = center ?? [0, 0, 0];
    move({ origin: [...c] as [number, number, number], normal: [...n] });
  };
  const flip = () => {
    if (section === null) {
      return;
    }
    const n = section.plane.normal;
    move({
      origin: [...section.plane.origin] as [number, number, number],
      normal: [-n[0], -n[1], -n[2]] as [number, number, number],
    });
  };
  const setOffset = (t: number) => {
    const c = center;
    if (c === null) {
      return;
    }
    move({
      origin: [c[0] + normal[0] * t, c[1] + normal[1] * t, c[2] + normal[2] * t],
      normal: [...normal] as [number, number, number],
    });
  };
  const remove = () => {
    if (section === null) {
      return;
    }
    try {
      store.removeSection(section.handle as SectionHandle);
    } catch {
      // Removed meanwhile.
    }
    app.setInspectSection(null);
  };
  return { enable, setAxis, flip, setOffset, remove };
}

type ClipBox = { min: [number, number, number]; max: [number, number, number] };

/**
 * The selected mesh's middle half in world coordinates: around its world
 * center, half the world radius (scale-aware) each way.
 */
export function fitWorldBox(scene: Scene | WorkerScene, item: Item): ClipBox | null {
  try {
    const m = scene.matrix4(item.mesh);
    const [cx, cy, cz] = worldCenter(scene, item.mesh);
    // How far the matrix takes a unit ball along each world axis (its rows' lengths):
    // non-uniform scale (#103) stretches the box, and a turned one along other axes.
    const sx = Math.sqrt(m[0] * m[0] + m[4] * m[4] + m[8] * m[8]);
    const sy = Math.sqrt(m[1] * m[1] + m[5] * m[5] + m[9] * m[9]);
    const sz = Math.sqrt(m[2] * m[2] + m[6] * m[6] + m[10] * m[10]);
    const r = Math.max(item.radius, 1e-6) / 2;
    return {
      min: [cx - sx * r, cy - sy * r, cz - sz * r],
      max: [cx + sx * r, cy + sy * r, cz + sz * r],
    };
  } catch {
    return null; // removed meanwhile
  }
}

/** The clip sliders' bounds: the box grown by its own span (at least 1e-6) on every side. */
export function clipSliderBounds(clip: ClipBox): { lo: [number, number, number]; hi: [number, number, number] } {
  const span: [number, number, number] = [
    Math.max(clip.max[0] - clip.min[0], 1e-6),
    Math.max(clip.max[1] - clip.min[1], 1e-6),
    Math.max(clip.max[2] - clip.min[2], 1e-6),
  ];
  const lo: [number, number, number] = [clip.min[0] - span[0], clip.min[1] - span[1], clip.min[2] - span[2]];
  const hi: [number, number, number] = [clip.max[0] + span[0], clip.max[1] + span[1], clip.max[2] + span[2]];
  return { lo, hi };
}

/** The box with one end of one axis moved to `v`, clamped so the min never passes the max. */
export function clipBoxWithEnd(clip: ClipBox, axis: number, end: 0 | 1, v: number): ClipBox {
  const min: [number, number, number] = [...clip.min];
  const max: [number, number, number] = [...clip.max];
  if (end === 0) {
    min[axis] = Math.min(v, max[axis]);
  } else {
    max[axis] = Math.max(v, min[axis]);
  }
  return { min, max };
}
