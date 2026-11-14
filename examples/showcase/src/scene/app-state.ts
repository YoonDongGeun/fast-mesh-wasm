// The app's own state, beside the SceneStore: the meshes it shows (name,
// material, size), the loads in progress, the collision pair and the
// contact group. A tiny
// external store, so the loader (outside React) and the components share it;
// read it with `useApp`.
import type { Analysis, MeshHandle, SectionHandle, SectionPlane } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { useSyncExternalStore } from "react";
import * as THREE from "three";
import type { Vec3 } from "../load/mesh-data";
import type { LoadedMesh } from "../load/protocol";
import type { SliceAxis } from "../panels/volume-slice";
import type { SculptBrush, SculptMask } from "../viewport/sculpt";
import {
  createGroup,
  dropMember,
  type NamedGroup,
  removeGroup,
  setColored,
  toggleCheck,
} from "../panels/groups";
import {
  clampSidebarWidth,
  DEFAULT_SIDEBAR_WIDTH,
  defaultSettings,
  type SculptFilterSettings,
  type SculptMethods,
  type SculptSymmetrySettings,
  type ShowcaseSettings,
  type ShowcaseTheme,
} from "../panels/settings";
import { type ImportScale, type LengthUnit, NO_IMPORT_SCALE } from "../panels/units";

export interface Item {
  /** A React key. */
  id: number;
  mesh: MeshHandle;
  name: string;
  /** Owned by the app (one per mesh, as `bindMesh` requires); disposed on removal. */
  material: THREE.MeshStandardMaterial;
  /** The mesh's colour (`0xrrggbb`), for its swatch: the material turns white while a colouring shows. */
  color: number;
  /** The triangle count when loaded (repairs change it; see `useTriangleCount`). */
  triangles: number;
  /** The mesh origin: its bounding box centre when loaded. */
  center: Vec3;
  /** Half the bounding box diagonal, for focusing the camera. */
  radius: number;
  /** The retained source volume's id (DICOM + phantoms), for the volume view. */
  volumeId: string | null;
  /** The suggested volume preset (a sample's descriptor hint), if any. */
  preset: string | null;
  /** How its numbers became millimetres on load: already applied, provenance only. */
  importScale: ImportScale;
}

export type JobPhase = "reading" | "parsing" | "adding" | "done" | "error";

/** One file (or the demo set) being loaded. */
export interface Job {
  id: number;
  name: string;
  phase: JobPhase;
  message?: string;
  /** Load progress, 0..1, while reading/parsing/adding (unset: indeterminate). */
  progress?: number;
  /** Post-load summary (triangles, time, decimation) shown on done jobs. */
  summary?: string;
}

/** The meshes the collision panel colours, each against the other; `null` for none picked. */
export interface CollisionPair {
  a: MeshHandle | null;
  b: MeshHandle | null;
}

/** The collision panel's mode: two picked meshes, or a contact group of several. */
export type CollisionMode = "pair" | "group";

/** The collision panel's contact group: the ticked meshes and which of them are measured to each other. */
export interface GroupSetup {
  members: readonly MeshHandle[];
  /** Every pair of members, or the listed ones (either way round). */
  pairs: readonly (readonly [MeshHandle, MeshHandle])[] | "all";
}

/** The collision heatmap's slider values, in mesh units (persisted: last-used). */
export interface HeatmapParams {
  depth: number;
  gap: number;
  tolerance: number;
}

/**
 * A brush-center pick in flight: armed for `mesh` while `point` is `null`,
 * landed once the viewport writes the local point. The repair panel consumes
 * it (into its center inputs) and clears it.
 */
export interface BrushPick {
  mesh: MeshHandle;
  point: Vec3 | null;
}

/**
 * The linked volume-view state: slice positions, window preset, scene
 * overlay, 3D preset, shown tab. The panel and the viewport overlay
 * steer the same state.
 */
export interface VolumeMpr {
  volumeId: string;
  slices: Record<SliceAxis, number>;
  window: number;
  inScene: boolean;
  preset: number;
  tab: "volume" | "slices";
}

/** One repair-issue kind drawn in 3D (see `viewport/repair-overlays`). */
export type RepairOverlayKind = "holes" | "nonManifold" | "degenerate" | "components";

export const REPAIR_OVERLAY_KINDS: readonly RepairOverlayKind[] = [
  "holes",
  "nonManifold",
  "degenerate",
  "components",
];

/** The repair panel's latest analysis, shared with the 3D overlay. */
export interface RepairAnalysis {
  mesh: MeshHandle;
  analysis: Analysis;
}

/** One inspection tool of the inspect panel: ruler, click-to-query, section plane or clip box. */
export type InspectTool = "ruler" | "query" | "section" | "clip";

/** The inspect section plane: the store's handle and its plane, for the panel and the gizmo. */
export interface InspectSection {
  handle: SectionHandle;
  plane: SectionPlane;
}

/** A click-to-query hit: the mesh, the snapped vertex slot, and its world point. */
export interface InspectQuery {
  mesh: MeshHandle;
  slot: number;
  point: Vec3;
}

export interface AppSnapshot {
  items: readonly Item[];
  jobs: readonly Job[];
  collision: CollisionPair;
  collisionMode: CollisionMode;
  group: GroupSetup;
  /** The meshes ticked in the list, in list order: the next named group's members. */
  checks: readonly MeshHandle[];
  /** The named mesh groups. */
  meshGroups: readonly NamedGroup<MeshHandle>[];
  /** The brush-center pick in flight, if any. */
  brushPick: BrushPick | null;
  /** The sculpt brush armed on a mesh, if any: hover shows its ball, press-and-hold paints. */
  sculpt: SculptBrush | null;
  /**
   * The painted protected region on a mesh, if any: deform dabs leave its
   * vertices in place (see `viewport/sculpt.ts`'s mask contract). `null` when never painted or cleared — loudly, with
   * `sculptMaskNotice`, when a topology edit takes a masked vertex.
   */
  sculptMask: SculptMask | null;
  /** Whether the mask's tint shows while a sculpt tool or the mask paint is armed. */
  sculptMaskShown: boolean;
  /** The mask paint is armed: press-and-hold paints the protected region. */
  maskPaint: boolean;
  /** The mask paint radius, in mesh units. */
  maskRadius: number;
  /** Why the mask was last cleared against its will (`null` when it stands or was cleared by hand). */
  sculptMaskNotice: string | null;
  /** A stroke's end also remeshes its whole region (in the same undo step on the sync scene). Off by default: it rewrites what the stroke did not touch. */
  sculptRemesh: boolean;
  /**
   * A stroke splits the triangles it stretches, so no triangle balloons mid-stroke.
   * Same undo step as the stroke (see `liveRemeshDue`'s semantics note in
   * `viewport/sculpt.ts`): the live passes repair inside the stroke's open
   * step and never open one of their own.
   */
  sculptLiveRemesh: boolean;
  /** Per-brush algorithm selections (#177, #192; persisted). */
  sculptMethods: SculptMethods;
  /** Kelvinlet grab Poisson ratio in `[0, 0.5]` (#177; persisted). */
  sculptGrabPoisson: number;
  /** Composable surface feature filter (#177, #192; persisted). */
  sculptFilter: SculptFilterSettings;
  /** Mirror symmetry axes across local coordinate planes (#192; persisted). */
  sculptSymmetry: SculptSymmetrySettings;
  /**
   * The width, in distance units, of the blend at the heatmap's colour
   * boundaries (fragment shading's `edgeSoftness`); 0 is a hard edge.
   */
  edgeSoftness: number;
  /** Draw every mesh wireframed (a display setting: not undoable). */
  wireframe: boolean;
  /**
   * The shift-clicked meshes in click order: the primary first (it doubles
   * as the store's single selection, so the gizmo and the panels follow
   * it), the Join sources after it.
   */
  multi: readonly MeshHandle[];
  /** The linked MPR state for the selected volume, if any panel shows it. */
  volumeMpr: VolumeMpr | null;
  /** Which repair-issue kinds draw in 3D (a display setting: not undoable). */
  repairOverlays: Record<RepairOverlayKind, boolean>;
  /** The repair panel's latest analysis (the overlay's geometry source). */
  repairAnalysis: RepairAnalysis | null;
  /**
   * The armed inspection tool, if any: ruler (click two vertices, world
   * points that survive placement undo), click-to-query (the picked
   * vertex's coloring value), section plane or clip box.
   */
  inspectTool: InspectTool | null;
  /** The ruler's endpoints in world coordinates (`b` is `null` until the second click). */
  ruler: { a: Vec3; b: Vec3 | null } | null;
  /** The click-to-query hit, if one landed. */
  inspectQuery: InspectQuery | null;
  /** The section plane cut, if one is armed (its handle lives in the store). */
  inspectSection: InspectSection | null;
  /** The axis-aligned clip box in world coordinates, if one is armed. */
  clipBox: { min: Vec3; max: Vec3 } | null;
  /** Heavy loads are decimated on load (#113, via the #118 simplify op). */
  decimateOnLoad: boolean;
  /** The app theme (a display setting: persisted, not undoable). */
  theme: ShowcaseTheme;
  /** Whether the sidebars show (a display setting: persisted, not undoable). */
  leftVisible: boolean;
  rightVisible: boolean;
  /** The sidebar widths, in pixels (a display setting: persisted, not undoable). */
  leftWidth: number;
  rightWidth: number;
  /**
   * The collision heatmap's slider values (`null` until a slider moves:
   * the defaults then follow the meshes' scale). Persisted: last-used.
   */
  heatmapParams: HeatmapParams | null;
  /** Fast heatmap mode (bounded fields; persisted: last-used, not undoable). */
  heatmapFast: boolean;
  /** The display unit: the project's (text only, every stored value stays in mm). */
  units: LengthUnit;
  /** The display unit new sessions and older project files start on (persisted). */
  defaultUnits: LengthUnit;
  /** What files loaded from here on are assumed to be in (not persisted: a surprise on the next load). */
  importScale: ImportScale;
}

const PALETTE = [0xd9735f, 0x6ea8d9, 0x8fc46e, 0xc9a0dc, 0xe0b45c, 0x5fc4b8, 0xe08bb0, 0xa3a3f0];

export class AppState {
  #snap: AppSnapshot = {
    items: [],
    jobs: [],
    collision: { a: null, b: null },
    collisionMode: "pair",
    group: { members: [], pairs: "all" },
    checks: [],
    meshGroups: [],
    brushPick: null,
    sculpt: null,
    sculptMask: null,
    sculptMaskShown: true,
    maskPaint: false,
    maskRadius: 0.25,
    sculptMaskNotice: null,
    sculptRemesh: false,
    sculptLiveRemesh: true,
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
    edgeSoftness: 0,
    wireframe: false,
    multi: [],
    volumeMpr: null,
    repairOverlays: { holes: true, nonManifold: true, degenerate: true, components: true },
    repairAnalysis: null,
    inspectTool: null,
    ruler: null,
    inspectQuery: null,
    inspectSection: null,
    clipBox: null,
    decimateOnLoad: true,
    theme: "dark",
    leftVisible: true,
    rightVisible: true,
    leftWidth: DEFAULT_SIDEBAR_WIDTH,
    rightWidth: DEFAULT_SIDEBAR_WIDTH,
    heatmapParams: null,
    heatmapFast: false,
    units: "mm",
    defaultUnits: "mm",
    importScale: NO_IMPORT_SCALE,
  };

  /** Starts with the stored display settings (`main.tsx` passes them; tests take the defaults). */
  constructor(initial: ShowcaseSettings = defaultSettings()) {
    this.#snap = {
      ...this.#snap,
      theme: initial.theme,
      leftVisible: initial.leftVisible,
      rightVisible: initial.rightVisible,
      leftWidth: initial.leftWidth,
      rightWidth: initial.rightWidth,
      wireframe: initial.wireframe,
      edgeSoftness: initial.edgeSoftness,
      repairOverlays: { ...initial.repairOverlays },
      collisionMode: initial.collisionMode,
      heatmapParams: initial.heatmap === null ? null : { ...initial.heatmap },
      heatmapFast: initial.heatmapFast,
      decimateOnLoad: initial.decimateOnLoad,
      sculptRemesh: initial.sculptEvenOut,
      sculptLiveRemesh: initial.sculptLiveRemesh,
      maskRadius: initial.maskRadius,
      sculptMethods: { ...initial.sculptMethods },
      sculptGrabPoisson: initial.sculptGrabPoisson,
      sculptFilter: { ...initial.sculptFilter },
      sculptSymmetry: { ...initial.sculptSymmetry },
      units: initial.defaultUnits,
      defaultUnits: initial.defaultUnits,
    };
  }
  /** The pair defaults to the first two meshes added, until it is picked by hand. */
  #autoPick = true;
  /** The group defaults to the first three meshes added (the demo set), until it is ticked by hand. */
  #autoGroup = true;
  readonly #listeners = new Set<() => void>();
  #nextItem = 0;
  #nextColor = 0;
  #nextGroup = 1;

  getSnapshot = (): AppSnapshot => this.#snap;

  subscribe = (listener: () => void): (() => void) => {
    this.#listeners.add(listener);
    return () => this.#listeners.delete(listener);
  };

  #set(patch: Partial<AppSnapshot>): void {
    this.#snap = { ...this.#snap, ...patch };
    for (const l of this.#listeners) {
      l();
    }
  }

  /** Adds a loaded mesh to the store (its arrays are transferred to a Worker scene) and lists it. */
  addMesh(store: SceneStore, loaded: LoadedMesh): Item {
    const triangles = loaded.indices.length / 3;
    const mesh = store.addMesh(loaded.positions, loaded.indices, { origin: loaded.center });
    const color = PALETTE[this.#nextColor++ % PALETTE.length];
    // Double-sided: open meshes (and their insides) stay visible.
    const material = new THREE.MeshStandardMaterial({ color, roughness: 0.55, side: THREE.DoubleSide });
    const item: Item = {
      id: this.#nextItem++,
      mesh,
      name: loaded.name,
      material,
      color,
      triangles,
      center: loaded.center,
      radius: loaded.radius,
      volumeId: loaded.volumeId,
      preset: loaded.preset,
      importScale: loaded.importScale ?? NO_IMPORT_SCALE,
    };
    let collision = this.#snap.collision;
    if (this.#autoPick) {
      if (collision.a === null) {
        collision = { ...collision, a: mesh };
      } else if (collision.b === null) {
        collision = { ...collision, b: mesh };
        this.#autoPick = false;
      }
    }
    let group = this.#snap.group;
    if (this.#autoGroup && group.members.length < 3) {
      group = { ...group, members: [...group.members, mesh] };
    }
    this.#set({ items: [...this.#snap.items, item], collision, group });
    return item;
  }

  /**
   * Removes a mesh from the store and the list, and disposes its material.
   * When it is one of the collision pair, its picker resets; in pair mode
   * the other mesh's colouring (measured to it) stops first, in the same
   * tick: the store would otherwise report the colouring's lost target as
   * an error. It also leaves the contact group (and its pairs); the store
   * takes a removed mesh out of its groups itself, and the other members
   * keep their remaining partners.
   */
  removeMesh(store: SceneStore, item: Item): void {
    const detached = this.#detach(store, item);
    store.removeMesh(item.mesh);
    item.material.dispose();
    this.#dropLists(item, detached);
  }

  /**
   * Drops a mesh the engine already removed (a join's source): the same list
   * cleanup as {@link removeMesh}, without the store call (which would report
   * the missing mesh as an error).
   */
  dropItem(store: SceneStore, item: Item): void {
    item.material.dispose();
    this.#dropLists(item, this.#detach(store, item));
  }

  /**
   * Unpicks a removed mesh from the collision pair (releasing the other
   * mesh's colouring first, in the same tick) and the contact group. The
   * store takes a removed mesh out of its groups itself.
   */
  #detach(store: SceneStore, item: Item): { collision: CollisionPair; group: GroupSetup } {
    let collision = this.#snap.collision;
    if (collision.a === item.mesh || collision.b === item.mesh) {
      const other = collision.a === item.mesh ? collision.b : collision.a;
      if (this.#snap.collisionMode === "pair" && other !== null && other !== item.mesh) {
        store.setDisplay(other, { coloring: null });
      }
      collision = { a: collision.a === item.mesh ? null : collision.a, b: collision.b === item.mesh ? null : collision.b };
    }
    const { members, pairs } = this.#snap.group;
    return {
      collision,
      group: {
        members: members.filter((m) => m !== item.mesh),
        pairs: pairs === "all" ? "all" : pairs.filter(([a, b]) => a !== item.mesh && b !== item.mesh),
      },
    };
  }

  #dropLists(item: Item, detached: { collision: CollisionPair; group: GroupSetup }): void {
    this.#set({
      items: this.#snap.items.filter((i) => i !== item),
      collision: detached.collision,
      group: detached.group,
      checks: this.#snap.checks.filter((m) => m !== item.mesh),
      multi: this.#snap.multi.filter((m) => m !== item.mesh),
      meshGroups: dropMember(this.#snap.meshGroups, item.mesh),
      sculpt: this.#snap.sculpt?.mesh === item.mesh ? null : this.#snap.sculpt,
      sculptMask: this.#snap.sculptMask?.mesh === item.mesh ? null : this.#snap.sculptMask,
      maskPaint: this.#snap.sculptMask?.mesh === item.mesh ? false : this.#snap.maskPaint,
      sculptMaskNotice: this.#snap.sculptMask?.mesh === item.mesh ? null : this.#snap.sculptMaskNotice,
      repairAnalysis: this.#snap.repairAnalysis?.mesh === item.mesh ? null : this.#snap.repairAnalysis,
      inspectQuery: this.#snap.inspectQuery?.mesh === item.mesh ? null : this.#snap.inspectQuery,
    });
  }

  /** Picks the collision pair's `side` (by hand: no more defaults). */
  pick(side: keyof CollisionPair, mesh: MeshHandle | null): void {
    this.#autoPick = false;
    this.#set({ collision: { ...this.#snap.collision, [side]: mesh } });
  }

  /** Switches the collision panel between the pair and the contact group. */
  setCollisionMode(mode: CollisionMode): void {
    this.#set({ collisionMode: mode });
  }

  /** Ticks (`on`) or unticks a mesh as a group member (by hand: no more defaults), kept in list order. */
  setGroupMember(mesh: MeshHandle, on: boolean): void {
    this.#autoGroup = false;
    const { members, pairs } = this.#snap.group;
    const next = on
      ? this.#snap.items.map((i) => i.mesh).filter((m) => m === mesh || members.includes(m))
      : members.filter((m) => m !== mesh);
    this.#set({ group: { members: next, pairs } });
  }

  /** Sets which group members are measured to each other: `"all"`, or the listed pairs. */
  setGroupPairs(pairs: GroupSetup["pairs"]): void {
    this.#set({ group: { ...this.#snap.group, pairs } });
  }

  /** Ticks (`on`) or unticks a mesh in the list, kept in list order. */
  toggleCheck(mesh: MeshHandle, on: boolean): void {
    const order = this.#snap.items.map((i) => i.mesh);
    this.#set({ checks: toggleCheck(this.#snap.checks, order, mesh, on) });
  }

  /** Groups the ticked meshes under `name` (blank for `Group <n>`). Needs at least one tick. */
  createGroup(name: string): void {
    const members = this.#snap.checks;
    if (members.length === 0) {
      return;
    }
    const id = this.#nextGroup++;
    this.#set({ meshGroups: createGroup(this.#snap.meshGroups, id, name, `Group ${id}`, members) });
  }

  /** Removes the named group `id` (its meshes stay). */
  removeGroup(id: number): void {
    this.#set({ meshGroups: removeGroup(this.#snap.meshGroups, id) });
  }

  /** Colours (`on`) or uncolours the named group `id` by its contacts. */
  setGroupColored(id: number, on: boolean): void {
    this.#set({ meshGroups: setColored(this.#snap.meshGroups, id, on) });
  }

  /** Arms, lands or clears (`null`) the brush-center pick. */
  setBrushPick(pick: BrushPick | null): void {
    this.#set({ brushPick: pick });
  }

  /** Arms the sculpt brush, updates it while armed, or disarms it (`null`). */
  setSculpt(sculpt: SculptBrush | null): void {
    this.#set({ sculpt });
  }

  /** Stores the painted mask (`null` clears it by hand, and clears its notice). */
  setSculptMask(mask: SculptMask | null): void {
    this.#set(mask === null ? { sculptMask: null, sculptMaskNotice: null } : { sculptMask: mask });
  }

  /** Shows (`true`) or hides the mask's tint; the mask itself stays. */
  setSculptMaskShown(on: boolean): void {
    this.#set({ sculptMaskShown: on });
  }

  /** Arms (`true`) or disarms the mask paint. */
  setMaskPaint(on: boolean): void {
    this.#set({ maskPaint: on });
  }

  /** Sets the mask paint radius, in mesh units. */
  setMaskRadius(radius: number): void {
    this.#set({ maskRadius: radius });
  }

  /**
   * Reports a mask cleared against its will (a topology edit took a masked
   * vertex): the loud half of the mask contract, shown by the toolbox until
   * the next paint or hand clear.
   */
  setSculptMaskNotice(notice: string | null): void {
    this.#set({ sculptMaskNotice: notice });
  }

  /** Sets the shift-clicked meshes, primary first (see `selectMesh`). */
  setMulti(multi: readonly MeshHandle[]): void {
    this.#set({ multi });
  }

  /** Sets or clears (`null`) the linked MPR state. */
  setVolumeMpr(mpr: VolumeMpr | null): void {
    this.#set({ volumeMpr: mpr });
  }

  /** Whether a stroke's end remeshes its region. */
  setSculptRemesh(on: boolean): void {
    this.#set({ sculptRemesh: on });
  }

  /** Whether a stroke is remeshed live while it paints. */
  setSculptLiveRemesh(on: boolean): void {
    this.#set({ sculptLiveRemesh: on });
  }

  /** Updates per-brush sculpt algorithm selections (persisted). */
  setSculptMethods(patch: Partial<SculptMethods>): void {
    this.#set({ sculptMethods: { ...this.#snap.sculptMethods, ...patch } });
  }

  /** Updates the Kelvinlet grab Poisson ratio in `[0, 0.5]` (persisted). */
  setSculptGrabPoisson(poisson: number): void {
    if (!Number.isFinite(poisson)) {
      return;
    }
    this.#set({ sculptGrabPoisson: Math.min(0.5, Math.max(0, poisson)) });
  }

  /** Updates composable surface filter settings clamped to `settings.ts` bounds (persisted). */
  setSculptFilter(patch: Partial<SculptFilterSettings>): void {
    const cur = this.#snap.sculptFilter;
    const kind =
      patch.kind === "none" || patch.kind === "peaksOnly" || patch.kind === "valleysOnly" || patch.kind === "normalAngle"
        ? patch.kind
        : cur.kind;
    const threshold =
      patch.threshold !== undefined && Number.isFinite(patch.threshold)
        ? Math.min(1, Math.max(0, patch.threshold))
        : cur.threshold;
    const contrast =
      patch.contrast !== undefined && Number.isFinite(patch.contrast)
        ? Math.min(1, Math.max(0, patch.contrast))
        : cur.contrast;
    const maxAngleDeg =
      patch.maxAngleDeg !== undefined && Number.isFinite(patch.maxAngleDeg) && patch.maxAngleDeg > 0
        ? Math.min(180, patch.maxAngleDeg)
        : cur.maxAngleDeg;
    this.#set({ sculptFilter: { kind, threshold, contrast, maxAngleDeg } });
  }

  /** Updates mirror symmetry axis toggles (persisted). */
  setSculptSymmetry(patch: Partial<SculptSymmetrySettings>): void {
    this.#set({ sculptSymmetry: { ...this.#snap.sculptSymmetry, ...patch } });
  }

  /** Sets the heatmap's boundary softness (a display setting: not undoable). */
  setEdgeSoftness(width: number): void {
    this.#set({ edgeSoftness: width });
  }

  /** Draws every mesh wireframed or solid (a display setting: not undoable). */
  setWireframe(on: boolean): void {
    this.#set({ wireframe: on });
  }

  /** Draws (`on`) or hides a repair-issue kind in 3D (a display setting: not undoable). */
  setRepairOverlay(kind: RepairOverlayKind, on: boolean): void {
    if (this.#snap.repairOverlays[kind] === on) {
      return;
    }
    this.#set({ repairOverlays: { ...this.#snap.repairOverlays, [kind]: on } });
  }

  /** Shares the repair panel's latest analysis with the 3D overlay (`null` clears it). */
  setRepairAnalysis(analysis: RepairAnalysis | null): void {
    this.#set({ repairAnalysis: analysis });
  }

  /** Arms the inspection tool (`null` disarms it, keeping its results). */
  setInspectTool(tool: InspectTool | null): void {
    this.#set({ inspectTool: tool });
  }

  /** Sets the ruler's endpoints in world coordinates (`null` clears it). */
  setRuler(ruler: { a: Vec3; b: Vec3 | null } | null): void {
    this.#set({ ruler });
  }

  /** Records a click-to-query hit (`null` clears it). */
  setInspectQuery(query: InspectQuery | null): void {
    this.#set({ inspectQuery: query });
  }

  /** Shares the inspect section plane with the 3D gizmo (`null` clears it). */
  setInspectSection(section: InspectSection | null): void {
    this.#set({ inspectSection: section });
  }

  /** Sets the axis-aligned clip box in world coordinates (`null` clears it). */
  setClipBox(box: { min: Vec3; max: Vec3 } | null): void {
    this.#set({ clipBox: box });
  }

  /** Whether heavy loads are decimated on load (the mesh list's checkbox). */
  setDecimateOnLoad(on: boolean): void {
    this.#set({ decimateOnLoad: on });
  }

  /** Sets the app theme (a display setting: persisted, not undoable). */
  setTheme(theme: ShowcaseTheme): void {
    this.#set({ theme });
  }

  /** Shows or hides the left sidebar (a display setting: persisted, not undoable). */
  setLeftVisible(visible: boolean): void {
    this.#set({ leftVisible: visible });
  }

  /** Shows or hides the right sidebar (a display setting: persisted, not undoable). */
  setRightVisible(visible: boolean): void {
    this.#set({ rightVisible: visible });
  }

  /** Sets the left sidebar's width in pixels, clamped (a display setting: persisted, not undoable). */
  setLeftWidth(width: number): void {
    this.#set({ leftWidth: clampSidebarWidth(width) });
  }

  /** Sets the right sidebar's width in pixels, clamped (a display setting: persisted, not undoable). */
  setRightWidth(width: number): void {
    this.#set({ rightWidth: clampSidebarWidth(width) });
  }

  /** Sets the heatmap sliders (`null`: follow the meshes' scale). Persisted: last-used. */
  setHeatmapParams(params: HeatmapParams | null): void {
    this.#set({ heatmapParams: params });
  }

  /** Sets fast heatmap mode (bounded fields). Persisted: last-used, not undoable. */
  setHeatmapFast(fast: boolean): void {
    this.#set({ heatmapFast: fast });
  }

  /** The display unit picked by hand: this project's, and the default from here on. Text only. */
  setUnits(units: LengthUnit): void {
    this.#set({ units, defaultUnits: units });
  }

  /** The display unit of a project just opened; `undefined` (an older file) takes the default, which it never changes. */
  setProjectUnits(units: LengthUnit | undefined): void {
    this.#set({ units: units ?? this.#snap.defaultUnits });
  }

  /** Sets what files loaded from here on are assumed to be in. Meshes already loaded stay as they are. */
  setImportScale(importScale: ImportScale): void {
    this.#set({ importScale });
  }

  /**
   * Restores every persisted display setting to its default (the status
   * bar's "reset to defaults" button). Meshes, selection and history stay.
   */
  resetDisplaySettings(): void {
    const d = defaultSettings();
    this.#set({
      theme: d.theme,
      leftVisible: d.leftVisible,
      rightVisible: d.rightVisible,
      leftWidth: d.leftWidth,
      rightWidth: d.rightWidth,
      wireframe: d.wireframe,
      edgeSoftness: d.edgeSoftness,
      repairOverlays: { ...d.repairOverlays },
      collisionMode: d.collisionMode,
      heatmapParams: d.heatmap,
      heatmapFast: d.heatmapFast,
      decimateOnLoad: d.decimateOnLoad,
      sculptRemesh: d.sculptEvenOut,
      sculptLiveRemesh: d.sculptLiveRemesh,
      maskRadius: d.maskRadius,
      sculptMethods: { ...d.sculptMethods },
      sculptGrabPoisson: d.sculptGrabPoisson,
      sculptFilter: { ...d.sculptFilter },
      sculptSymmetry: { ...d.sculptSymmetry },
      units: d.defaultUnits,
      defaultUnits: d.defaultUnits,
      importScale: NO_IMPORT_SCALE,
    });
  }

  setJob(job: Job): void {
    const jobs = this.#snap.jobs.some((j) => j.id === job.id)
      ? this.#snap.jobs.map((j) => (j.id === job.id ? job : j))
      : [...this.#snap.jobs, job];
    this.#set({ jobs });
  }

  dropJob(id: number): void {
    this.#set({ jobs: this.#snap.jobs.filter((j) => j.id !== id) });
  }
}

/** A slice of the app state; the component re-renders when it changes (`Object.is`). */
export function useApp<T>(app: AppState, select: (s: AppSnapshot) => T): T {
  const getSlice = () => select(app.getSnapshot());
  return useSyncExternalStore(app.subscribe, getSlice, getSlice);
}
