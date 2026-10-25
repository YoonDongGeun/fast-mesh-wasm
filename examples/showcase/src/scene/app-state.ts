// The app's own state, beside the SceneStore: the meshes it shows (name,
// material, size), the loads in progress, the collision pair and the
// contact group. A tiny
// external store, so the loader (outside React) and the components share it;
// read it with `useApp`.
import type { MeshHandle } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { useSyncExternalStore } from "react";
import * as THREE from "three";
import type { Vec3 } from "../load/mesh-data";
import type { LoadedMesh } from "../load/protocol";
import {
  createGroup,
  dropMember,
  type NamedGroup,
  removeGroup,
  setColored,
  toggleCheck,
} from "../panels/groups";

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
}

export type JobPhase = "reading" | "parsing" | "adding" | "done" | "error";

/** One file (or the demo set) being loaded. */
export interface Job {
  id: number;
  name: string;
  phase: JobPhase;
  message?: string;
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

/**
 * A brush-center pick in flight: armed for `mesh` while `point` is `null`,
 * landed once the viewport writes the local point. The repair panel consumes
 * it (into its center inputs) and clears it.
 */
export interface BrushPick {
  mesh: MeshHandle;
  point: Vec3 | null;
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
  /**
   * The width, in distance units, of the blend at the heatmap's colour
   * boundaries (fragment shading's `edgeSoftness`); 0 is a hard edge.
   */
  edgeSoftness: number;
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
    edgeSoftness: 0,
  };
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
    let collision = this.#snap.collision;
    if (collision.a === item.mesh || collision.b === item.mesh) {
      const other = collision.a === item.mesh ? collision.b : collision.a;
      if (this.#snap.collisionMode === "pair" && other !== null && other !== item.mesh) {
        store.setDisplay(other, { coloring: null });
      }
      collision = { a: collision.a === item.mesh ? null : collision.a, b: collision.b === item.mesh ? null : collision.b };
    }
    const { members, pairs } = this.#snap.group;
    const group: GroupSetup = {
      members: members.filter((m) => m !== item.mesh),
      pairs: pairs === "all" ? "all" : pairs.filter(([a, b]) => a !== item.mesh && b !== item.mesh),
    };
    store.removeMesh(item.mesh);
    item.material.dispose();
    this.#set({
      items: this.#snap.items.filter((i) => i !== item),
      collision,
      group,
      checks: this.#snap.checks.filter((m) => m !== item.mesh),
      meshGroups: dropMember(this.#snap.meshGroups, item.mesh),
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

  /** Sets the heatmap's boundary softness (a display setting: not undoable). */
  setEdgeSoftness(width: number): void {
    this.#set({ edgeSoftness: width });
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
  return useSyncExternalStore(app.subscribe, () => select(app.getSnapshot()));
}
