// The app's own state, beside the SceneStore: the meshes it shows (name,
// material, size) and the loads in progress. A tiny external store, so the
// loader (outside React) and the components share it; read it with `useApp`.
import type { MeshHandle } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { useSyncExternalStore } from "react";
import * as THREE from "three";
import type { Vec3 } from "../load/mesh-data";
import type { LoadedMesh } from "../load/protocol";

export interface Item {
  /** A React key. */
  id: number;
  mesh: MeshHandle;
  name: string;
  /** Owned by the app (one per mesh, as `bindMesh` requires); disposed on removal. */
  material: THREE.MeshStandardMaterial;
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

export interface AppSnapshot {
  items: readonly Item[];
  jobs: readonly Job[];
}

const PALETTE = [0xd9735f, 0x6ea8d9, 0x8fc46e, 0xc9a0dc, 0xe0b45c, 0x5fc4b8, 0xe08bb0, 0xa3a3f0];

export class AppState {
  #snap: AppSnapshot = { items: [], jobs: [] };
  readonly #listeners = new Set<() => void>();
  #nextItem = 0;
  #nextColor = 0;

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
      triangles,
      center: loaded.center,
      radius: loaded.radius,
    };
    this.#set({ items: [...this.#snap.items, item] });
    return item;
  }

  /** Removes a mesh from the store and the list, and disposes its material. */
  removeMesh(store: SceneStore, item: Item): void {
    store.removeMesh(item.mesh);
    item.material.dispose();
    this.#set({ items: this.#snap.items.filter((i) => i !== item) });
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
