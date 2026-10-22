// The page side of the loader Worker: one job per file (or for the demo set),
// tracked in the app state as reading -> parsing -> adding -> done (or error).
// The page only adds the parsed arrays to the store.
import type { SceneStore } from "fast-mesh-wasm/store";
import type { AppState, Item } from "../scene/app-state";
import type { LoaderReply, LoaderRequest } from "./protocol";

/** How long a finished job stays listed. */
const DONE_LINGER_MS = 2500;

export class Loader {
  readonly #worker: Worker;
  readonly #store: SceneStore;
  readonly #app: AppState;
  readonly #waiting = new Map<number, { name: string; resolve: (items: Item[]) => void }>();
  #nextJob = 0;

  constructor(store: SceneStore, app: AppState) {
    this.#store = store;
    this.#app = app;
    this.#worker = new Worker(new URL("./parser.worker.ts", import.meta.url), { type: "module" });
    this.#worker.onmessage = (e: MessageEvent<LoaderReply>) => this.#receive(e.data);
    this.#worker.onerror = (e) => {
      // The Worker itself failed (its script did not load): fail every job.
      for (const [id, job] of this.#waiting) {
        this.#app.setJob({ id, name: job.name, phase: "error", message: e.message || "the loader Worker failed" });
        job.resolve([]);
      }
      this.#waiting.clear();
    };
  }

  /** Loads files; resolves with the meshes added (none for a file that failed: its job shows why). */
  async loadFiles(files: Iterable<File>): Promise<Item[]> {
    const all = await Promise.all([...files].map((file) => this.#start(file.name, { kind: "file", id: 0, file })));
    return all.flat();
  }

  /** Generates and adds the demo meshes A, B and C. */
  loadDemo(big: boolean): Promise<Item[]> {
    return this.#start(big ? "demo meshes (big)" : "demo meshes", { kind: "demo", id: 0, big });
  }

  #start(name: string, request: LoaderRequest): Promise<Item[]> {
    const id = this.#nextJob++;
    this.#app.setJob({ id, name, phase: "reading" });
    return new Promise((resolve) => {
      this.#waiting.set(id, { name, resolve });
      this.#worker.postMessage({ ...request, id });
    });
  }

  #receive(reply: LoaderReply): void {
    const job = this.#waiting.get(reply.id);
    if (job === undefined) {
      return;
    }
    const { name } = job;
    if (reply.kind === "parsing") {
      this.#app.setJob({ id: reply.id, name, phase: "parsing" });
      return;
    }
    this.#waiting.delete(reply.id);
    if (reply.kind === "error") {
      this.#app.setJob({ id: reply.id, name, phase: "error", message: reply.message });
      job.resolve([]);
      return;
    }
    this.#app.setJob({ id: reply.id, name, phase: "adding" });
    let items: Item[] = [];
    try {
      items = reply.meshes.map((m) => this.#app.addMesh(this.#store, m));
    } catch (error) {
      this.#app.setJob({ id: reply.id, name, phase: "error", message: String(error) });
      job.resolve([]);
      return;
    }
    // Adding until the scene reports every mesh added (its mirror arrived from
    // a Worker scene; the next frame on a local one).
    const pending = new Set(items.map((i) => i.mesh));
    const done = () => {
      off();
      this.#app.setJob({ id: reply.id, name, phase: "done" });
      setTimeout(() => this.#app.dropJob(reply.id), DONE_LINGER_MS);
      job.resolve(items);
    };
    const off = this.#store.scene.onChanges((changes) => {
      for (const c of changes) {
        if (c.added || c.removed) {
          pending.delete(c.mesh);
        }
      }
      if (pending.size === 0) {
        done();
      }
    });
    if (pending.size === 0) {
      done();
    }
  }
}
