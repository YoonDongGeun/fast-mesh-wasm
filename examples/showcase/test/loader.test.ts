// The loader's drop-batch sibling rule: `.bin` files dropped with a
// `.gltf`/`.glb` are its external buffers (attached to its job); a lone
// `.bin` stays a regular file (its job reports it as unreadable, as before).
import type { MeshHandle } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BIG_DECIMATE_TARGET, BIG_DECIMATE_TRIANGLES } from "../src/load/big-file";
import { Loader, siblingBinFiles } from "../src/load/loader";
import type { LoaderReply, LoaderRequest } from "../src/load/protocol";
import { importScaleOf } from "../src/panels/units";
import { AppState } from "../src/scene/app-state";

function named(name: string): File {
  return { name } as File;
}

describe("siblingBinFiles", () => {
  it("attaches .bin files to a dropped .gltf", () => {
    const files = [named("tri.gltf"), named("tri.bin"), named("extra.bin")];
    expect(siblingBinFiles(files).map((f) => f.name)).toEqual(["tri.bin", "extra.bin"]);
  });

  it("attaches .bin files to a dropped .glb, case-insensitively", () => {
    const files = [named("scene.GLB"), named("scene.BIN")];
    expect(siblingBinFiles(files).map((f) => f.name)).toEqual(["scene.BIN"]);
  });

  it("leaves a .bin without a .gltf or .glb alone", () => {
    expect(siblingBinFiles([named("tri.bin"), named("part.stl")])).toEqual([]);
    expect(siblingBinFiles([named("part.stl")])).toEqual([]);
  });
});

describe("the import scale's reach (issue #145)", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  /** The requests a Loader posts, with what files are assumed to be in set to inches. */
  function inches(): { loader: Loader; posted: LoaderRequest[] } {
    const posted: LoaderRequest[] = [];
    vi.stubGlobal(
      "Worker",
      class {
        postMessage(request: LoaderRequest) {
          posted.push(request);
        }
      },
    );
    const app = new AppState();
    app.setImportScale(importScaleOf("in"));
    return { loader: new Loader({} as SceneStore, app), posted };
  }

  it("rides on mesh files", async () => {
    const { loader, posted } = inches();
    void loader.loadFiles([named("part.stl"), named("tri.gltf"), named("tri.bin")]);
    await vi.waitFor(() => expect(posted.length).toBe(2));
    for (const request of posted) {
      expect(request).toMatchObject({ kind: "file", importScale: { factor: 25.4, units: "in" } });
    }
  });

  it("never reaches a DICOM series, a phantom, a sample or the demo: those are already in mm", async () => {
    const { loader, posted } = inches();
    void loader.loadFiles([named("slice1.dcm"), named("slice2.dcm")]);
    void loader.loadDicomDemo("vessels");
    void loader.loadSample("headsq");
    void loader.loadDemo(false);
    await vi.waitFor(() => expect(posted.length).toBe(4));
    expect(posted.map((r) => r.kind).sort()).toEqual(["demo", "dicom", "phantom", "sample"]);
    for (const request of posted) {
      expect(request).not.toHaveProperty("importScale");
    }
  });
});

describe("worker-side decimate-on-load (#198)", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function setupHarness(decimateOnLoad = true) {
    const posted: LoaderRequest[] = [];
    let workerInstance: {
      onmessage: ((e: MessageEvent<LoaderReply>) => void) | null;
      postMessage: (req: LoaderRequest) => void;
    } | null = null;
    vi.stubGlobal(
      "Worker",
      class {
        onmessage: ((e: MessageEvent<LoaderReply>) => void) | null = null;
        onerror: ((e: unknown) => void) | null = null;
        onmessageerror: ((e: unknown) => void) | null = null;
        constructor() {
          workerInstance = this;
        }
        postMessage(request: LoaderRequest) {
          posted.push(request);
        }
        terminate() {}
      },
    );

    const changeListeners = new Set<(changes: { mesh: MeshHandle; added?: boolean; removed?: boolean }[]) => void>();
    const meshHandle = { id: 42 } as unknown as MeshHandle;
    const repair = vi.fn();
    const store = {
      addMesh: vi.fn(() => meshHandle),
      removeMesh: vi.fn(),
      repair,
      getSnapshot: () => ({ status: "idle", meshes: new Map([[meshHandle, { status: "ready" }]]) }),
      subscribe: () => () => {},
      scene: {
        onChanges: (cb: (changes: { mesh: MeshHandle; added?: boolean; removed?: boolean }[]) => void) => {
          changeListeners.add(cb);
          return () => changeListeners.delete(cb);
        },
      },
    } as unknown as SceneStore;

    const app = new AppState();
    app.setDecimateOnLoad(decimateOnLoad);
    const loader = new Loader(store, app);

    const emit = (reply: LoaderReply) => {
      workerInstance?.onmessage?.({ data: reply } as MessageEvent<LoaderReply>);
    };
    const confirmAdded = () => {
      for (const cb of changeListeners) {
        cb([{ mesh: meshHandle, added: true }]);
      }
    };

    return { app, loader, store, repair, posted, emit, confirmAdded };
  }

  it("attaches decimate config only when decimateOnLoad is enabled", async () => {
    const enabled = setupHarness(true);
    void enabled.loader.loadFiles([named("huge.stl")]);
    await vi.waitFor(() => expect(enabled.posted.length).toBe(1));
    expect(enabled.posted[0]).toMatchObject({
      kind: "file",
      decimate: { threshold: BIG_DECIMATE_TRIANGLES, target: BIG_DECIMATE_TARGET },
    });

    const disabled = setupHarness(false);
    void disabled.loader.loadFiles([named("huge.stl")]);
    await vi.waitFor(() => expect(disabled.posted.length).toBe(1));
    expect(disabled.posted[0]).not.toHaveProperty("decimate");
  });

  it("tracks worker decimating progress and finishes without main-thread repair", async () => {
    const { app, loader, store, repair, posted, emit, confirmAdded } = setupHarness(true);
    const pending = loader.loadFiles([named("scan.stl")]);
    await vi.waitFor(() => expect(posted.length).toBe(1));
    const reqId = (posted[0] as { id: number }).id;

    emit({ kind: "progress", id: reqId, phase: "decimating", fraction: 0.84 });
    expect(app.getSnapshot().jobs[0]).toMatchObject({ id: reqId, phase: "decimating", progress: 0.84 });

    emit({
      kind: "meshes",
      id: reqId,
      meshes: [
        {
          name: "scan",
          positions: new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          indices: new Uint32Array([0, 1, 2]),
          center: [0.5, 0.5, 0],
          radius: 1,
          volumeId: null,
          preset: null,
          decimated: { from: 1_000_000, to: 100_000 },
        },
      ],
    });
    await vi.waitFor(() => expect((store.addMesh as ReturnType<typeof vi.fn>).mock.calls.length).toBe(1));
    confirmAdded();
    const items = await pending;
    expect(items).toHaveLength(1);
    expect(repair).not.toHaveBeenCalled();
    expect(app.getSnapshot().jobs[0]).toMatchObject({
      id: reqId,
      phase: "done",
      progress: 1,
    });
    expect(app.getSnapshot().jobs[0]?.summary).toContain("decimated 1.00 M → 100,000");
  });

  it("cancels cleanly mid-decimating without adding a mesh", async () => {
    const { app, loader, store, posted, emit } = setupHarness(true);
    const pending = loader.loadFiles([named("huge.stl")]);
    await vi.waitFor(() => expect(posted.length).toBe(1));
    const reqId = (posted[0] as { id: number }).id;

    emit({ kind: "progress", id: reqId, phase: "decimating", fraction: 0.78 });
    expect(app.getSnapshot().jobs[0]?.phase).toBe("decimating");

    loader.cancelJob(reqId);
    const items = await pending;
    expect(items).toEqual([]);
    expect(store.addMesh).not.toHaveBeenCalled();
    expect(posted.some((r) => r.kind === "cancel" && r.id === reqId)).toBe(true);
    expect(app.getSnapshot().jobs[0]).toMatchObject({ id: reqId, phase: "error", message: "cancelled" });
  });
});
