// The loader's drop-batch sibling rule: `.bin` files dropped with a
// `.gltf`/`.glb` are its external buffers (attached to its job); a lone
// `.bin` stays a regular file (its job reports it as unreadable, as before).
import type { SceneStore } from "fast-mesh-wasm/store";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Loader, siblingBinFiles } from "../src/load/loader";
import type { LoaderRequest } from "../src/load/protocol";
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
