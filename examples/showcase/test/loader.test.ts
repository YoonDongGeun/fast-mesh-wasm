// The loader's drop-batch sibling rule: `.bin` files dropped with a
// `.gltf`/`.glb` are its external buffers (attached to its job); a lone
// `.bin` stays a regular file (its job reports it as unreadable, as before).
import { describe, expect, it } from "vitest";
import { siblingBinFiles } from "../src/load/loader";

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
