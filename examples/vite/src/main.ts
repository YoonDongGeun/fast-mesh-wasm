import { computeBounds, computeVertexNormals, init } from "fast-mesh-wasm";

await init();

// Unit quad on the XY plane.
const positions = new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]);
const indices = new Uint32Array([0, 1, 2, 0, 2, 3]);

const normals = computeVertexNormals(positions, indices);
using bounds = computeBounds(positions);

document.querySelector("#output")!.textContent = [
  `normals: ${Array.from(normals).join(", ")}`,
  `bounds.min: ${Array.from(bounds.min).join(", ")}`,
  `bounds.max: ${Array.from(bounds.max).join(", ")}`,
].join("\n");
