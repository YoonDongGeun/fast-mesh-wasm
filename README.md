# fast-mesh-wasm

[![npm](https://img.shields.io/npm/v/fast-mesh-wasm)](https://www.npmjs.com/package/fast-mesh-wasm)
[![CI](https://github.com/YoonDongGeun/fast-mesh-wasm/actions/workflows/ci.yml/badge.svg)](https://github.com/YoonDongGeun/fast-mesh-wasm/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)

Fast triangle mesh utilities for JavaScript and TypeScript, written in Rust and compiled to WebAssembly.

- Works in browsers (Vite, webpack, Next.js), Node.js and Bun
- Installs the same way with npm, pnpm and bun. No native build step, no install scripts, zero dependencies
- Full TypeScript types and inline docs, generated from the Rust source
- A `Scene` for interactive work: placements, queries, distance-field colouring, sections, repair and undo/redo
- Uses the flat buffer layout of WebGL and three.js `BufferGeometry`, so data goes in and out without conversion

> **Status:** early development (0.x). The API may change between minor versions.

## Install

```sh
npm install fast-mesh-wasm
pnpm add fast-mesh-wasm
bun add fast-mesh-wasm
```

## Usage

```ts
import { init, computeVertexNormals, computeBounds } from "fast-mesh-wasm";

await init(); // loads the .wasm file; required in browsers, a no-op in Node.js and Bun

// A unit quad: 4 vertices, 2 triangles
const positions = new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]);
const indices = new Uint32Array([0, 1, 2, 0, 2, 3]);

const normals = computeVertexNormals(positions, indices);
// Float32Array [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]

const bounds = computeBounds(positions);
// { min: [0, 0, 0], max: [1, 1, 0], center: [0.5, 0.5, 0], size: [1, 1, 0] }
```

Inputs can be typed arrays or plain arrays: `Float32Array` positions and `Uint32Array` indices are used without copying, while `Uint16Array`, `Float64Array` and `number[]` are converted once.

### With three.js

```ts
const geometry = new THREE.BufferGeometry();
geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
geometry.setIndex(new THREE.BufferAttribute(indices, 1));

const normals = computeVertexNormals(geometry.attributes.position.array, geometry.index!.array);
geometry.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
```

### three.js adapter

`fast-mesh-wasm/three` binds `Scene` meshes to `THREE.Mesh` objects that follow the scene on every `scene.commitFrame()`. It needs `three` (an optional peer dependency, `>=0.160`) and, for TypeScript, `@types/three`:

```sh
npm install three
npm install -D @types/three
```

```ts
import * as THREE from "three";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { Scene } from "fast-mesh-wasm";
import { attachTransformControls, bindMesh, bindSection } from "fast-mesh-wasm/three";

const scene = new Scene();
const a = scene.addMesh(positionsA, indicesA);
const b = scene.addMesh(positionsB, indicesB);

const threeScene = new THREE.Scene();
const renderer = new THREE.WebGLRenderer();
renderer.localClippingEnabled = true; // needed for bindSection's clipping plane

const boundA = bindMesh(scene, a); // a THREE.Mesh in boundA.object
const boundB = bindMesh(scene, b, { color: 0x88aaff, opacity: 0.6 });
threeScene.add(boundA.object, boundB.object);

// Drag B with a TransformControls gizmo: one gesture is one scene drag and one undo step.
const controls = new TransformControls(camera, renderer.domElement);
controls.attach(boundB.object);
threeScene.add(controls.getHelper()); // three >= 0.169; on older three use threeScene.add(controls)
attachTransformControls(controls, scene, boundB);

// A section: contours and fill, and the bound meshes are clipped at its plane.
const section = scene.addSection({ meshes: [a, b], plane: { origin: [0, 0, 0], normal: [0, 0, 1] } });
const boundSection = bindSection(scene, section, [boundA, boundB]);
threeScene.add(boundSection.group);

function frame() {
  scene.commitFrame(); // once per animation frame, before rendering: the adapters apply the changes here
  renderer.render(threeScene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
```

- Colouring (`scene.setColoring`) shows up as a vertex-colour attribute with no code in the app. Undo and redo move the objects through the same change stream.
- The engine's colour maps are sRGB-encoded bytes. The adapter converts them to three's linear working space as it copies them (a normalized `Uint16` colour attribute), so they look the same as in a renderer that passes the bytes through to a non-sRGB canvas.
- A material you pass to `bindMesh` is used as given and must not be shared between bound meshes (the adapter controls its `color`, `vertexColors`, `opacity`, `transparent` and `depthWrite`).
- The adapter only reads `scene.onChanges`; it never calls `takeChanges`.

A renderer-neutral version of the same loop, with raw WebGPU and no 3D library, is in [`examples/webgpu`](examples/webgpu).

## Scene API

For interactive work with several meshes, the `Scene` class keeps the meshes in WebAssembly memory and does the heavy parts in Rust: placements, a BVH per mesh, queries, signed distance fields with colour maps, planar sections, hole repair and undo/redo. Your renderer reads the results through zero-copy typed-array views, and you upload only the ranges that changed.

```ts
import { init, type MeshHandle, Scene, type Quat, type Vec3 } from "fast-mesh-wasm";

await init();
const scene = new Scene();

// Provided by your app (not part of this package): mesh data and the
// renderer's upload functions.
declare const upperPositions: Float64Array, upperIndices: Uint32Array;
declare const lowerPositions: Float64Array, lowerIndices: Uint32Array;
declare function uploadMatrix(mesh: MeshHandle, matrix: Float64Array): void;
declare function uploadColors(mesh: MeshHandle, startVertex: number, rgba: Uint8Array): void;

// Add meshes. Positions are in world space; the scene stores them relative to
// the mesh's bounding-box centre so single precision stays accurate.
const upper = scene.addMesh(upperPositions, upperIndices);
const lower = scene.addMesh(lowerPositions, lowerIndices);

// Colour the upper mesh by its signed distance to the lower one.
const gap = scene.addDistanceField({ name: "gap", on: upper, to: lower, maxDistance: 2 });
const colour = (field: typeof gap) =>
  scene.setColoring(upper, { field, map: "turbo", range: [-0.5, 1] });
colour(gap);

// Drag the lower mesh. The whole drag is one undo step. Each `move`
// recomputes the eager fields, so coalesce pointer events to one `move` per
// animation frame.
const drag = scene.drag(lower);
let pending: { rotation: Quat; translation: Vec3 } | undefined;
function onPointerMove(rotation: Quat, translation: Vec3) {
  pending = { rotation, translation }; // keep only the latest pose
}
function onPointerUp() {
  if (pending !== undefined) drag.move(pending);
  pending = undefined;
  drag.end();
}

// Once per animation frame: take what changed and upload only that.
function frame() {
  if (pending !== undefined) {
    drag.move(pending); // fields and colours follow
    pending = undefined;
  }
  for (const change of scene.commitFrame()) {
    const views = scene.views(change.mesh); // fresh views; do not keep them
    if (change.placementChanged) {
      uploadMatrix(change.mesh, scene.matrix4(change.mesh)); // Float64Array(16)
    }
    const colors = views.colors();
    if (colors !== null && change.colors !== "all") {
      for (const [start, count] of change.colors) {
        uploadColors(change.mesh, start, colors.subarray(4 * start, 4 * (start + count)));
      }
    } // "all" (a new mesh, or buffers that grew): re-upload the whole buffer
    // positions, normals and indices work the same way
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// Toggle the colouring (display settings are not part of the undo history).
scene.setColoring(upper, null);
colour(gap);

// Undo and redo placements and edits.
if (scene.canUndo()) scene.undo();
scene.redo();

// Queries are in world space.
const hit = scene.raycast([0, 0, 50], [0, 0, -1]); // nearest hit over all meshes, or null
const closest = scene.meshDistance(upper, lower); // { distance, a, b, ... }

scene.free(); // or `using scene = new Scene()`
```

Errors are `FastMeshError`s with a `code` (`UNKNOWN_MESH`, `INVALID_ARGUMENT`, `STALE_BVH`, ...). After a trap inside the WebAssembly module every later call throws `INSTANCE_POISONED`. A view is only valid until the next scene call that may allocate or move buffers: a kept view may read other data or be empty, so re-acquire views after any scene call. `new Scene({ debug: true })` makes views warn when they are read after such a call. Mesh edits (`scene.edit`), `analyze`, `repair`, sections (`addSection`, `section`, `sectionGap`) and `findContacts` are documented in the types.

## Functions

| Function | Returns | Description |
| --- | --- | --- |
| `init(input?)` | `Promise<void>` | Loads the WebAssembly module. Required once in browsers; resolves immediately in Node.js and Bun. Safe to call repeatedly. `input` optionally overrides where the `.wasm` file comes from (`InitInput`). |
| `computeVertexNormals(positions, indices)` | `Float32Array` | Smooth, area-weighted per-vertex normals. Same length as `positions`. |
| `computeBounds(positions)` | `Bounds` | Plain object `{ min, max, center, size }`, each a `[x, y, z]` array. |

All functions throw an `Error` with a readable message when a buffer length is not a multiple of 3 or an index is out of range, and when called before `init()` has finished in the browser.

## Supported environments

| Environment | Entry | Notes |
| --- | --- | --- |
| Browsers via a bundler | `dist/web.js` | Vite, webpack 5, Next.js and others copy the `.wasm` asset automatically |
| Node.js 22+ | `dist/node.js` | The module is ready on import, also inside webpack server bundles (e.g. Next.js) |
| Bun | `dist/node.js` | Same as Node.js |

The right entry is picked automatically through the `exports` field of `package.json`. The package is ESM only.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, the build pipeline and the release process. Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
