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

// A section: contours and fill, and the bound meshes are clipped at its plane
// (the side the section normal points to stays).
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

## Worker scene

`fast-mesh-wasm/worker` runs the same engine in a Web Worker, so adding a large mesh, building fields or recolouring never blocks the page. `createWorkerScene()` starts the Worker and resolves to a `WorkerScene` with the `Scene` methods. Your renderer keeps reading synchronous views: every result carries the bytes of its dirty ranges, which are copied into main-thread mirrors.

```ts
import { createWorkerScene } from "fast-mesh-wasm/worker";
import { bindMesh } from "fast-mesh-wasm/three"; // optional: the three.js adapter works on a WorkerScene too

const scene = await createWorkerScene(); // no init() needed: the Worker loads the wasm

// Handles come back at once; the work runs in the Worker.
const upper = scene.addMesh(upperPositions, upperIndices);
const lower = scene.addMesh(lowerPositions, lowerIndices);
const gap = scene.addDistanceField({ name: "gap", on: upper, to: lower, maxDistance: 2 });
scene.setColoring(upper, { field: gap, map: "turbo", range: [-0.5, 1] });
const boundUpper = bindMesh(scene, upper);

// Once per animation frame: send this frame's requests, receive the results so far.
function frame() {
  if (pending !== undefined) drag?.move(pending); // at most one move per frame
  pending = undefined;
  scene.commitFrame(); // the adapters' listeners apply the changes here, as with Scene
  renderer.render(threeScene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// Status of each piece of state: "idle" | "computing" | "ready" | "error".
scene.onStatus(() => {
  const coloring = scene.status(`display:${scene.idOf(upper)}`);
  button.classList.toggle("busy", coloring.status === "computing");
  undoButton.disabled = !scene.canUndo; // canUndo / canRedo are getters here
});

// Queries are async.
const hit = await scene.raycast([0, 0, 50], [0, 0, -1]);
const distance = await scene.meshDistance(upper, lower);
const values = await scene.fetchFieldValues(gap); // `fieldValues` on Scene
```

- **The frame loop.** `commitFrame()` sends everything queued since the last frame as one batch and returns the change records of the results received meanwhile (already applied to the mirrors, so `views`, `matrix4`, `placement`, `section` and `sectionPlane` read the new state). Results arrive between frames; nothing changes in the mirrors outside a message.
- **Requests run in order.** A newer colouring, section plane/meshes/fill or drag move drops the older one if it has not started yet, so fast toggling settles on the last state without doing the steps in between. Edits, repairs, `setPlacement`, adds, removals and undo/redo are never dropped. Each `setPlacement` is an undo step: for pointer-rate updates use a drag.
- **Drags are optimistic.** `scene.drag(mesh).move(pose)` moves the placement mirror at once, so the object follows the pointer; colours and sections catch up when the Worker's result arrives (their keys show `"computing"` until then). `end()` and `cancel()` return at once; `cancel()` restores the start pose and records nothing.
- **Queries and anything that returns data** (`raycast`, `closestPoint`, `meshDistance`, `intersects`, `findContacts`, `analyze`, `edit`, `repair`, `fetchFieldValues`, `fieldStatus`, `sectionGap`) return promises. Once a frame loop runs they go out with the next `commitFrame()` (or after 100 ms if no frame comes); before the first `commitFrame()` they are sent on a microtask. For latency-critical picking, keep a light picking copy in a main-thread `Scene`.
- **Errors.** A failed state change shows on its key (`status(key).error`, a `FastMeshError` with the same `code` as on `Scene`); a failed promise rejects with it. After a trap inside the Worker every call fails with `INSTANCE_POISONED`. `await scene.restart()` starts a new Worker and replays the scene from the main thread's mirrors: meshes (compacted: slot ids change, handles do not), placements, fields, colourings and sections. History is cleared, and edits or queries that had not finished fail and must be re-issued.
- **Not in the Worker scene:** user-opened steps (`beginStep`/`endStep`/`cancelStep`), Lazy and Manual modes (it is always eager) and `takeChanges` (use `onChanges` or the return value of `commitFrame()`). `terminate()` stops the Worker.
- **Large meshes.** `addMesh` copies its inputs on the main thread (they are then transferred to the Worker) and, without an `origin` option, scans the positions for the bounding-box centre. That takes about 9 ms for 1M triangles and about 63 ms for 10M (see [docs/benchmarks.md](docs/benchmarks.md)); passing `origin` skips the scan.

**Bundlers.** The Worker script is a separate file, `worker-entry.js`, loaded with `new Worker(new URL("./worker-entry.js", import.meta.url), { type: "module" })`. Vite and other bundlers that understand this pattern pick it up and emit it with the `.wasm` it loads: [`examples/webgpu`](examples/webgpu) uses the default with Vite, in both `vite build` and `vite dev` (open it with `?worker`). If your setup does not handle the pattern, serve the package's `dist/` folder (the Worker script loads the files next to it) and pass the script's URL, or create the Worker yourself:

```ts
const scene = await createWorkerScene({ workerUrl: "/fast-mesh-wasm/worker-entry.js" });
// or
const scene = await createWorkerScene({ createWorker: () => new Worker(myUrl, { type: "module" }) });
```

`fast-mesh-wasm/worker` is for browsers; it has no Node.js export (Node's `worker_threads` is used only by the tests).

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
