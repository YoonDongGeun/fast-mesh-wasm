# fast-mesh-wasm + raw WebGPU

The owner's scene (a sphere, a torus and a wavy plate) drawn with WebGPU and
no 3D library. It shows the renderer-neutral way to use the `Scene` API: the
scene owns geometry, placements, history, distance fields and sections, and
the renderer only mirrors its buffers onto the GPU.

## The renderer-neutral pattern

Once per animation frame (see `src/main.ts` and `Renderer.applyChange` in `src/gpu.ts`):

```ts
for (const change of scene.commitFrame()) {
  if (change.removed) { destroyBuffers(change.mesh); continue; }
  if (change.added || change.capacityChanged) { recreateBuffers(change.mesh); continue; } // every buffer is "all"
  const v = scene.views(change.mesh);            // 1. read the views, fresh, this frame
  for (const [start, count] of ranges(change.positions)) // 2. upload only the dirty ranges ("all": the whole view)
    queue.writeBuffer(posBuf, start * 12, v.positions().subarray(3 * start, 3 * (start + count)));
  // normals: same as positions; colours: 4 bytes per vertex (start * 4); indices: 3 u32 per triangle
}
for (const mesh of meshes) {
  const m = scene.matrix4(mesh);                 // 4. apply the placement: column-major f64
  queue.writeBuffer(objectUniform(mesh), 0, Float32Array.from(m)); // → f32 model matrix
}
```

- Views are only valid until the next scene call: get them inside the frame handler and never keep them.
- Call `.subarray()` before `writeBuffer` (with `new Scene({ debug: true })` views are Proxies).
- `"all"` means upload the whole buffer; on `capacityChanged` (3.) the slot counts changed, so recreate the buffers.
- The index buffer is `Uint32` with removed triangles as `(0, 0, 0)`: draw `indices().length` indices; dead slots are degenerate.
- `colors()` is `null` while colouring is off. This example always allocates a 4-byte-per-vertex colour buffer
  (so one pipeline fits every mesh) and a uniform flag picks the vertex colour or the object's base colour.
- Drags: `scene.drag(mesh)` per gesture, one `move` per animation frame (the latest pose, before
  `commitFrame()`), `end()` on release. Each drag is one undo step; undo/redo arrive as ordinary changes.

## Run it

```sh
pnpm install
pnpm build                                          # builds the fast-mesh-wasm package
pnpm --filter fast-mesh-wasm-example-webgpu dev     # then open the printed URL
```

Needs a browser with WebGPU (recent Chrome, Edge or Safari). Without it the page shows a message.

## Controls

- Click: pick a mesh. Drag it: move it in a camera-facing plane. Shift+drag: rotate it (arcball).
- Drag empty space: orbit. Wheel: zoom.
- Ctrl+Z: undo. Ctrl+Y or Ctrl+Shift+Z: redo.
- C: distance colouring of the picked mesh against its nearest other mesh (turbo map, `maxDistance` = 10% of
  the scene diagonal; farther vertices stay grey). C again turns it off.
- 1–9: picked mesh opacity 0.1–0.9; 0: opaque.
- S: section plane (normal +z) through the scene centre: yellow contours, and the +z half is clipped away.
- Drop a binary STL file on the page to add it.

## Manual test checklist

1. The page shows three lit meshes; the console has no errors or WebGPU validation warnings.
2. Click the sphere: the status line names it and it gets an orange tint. Click empty space: nothing picked.
3. Drag the sphere: it follows the pointer. Release, Ctrl+Z: it jumps back. Ctrl+Y (and Ctrl+Shift+Z): it returns.
4. Shift+drag the torus: it tilts about its centre. Ctrl+Z undoes the whole drag in one step.
5. Pick the sphere, press C: the side facing the torus is coloured. Drag it closer: the colours follow each frame.
   Press C: the sphere is plain again.
6. Pick the torus, press 5: it is half transparent and the meshes behind show through. Make a second mesh
   transparent too: they blend in a stable order. Press 0: opaque again.
7. Press S: yellow contours on every mesh and the near half is cut away. Move a mesh: its contour follows.
   Press S: the plane and the cut are gone.
8. Drop a binary STL: it appears, the camera refits, and it can be picked, dragged and coloured.
9. Open the page in a browser without WebGPU: a message explains it, and the console shows no exception.
