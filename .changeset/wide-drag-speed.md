---
"fast-mesh-wasm": patch
---

Faster wide-range distance colouring while dragging: when most of a mesh is within a field's `maxDistance`, the per-pose update now recomputes every vertex instead of first listing every nearby triangle pair, which makes a 1M-triangle drag with `maxDistance` 8 about 29 times faster per pose. Values and colours are unchanged, bit for bit. In the Worker scene (`fast-mesh-wasm/worker`), the field recompute after a drag move now runs in small steps, and a recompute whose pose is already stale (a newer move, `setPlacement`, edit, repair, compact, undo/redo or removal of either mesh is queued) is dropped, so colours follow the latest pose without a backlog. Each pose's colours are still applied all at once. No API change.
