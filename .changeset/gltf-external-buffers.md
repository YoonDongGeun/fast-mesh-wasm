---
"fast-mesh-wasm": patch
---

`importGLTFAsync` loads `.gltf` assets with external `.bin` buffers through a new `resolveBuffer` option: a callback returning each buffer's bytes by URI (fetch / FileReader / fs on the caller's side; the importer itself never fetches). The sync `importGLTF` still throws on external buffers, now pointing at the async form.
