---
"fast-mesh-wasm": patch
---

Queries between meshes of different scales take both mesh origins into account (#288). `meshDistance`, `intersects`, `findContacts`, `trianglesNear` and `meshDeviation` placed the second mesh off by the first one's origin when the pair was scaled and the first mesh's scale was uniform, and distance fields did so with unevenly scaled targets, so a mesh away from its own zero gave wrong distances and missed or invented contacts.
