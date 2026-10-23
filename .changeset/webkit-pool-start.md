---
"fast-mesh-wasm": patch
---

Multithreaded Worker scene: on WebKit 26 (Safari 26) the scene now starts single-threaded in about 50 ms instead of after the 10 s thread-pool timeout. WebKit 26 traps `memory.fill` and `memory.copy` on a shared memory that another thread grew, so its thread pool could never start. A probe of about 10 ms now detects this before the threaded files are fetched. Other browsers are unchanged and keep the threaded engine. See "Older WebKit" in the README.
