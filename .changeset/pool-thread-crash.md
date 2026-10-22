---
"fast-mesh-wasm": patch
---

Multithreaded Worker scene: a crash on one of the engine's pool threads no longer hangs the scene. Before, a Rust panic or a WebAssembly trap in work split across threads killed only that thread, and pending promises never settled. Now the scene fails every pending call with `INSTANCE_POISONED` and terminates the Worker, as the single-threaded engine does; `restart()` recovers with a new pool. Pool threads report the crash on a `BroadcastChannel`, so nothing is polled and single-threaded scenes are unchanged. A few failures still cannot be detected (a stack overflow in WebAssembly memory, a Worker the browser ends without an exception, a thread that hangs); see "Crashes on a pool thread" in the README.
