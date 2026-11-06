// Checks of `register`'s options, on the main thread and without WebAssembly:
// `Scene` and `WorkerScene` both run them before ending a drag, so that a call
// that is going to fail changes nothing. The engine repeats them.
import { FastMeshError } from "./core.js";
import type { RegisterOptions } from "./scene.js";

function invalid(message: string): FastMeshError {
  return new FastMeshError("INVALID_ARGUMENT", message);
}

function isNumber(v: unknown): v is number {
  return typeof v === "number" && !Number.isNaN(v);
}

/**
 * Throws `INVALID_ARGUMENT` for an option out of range or for one mesh twice.
 * Unset options are valid: they take their defaults.
 */
export function checkRegisterOptions(sameMesh: boolean, o: RegisterOptions): void {
  if (sameMesh) {
    throw invalid("cannot register a mesh to itself");
  }
  const { maxIterations: it, tolerance: tol, overlap, maxDistance: far, maxSamples: cap, maxRmse: rmse } = o;
  if (it !== undefined && !(Number.isInteger(it) && it >= 0)) {
    throw invalid("maxIterations must be a non-negative integer");
  }
  if (tol !== undefined && !(isNumber(tol) && Number.isFinite(tol) && tol >= 0)) {
    throw invalid("tolerance must be finite and zero or greater");
  }
  if (overlap !== undefined && !(isNumber(overlap) && overlap > 0 && overlap <= 1)) {
    throw invalid("overlap must be in (0, 1]");
  }
  if (far !== undefined && !(isNumber(far) && far > 0)) {
    throw invalid("maxDistance must be positive");
  }
  if (cap !== undefined && !(Number.isInteger(cap) && cap >= 1)) {
    throw invalid("maxSamples must be a positive integer");
  }
  if (rmse !== undefined && !(isNumber(rmse) && rmse >= 0)) {
    throw invalid("maxRmse must be zero or greater");
  }
  if (o.apply !== undefined && typeof o.apply !== "boolean") {
    throw invalid("apply must be a boolean");
  }
}
