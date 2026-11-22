// The face selection tools' 2D marquee (#233): the live box or lasso in
// canvas pixels, published from the viewport's pointer handling to the
// <svg> overlay outside the canvas, with no prop plumbing between them.

/** The marquee being dragged, in canvas pixels with the canvas size; `null` when none. */
export type FaceSelectionMarqueeState =
  | {
      kind: "box";
      x0: number;
      y0: number;
      x1: number;
      y1: number;
      width: number;
      height: number;
    }
  | {
      kind: "lasso";
      points: readonly (readonly [number, number])[];
      width: number;
      height: number;
    }
  | null;

let currentMarquee: FaceSelectionMarqueeState = null;
const marqueeListeners = new Set<() => void>();

/** Returns the active 2D marquee overlay state used by the `FaceSelectionMarquee` overlay. */
export function getFaceSelectionMarquee(): FaceSelectionMarqueeState {
  return currentMarquee;
}

/** Publishes a new 2D marquee overlay state and notifies subscribers. */
export function setFaceSelectionMarquee(next: FaceSelectionMarqueeState): void {
  currentMarquee = next;
  for (const listener of marqueeListeners) {
    listener();
  }
}

/** Subscribes to changes of the 2D marquee overlay state. */
export function subscribeFaceSelectionMarquee(listener: () => void): () => void {
  marqueeListeners.add(listener);
  return () => {
    marqueeListeners.delete(listener);
  };
}
