// What touch changes for the canvas's pointer listeners: only the first finger
// on the canvas may start a tool, whatever is touching the rest of the page,
// and a tap's finger jitters further than a mouse.

/** How far (CSS px) a finger may wander during a tap. Chrome's own click slop is 15. */
const TOUCH_TAP_SLOP = 16;

/** The fingers on the glass, counted by where they landed. A pointer keeps its target until it lifts (touch's implicit capture). */
export class FingerTracker {
  private readonly canvas = new Set<number>();
  private readonly elsewhere = new Set<number>();
  private first = -1;
  private firstAt = { x: 0, y: 0 };
  private overlapped = false;
  private pinched = false;

  down(id: number, onCanvas: boolean, isPrimary: boolean, x: number, y: number): void {
    if (isPrimary) {
      // No other finger is down, whatever a lost pointerup left behind.
      this.canvas.clear();
      this.elsewhere.clear();
    }
    if (!onCanvas) {
      this.elsewhere.add(id);
      this.overlapped ||= this.canvas.size > 0;
      return;
    }
    if (this.canvas.size === 0) {
      this.first = id;
      this.firstAt = { x, y };
      this.overlapped = this.elsewhere.size > 0;
      this.pinched = false;
    } else {
      this.pinched = true;
    }
    this.canvas.add(id);
  }

  /**
   * A finger left the glass. True if it ended a tap the browser will not turn
   * into a click: it does not for a finger pressed while another touches the
   * page elsewhere, however far apart they are.
   */
  up(id: number, cancelled: boolean, x: number, y: number): boolean {
    this.elsewhere.delete(id);
    const wasOnCanvas = this.canvas.delete(id);
    return (
      wasOnCanvas &&
      !cancelled &&
      id === this.first &&
      this.overlapped &&
      !this.pinched &&
      Math.hypot(x - this.firstAt.x, y - this.firstAt.y) <= TOUCH_TAP_SLOP
    );
  }

  has(id: number): boolean {
    return this.canvas.has(id);
  }

  /** Whether this pointer landed on the canvas after the gesture's first finger. It stays so until the canvas is empty and a new gesture starts. */
  isExtra(id: number): boolean {
    return id !== this.first;
  }

  reset(): void {
    this.canvas.clear();
    this.elsewhere.clear();
    this.first = -1;
    this.overlapped = false;
    this.pinched = false;
  }
}

const fingers = new FingerTracker();

/**
 * Keeps the finger count for `isExtraFinger`, and keeps fingers off the canvas
 * from the camera. TrackballControls takes `pointermove` and `pointerup` on the
 * document, from every pointer, once it holds one finger: a finger on a panel
 * would write its move state. Only touch is filtered, the mouse passes.
 *
 * Call before creating the TrackballControls: this listener must come first on
 * the document for its stopImmediatePropagation to hold those events back. It
 * runs after the event has bubbled through the page, so the page's own handlers
 * have seen it. Returns the disposer.
 */
export function watchFingers(canvas: HTMLElement): () => void {
  const doc = canvas.ownerDocument;
  const onDown = (e: PointerEvent) => {
    if (e.pointerType === "touch") {
      fingers.down(e.pointerId, e.target === canvas, e.isPrimary, e.clientX, e.clientY);
    }
  };
  const onMove = (e: PointerEvent) => {
    if (e.pointerType === "touch" && !fingers.has(e.pointerId)) {
      e.stopImmediatePropagation();
    }
  };
  const onUp = (e: PointerEvent) => {
    if (e.pointerType !== "touch") {
      return;
    }
    onMove(e);
    if (fingers.up(e.pointerId, e.type === "pointercancel", e.clientX, e.clientY)) {
      canvas.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true, clientX: e.clientX, clientY: e.clientY }));
    }
  };
  // Capture: counted before any listener on the canvas asks.
  doc.addEventListener("pointerdown", onDown, true);
  doc.addEventListener("pointermove", onMove);
  doc.addEventListener("pointerup", onUp);
  doc.addEventListener("pointercancel", onUp);
  return () => {
    doc.removeEventListener("pointerdown", onDown, true);
    doc.removeEventListener("pointermove", onMove);
    doc.removeEventListener("pointerup", onUp);
    doc.removeEventListener("pointercancel", onUp);
    fingers.reset();
  };
}

/**
 * A finger that landed on the canvas while another was already on it. It belongs
 * to the camera's pinch: tools (the gizmo, slice and section handles, brushes)
 * never start from it. Fingers off the canvas do not count. Mouse and pen
 * pointers are never extra.
 */
export function isExtraFinger(e: { pointerType: string; pointerId: number }): boolean {
  return e.pointerType === "touch" && fingers.isExtra(e.pointerId);
}

/** How far a press may travel and still be a tap: `mouseSlop` unless the press was a finger. */
export function tapSlop(pointerType: string, mouseSlop: number): number {
  return pointerType === "touch" ? TOUCH_TAP_SLOP : mouseSlop;
}
