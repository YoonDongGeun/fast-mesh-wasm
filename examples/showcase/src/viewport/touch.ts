// What touch changes for the canvas's pointer listeners: only the first
// finger of a gesture may start a tool, and a tap's finger jitters further
// than a mouse.

/** How far (CSS px) a finger may wander during a tap. Chrome's own click slop is 15. */
const TOUCH_TAP_SLOP = 16;

/**
 * A finger that landed while another was already down. It belongs to the
 * camera's pinch: tools (the gizmo, slice and section handles, brushes) never
 * start from it. Mouse and pen pointers are always primary.
 */
export function isExtraFinger(e: { pointerType: string; isPrimary: boolean }): boolean {
  return e.pointerType === "touch" && !e.isPrimary;
}

/** How far a press may travel and still be a tap: `mouseSlop` unless the press was a finger. */
export function tapSlop(pointerType: string, mouseSlop: number): number {
  return pointerType === "touch" ? TOUCH_TAP_SLOP : mouseSlop;
}
