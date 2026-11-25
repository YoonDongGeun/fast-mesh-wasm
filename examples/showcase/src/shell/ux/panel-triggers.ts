// The ribbon reaches the panels through their DOM: a shelf button presses a
// panel's own control (so the panel's run logic stays the one path) or
// scrolls to and focuses the panel's section, and Open clicks the hidden
// file input. A panel whose code is still on its way (#285) is waited for.
import { loadWaitingParts, partsArriving } from "../../shared/ux/lazy-chunk";

/** How often a wait looks for its control. */
const LOOK_MS = 50;
/**
 * How long a wait goes on once every panel is here: a control may come with
 * what its panel reads first (the repair panel's, with the analysis).
 */
const SETTLE_MS = 3000;

/**
 * Runs `act` on the first element matching `selector`: now if there is one,
 * else, while panels are still loading, once theirs has brought it and it is
 * enabled. Does nothing when no panel is on its way, or none brings it.
 * Returns the function that gives up the wait.
 */
export function whenPresent(selector: string, act: (el: HTMLElement) => void): () => void {
  const now = document.querySelector<HTMLElement>(selector);
  if (now !== null) {
    act(now);
    return () => {};
  }
  if (!loadWaitingParts()) {
    return () => {};
  }
  let timer: ReturnType<typeof setTimeout> | undefined;
  let left = SETTLE_MS;
  const look = () => {
    const el = document.querySelector<HTMLElement>(selector);
    if (el !== null && !el.matches(":disabled")) {
      act(el);
      return;
    }
    if (!partsArriving()) {
      left -= LOOK_MS;
    }
    if (left > 0) {
      timer = setTimeout(look, LOOK_MS);
    }
  };
  timer = setTimeout(look, LOOK_MS);
  return () => clearTimeout(timer);
}

/** Gives up the latest command's wait, if it is still waiting. */
let dropLatest: () => void = () => {};

/** `whenPresent` for a command: a later command takes its place. */
function command(selector: string, act: (el: HTMLElement) => void): void {
  dropLatest();
  dropLatest = whenPresent(selector, act);
}

/** Clicks the first element matching `selector`, if there is one (or its panel brings one). */
export function triggerClick(selector: string): void {
  command(selector, (el) => el.click());
}

/** Scrolls the first element matching `selector` into view and focuses it. */
export function focusElement(selector: string): void {
  command(selector, (el) => {
    el.scrollIntoView({ block: "nearest" });
    el.focus();
  });
}

/** Opens the file picker behind the mesh list's hidden `#file-input`. */
export function openFilePicker(): void {
  document.getElementById("file-input")?.click();
}
