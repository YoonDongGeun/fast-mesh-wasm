// The ribbon reaches the panels through their DOM: a shelf button presses a
// panel's own control (so the panel's run logic stays the one path) or
// scrolls to and focuses the panel's section, and Open clicks the hidden
// file input.

/** Clicks the first element matching `selector`, if there is one. */
export function triggerClick(selector: string): void {
  const el = document.querySelector<HTMLElement>(selector);
  el?.click();
}

/** Scrolls the first element matching `selector` into view and focuses it. */
export function focusElement(selector: string): void {
  const el = document.querySelector<HTMLElement>(selector);
  if (!el) return;
  el.scrollIntoView({ block: "nearest" });
  el.focus();
}

/** Opens the file picker behind the mesh list's hidden `#file-input`. */
export function openFilePicker(): void {
  document.getElementById("file-input")?.click();
}
