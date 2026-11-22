// The command palette's behaviour (#235): a fresh query and the search field
// focused each time it opens, the commands ranked for the query, and keyboard
// navigation in the field (ArrowUp / ArrowDown wrap, Enter runs the active
// command, Escape closes). The app's shortcuts open it (Ctrl+K / Cmd+K, `/`:
// app/ux/use-keyboard-shortcuts.ts); a second window listener here toggled it
// again on the same keypress, so Ctrl+K opened and closed it at once.
import { type KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import { useApp } from "../../app/state/app-state";
import { buildCommandPaletteItems, type CommandPaletteContext, type CommandPaletteItem } from "./command-registry";
import { filterCommands } from "./command-search";

export function useCommandPalette(ctx: CommandPaletteContext) {
  const { app } = ctx;
  const open = useApp(app, (s) => s.commandPaletteOpen);
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (open) {
      setQuery("");
      setActiveIndex(0);
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  const allCommands = useMemo(
    () => buildCommandPaletteItems(ctx),
    [ctx.app, ctx.items, ctx.eligibleItems, ctx.selection, ctx.canUndo, ctx.canRedo, ctx.ortho, ctx.units],
  );

  const filtered = useMemo(() => filterCommands(allCommands, query), [allCommands, query]);

  useEffect(() => {
    setActiveIndex(0);
  }, [query]);

  const close = () => app.setCommandPaletteOpen(false);

  /** Closes the palette and runs `cmd`, unless it is missing or disabled. */
  const execute = (cmd: CommandPaletteItem | undefined) => {
    if (!cmd || cmd.disabled) return;
    app.setCommandPaletteOpen(false);
    cmd.run();
  };

  const onInputKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Escape") {
      e.preventDefault();
      app.setCommandPaletteOpen(false);
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIndex((i) => (filtered.length === 0 ? 0 : (i + 1) % filtered.length));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIndex((i) => (filtered.length === 0 ? 0 : (i - 1 + filtered.length) % filtered.length));
    } else if (e.key === "Enter") {
      e.preventDefault();
      execute(filtered[activeIndex]);
    }
  };

  return { open, query, setQuery, activeIndex, setActiveIndex, inputRef, filtered, close, execute, onInputKeyDown };
}
