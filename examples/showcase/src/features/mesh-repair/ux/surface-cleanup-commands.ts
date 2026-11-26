// The surface cleanup's commands (#264), apart from what they run: the ribbon
// and the command palette list them before the repair chunk is here.

/** The surface cleanup commands the ribbon and the command palette offer, by the panel button they press. */
export const SURFACE_CLEANUP_COMMANDS: readonly {
  op: "denoise" | "delaunayFlips";
  label: string;
  short: string;
  keywords: string;
}[] = [
  {
    op: "denoise",
    label: "Repair: Denoise (Keep Sharp Edges)",
    short: "Denoise",
    keywords: "repair denoise smooth scan noise bilateral normal filter feature preserving sharp edges",
  },
  {
    op: "delaunayFlips",
    label: "Repair: Delaunay Edge Flips",
    short: "Delaunay Flips",
    keywords: "repair delaunay flip edges triangle quality minimum angle dihedral",
  },
];
