// Images dropped on the page (#267) open the Image to Relief dialog, with a
// height map's JSON if one is dropped along; every other file loads as before.
import { useCallback } from "react";
import type { AppState } from "../../../app/state/app-state";
import { splitReliefFiles } from "./relief-files";
import { reliefDialogOf } from "./relief-dialog-state";

export function useReliefDrop(
  app: AppState,
  load: (files: Iterable<File>) => void,
): (files: Iterable<File>) => void {
  return useCallback(
    (files: Iterable<File>) => {
      const { relief, rest } = splitReliefFiles(files);
      if (relief !== null) {
        reliefDialogOf(app).take(relief);
      }
      if (rest.length > 0) {
        load(rest);
      }
    },
    [app, load],
  );
}
