// Files dropped anywhere on the page load. The result is true while files
// are dragged over the window (it shows the drop overlay); leaving the
// window, or dropping, clears it.
import { useEffect, useState } from "react";

export function useFileDrop(load: (files: Iterable<File>) => void): boolean {
  const [dropping, setDropping] = useState(false);
  useEffect(() => {
    const hasFiles = (e: DragEvent) => e.dataTransfer?.types.includes("Files") === true;
    const onDragOver = (e: DragEvent) => {
      if (hasFiles(e)) {
        e.preventDefault();
        setDropping(true);
      }
    };
    const onDragLeave = (e: DragEvent) => {
      // Leaving the window, not moving between elements.
      if (e.relatedTarget === null) {
        setDropping(false);
      }
    };
    const onDrop = (e: DragEvent) => {
      if (!hasFiles(e)) {
        return;
      }
      e.preventDefault();
      setDropping(false);
      load(e.dataTransfer?.files ?? []);
    };
    window.addEventListener("dragover", onDragOver);
    window.addEventListener("dragleave", onDragLeave);
    window.addEventListener("drop", onDrop);
    return () => {
      window.removeEventListener("dragover", onDragOver);
      window.removeEventListener("dragleave", onDragLeave);
      window.removeEventListener("drop", onDrop);
    };
  }, [load]);
  return dropping;
}
