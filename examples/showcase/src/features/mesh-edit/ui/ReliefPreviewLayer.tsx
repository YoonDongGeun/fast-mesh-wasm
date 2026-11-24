// The Image to Relief dialog's live preview (#267): a coarse relief drawn
// translucent at the dialog's centre and rotation, as the Create dialog draws
// its shape (the same scene objects). It takes no clicks. The geometry is
// rebuilt when a new relief arrives; centre and rotation only move it.
import { useThree } from "@react-three/fiber";
import { useEffect, useLayoutEffect, useMemo } from "react";
import * as THREE from "three";
import type { AppState } from "../../../app/state/app-state";
import { buildCreatePreview } from "../../home/ui/CreatePreviewLayer";
import { useReliefDialog } from "../ux/relief-dialog-state";

export function ReliefPreviewLayer(props: { app: AppState }) {
  const preview = useReliefDialog(props.app, (s) => s.preview);
  const invalidate = useThree((s) => s.invalidate);
  const mesh = preview?.mesh ?? null;
  const built = useMemo(() => (mesh === null ? null : buildCreatePreview(mesh)), [mesh]);
  useEffect(() => () => built?.dispose(), [built]);
  useLayoutEffect(() => {
    if (built !== null && preview !== null) {
      const [x, y, z] = preview.centre;
      const [rx, ry, rz] = preview.rotationDeg.map((d) => THREE.MathUtils.degToRad(d));
      built.root.position.set(x, y, z);
      built.root.rotation.set(rx, ry, rz, "XYZ");
      built.root.name = "relief-preview";
    }
    invalidate();
  }, [built, preview, invalidate]);
  return built === null ? null : <primitive object={built.root} />;
}
