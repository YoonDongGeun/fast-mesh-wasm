// The panels, layers and dialogs of the feature chunks (#285), under the names
// they have in their features: the shell uses them as if it had imported them.
// Their stylesheets are small and come with the page's, so that their order in
// the cascade stays and no part shows unstyled.
import { useSelection } from "fast-mesh-wasm/react";
import { createDialogOf, useCreateDialog } from "../../features/home/ux/create-dialog-state";
import { reliefDialogOf, useReliefDialog } from "../../features/mesh-edit/ux/relief-dialog-state";
import { textDialogOf, useTextDialog } from "../../features/mesh-edit/ux/text-dialog-state";
import { lazyDialog, lazyPanel, lazyPart } from "../../shared/ui/LazyPart";
import { type AppState, useApp } from "../state/app-state";
import { featureChunks } from "../ux/feature-chunks";
import "../../features/build-plate/ui/build-plate.css";
import "../../features/home/ui/create-dialog.css";
import "../../features/home/ui/height-map.css";
import "../../features/mesh-edit/ui/relief-dialog.css";
import "../../features/mesh-edit/ui/text-dialog.css";
import "../../features/polylines/ui/polylines.css";

// ---- Panels (heights: with nothing selected, at the sidebar's default width) ----

export const BuildPlatePanel = lazyPanel(featureChunks.buildPlate, (m) => m.BuildPlatePanel, {
  title: "Build Plate",
  height: 427,
  className: "build-plate-panel",
});
export const ExportPanel = lazyPanel(featureChunks.home, (m) => m.ExportPanel, { title: "Export", height: 320 });
export const SelectPanel = lazyPanel(featureChunks.select, (m) => m.SelectPanel, { title: "Select", height: 775 });
export const DeformPanel = lazyPanel(featureChunks.sculpt, (m) => m.DeformPanel, {
  title: "Free-Form Deformation",
  height: 510,
});
export const ModelingPanel = lazyPanel(featureChunks.meshEdit, (m) => m.ModelingPanel, {
  title: "Modeling & Resampling",
  height: 2844,
});
export const RepairPanel = lazyPanel(featureChunks.meshRepair, (m) => m.RepairPanel, { title: "Repair", height: 42 });
export const PointsPanel = lazyPanel(featureChunks.points, (m) => m.PointsPanel, {
  title: "Point Cloud Source",
  height: 595,
});
export const AlignPanel = lazyPanel(featureChunks.transform, (m) => m.AlignPanel, { title: "Align", height: 181 });
export const CollisionPanel = lazyPanel(featureChunks.collision, (m) => m.CollisionPanel, {
  title: "Collision",
  height: 1008,
});
export const InspectPanel = lazyPanel(featureChunks.inspect, (m) => m.InspectPanel, { title: "Inspect", height: 995 });
export const PolylinePanel = lazyPanel(featureChunks.polylines, (m) => m.PolylinePanel, {
  title: "Polylines",
  height: 89,
});
export const VolumePanel = lazyPanel(featureChunks.ct, (m) => m.VolumePanel, {
  title: "Volume",
  height: 514,
  useShown: ({ app }) => {
    const [selection] = useSelection();
    return useApp(app, (s) => s.items.some((i) => i.mesh === selection && i.volumeId !== null));
  },
});

// ---- Over the viewport ----

export const HoverTip = lazyPart(featureChunks.collision, (m) => m.HoverTip);
export const FaceSelectionMarquee = lazyPart(featureChunks.select, (m) => m.FaceSelectionMarquee);

type DialogProps = { app: AppState };

export const CreatePrimitiveDialog = lazyDialog(featureChunks.home, (m) => m.CreatePrimitiveDialog, {
  title: "Create shape",
  useOpen: ({ app }: DialogProps) => ({
    open: useCreateDialog(app, (s) => s.open),
    close: () => createDialogOf(app).close(),
  }),
});
export const TextDialog = lazyDialog(featureChunks.meshEdit, (m) => m.TextDialog, {
  title: "Text",
  useOpen: ({ app }: DialogProps) => ({
    open: useTextDialog(app, (s) => s.open),
    close: () => textDialogOf(app).close(),
  }),
});
export const ReliefDialog = lazyDialog(featureChunks.meshEdit, (m) => m.ReliefDialog, {
  title: "Image to relief",
  useOpen: ({ app }: DialogProps) => ({
    open: useReliefDialog(app, (s) => s.open),
    close: () => reliefDialogOf(app).close(),
  }),
});

// ---- In the viewport ----

export const AlignToolsLayer = lazyPart(featureChunks.transform, (m) => m.AlignToolsLayer);
export const BuildPlateLayer = lazyPart(featureChunks.buildPlate, (m) => m.BuildPlateLayer);
export const CreatePreviewLayer = lazyPart(featureChunks.home, (m) => m.CreatePreviewLayer);
export const DeformHandlesLayer = lazyPart(featureChunks.sculpt, (m) => m.DeformHandlesLayer);
export const FaceSelectionLayer = lazyPart(featureChunks.select, (m) => m.FaceSelectionLayer);
export const HollowPlaceToolsLayer = lazyPart(featureChunks.meshEdit, (m) => m.HollowPlaceToolsLayer);
export const InspectLayers = lazyPart(featureChunks.inspect, (m) => m.InspectLayers);
export const LatticePreviewLayer = lazyPart(featureChunks.meshEdit, (m) => m.LatticePreviewLayer);
export const LatticeSeeThrough = lazyPart(featureChunks.meshEdit, (m) => m.LatticeSeeThrough);
export const MaskLayer = lazyPart(featureChunks.sculpt, (m) => m.MaskLayer);
export const MetrologyOverlaysLayer = lazyPart(featureChunks.points, (m) => m.MetrologyOverlaysLayer);
export const PlacementPicksLayer = lazyPart(featureChunks.transform, (m) => m.PlacementPicksLayer);
export const PolylineLayer = lazyPart(featureChunks.polylines, (m) => m.PolylineLayer);
export const ReliefPreviewLayer = lazyPart(featureChunks.meshEdit, (m) => m.ReliefPreviewLayer);
export const SculptLayer = lazyPart(featureChunks.sculpt, (m) => m.SculptLayer);
export const SculptMaskOverlay = lazyPart(featureChunks.sculpt, (m) => m.SculptMaskOverlay);
export const SurfacePathLayer = lazyPart(featureChunks.inspect, (m) => m.SurfacePathLayer);
export const TextPreviewLayer = lazyPart(featureChunks.meshEdit, (m) => m.TextPreviewLayer);
