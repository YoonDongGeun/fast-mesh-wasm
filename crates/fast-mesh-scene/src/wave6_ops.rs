//! Scene-level wrappers for Wave 6 geometry, selection, deformation, inspection, and solid-modeling domains (`#233`, `#193`, `#234`, `#236`).

use std::f64::consts::PI;

use fast_mesh_ops::OpsError;
use fast_mesh_ops::boolean::{
    DrainHoleOptions, DrainHoleReport, DrainHoleSpec, HollowOptions, HollowReport, PlaceMeshParams,
    PlaceMeshReport, PlaceMode, compute_place_transform, hollow_mesh, place_mesh, punch_drain_hole,
};
use fast_mesh_ops::deform::{
    FfdBasis, FfdLattice, FreeformOp, FreeformReport, apply_freeform, apply_freeform_protected,
};
use fast_mesh_ops::inspect::{
    ContourSlice, FitPrimitiveOptions, FitPrimitiveReport, FittedPrimitive, PointCloud,
    PrimitiveKind, SurfaceReconstructOptions, datum_alignment_rigid, fit_primitive_mesh,
    reconstruct_surface, slice_contour_stack, slice_contours,
};
use fast_mesh_ops::repair::{
    AdjacencyKind, FaceEditReport, FaceSelection, ScreenSelectOptions, ScreenSelectionShape,
    delete_selected_faces, extract_selected_faces, face_selection_to_vertex_mask,
    fill_selection_holes, grow_face_selection_with, select_by_crease_angle, select_connected_shell,
    select_faces_in_sphere, select_faces_screen, select_small_face_islands, shrink_face_selection,
    smooth_selection_boundary, split_selected_faces, vertex_mask_to_face_selection,
};
use fast_mesh_store::{Mesh, StoreError, Transaction, TriangleId, UV_CHANNEL, VertexId};

use crate::fields::baked_world_mesh;
use crate::placement::{Placement, local_to_world, orthonormalize, world_to_local};
use crate::{MeshId, MeshOptions, Result, Scene, SceneError};

/// World-space 3D vector in double precision.
pub type Vec3 = [f64; 3];

fn finite_vec3(v: Vec3, msg: &'static str) -> Result<()> {
    if v.iter().all(|c| c.is_finite()) {
        Ok(())
    } else {
        Err(SceneError::Ops(OpsError::InvalidInput(msg)))
    }
}

fn dot3(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn unit3(v: Vec3, msg: &'static str) -> Result<Vec3> {
    finite_vec3(v, msg)?;
    let len = dot3(v, v).sqrt();
    if !len.is_finite() || len <= 1e-12 {
        return Err(SceneError::Ops(OpsError::InvalidInput(msg)));
    }
    Ok([v[0] / len, v[1] / len, v[2] / len])
}

fn vec3_to_f32(v: Vec3, msg: &'static str) -> Result<[f32; 3]> {
    finite_vec3(v, msg)?;
    let out = [v[0] as f32, v[1] as f32, v[2] as f32];
    if !out.iter().all(|c| c.is_finite()) {
        return Err(SceneError::Ops(OpsError::InvalidInput(msg)));
    }
    Ok(out)
}

fn mul_mat3(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|r| {
        std::array::from_fn(|c| a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c])
    })
}

fn mat3_vec3(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn mat4_mul_f64(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut out = [[0.0_f64; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            out[col][row] = a[0][row] * b[col][0]
                + a[1][row] * b[col][1]
                + a[2][row] * b[col][2]
                + a[3][row] * b[col][3];
        }
    }
    out
}

/// Column-major 4x4 affine matrix mapping stored `Mesh::positions()` (`p`)
/// directly to world coordinates (`R * (S * (origin + p)) + t`).
fn model_matrix_stored_to_world(placement: &Placement, origin: [f64; 3]) -> [[f64; 4]; 4] {
    let r = &placement.rotation;
    let s = &placement.scale;
    let w0 = local_to_world(placement, origin, [0.0, 0.0, 0.0]);
    [
        [r[0][0] * s[0], r[1][0] * s[0], r[2][0] * s[0], 0.0],
        [r[0][1] * s[1], r[1][1] * s[1], r[2][1] * s[1], 0.0],
        [r[0][2] * s[2], r[1][2] * s[2], r[2][2] * s[2], 0.0],
        [w0[0], w0[1], w0[2], 1.0],
    ]
}

fn world_pose_for_local_origin_mesh(placement: &Placement, parent_origin: [f64; 3]) -> Placement {
    Placement {
        rotation: placement.rotation,
        scale: placement.scale,
        translation: local_to_world(placement, parent_origin, [0.0, 0.0, 0.0]),
    }
}

fn replace_tx_world_to_local(
    tx: &mut Transaction<'_>,
    world_mesh: &Mesh,
    placement: &Placement,
    origin: [f64; 3],
) -> Result<()> {
    let mut drop_tris: Vec<TriangleId> = Vec::new();
    let mut drop_verts: Vec<VertexId> = Vec::new();
    {
        let cur = tx.mesh();
        drop_tris
            .try_reserve(cur.triangle_count())
            .map_err(|_| StoreError::OutOfMemory)?;
        for t in 0..cur.triangle_slots() as TriangleId {
            if cur.is_triangle_alive(t) {
                drop_tris.push(t);
            }
        }
        drop_verts
            .try_reserve(cur.vertex_count())
            .map_err(|_| StoreError::OutOfMemory)?;
        for v in 0..cur.vertex_slots() as VertexId {
            if cur.is_vertex_alive(v) {
                drop_verts.push(v);
            }
        }
    }

    let mut remap: Vec<u32> = vec![u32::MAX; world_mesh.vertex_slots()];
    let mut add_pos: Vec<[f32; 3]> = Vec::new();
    let mut add_tris: Vec<[u32; 3]> = Vec::new();
    let w_orig = world_mesh.origin();
    let w_pos = world_mesh.positions();

    for t in 0..world_mesh.triangle_slots() as TriangleId {
        let Some(tri) = world_mesh.triangle(t) else {
            continue;
        };
        let mut mapped = [0u32; 3];
        for (k, v) in tri.into_iter().enumerate() {
            let idx = if remap[v as usize] != u32::MAX {
                remap[v as usize]
            } else {
                let next = u32::try_from(add_pos.len()).map_err(|_| StoreError::OutOfMemory)?;
                let p = w_pos[v as usize];
                let wp = [
                    w_orig[0] + f64::from(p[0]),
                    w_orig[1] + f64::from(p[1]),
                    w_orig[2] + f64::from(p[2]),
                ];
                let lp = world_to_local(placement, origin, wp);
                let lf = vec3_to_f32(lp, "transformed local vertex leaves f32 range")?;
                add_pos.push(lf);
                remap[v as usize] = next;
                next
            };
            mapped[k] = idx;
        }
        add_tris.push(mapped);
    }

    if tx.mesh().uvs().is_some() {
        tx.drop_channel(UV_CHANNEL)?;
    }
    if !drop_tris.is_empty() {
        tx.remove_triangles(&drop_tris)?;
    }
    if !drop_verts.is_empty() {
        tx.remove_vertices(&drop_verts)?;
    }
    if !add_pos.is_empty() {
        let added = tx.add_vertices(&add_pos)?;
        for tri in &mut add_tris {
            *tri = tri.map(|i| added[i as usize]);
        }
        if !add_tris.is_empty() {
            tx.add_triangles(&add_tris)?;
        }
    }
    Ok(())
}

fn shrink_vertex_adjacent(mesh: &Mesh, sel: &FaceSelection, rings: usize) -> Result<FaceSelection> {
    let slots = mesh.triangle_slots();
    let v_slots = mesh.vertex_slots();
    if rings == 0 || sel.is_empty() || mesh.triangle_count() == 0 {
        let copy = FaceSelection::from_triangles(mesh, &sel.to_triangles(mesh))?;
        return Ok(copy);
    }

    // Identify mesh boundary vertices once: any edge with < 2 alive incident triangles exposes its endpoints.
    let mut edge_counts: std::collections::HashMap<(u32, u32), u8> =
        std::collections::HashMap::new();
    for slot in 0..slots {
        let Some(tri) = mesh.triangle(slot as TriangleId) else {
            continue;
        };
        for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            let key = if a <= b { (a, b) } else { (b, a) };
            let entry = edge_counts.entry(key).or_insert(0);
            *entry = entry.saturating_add(1);
        }
    }
    let mut mesh_border_vert = vec![false; v_slots];
    for ((a, b), count) in edge_counts {
        if count < 2 {
            mesh_border_vert[a as usize] = true;
            mesh_border_vert[b as usize] = true;
        }
    }

    let mut cur = sel.clone();
    let mut border_vert = vec![false; v_slots];

    for _ in 0..rings {
        border_vert.copy_from_slice(&mesh_border_vert);
        for slot in 0..slots {
            let t = slot as TriangleId;
            let Some(tri) = mesh.triangle(t) else {
                continue;
            };
            if !cur.contains(t) {
                border_vert[tri[0] as usize] = true;
                border_vert[tri[1] as usize] = true;
                border_vert[tri[2] as usize] = true;
            }
        }
        let mut to_remove = Vec::new();
        for t in cur.to_triangles(mesh) {
            let Some(tri) = mesh.triangle(t) else {
                continue;
            };
            if border_vert[tri[0] as usize]
                || border_vert[tri[1] as usize]
                || border_vert[tri[2] as usize]
            {
                to_remove.push(t);
            }
        }
        if to_remove.is_empty() {
            break;
        }
        for t in to_remove {
            cur.remove(t);
        }
        if cur.is_empty() {
            break;
        }
    }
    Ok(cur)
}

fn stamp_mesh_with_scale_and_origin(entry: &crate::scene::Entry) -> Result<Mesh> {
    let scale = entry.placement.scale;
    let origin = entry.mesh.origin();
    let m = &entry.mesh;
    let mut flat_pos = Vec::new();
    flat_pos
        .try_reserve(m.vertex_count().saturating_mul(3))
        .map_err(|_| StoreError::OutOfMemory)?;
    let mut remap = vec![u32::MAX; m.vertex_slots()];
    let mut flat_tris = Vec::new();
    flat_tris
        .try_reserve(m.triangle_count().saturating_mul(3))
        .map_err(|_| StoreError::OutOfMemory)?;
    let pos = m.positions();
    for t in 0..m.triangle_slots() as TriangleId {
        let Some(tri) = m.triangle(t) else {
            continue;
        };
        for v in tri {
            let idx = if remap[v as usize] != u32::MAX {
                remap[v as usize]
            } else {
                let next =
                    u32::try_from(flat_pos.len() / 3).map_err(|_| StoreError::OutOfMemory)?;
                let p = pos[v as usize];
                for k in 0..3 {
                    let coord = scale[k] * (origin[k] + f64::from(p[k]));
                    if !coord.is_finite() {
                        return Err(SceneError::InvalidInput("stamp vertex coordinate overflow"));
                    }
                    flat_pos.push(coord);
                }
                remap[v as usize] = next;
                next
            };
            flat_tris.push(idx);
        }
    }
    Ok(Mesh::from_buffers_f64_with_origin(
        &flat_pos,
        &flat_tris,
        [0.0, 0.0, 0.0],
    )?)
}

impl Scene {
    // -----------------------------------------------------------------------
    // #233 Face Selection
    // -----------------------------------------------------------------------

    /// Selects triangles on mesh `id` lying inside (or touching) a 2D screen-space
    /// rectangle or lasso polygon, using world camera/projection options.
    pub fn select_faces_screen(
        &mut self,
        id: MeshId,
        shape: &ScreenSelectionShape,
        world_opts: &ScreenSelectOptions,
    ) -> Result<Vec<u32>> {
        self.drop_field_unit();
        self.entry(id)?;
        self.prepare(Some(&[id]))?;

        for col in &world_opts.view_proj {
            if !col.iter().all(|c| c.is_finite()) {
                return Err(SceneError::Ops(OpsError::InvalidInput(
                    "view_proj must have finite entries",
                )));
            }
        }
        let cam_origin_world = world_opts.camera_origin.map(f64::from);
        finite_vec3(cam_origin_world, "camera_origin must be finite")?;
        let cam_forward_world = world_opts.camera_forward.map(f64::from);
        let _ = unit3(cam_forward_world, "camera_forward must be non-zero finite")?;

        let entry = self.entry(id)?;
        let model = model_matrix_stored_to_world(&entry.placement, entry.mesh.origin());
        let vp_world_f64 =
            std::array::from_fn(|c| std::array::from_fn(|r| f64::from(world_opts.view_proj[c][r])));
        let vp_local_f64 = mat4_mul_f64(&vp_world_f64, &model);
        let view_proj_local =
            std::array::from_fn(|c| std::array::from_fn(|r| vp_local_f64[c][r] as f32));
        for col in &view_proj_local {
            if !col.iter().all(|c| c.is_finite()) {
                return Err(SceneError::Ops(OpsError::InvalidInput(
                    "composed view_proj overflows f32",
                )));
            }
        }

        let cam_origin_local = vec3_to_f32(
            world_to_local(&entry.placement, entry.mesh.origin(), cam_origin_world),
            "camera_origin in mesh space overflows f32",
        )?;
        let cam_fwd_local = unit3(
            entry.placement.inverse_direction(cam_forward_world),
            "camera_forward in mesh space is invalid",
        )?;
        let local_opts = ScreenSelectOptions {
            view_proj: view_proj_local,
            camera_origin: cam_origin_local,
            camera_forward: vec3_to_f32(cam_fwd_local, "camera_forward overflows f32")?,
            is_orthographic: world_opts.is_orthographic,
            visible_only: world_opts.visible_only,
            reject_backfaces: world_opts.reject_backfaces,
            include_touching: world_opts.include_touching,
        };

        let bvh = if world_opts.visible_only {
            Some(&entry.bvh)
        } else {
            None
        };
        let sel = select_faces_screen(&entry.mesh, bvh, shape, &local_opts)?;
        Ok(sel.to_triangles(&entry.mesh))
    }

    /// Selects triangles on mesh `id` intersecting the world sphere `(world_center, world_radius)`,
    /// optionally constrained by `world_normal_cone = Some((world_axis, max_angle_rad))`.
    pub fn select_faces_in_sphere_world(
        &mut self,
        id: MeshId,
        world_center: Vec3,
        world_radius: f64,
        world_normal_cone: Option<(Vec3, f64)>,
    ) -> Result<Vec<u32>> {
        self.drop_field_unit();
        self.entry(id)?;
        finite_vec3(world_center, "sphere center must be finite")?;
        if !world_radius.is_finite() || world_radius < 0.0 {
            return Err(SceneError::Ops(OpsError::InvalidInput(
                "sphere radius must be finite and non-negative",
            )));
        }
        if let Some((axis, max_angle_rad)) = world_normal_cone {
            let _ = unit3(axis, "normal_cone axis must be non-zero finite")?;
            if !max_angle_rad.is_finite() || !(0.0..=PI).contains(&max_angle_rad) {
                return Err(SceneError::Ops(OpsError::InvalidInput(
                    "normal_cone max angle must be finite radians in [0, pi]",
                )));
            }
        }

        self.prepare(Some(&[id]))?;
        let entry = self.entry(id)?;
        if let Some(scale) = entry.placement.uniform_scale() {
            let local_center = vec3_to_f32(
                world_to_local(&entry.placement, entry.mesh.origin(), world_center),
                "sphere center in local space overflows f32",
            )?;
            let local_radius = (world_radius / scale) as f32;
            if !local_radius.is_finite() {
                return Err(SceneError::Ops(OpsError::InvalidInput(
                    "sphere radius in local space overflows f32",
                )));
            }
            let local_cone = match world_normal_cone {
                Some((axis, max_angle_rad)) => {
                    let local_axis = unit3(
                        entry.placement.inverse_direction(axis),
                        "normal_cone axis invalid in local space",
                    )?;
                    let deg = (max_angle_rad.to_degrees() as f32).clamp(0.0, 180.0);
                    Some((
                        vec3_to_f32(local_axis, "normal_cone axis overflows f32")?,
                        deg,
                    ))
                }
                None => None,
            };
            let sel = select_faces_in_sphere(
                &entry.mesh,
                Some(&entry.bvh),
                local_center,
                local_radius,
                local_cone,
            )?;
            Ok(sel.to_triangles(&entry.mesh))
        } else {
            let (baked, baked_bvh) = baked_world_mesh(entry)?;
            let world_cone = match world_normal_cone {
                Some((axis, max_angle_rad)) => {
                    let deg = (max_angle_rad.to_degrees() as f32).clamp(0.0, 180.0);
                    Some((vec3_to_f32(axis, "normal_cone axis overflows f32")?, deg))
                }
                None => None,
            };
            let sel = select_faces_in_sphere(
                &baked,
                Some(&baked_bvh),
                vec3_to_f32(world_center, "sphere center overflows f32")?,
                world_radius as f32,
                world_cone,
            )?;
            Ok(sel.to_triangles(&entry.mesh))
        }
    }

    /// Flood-selects triangles edge-connected to `seed_tri` while adjacent dihedral angle
    /// remains within `max_dihedral_rad` radians in world space.
    pub fn select_by_crease_angle(
        &self,
        id: MeshId,
        seed_tri: TriangleId,
        max_dihedral_rad: f64,
    ) -> Result<Vec<u32>> {
        let entry = self.entry(id)?;
        if !max_dihedral_rad.is_finite() || !(0.0..=PI).contains(&max_dihedral_rad) {
            return Err(SceneError::Ops(OpsError::InvalidInput(
                "max_dihedral_rad must be finite and within [0, pi]",
            )));
        }
        let max_deg = (max_dihedral_rad.to_degrees() as f32).clamp(0.0, 180.0);
        if entry.placement.uniform_scale().is_some() {
            let sel = select_by_crease_angle(&entry.mesh, &[seed_tri], max_deg)?;
            Ok(sel.to_triangles(&entry.mesh))
        } else {
            let (baked, _) = baked_world_mesh(entry)?;
            let sel = select_by_crease_angle(&baked, &[seed_tri], max_deg)?;
            Ok(sel.to_triangles(&entry.mesh))
        }
    }

    /// Selects the entire edge-connected shell containing `seed_tri` on mesh `id`.
    pub fn select_connected_shell(&self, id: MeshId, seed_tri: TriangleId) -> Result<Vec<u32>> {
        let entry = self.entry(id)?;
        let sel = select_connected_shell(&entry.mesh, &[seed_tri])?;
        Ok(sel.to_triangles(&entry.mesh))
    }

    /// Selects edge-connected face islands with at most `max_face_count` triangles and optional
    /// world surface area limit `max_area`.
    pub fn select_small_face_islands(
        &self,
        id: MeshId,
        max_face_count: usize,
        max_area: Option<f64>,
    ) -> Result<Vec<u32>> {
        let entry = self.entry(id)?;
        if let Some(area) = max_area
            && (!area.is_finite() || area < 0.0)
        {
            return Err(SceneError::Ops(OpsError::InvalidInput(
                "max_area must be finite and non-negative",
            )));
        }
        if let Some(scale) = entry.placement.uniform_scale() {
            let local_max_area = match max_area {
                Some(a) => (a / (scale * scale)) as f32,
                None => f32::MAX,
            };
            let sel = select_small_face_islands(&entry.mesh, max_face_count, local_max_area)?;
            Ok(sel.to_triangles(&entry.mesh))
        } else {
            let (baked, _) = baked_world_mesh(entry)?;
            let area = max_area.map_or(f32::MAX, |a| a as f32);
            let sel = select_small_face_islands(&baked, max_face_count, area)?;
            Ok(sel.to_triangles(&entry.mesh))
        }
    }

    /// Grows `selected_tris` outward by `rings` steps across `adjacency`.
    pub fn grow_face_selection(
        &self,
        id: MeshId,
        selected_tris: &[u32],
        rings: usize,
        adjacency: AdjacencyKind,
    ) -> Result<Vec<u32>> {
        let entry = self.entry(id)?;
        let sel = FaceSelection::from_triangles(&entry.mesh, selected_tris)?;
        let out = grow_face_selection_with(&entry.mesh, &sel, rings, adjacency)?;
        Ok(out.to_triangles(&entry.mesh))
    }

    /// Shrinks `selected_tris` inward by `rings` boundary layers across `adjacency`.
    pub fn shrink_face_selection(
        &self,
        id: MeshId,
        selected_tris: &[u32],
        rings: usize,
        adjacency: AdjacencyKind,
    ) -> Result<Vec<u32>> {
        let entry = self.entry(id)?;
        let sel = FaceSelection::from_triangles(&entry.mesh, selected_tris)?;
        let out = match adjacency {
            AdjacencyKind::Edge => shrink_face_selection(&entry.mesh, &sel, rings)?,
            AdjacencyKind::Vertex => shrink_vertex_adjacent(&entry.mesh, &sel, rings)?,
        };
        Ok(out.to_triangles(&entry.mesh))
    }

    /// Fills enclosed unselected interior face pockets up to `max_hole_tris` triangles.
    pub fn fill_selection_holes(
        &self,
        id: MeshId,
        selected_tris: &[u32],
        max_hole_tris: usize,
    ) -> Result<Vec<u32>> {
        let entry = self.entry(id)?;
        let sel = FaceSelection::from_triangles(&entry.mesh, selected_tris)?;
        let out = fill_selection_holes(&entry.mesh, &sel, max_hole_tris)?;
        Ok(out.to_triangles(&entry.mesh))
    }

    /// Smooths the border of `selected_tris` via `iterations` majority-voting passes.
    pub fn smooth_selection_boundary(
        &self,
        id: MeshId,
        selected_tris: &[u32],
        iterations: usize,
    ) -> Result<Vec<u32>> {
        let entry = self.entry(id)?;
        let sel = FaceSelection::from_triangles(&entry.mesh, selected_tris)?;
        let out = smooth_selection_boundary(&entry.mesh, &sel, iterations)?;
        Ok(out.to_triangles(&entry.mesh))
    }

    /// Converts `selected_tris` to a per-vertex-slot weight mask in `[0.0, 1.0]`
    /// (length `mesh.vertex_slots()`).
    pub fn face_selection_to_vertex_mask(
        &self,
        id: MeshId,
        selected_tris: &[u32],
    ) -> Result<Vec<f32>> {
        let entry = self.entry(id)?;
        let sel = FaceSelection::from_triangles(&entry.mesh, selected_tris)?;
        let pairs = face_selection_to_vertex_mask(&entry.mesh, &sel, 0)?;
        let mut mask = vec![0.0_f32; entry.mesh.vertex_slots()];
        for (v, w) in pairs {
            mask[v as usize] = w;
        }
        Ok(mask)
    }

    /// Converts a dense per-vertex-slot `mask` (`[0.0, 1.0]`, length `mesh.vertex_slots()`)
    /// into sorted selected triangle indices.
    pub fn vertex_mask_to_face_selection(
        &self,
        id: MeshId,
        mask: &[f32],
        threshold: f32,
        require_all_vertices: bool,
    ) -> Result<Vec<u32>> {
        let entry = self.entry(id)?;
        let mesh = &entry.mesh;
        if mask.len() != mesh.vertex_slots() {
            return Err(SceneError::Ops(OpsError::InvalidInput(
                "vertex mask length must match vertex_slots()",
            )));
        }
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(SceneError::Ops(OpsError::InvalidInput(
                "threshold must be finite and within [0, 1]",
            )));
        }
        if require_all_vertices {
            let mut pairs = Vec::with_capacity(mesh.vertex_count());
            for (slot, &w) in mask.iter().enumerate() {
                let v = slot as VertexId;
                if mesh.is_vertex_alive(v) {
                    pairs.push((v, w));
                } else if !w.is_finite() {
                    return Err(SceneError::Ops(OpsError::InvalidInput(
                        "vertex mask weight must be finite",
                    )));
                }
            }
            let sel = vertex_mask_to_face_selection(mesh, &pairs, threshold)?;
            return Ok(sel.to_triangles(mesh));
        }

        for (slot, &w) in mask.iter().enumerate() {
            if mesh.is_vertex_alive(slot as VertexId)
                && (!w.is_finite() || !(0.0..=1.0).contains(&w))
            {
                return Err(SceneError::Ops(OpsError::InvalidInput(
                    "vertex weight must be finite and within [0, 1]",
                )));
            }
        }
        let mut out = Vec::new();
        for slot in 0..mesh.triangle_slots() {
            let t = slot as TriangleId;
            let Some(tri) = mesh.triangle(t) else {
                continue;
            };
            let max_w = mask[tri[0] as usize]
                .max(mask[tri[1] as usize])
                .max(mask[tri[2] as usize]);
            let selected = if threshold == 0.0 {
                max_w > 0.0
            } else {
                max_w >= threshold
            };
            if selected {
                out.push(t);
            }
        }
        Ok(out)
    }

    /// Deletes `selected_tris` and orphaned vertices from `id` in one undoable step.
    pub fn delete_selected_faces(
        &mut self,
        id: MeshId,
        selected_tris: &[u32],
    ) -> Result<FaceEditReport> {
        self.drop_field_unit();
        let sel = {
            let entry = self.entry(id)?;
            FaceSelection::from_triangles(&entry.mesh, selected_tris)?
        };
        self.edit(id, |tx| delete_selected_faces(tx, &sel))
    }

    /// Extracts `selected_tris` from `id` as a compact new scene mesh preserving exact world coordinates.
    pub fn extract_selected_faces(&mut self, id: MeshId, selected_tris: &[u32]) -> Result<MeshId> {
        self.drop_field_unit();
        let (extracted, placement, mode) = {
            let entry = self.entry(id)?;
            let sel = FaceSelection::from_triangles(&entry.mesh, selected_tris)?;
            let ext = extract_selected_faces(&entry.mesh, &sel)?;
            (ext, entry.placement, entry.mode)
        };
        self.begin_step();
        let res = self.add_mesh(
            extracted,
            MeshOptions {
                placement,
                mode,
                ..MeshOptions::default()
            },
        );
        match res {
            Ok(new_id) => {
                self.end_step()?;
                Ok(new_id)
            }
            Err(err) => {
                let _ = self.cancel_step();
                Err(err)
            }
        }
    }

    /// Splits `selected_tris` out of `id` into a new mesh at the exact matching world pose,
    /// recording `id`'s edit in one undoable step.
    pub fn split_selected_faces(
        &mut self,
        id: MeshId,
        selected_tris: &[u32],
    ) -> Result<(MeshId, FaceEditReport)> {
        self.drop_field_unit();
        let (sel, placement, mode) = {
            let entry = self.entry(id)?;
            let sel = FaceSelection::from_triangles(&entry.mesh, selected_tris)?;
            (sel, entry.placement, entry.mode)
        };
        self.begin_step();
        let edit_res = self.edit(id, |tx| split_selected_faces(tx, &sel));
        let (detached, report) = match edit_res {
            Ok(v) => v,
            Err(err) => {
                let _ = self.cancel_step();
                return Err(err);
            }
        };
        let add_res = self.add_mesh(
            detached,
            MeshOptions {
                placement,
                mode,
                ..MeshOptions::default()
            },
        );
        let new_id = match add_res {
            Ok(v) => v,
            Err(err) => {
                let _ = self.cancel_step();
                return Err(err);
            }
        };
        self.end_step()?;
        Ok((new_id, report))
    }

    // -----------------------------------------------------------------------
    // #193 Free-Form Deformation
    // -----------------------------------------------------------------------

    /// Builds an identity-offset FFD control lattice tightly enclosing `id` in mesh-local coordinates.
    pub fn build_ffd_lattice(
        &self,
        id: MeshId,
        dims: [usize; 3],
        basis: FfdBasis,
        padding: f64,
    ) -> Result<FfdLattice> {
        let entry = self.entry(id)?;
        if !padding.is_finite() || padding < 0.0 {
            return Err(SceneError::Ops(OpsError::InvalidInput(
                "FFD padding must be finite and non-negative",
            )));
        }
        Ok(FfdLattice::with_mesh(
            &entry.mesh,
            dims,
            basis,
            padding as f32,
        )?)
    }

    /// Applies an atomic FFD, ARAP, Kelvinlet, or XPBD freeform deformation on `id`.
    pub fn apply_freeform(&mut self, id: MeshId, op: &FreeformOp) -> Result<FreeformReport> {
        self.edit(id, |tx| apply_freeform(tx, op))
    }

    /// Applies a freeform deformation on `id` with protected vertices (frozen with weight `0.0` when
    /// `protection_weights` is `None`, or blended using per-vertex weights in `[0.0, 1.0]`).
    pub fn apply_freeform_protected(
        &mut self,
        id: MeshId,
        op: &FreeformOp,
        protected_vertices: &[u32],
        protection_weights: Option<&[f32]>,
    ) -> Result<FreeformReport> {
        self.drop_field_unit();
        self.entry(id)?;
        let weights_owned: Vec<f32>;
        let weights_slice = match protection_weights {
            Some(w) => w,
            None => {
                weights_owned = vec![0.0_f32; protected_vertices.len()];
                &weights_owned
            }
        };
        self.edit(id, |tx| {
            apply_freeform_protected(tx, op, protected_vertices, weights_slice)
        })
    }

    // -----------------------------------------------------------------------
    // #234 Primitive Fit, Contours & Point Cloud
    // -----------------------------------------------------------------------

    /// Fits an analytical geometric primitive to `id` (or its `subset_tris` when non-empty)
    /// in world coordinates.
    pub fn fit_primitive_world(
        &self,
        id: MeshId,
        subset_tris: Option<&[u32]>,
        kind: Option<PrimitiveKind>,
        options: &FitPrimitiveOptions,
    ) -> Result<FitPrimitiveReport> {
        let entry = self.entry(id)?;
        let (world_mesh, _) = baked_world_mesh(entry)?;
        let tri_subset = subset_tris.filter(|s| !s.is_empty());
        Ok(fit_primitive_mesh(
            &world_mesh,
            tri_subset,
            kind.unwrap_or(PrimitiveKind::Auto),
            options,
        )?)
    }

    /// Rigidly aligns `id`'s world placement so `primitive`'s datum axis and origin map onto
    /// `(target_axis, target_origin)` in one undoable placement step.
    pub fn align_to_fitted_primitive(
        &mut self,
        id: MeshId,
        primitive: &FittedPrimitive,
        target_axis: Vec3,
        target_origin: Vec3,
    ) -> Result<Placement> {
        self.drop_field_unit();
        let old = self.entry(id)?.placement;
        let rigid = datum_alignment_rigid(primitive, target_axis, target_origin)?;
        let next = Placement {
            rotation: orthonormalize(mul_mat3(rigid.rotation, old.rotation)),
            translation: {
                let rotated = mat3_vec3(rigid.rotation, old.translation);
                [
                    rotated[0] + rigid.translation[0],
                    rotated[1] + rigid.translation[1],
                    rotated[2] + rigid.translation[2],
                ]
            },
            scale: old.scale,
        };
        self.set_placement(id, next)?;
        Ok(next)
    }

    /// Slices closed and open polyline contours through `id` on a world plane `(origin, normal)`.
    pub fn slice_contours_world(
        &self,
        id: MeshId,
        origin: Vec3,
        normal: Vec3,
    ) -> Result<ContourSlice> {
        let entry = self.entry(id)?;
        let o32 = vec3_to_f32(origin, "contour slice origin must be finite")?;
        let n32 = vec3_to_f32(normal, "contour slice normal must be finite")?;
        let (world_mesh, _) = baked_world_mesh(entry)?;
        Ok(slice_contours(&world_mesh, o32, n32)?)
    }

    /// Slices `count` parallel contour layers starting at `(origin, normal)` separated by world `spacing`.
    pub fn slice_contour_stack_world(
        &self,
        id: MeshId,
        origin: Vec3,
        normal: Vec3,
        spacing: f64,
        count: usize,
    ) -> Result<Vec<ContourSlice>> {
        let entry = self.entry(id)?;
        let o32 = vec3_to_f32(origin, "contour slice origin must be finite")?;
        let n32 = vec3_to_f32(normal, "contour slice normal must be finite")?;
        if !spacing.is_finite() {
            return Err(SceneError::Ops(OpsError::InvalidInput(
                "contour stack spacing must be finite",
            )));
        }
        let (world_mesh, _) = baked_world_mesh(entry)?;
        Ok(slice_contour_stack(
            &world_mesh,
            o32,
            n32,
            spacing as f32,
            count,
        )?)
    }

    /// Reconstructs a surface mesh from `cloud` and adds it to the scene at identity placement.
    pub fn reconstruct_point_cloud(
        &mut self,
        cloud: &PointCloud,
        options: &SurfaceReconstructOptions,
    ) -> Result<MeshId> {
        let mesh = reconstruct_surface(cloud, options)?;
        self.add_mesh(mesh, MeshOptions::default())
    }

    // -----------------------------------------------------------------------
    // #236 Hollow, Drain Holes & Place Mesh
    // -----------------------------------------------------------------------

    /// Hollows solid mesh `id` into an inner cavity shell in one undoable edit step.
    pub fn hollow_mesh(&mut self, id: MeshId, options: &HollowOptions) -> Result<HollowReport> {
        self.edit(id, |tx| hollow_mesh(tx, options))
    }

    /// Punches a cylindrical/tapered drain hole specified in world space into `id`, optionally
    /// adding generated plug meshes positioned inside the hole in world space.
    pub fn punch_drain_hole_world(
        &mut self,
        id: MeshId,
        world_spec: &DrainHoleSpec,
        options: &DrainHoleOptions,
    ) -> Result<(DrainHoleReport, Vec<MeshId>)> {
        self.drop_field_unit();
        let (placement, origin, mode, uniform_scale) = {
            let entry = self.entry(id)?;
            (
                entry.placement,
                entry.mesh.origin(),
                entry.mode,
                entry.placement.uniform_scale(),
            )
        };

        self.begin_step();
        let outcome = if let Some(s) = uniform_scale {
            let local_pt = vec3_to_f32(
                world_to_local(&placement, origin, world_spec.surface_point.map(f64::from)),
                "drain hole point in local space overflows f32",
            )?;
            let local_n = vec3_to_f32(
                placement.normal_to_local(world_spec.surface_normal.map(f64::from)),
                "drain hole normal in local space overflows f32",
            )?;
            let local_spec = DrainHoleSpec {
                surface_point: local_pt,
                surface_normal: local_n,
                diameter: (f64::from(world_spec.diameter) / s) as f32,
                depth: world_spec.depth.map(|d| (f64::from(d) / s) as f32),
                outer_chamfer: (f64::from(world_spec.outer_chamfer) / s) as f32,
                taper_angle_deg: world_spec.taper_angle_deg,
                segments: world_spec.segments,
            };
            let local_opts = DrainHoleOptions {
                plug_clearance: (f64::from(options.plug_clearance) / s) as f32,
                ..*options
            };
            let edit_res = self.edit(id, |tx| punch_drain_hole(tx, &local_spec, &local_opts));
            let plug_placement = world_pose_for_local_origin_mesh(&placement, origin);
            edit_res.map(|rep| (rep, plug_placement))
        } else {
            let world_run = (|| -> Result<(Mesh, DrainHoleReport)> {
                let (mut baked, _) = baked_world_mesh(self.entry(id)?)?;
                let mut tx = baked.begin_edit();
                let rep = punch_drain_hole(&mut tx, world_spec, options)?;
                tx.commit()?;
                Ok((baked, rep))
            })();
            match world_run {
                Ok((baked_res, rep)) => {
                    match self.edit(id, |tx| {
                        replace_tx_world_to_local(tx, &baked_res, &placement, origin)
                    }) {
                        Ok(()) => Ok((rep, Placement::IDENTITY)),
                        Err(e) => Err(e),
                    }
                }
                Err(e) => Err(e),
            }
        };

        let (mut report, plug_placement) = match outcome {
            Ok(v) => v,
            Err(err) => {
                let _ = self.cancel_step();
                return Err(err);
            }
        };

        let mut plug_ids = Vec::with_capacity(report.plug_meshes.len());
        let plugs = std::mem::take(&mut report.plug_meshes);
        for plug in plugs {
            match self.add_mesh(
                plug,
                MeshOptions {
                    placement: plug_placement,
                    mode,
                    ..MeshOptions::default()
                },
            ) {
                Ok(pid) => plug_ids.push(pid),
                Err(err) => {
                    let _ = self.cancel_step();
                    return Err(err);
                }
            }
        }
        self.end_step()?;
        Ok((report, plug_ids))
    }

    /// Places `stamp` relative to `target` at world `(hit_point_world, hit_normal_world)` in one
    /// undoable step. Returns `(report, Some(new_id))` in `PlaceMode::KeepSeparate` or
    /// `(report, None)` for CSG / shell merge modes.
    pub fn place_mesh_world(
        &mut self,
        target: MeshId,
        stamp: MeshId,
        hit_point_world: Vec3,
        hit_normal_world: Vec3,
        params: &PlaceMeshParams,
    ) -> Result<(PlaceMeshReport, Option<MeshId>)> {
        self.drop_field_unit();
        self.entry(target)?;
        let (target_placement, target_origin, target_mode, target_tris, stamp_mesh) = {
            let t_entry = self.entry(target)?;
            let s_entry = self.entry(stamp)?;
            let sm = stamp_mesh_with_scale_and_origin(s_entry)?;
            (
                t_entry.placement,
                t_entry.mesh.origin(),
                t_entry.mode,
                t_entry.mesh.triangle_count(),
                sm,
            )
        };

        let hp32 = vec3_to_f32(hit_point_world, "hit_point_world must be finite")?;
        let hn32 = vec3_to_f32(hit_normal_world, "hit_normal_world must be finite")?;
        let world_params = PlaceMeshParams {
            hit_point: hp32,
            hit_normal: hn32,
            ..*params
        };
        let world_transform = compute_place_transform(&world_params)?;

        if params.mode == PlaceMode::KeepSeparate {
            let mut report = if params.conform_to_surface {
                let (mut world_target, _) = baked_world_mesh(self.entry(target)?)?;
                let mut tx = world_target.begin_edit();
                let rep = place_mesh(&mut tx, &stamp_mesh, &world_params)?;
                tx.commit()?;
                rep
            } else {
                let mut empty = Mesh::new();
                let mut tx = empty.begin_edit();
                let mut rep = place_mesh(&mut tx, &stamp_mesh, &world_params)?;
                tx.commit()?;
                rep.kept_triangles = target_tris;
                rep
            };
            report.transform = world_transform;
            let placed = report.placed_mesh.take().ok_or(SceneError::Corrupted)?;
            let new_id = self.add_mesh(
                placed,
                MeshOptions {
                    placement: Placement::IDENTITY,
                    mode: target_mode,
                    ..MeshOptions::default()
                },
            )?;
            return Ok((report, Some(new_id)));
        }

        if let Some(s_t) = target_placement.uniform_scale() {
            let local_hp = vec3_to_f32(
                world_to_local(&target_placement, target_origin, hit_point_world),
                "hit_point in target space overflows f32",
            )?;
            let local_hn = vec3_to_f32(
                target_placement.normal_to_local(hit_normal_world),
                "hit_normal in target space overflows f32",
            )?;
            let local_params = PlaceMeshParams {
                hit_point: local_hp,
                hit_normal: local_hn,
                uniform_scale: (f64::from(params.uniform_scale) / s_t) as f32,
                embed_depth: (f64::from(params.embed_depth) / s_t) as f32,
                ..*params
            };
            let mut rep = self.edit(target, |tx| place_mesh(tx, &stamp_mesh, &local_params))?;
            rep.transform = world_transform;
            return Ok((rep, None));
        }

        let (mut baked_target, _) = baked_world_mesh(self.entry(target)?)?;
        let mut rep = {
            let mut tx = baked_target.begin_edit();
            let rep = place_mesh(&mut tx, &stamp_mesh, &world_params)?;
            tx.commit()?;
            rep
        };
        self.edit(target, |tx| {
            replace_tx_world_to_local(tx, &baked_target, &target_placement, target_origin)
        })?;
        rep.transform = world_transform;
        Ok((rep, None))
    }
}
