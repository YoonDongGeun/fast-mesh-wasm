//! Brush dabs whose cost follows the brush region, not the mesh.

use fast_mesh_ops::OpsError;
use fast_mesh_ops::protect::{check_protection, with_protection_band};
use fast_mesh_ops::sculpt::{
    Brush, ClayOptions, ClayProfile, CreaseOptions, FlattenOptions, GrabOptions, InflateOptions,
    STROKE_POS_CHANNEL, SmoothOptions, SurfaceFilter, Symmetry, TwistOptions, clay, clay_region,
    crease, crease_region, flatten, flatten_region, grab, grab_region_with_stroke,
    grab_widest_reach, inflate, inflate_region_with_stroke, smooth, smooth_region, twist,
    twist_region,
};
use fast_mesh_query::{Region, vertices_in};
use fast_mesh_store::{StoreError, VertexId};

use crate::history::Change;
use crate::scene::Entry;
use crate::{MeshId, Result, Scene, SceneError, UpdateMode};

/// One brush dab for [`Scene::dab`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Dab {
    /// [`inflate`](fast_mesh_ops::sculpt::inflate): along vertex, average, or stroke normals.
    Inflate(InflateOptions),
    /// [`smooth`](fast_mesh_ops::sculpt::smooth): Taubin, Laplacian, HC-Laplacian, or Cotangent relaxation.
    Smooth(SmoothOptions),
    /// [`grab`](fast_mesh_ops::sculpt::grab): sphere, geodesic, or Kelvinlet grab.
    Grab(GrabOptions),
    /// [`flatten`](fast_mesh_ops::sculpt::flatten): flatten, scrape peaks, or fill valleys.
    Flatten(FlattenOptions),
    /// [`crease`](fast_mesh_ops::sculpt::crease): sharp groove carving and/or tangent pinch.
    Crease(CreaseOptions),
    /// [`clay`](fast_mesh_ops::sculpt::clay): round clay buildup or square clay strips.
    Clay(ClayOptions),
    /// [`twist`](fast_mesh_ops::sculpt::twist): vortex or rotational Kelvinlet twist.
    Twist(TwistOptions),
}

fn brush_valid(brush: &Brush) -> bool {
    brush.radius.is_finite()
        && brush.radius > 0.0
        && brush.center.iter().all(|c| c.is_finite())
        && brush.strength.is_finite()
}

fn nonzero_finite_vec(v: [f32; 3]) -> bool {
    if v.iter().any(|x| !x.is_finite()) {
        return false;
    }
    let len = (f64::from(v[0]).powi(2) + f64::from(v[1]).powi(2) + f64::from(v[2]).powi(2)).sqrt();
    len > 0.0 && len.is_finite()
}

fn filter_valid(filter: &SurfaceFilter) -> bool {
    match filter {
        SurfaceFilter::None => true,
        SurfaceFilter::PeaksOnly {
            threshold,
            contrast,
        }
        | SurfaceFilter::ValleysOnly {
            threshold,
            contrast,
        } => (0.0..=1.0).contains(threshold) && (0.0..=1.0).contains(contrast),
        SurfaceFilter::NormalAngle {
            normal,
            max_angle_deg,
        } => {
            *max_angle_deg > 0.0 && *max_angle_deg <= 180.0 && normal.is_none_or(nonzero_finite_vec)
        }
    }
}

fn reflect_vec(bits: u8, v: [f32; 3]) -> [f32; 3] {
    [
        if bits & 1 != 0 { -v[0] } else { v[0] },
        if bits & 2 != 0 { -v[1] } else { v[1] },
        if bits & 4 != 0 { -v[2] } else { v[2] },
    ]
}

impl Dab {
    fn brush(&self) -> &Brush {
        match self {
            Dab::Inflate(o) => &o.brush,
            Dab::Smooth(o) => &o.brush,
            Dab::Grab(o) => &o.brush,
            Dab::Flatten(o) => &o.brush,
            Dab::Crease(o) => &o.brush,
            Dab::Clay(o) => &o.brush,
            Dab::Twist(o) => &o.brush,
        }
    }

    fn symmetry(&self) -> Symmetry {
        match self {
            Dab::Inflate(o) => o.symmetry,
            Dab::Smooth(o) => o.symmetry,
            Dab::Grab(o) => o.symmetry,
            Dab::Flatten(o) => o.symmetry,
            Dab::Crease(o) => o.symmetry,
            Dab::Clay(o) => o.symmetry,
            Dab::Twist(o) => o.symmetry,
        }
    }

    fn query_radius(&self) -> f32 {
        match self {
            Dab::Clay(o) if o.profile == ClayProfile::Strips => o.brush.radius * 3.0_f32.sqrt(),
            Dab::Grab(o) if o.adaptive_reach => grab_widest_reach(&o.brush, o.direction, o.method),
            _ => self.brush().radius,
        }
    }

    fn is_valid(&self) -> bool {
        match self {
            Dab::Inflate(o) => brush_valid(&o.brush) && filter_valid(&o.filter),
            Dab::Smooth(o) => brush_valid(&o.brush) && filter_valid(&o.filter),
            Dab::Grab(o) => {
                brush_valid(&o.brush)
                    && o.direction.iter().all(|c| c.is_finite())
                    && (0.0..=0.5).contains(&o.poisson)
                    && filter_valid(&o.filter)
            }
            Dab::Flatten(o) => {
                brush_valid(&o.brush)
                    && o.plane_origin
                        .is_none_or(|p| p.iter().all(|x| x.is_finite()))
                    && o.plane_normal.is_none_or(nonzero_finite_vec)
                    && filter_valid(&o.filter)
            }
            Dab::Crease(o) => {
                brush_valid(&o.brush) && (0.0..=1.0).contains(&o.pinch) && filter_valid(&o.filter)
            }
            Dab::Clay(o) => {
                brush_valid(&o.brush)
                    && o.tangent.is_none_or(nonzero_finite_vec)
                    && filter_valid(&o.filter)
            }
            Dab::Twist(o) => {
                brush_valid(&o.brush)
                    && o.axis.is_none_or(nonzero_finite_vec)
                    && filter_valid(&o.filter)
            }
        }
    }
}

/// What [`Scene::dab`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DabReport {
    /// How many vertices moved (displacement above 0).
    pub moved: usize,
    /// The largest displacement applied, in local units; 0 when nothing moved.
    pub max_displacement: f32,
}

/// The ascending candidate vertices under `dab`, through the mesh's BVH, with
/// the incidence index the region sculpt kernels read. `None` when `dab` cannot
/// be searched for (an invalid brush/filter, which the whole-mesh function then
/// reports) or the BVH cannot serve: stale under Manual, or a failed update.
fn region_of(entry: &mut Entry, dab: &Dab) -> Option<Vec<VertexId>> {
    if !dab.is_valid() || (entry.mode == UpdateMode::Manual && !entry.is_current()) {
        return None;
    }
    let grab_from_stroke = matches!(dab, Dab::Grab(o) if o.from_stroke);
    if grab_from_stroke
        && entry.stroke.epoch == 0
        && entry.mesh.channel(STROKE_POS_CHANNEL).is_some()
    {
        return None;
    }
    let q_radius = dab.query_radius();
    if !(q_radius.is_finite() && q_radius > 0.0) {
        return None;
    }
    entry.update_bvh().ok()?;
    entry.mesh.ensure_incidence().ok()?;

    let mut region = Vec::new();
    let (masks, mask_count) = dab.symmetry().active_masks();
    let base_center = dab.brush().center;
    for &bits in &masks[..mask_count] {
        let sphere = Region::Sphere {
            center: reflect_vec(bits, base_center),
            radius: q_radius,
        };
        vertices_in(&entry.mesh, &entry.bvh, &sphere, &mut region).ok()?;
    }

    let union_touched =
        grab_from_stroke && entry.stroke.epoch != 0 && !entry.stroke.touched.is_empty();
    if union_touched {
        region.try_reserve(entry.stroke.touched.len()).ok()?;
        for &v in &entry.stroke.touched {
            if entry.mesh.is_vertex_alive(v) {
                region.push(v);
            }
        }
    }
    if mask_count > 1 || union_touched {
        region.sort_unstable();
        region.dedup();
    }
    Some(region)
}

/// Whether every vertex of the ascending `region` is listed at weight 0.
fn fully_held(
    region: &[VertexId],
    vertices: &[VertexId],
    weights: &[f32],
) -> std::result::Result<bool, StoreError> {
    let mut left = region.len();
    let mut seen: Vec<bool> = Vec::new();
    seen.try_reserve(region.len())
        .map_err(|_| StoreError::OutOfMemory)?;
    seen.resize(region.len(), false);
    for (v, w) in vertices.iter().zip(weights) {
        if *w == 0.0
            && let Ok(k) = region.binary_search(v)
            && !seen[k]
        {
            seen[k] = true;
            left -= 1;
        }
    }
    Ok(left == 0)
}

impl Scene {
    /// Applies one brush dab as one edit: the same history, reports and
    /// failure behaviour as [`Scene::edit`] with the matching
    /// [`sculpt`](fast_mesh_ops::sculpt) function, and the same result, bit
    /// for bit, except that a vertex no triangle uses is out of the brush's
    /// reach. The cost follows the vertices under the brush, plus a walk down
    /// the BVH, however large the mesh is; where the BVH cannot serve (stale
    /// under [`UpdateMode::Manual`]) it scans the mesh instead.
    ///
    /// The first dab on a mesh builds its vertex-to-triangles index, once.
    ///
    /// # Errors
    /// As for [`Scene::edit`] with the sculpt function:
    /// [`SceneError::UnknownMesh`] for a stale id, [`SceneError::Ops`] for an
    /// invalid brush or filter, [`SceneError::Store`] when memory runs out.
    /// Nothing changes in any of these cases.
    pub fn dab(&mut self, id: MeshId, dab: &Dab) -> Result<DabReport> {
        self.dab_protected_band(id, dab, &[], &[], 0.0)
    }

    /// [`Scene::dab`] under [`with_protection_band`] with `band = 0.0`: each
    /// listed vertex's move is scaled by its weight, at a cost that follows
    /// the brush region plus the list.
    ///
    /// A dab with every vertex under its brush held fully (weight 0) moves
    /// nothing, so it edits nothing either: it reports 0 and leaves no undo step.
    ///
    /// # Errors
    /// As for [`Scene::dab`], and as [`check_protection`] for the list.
    /// Nothing changes in any of these cases.
    pub fn dab_protected(
        &mut self,
        id: MeshId,
        dab: &Dab,
        vertices: &[VertexId],
        weights: &[f32],
    ) -> Result<DabReport> {
        self.dab_protected_band(id, dab, vertices, weights, 0.0)
    }

    /// [`Scene::dab_protected`] with a geodesic soft-falloff transition band
    /// of width `band` around `0.0`-weight protected vertices (see
    /// [`with_protection_band`]).
    ///
    /// # Errors
    /// As for [`Scene::dab_protected`], plus [`SceneError::Ops`] when `band`
    /// is negative or non-finite. Nothing changes in any of these cases.
    pub fn dab_protected_band(
        &mut self,
        id: MeshId,
        dab: &Dab,
        vertices: &[VertexId],
        weights: &[f32],
        band: f32,
    ) -> Result<DabReport> {
        self.drop_field_unit();
        let entry = self.entries.get_mut(id).ok_or(SceneError::UnknownMesh)?;
        check_protection(&entry.mesh, vertices, weights)?;
        if !band.is_finite() || band < 0.0 {
            return Err(SceneError::Ops(OpsError::InvalidInput(
                "protection band must be finite and non-negative",
            )));
        }
        let region = region_of(entry, dab);
        if let Some(r) = &region
            && !vertices.is_empty()
            && fully_held(r, vertices, weights)?
        {
            return Ok(DabReport::default());
        }

        if entry.stroke.epoch != 0 {
            match &region {
                Some(r) => entry.stroke.capture_region_with_one_ring(&entry.mesh, r)?,
                None if dab.is_valid() => {
                    entry
                        .stroke
                        .materialize_channels_for_fallback(&mut entry.mesh)?;
                }
                None => {}
            }
        }

        self.history.reserve()?;
        let max_moved = region
            .as_ref()
            .map_or_else(|| entry.mesh.vertex_slots(), Vec::len);
        entry
            .stroke
            .reserve_for_moved(entry.mesh.vertex_slots(), max_moved)?;

        let (moved, max_displacement, patch) = {
            let stroke_slice = entry.stroke.as_slice();
            let mut tx = entry.mesh.begin_edit();
            let (moved, max_displacement) = with_protection_band(
                &mut tx,
                region.as_deref(),
                vertices,
                weights,
                band,
                |tx| -> std::result::Result<(usize, f32), SceneError> {
                    Ok(match (dab, &region) {
                        (Dab::Inflate(o), Some(r)) => {
                            let r = inflate_region_with_stroke(tx, o, r, stroke_slice)?;
                            (r.moved, r.max_displacement)
                        }
                        (Dab::Inflate(o), None) => {
                            let r = inflate(tx, o)?;
                            (r.moved, r.max_displacement)
                        }
                        (Dab::Smooth(o), Some(r)) => {
                            let r = smooth_region(tx, o, r)?;
                            (r.moved, r.max_displacement)
                        }
                        (Dab::Smooth(o), None) => {
                            let r = smooth(tx, o)?;
                            (r.moved, r.max_displacement)
                        }
                        (Dab::Grab(o), Some(r)) => {
                            let r = grab_region_with_stroke(tx, o, r, stroke_slice, &[])?;
                            (r.moved, r.max_displacement)
                        }
                        (Dab::Grab(o), None) => {
                            let r = grab(tx, o)?;
                            (r.moved, r.max_displacement)
                        }
                        (Dab::Flatten(o), Some(r)) => {
                            let r = flatten_region(tx, o, r)?;
                            (r.moved, r.max_displacement)
                        }
                        (Dab::Flatten(o), None) => {
                            let r = flatten(tx, o)?;
                            (r.moved, r.max_displacement)
                        }
                        (Dab::Crease(o), Some(r)) => {
                            let r = crease_region(tx, o, r)?;
                            (r.moved, r.max_displacement)
                        }
                        (Dab::Crease(o), None) => {
                            let r = crease(tx, o)?;
                            (r.moved, r.max_displacement)
                        }
                        (Dab::Clay(o), Some(r)) => {
                            let r = clay_region(tx, o, r)?;
                            (r.moved, r.max_displacement)
                        }
                        (Dab::Clay(o), None) => {
                            let r = clay(tx, o)?;
                            (r.moved, r.max_displacement)
                        }
                        (Dab::Twist(o), Some(r)) => {
                            let r = twist_region(tx, o, r)?;
                            (r.moved, r.max_displacement)
                        }
                        (Dab::Twist(o), None) => {
                            let r = twist(tx, o)?;
                            (r.moved, r.max_displacement)
                        }
                    })
                },
            )?;
            let patch = tx.commit()?;
            (moved, max_displacement, patch)
        };

        if patch.is_empty() {
            return Ok(DabReport::default());
        }
        entry
            .stroke
            .record_moved_reserved(&patch.changes().vertices_moved);
        entry.record_changes(patch.changes());
        self.history
            .record(&mut self.entries, Change::Edit(id, patch));
        if let Some(entry) = self.entries.get_mut(id) {
            entry.after_change();
        }
        self.run_eager(Some(&[id]));
        Ok(DabReport {
            moved,
            max_displacement,
        })
    }

    /// Returns the strictly ascending live triangle-used [`VertexId`]s of mesh
    /// `id` inside the closed local-space sphere `(center, radius)` via the
    /// mesh's BVH.
    ///
    /// Brings a [`UpdateMode::Lazy`] BVH up to date first; refuses a stale
    /// [`UpdateMode::Manual`] BVH with [`SceneError::StaleBvh`].
    ///
    /// # Errors
    /// - [`SceneError::UnknownMesh`] for a stale id.
    /// - [`SceneError::Ops`] ([`OpsError::InvalidInput`]) when `center` is
    ///   non-finite or `radius` is negative or non-finite.
    /// - [`SceneError::StaleBvh`] when `id` is in [`UpdateMode::Manual`] with a
    ///   stale BVH.
    /// - [`SceneError::Store`] / BVH error on out-of-memory.
    pub fn vertices_in_sphere(
        &mut self,
        id: MeshId,
        center: [f32; 3],
        radius: f32,
    ) -> Result<Vec<VertexId>> {
        self.drop_field_unit();
        self.entry(id)?;
        if center.iter().any(|c| !c.is_finite()) {
            return Err(SceneError::Ops(OpsError::InvalidInput(
                "sphere center must be finite",
            )));
        }
        if !radius.is_finite() || radius < 0.0 {
            return Err(SceneError::Ops(OpsError::InvalidInput(
                "sphere radius must be finite and non-negative",
            )));
        }
        self.prepare(Some(&[id]))?;
        let entry = self.entry(id)?;
        let mut out = Vec::new();
        vertices_in(
            &entry.mesh,
            &entry.bvh,
            &Region::Sphere { center, radius },
            &mut out,
        )?;
        Ok(out)
    }
}

#[cfg(test)]
mod protected_tests {
    use fast_mesh_ops::deform::Falloff;
    use fast_mesh_store::Mesh;

    use super::*;
    use crate::{MeshOptions, SceneOptions};

    /// A flat `n x n` grid, with its vertices left of the middle held.
    fn setup(n: u32) -> (Scene, MeshId, Vec<VertexId>, Dab) {
        let mut positions = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                positions.extend([x as f32, y as f32, 0.0]);
            }
        }
        let mut indices = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let a = y * (n + 1) + x;
                indices.extend([a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]);
            }
        }
        let mut scene = Scene::new(SceneOptions::default());
        let id = scene
            .add_mesh(
                Mesh::from_buffers(&positions, &indices).unwrap(),
                MeshOptions::default(),
            )
            .unwrap();
        let local = scene.mesh(id).unwrap().positions().to_vec();
        let middle = local[((n / 2) * (n + 1) + n / 2) as usize];
        let held = local
            .iter()
            .enumerate()
            .filter(|(_, p)| p[0] < middle[0])
            .map(|(v, _)| v as VertexId)
            .collect();
        let dab = Dab::Inflate(InflateOptions {
            brush: Brush {
                center: middle,
                radius: n as f32 / 4.0,
                falloff: Falloff::Smooth,
                strength: 0.5,
            },
            ..Default::default()
        });
        (scene, id, held, dab)
    }

    #[test]
    fn holds_listed_vertices_and_matches_the_transaction_form() {
        let (mut a, id_a, held, dab) = setup(20);
        let weights = vec![0.0; held.len()];
        let before = a.mesh(id_a).unwrap().positions().to_vec();
        let report = a.dab_protected(id_a, &dab, &held, &weights).unwrap();
        assert!(report.moved > 0);
        let after = a.mesh(id_a).unwrap().positions().to_vec();
        for &v in &held {
            let (p, q) = (before[v as usize], after[v as usize]);
            assert_eq!(p.map(f32::to_bits), q.map(f32::to_bits));
        }
        assert_ne!(before, after);
        // The same as the whole-mesh deform under the same protection, bit for bit.
        let (mut b, id_b, _, _) = setup(20);
        let Dab::Inflate(o) = dab else { unreachable!() };
        b.edit(id_b, |tx| {
            fast_mesh_ops::with_protection(tx, &held, &weights, |tx| inflate(tx, &o))
        })
        .unwrap();
        let other = b.mesh(id_b).unwrap().positions().to_vec();
        assert_eq!(after, other);
        // One undo step, restoring everything.
        a.undo().unwrap();
        assert_eq!(a.mesh(id_a).unwrap().positions(), before.as_slice());
    }

    #[test]
    fn a_fully_held_brush_edits_nothing() {
        let (mut scene, id, _, dab) = setup(20);
        let all: Vec<VertexId> = (0..scene.mesh(id).unwrap().vertex_slots() as u32).collect();
        let weights = vec![0.0; all.len()];
        let report = scene.dab_protected(id, &dab, &all, &weights).unwrap();
        assert_eq!(report, DabReport::default());
        assert!(!scene.can_undo());
    }

    #[test]
    fn a_soft_weight_is_not_fully_held() {
        let (mut scene, id, _, dab) = setup(20);
        let all: Vec<VertexId> = (0..scene.mesh(id).unwrap().vertex_slots() as u32).collect();
        let weights = vec![0.5; all.len()];
        assert!(scene.dab_protected(id, &dab, &all, &weights).unwrap().moved > 0);
        assert!(scene.can_undo());
    }

    #[test]
    fn rejects_a_bad_list_and_changes_nothing() {
        let (mut scene, id, _, dab) = setup(4);
        assert!(scene.dab_protected(id, &dab, &[9999], &[0.0]).is_err());
        assert!(scene.dab_protected(id, &dab, &[0], &[2.0]).is_err());
        assert!(scene.dab_protected(id, &dab, &[0], &[]).is_err());
        assert!(!scene.can_undo());
    }
}

#[cfg(test)]
mod tests {
    use fast_mesh_ops::deform::{Falloff, SmoothMethod};
    use fast_mesh_store::Mesh;

    use super::*;
    use crate::{MeshOptions, SceneOptions};

    fn terrain(n: u32) -> Mesh {
        let mut positions = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                let (fx, fy) = (x as f32, y as f32);
                positions.extend([fx, fy, (fx * 0.3).sin() * (fy * 0.2).cos()]);
            }
        }
        let mut indices = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let a = y * (n + 1) + x;
                indices.extend([a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]);
            }
        }
        Mesh::from_buffers(&positions, &indices).unwrap()
    }

    fn scene_with(mode: UpdateMode) -> (Scene, MeshId) {
        let mut scene = Scene::new(SceneOptions::default());
        let options = MeshOptions {
            mode,
            ..MeshOptions::default()
        };
        let id = scene.add_mesh(terrain(24), options).unwrap();
        (scene, id)
    }

    fn brush(center: [f32; 3]) -> Brush {
        Brush {
            center,
            radius: 3.5,
            falloff: Falloff::Smooth,
            strength: 0.4,
        }
    }

    fn dabs() -> Vec<Dab> {
        let mut out = Vec::new();
        for k in 0..12 {
            let b = brush([6.0 + k as f32, 9.0, 0.5]);
            out.push(match k % 3 {
                0 => Dab::Inflate(InflateOptions {
                    brush: b,
                    ..Default::default()
                }),
                1 => Dab::Smooth(SmoothOptions {
                    brush: b,
                    cycles: 2,
                    method: SmoothMethod::Laplacian,
                    ..Default::default()
                }),
                _ => Dab::Grab(GrabOptions {
                    brush: b,
                    direction: [0.2, 0.1, 0.5],
                    ..Default::default()
                }),
            });
        }
        out
    }

    fn whole(scene: &mut Scene, id: MeshId, dab: &Dab) -> DabReport {
        scene
            .edit(id, |tx| -> std::result::Result<_, SceneError> {
                Ok(match dab {
                    Dab::Inflate(o) => {
                        let r = inflate(tx, o)?;
                        DabReport {
                            moved: r.moved,
                            max_displacement: r.max_displacement,
                        }
                    }
                    Dab::Smooth(o) => {
                        let r = smooth(tx, o)?;
                        DabReport {
                            moved: r.moved,
                            max_displacement: r.max_displacement,
                        }
                    }
                    Dab::Grab(o) => {
                        let r = grab(tx, o)?;
                        DabReport {
                            moved: r.moved,
                            max_displacement: r.max_displacement,
                        }
                    }
                    Dab::Flatten(o) => {
                        let r = flatten(tx, o)?;
                        DabReport {
                            moved: r.moved,
                            max_displacement: r.max_displacement,
                        }
                    }
                    Dab::Crease(o) => {
                        let r = crease(tx, o)?;
                        DabReport {
                            moved: r.moved,
                            max_displacement: r.max_displacement,
                        }
                    }
                    Dab::Clay(o) => {
                        let r = clay(tx, o)?;
                        DabReport {
                            moved: r.moved,
                            max_displacement: r.max_displacement,
                        }
                    }
                    Dab::Twist(o) => {
                        let r = twist(tx, o)?;
                        DabReport {
                            moved: r.moved,
                            max_displacement: r.max_displacement,
                        }
                    }
                })
            })
            .unwrap()
    }

    fn bits(scene: &Scene, id: MeshId) -> Vec<[u32; 3]> {
        scene
            .mesh(id)
            .unwrap()
            .positions()
            .iter()
            .map(|p| p.map(f32::to_bits))
            .collect()
    }

    #[test]
    fn dabs_match_the_whole_mesh_deforms_bit_for_bit_in_every_mode() {
        for mode in [UpdateMode::Eager, UpdateMode::Lazy, UpdateMode::Manual] {
            let (mut a, ia) = scene_with(mode);
            let (mut b, ib) = scene_with(mode);
            for dab in dabs() {
                let ra = a.dab(ia, &dab).unwrap();
                let rb = whole(&mut b, ib, &dab);
                assert_eq!(ra, rb, "{mode:?}");
                assert_eq!(bits(&a, ia), bits(&b, ib), "{mode:?}");
            }
        }
    }

    #[test]
    fn a_dab_is_one_undo_step_and_reports_its_change() {
        let (mut scene, id) = scene_with(UpdateMode::Eager);
        let before = bits(&scene, id);
        scene.take_changes();
        let report = scene
            .dab(
                id,
                &Dab::Inflate(InflateOptions {
                    brush: brush([12.0, 12.0, 0.0]),
                    ..Default::default()
                }),
            )
            .unwrap();
        assert!(report.moved > 0);
        assert_eq!(scene.take_changes().len(), 1);
        assert_ne!(bits(&scene, id), before);
        scene.undo().unwrap();
        assert_eq!(bits(&scene, id), before);
        assert!(!scene.can_undo());
    }

    #[test]
    fn dabs_in_an_open_step_undo_together() {
        let (mut scene, id) = scene_with(UpdateMode::Eager);
        let before = bits(&scene, id);
        scene.begin_step();
        for dab in dabs() {
            scene.dab(id, &dab).unwrap();
        }
        scene.end_step().unwrap();
        assert_ne!(bits(&scene, id), before);
        scene.undo().unwrap();
        assert_eq!(bits(&scene, id), before);
    }

    #[test]
    fn an_empty_region_records_nothing_and_a_bad_brush_fails() {
        let (mut scene, id) = scene_with(UpdateMode::Eager);
        let far = Dab::Grab(GrabOptions {
            brush: brush([1000.0, 0.0, 0.0]),
            direction: [0.0, 0.0, 1.0],
            ..Default::default()
        });
        assert_eq!(scene.dab(id, &far).unwrap(), DabReport::default());
        assert!(!scene.can_undo());
        let mut bad = brush([0.0; 3]);
        bad.radius = -1.0;
        assert!(matches!(
            scene.dab(
                id,
                &Dab::Inflate(InflateOptions {
                    brush: bad,
                    ..Default::default()
                })
            ),
            Err(SceneError::Ops(_))
        ));
    }

    #[test]
    fn a_stale_manual_mesh_still_dabs_by_scanning() {
        let (mut scene, id) = scene_with(UpdateMode::Manual);
        let dab = Dab::Inflate(InflateOptions {
            brush: brush([12.0, 12.0, 0.0]),
            ..Default::default()
        });
        scene.dab(id, &dab).unwrap();
        // The BVH is stale now: the next dab cannot use it, and must not fail.
        assert!(scene.dab(id, &dab).unwrap().moved > 0);
    }

    #[test]
    fn a_dab_reaches_vertices_grown_into_the_region_by_earlier_dabs() {
        // Drag a lump far from where the BVH was built, then dab where it is
        // now: the region search must follow the moved surface.
        let (mut scene, id) = scene_with(UpdateMode::Eager);
        let mut at = [12.0, 12.0, 0.0];
        for _ in 0..30 {
            let dab = Dab::Grab(GrabOptions {
                brush: Brush {
                    center: at,
                    radius: 2.0,
                    falloff: Falloff::Smooth,
                    strength: 1.0,
                },
                direction: [0.0, 0.0, 1.0],
                ..Default::default()
            });
            assert!(scene.dab(id, &dab).unwrap().moved > 0);
            at[2] += 1.0;
        }
        let top = scene
            .mesh(id)
            .unwrap()
            .positions()
            .iter()
            .map(|p| p[2])
            .fold(f32::MIN, f32::max);
        assert!(top > 25.0, "the lump followed: {top}");
    }

    /// Mirror-symmetric wavy terrain centered at the origin (`x, y in [-half, half]`).
    fn symmetric_terrain(half: i32) -> Mesh {
        let side = (2 * half + 1) as u32;
        let mut positions = Vec::new();
        for iy in -half..=half {
            for ix in -half..=half {
                let (x, y) = (ix as f32 * 0.5, iy as f32 * 0.5);
                let z = 0.35 * (0.7 * x).sin() * (0.6 * y).cos() + 0.08 * (x * x - y * y) * 0.1;
                positions.extend([x, y, z]);
            }
        }
        let at = |ix: i32, iy: i32| ((iy + half) as u32) * side + (ix + half) as u32;
        let mut indices = Vec::new();
        for iy in -half..half {
            for ix in -half..half {
                let (a, b, c, d) = (
                    at(ix, iy),
                    at(ix + 1, iy),
                    at(ix + 1, iy + 1),
                    at(ix, iy + 1),
                );
                if (ix >= 0) == (iy >= 0) {
                    indices.extend([a, b, c, a, c, d]);
                } else {
                    indices.extend([b, c, d, b, d, a]);
                }
            }
        }
        Mesh::from_buffers(&positions, &indices).unwrap()
    }

    fn apply_whole_dab(scene: &mut Scene, id: MeshId, dab: &Dab) -> DabReport {
        use fast_mesh_ops::sculpt::{clay, crease, flatten, grab, inflate, smooth, twist};
        scene
            .edit(id, |tx| -> std::result::Result<DabReport, SceneError> {
                Ok(match dab {
                    Dab::Inflate(o) => {
                        let r = inflate(tx, o)?;
                        DabReport {
                            moved: r.moved,
                            max_displacement: r.max_displacement,
                        }
                    }
                    Dab::Smooth(o) => {
                        let r = smooth(tx, o)?;
                        DabReport {
                            moved: r.moved,
                            max_displacement: r.max_displacement,
                        }
                    }
                    Dab::Grab(o) => {
                        let r = grab(tx, o)?;
                        DabReport {
                            moved: r.moved,
                            max_displacement: r.max_displacement,
                        }
                    }
                    Dab::Flatten(o) => {
                        let r = flatten(tx, o)?;
                        DabReport {
                            moved: r.moved,
                            max_displacement: r.max_displacement,
                        }
                    }
                    Dab::Crease(o) => {
                        let r = crease(tx, o)?;
                        DabReport {
                            moved: r.moved,
                            max_displacement: r.max_displacement,
                        }
                    }
                    Dab::Clay(o) => {
                        let r = clay(tx, o)?;
                        DabReport {
                            moved: r.moved,
                            max_displacement: r.max_displacement,
                        }
                    }
                    Dab::Twist(o) => {
                        let r = twist(tx, o)?;
                        DabReport {
                            moved: r.moved,
                            max_displacement: r.max_displacement,
                        }
                    }
                })
            })
            .unwrap()
    }

    #[test]
    fn all_brushes_methods_filters_and_symmetry_match_whole_mesh_bit_for_bit() {
        use fast_mesh_ops::sculpt::{
            ClayOptions, ClayProfile, CreaseOptions, FlattenMode, FlattenOptions, GrabMethod,
            InflateMethod, SurfaceFilter, Symmetry, TwistMethod, TwistOptions,
            capture_stroke_state as capture_mesh_stroke,
        };

        let b = Brush {
            center: [1.35, -0.85, 0.15],
            radius: 2.3,
            falloff: Falloff::Smooth,
            strength: 0.42,
        };
        let b_overlap = Brush {
            center: [0.25, 0.15, 0.1],
            radius: 2.4,
            falloff: Falloff::Smooth,
            strength: 0.35,
        };
        let filters = [
            SurfaceFilter::None,
            SurfaceFilter::PeaksOnly {
                threshold: 0.01,
                contrast: 0.55,
            },
            SurfaceFilter::ValleysOnly {
                threshold: 0.01,
                contrast: 0.55,
            },
            SurfaceFilter::NormalAngle {
                normal: None,
                max_angle_deg: 55.0,
            },
            SurfaceFilter::NormalAngle {
                normal: Some([0.1, -0.15, 1.0]),
                max_angle_deg: 65.0,
            },
        ];
        let symmetries = [
            Symmetry::NONE,
            Symmetry {
                x: true,
                y: false,
                z: false,
            },
            Symmetry {
                x: true,
                y: true,
                z: false,
            },
            Symmetry {
                x: true,
                y: true,
                z: true,
            },
        ];

        let mut cases: Vec<Dab> = Vec::new();
        // 1. Inflate x 3 methods + filters + symmetry
        for method in [
            InflateMethod::VertexNormal,
            InflateMethod::AverageNormal,
            InflateMethod::StrokeNormal,
        ] {
            for &filter in &filters {
                cases.push(Dab::Inflate(InflateOptions {
                    brush: b,
                    method,
                    filter,
                    symmetry: symmetries[1],
                }));
            }
            for &symmetry in &symmetries {
                cases.push(Dab::Inflate(InflateOptions {
                    brush: b_overlap,
                    method,
                    filter: SurfaceFilter::None,
                    symmetry,
                }));
            }
        }
        // 2. Smooth x 4 methods + filters + symmetry
        for method in [
            SmoothMethod::Taubin,
            SmoothMethod::Laplacian,
            SmoothMethod::HcLaplacian,
            SmoothMethod::Cotangent,
        ] {
            for &filter in &filters {
                cases.push(Dab::Smooth(SmoothOptions {
                    brush: b,
                    cycles: 2,
                    method,
                    filter,
                    symmetry: symmetries[2],
                }));
            }
            for &symmetry in &symmetries {
                cases.push(Dab::Smooth(SmoothOptions {
                    brush: b_overlap,
                    cycles: 2,
                    method,
                    filter: SurfaceFilter::None,
                    symmetry,
                }));
            }
        }
        // 3. Grab x 3 methods + adaptive_reach + filters + symmetry
        for method in [
            GrabMethod::Sphere,
            GrabMethod::Geodesic,
            GrabMethod::Kelvinlet,
        ] {
            for adaptive_reach in [false, true] {
                for &symmetry in &symmetries {
                    cases.push(Dab::Grab(GrabOptions {
                        brush: b,
                        direction: [0.85, -0.35, 0.25],
                        method,
                        poisson: 0.42,
                        from_stroke: false,
                        adaptive_reach,
                        filter: filters[1],
                        symmetry,
                    }));
                }
            }
        }
        // 4. Flatten x 3 modes (auto & explicit plane) + symmetry
        for mode in [FlattenMode::Both, FlattenMode::Peaks, FlattenMode::Valleys] {
            cases.push(Dab::Flatten(FlattenOptions {
                brush: b,
                mode,
                plane_origin: None,
                plane_normal: None,
                filter: filters[3],
                symmetry: symmetries[2],
            }));
            cases.push(Dab::Flatten(FlattenOptions {
                brush: b_overlap,
                mode,
                plane_origin: Some([0.0, 0.0, 0.0]),
                plane_normal: Some([0.0, 0.0, 1.0]),
                filter: SurfaceFilter::None,
                symmetry: symmetries[3],
            }));
        }
        // 5. Crease (groove & pure pinch)
        for (strength, pinch) in [(-0.4, 0.6), (0.0, 0.7)] {
            cases.push(Dab::Crease(CreaseOptions {
                brush: Brush { strength, ..b },
                pinch,
                filter: filters[4],
                symmetry: symmetries[1],
            }));
        }
        // 6. Clay x Round/Strips (auto & explicit tangent)
        for profile in [ClayProfile::Round, ClayProfile::Strips] {
            for tangent in [None, Some([0.9, 0.3, 0.0])] {
                cases.push(Dab::Clay(ClayOptions {
                    brush: b,
                    profile,
                    tangent,
                    filter: SurfaceFilter::None,
                    symmetry: symmetries[2],
                }));
            }
        }
        // 7. Twist x 2 methods (auto & explicit axis)
        for method in [TwistMethod::Vortex, TwistMethod::Kelvinlet] {
            for axis in [None, Some([0.0, 0.1, 1.0])] {
                cases.push(Dab::Twist(TwistOptions {
                    brush: b,
                    axis,
                    method,
                    filter: filters[2],
                    symmetry: symmetries[1],
                }));
            }
        }

        for mode in [UpdateMode::Eager, UpdateMode::Lazy, UpdateMode::Manual] {
            for (idx, dab) in cases.iter().enumerate() {
                let mut sa = Scene::new(SceneOptions::default());
                let ia = sa
                    .add_mesh(
                        symmetric_terrain(8),
                        MeshOptions {
                            mode,
                            ..MeshOptions::default()
                        },
                    )
                    .unwrap();
                let mut ref_mesh = symmetric_terrain(8);

                // For StrokeNormal, capture stroke state first and apply a preliminary move so
                // stroke-start normals differ from current normals (and Manual BVH goes stale!).
                if matches!(
                    dab,
                    Dab::Inflate(InflateOptions {
                        method: InflateMethod::StrokeNormal,
                        ..
                    })
                ) {
                    sa.capture_stroke_state(ia, false).unwrap();
                    capture_mesh_stroke(&mut ref_mesh).unwrap();
                }

                let mut sb = Scene::new(SceneOptions::default());
                let ib = sb
                    .add_mesh(
                        ref_mesh,
                        MeshOptions {
                            mode,
                            ..MeshOptions::default()
                        },
                    )
                    .unwrap();

                let pre = Dab::Grab(GrabOptions {
                    brush: Brush {
                        center: [1.0, -0.5, 0.0],
                        radius: 2.0,
                        falloff: Falloff::Smooth,
                        strength: 0.3,
                    },
                    direction: [0.1, -0.15, 0.4],
                    ..Default::default()
                });
                let r_pre_a = sa.dab(ia, &pre).unwrap();
                let r_pre_b = apply_whole_dab(&mut sb, ib, &pre);
                assert_eq!(r_pre_a, r_pre_b, "pre-dab {mode:?} case {idx}");

                let ra = sa.dab(ia, dab).unwrap();
                let rb = apply_whole_dab(&mut sb, ib, dab);
                assert_eq!(ra, rb, "{mode:?} case {idx}: {dab:?}");
                assert_eq!(bits(&sa, ia), bits(&sb, ib), "{mode:?} case {idx}: {dab:?}");
            }
        }
    }

    #[test]
    fn lazy_stroke_snapshot_split_stretched_snapback_and_invalidation_parity() {
        use fast_mesh_ops::remesh::Region as StretchRegion;
        use fast_mesh_ops::sculpt::{
            GrabMethod, InflateMethod, Symmetry, capture_stroke_state as capture_mesh_stroke, grab,
            inflate,
        };
        use fast_mesh_ops::stretch::{
            StretchOptions, capture_rest_sizes as capture_mesh_rest_sizes,
            split_stretched as mesh_split_stretched,
        };

        for mode in [UpdateMode::Eager, UpdateMode::Lazy, UpdateMode::Manual] {
            let mut scene = Scene::new(SceneOptions::default());
            let id = scene
                .add_mesh(
                    symmetric_terrain(10),
                    MeshOptions {
                        mode,
                        ..MeshOptions::default()
                    },
                )
                .unwrap();
            let mut standalone = symmetric_terrain(10);

            scene.capture_rest_sizes(id, false).unwrap();
            scene.capture_stroke_state(id, false).unwrap();
            assert!(
                !scene.can_undo(),
                "capture_stroke_state is not an undo step"
            );

            capture_mesh_rest_sizes(&mut standalone).unwrap();
            capture_mesh_stroke(&mut standalone).unwrap();

            let base_brush = Brush {
                center: [0.5, -0.5, 0.1],
                radius: 0.9,
                falloff: Falloff::Smooth,
                strength: 1.0,
            };
            let sym_x = Symmetry {
                x: true,
                y: false,
                z: false,
            };

            // Step 1: Wide tangential + normal grab from stroke with adaptive_reach.
            let wide_grab = GrabOptions {
                brush: base_brush,
                direction: [2.2, 0.4, 0.8],
                method: GrabMethod::Sphere,
                from_stroke: true,
                adaptive_reach: true,
                symmetry: sym_x,
                ..Default::default()
            };
            let r1_scene = scene.dab(id, &Dab::Grab(wide_grab)).unwrap();
            let r1_mesh = {
                let mut tx = standalone.begin_edit();
                let r = grab(&mut tx, &wide_grab).unwrap();
                tx.commit().unwrap();
                DabReport {
                    moved: r.moved,
                    max_displacement: r.max_displacement,
                }
            };
            assert!(r1_scene.moved > 10);
            assert_eq!(r1_scene, r1_mesh, "{mode:?} step 1 report");

            // `only_if_missing: true` keeps the stroke snapshot from before Step 1!
            scene.capture_stroke_state(id, true).unwrap();

            // Step 2: Live `split_stretched` mid-stroke interpolates rest_pos & rest_normal!
            let stretch_opts = StretchOptions {
                region: StretchRegion {
                    center: [0.0, 0.0, 0.0],
                    radius: 20.0,
                },
                limit: 1.08,
                ..StretchOptions::default()
            };
            let split_scene = scene.split_stretched(id, &stretch_opts).unwrap();
            let split_mesh = {
                let mut tx = standalone.begin_edit();
                let r = mesh_split_stretched(&mut tx, &stretch_opts).unwrap();
                tx.commit().unwrap();
                r
            };
            assert!(split_scene.splits > 0, "must split stretched edges");
            assert_eq!(split_scene, split_mesh, "{mode:?} split_stretched report");

            // Step 3: Shortened grab snaps outer & newly-split outer vertices back to interpolated rest_pos!
            let short_grab = GrabOptions {
                direction: [0.15, 0.05, 0.1],
                ..wide_grab
            };
            let r3_scene = scene.dab(id, &Dab::Grab(short_grab)).unwrap();
            let r3_mesh = {
                let mut tx = standalone.begin_edit();
                let r = grab(&mut tx, &short_grab).unwrap();
                tx.commit().unwrap();
                DabReport {
                    moved: r.moved,
                    max_displacement: r.max_displacement,
                }
            };
            assert!(r3_scene.moved > 0);
            assert_eq!(r3_scene, r3_mesh, "{mode:?} step 3 shortened grab report");

            // Step 4: StrokeNormal inflate over split vertices uses interpolated stroke normals!
            let stroke_inf = InflateOptions {
                brush: Brush {
                    center: [0.5, -0.5, 0.1],
                    radius: 3.0,
                    falloff: Falloff::Smooth,
                    strength: 0.35,
                },
                method: InflateMethod::StrokeNormal,
                symmetry: sym_x,
                ..Default::default()
            };
            let r4_scene = scene.dab(id, &Dab::Inflate(stroke_inf)).unwrap();
            let r4_mesh = {
                let mut tx = standalone.begin_edit();
                let r = inflate(&mut tx, &stroke_inf).unwrap();
                tx.commit().unwrap();
                DabReport {
                    moved: r.moved,
                    max_displacement: r.max_displacement,
                }
            };
            assert_eq!(r4_scene, r4_mesh, "{mode:?} step 4 StrokeNormal report");

            let want_bits: Vec<[u32; 3]> = standalone
                .positions()
                .iter()
                .map(|p| p.map(f32::to_bits))
                .collect();
            assert_eq!(bits(&scene, id), want_bits, "{mode:?} final positions");

            // Undo all 4 steps and redo them: exact bit-for-bit restoration of all live vertices!
            let live_bits = |m: &Mesh| -> Vec<(VertexId, [u32; 3])> {
                (0..m.vertex_slots() as VertexId)
                    .filter(|&v| m.is_vertex_alive(v))
                    .map(|v| (v, m.positions()[v as usize].map(f32::to_bits)))
                    .collect()
            };
            let initial_live = live_bits(&symmetric_terrain(10));
            let want_live = live_bits(&standalone);
            for _ in 0..4 {
                assert!(scene.undo().unwrap());
            }
            assert_eq!(
                live_bits(scene.mesh(id).unwrap()),
                initial_live,
                "{mode:?} after full undo"
            );
            for _ in 0..4 {
                assert!(scene.redo().unwrap());
            }
            assert_eq!(
                live_bits(scene.mesh(id).unwrap()),
                want_live,
                "{mode:?} after full redo"
            );
        }

        // Verify stroke invalidation on undo, cancel_step, compact, and non-split_stretched topology edit:
        // once invalidated, `from_stroke: true` with zero direction moves 0 vertices (no stale snap-back!).
        let zero_snap_probe = Dab::Grab(GrabOptions {
            brush: Brush {
                center: [0.0, 0.0, 0.0],
                radius: 1.5,
                falloff: Falloff::Linear,
                strength: 1.0,
            },
            direction: [0.0, 0.0, 0.0],
            from_stroke: true,
            ..Default::default()
        });
        let pull = Dab::Grab(GrabOptions {
            brush: Brush {
                center: [0.0, 0.0, 0.0],
                radius: 2.0,
                falloff: Falloff::Smooth,
                strength: 0.6,
            },
            direction: [0.0, 0.0, 1.0],
            from_stroke: true,
            ..Default::default()
        });

        let non_stroke_pull = Dab::Grab(GrabOptions {
            brush: Brush {
                center: [0.0, 0.0, 0.0],
                radius: 2.0,
                falloff: Falloff::Smooth,
                strength: 0.6,
            },
            direction: [0.0, 0.0, 1.0],
            from_stroke: false,
            ..Default::default()
        });

        // (a) Non-split_stretched topology edit invalidates stroke
        {
            let mut s = Scene::new(SceneOptions::default());
            let id = s
                .add_mesh(symmetric_terrain(5), MeshOptions::default())
                .unwrap();
            s.capture_stroke_state(id, false).unwrap();
            assert!(s.dab(id, &pull).unwrap().moved > 0);
            s.edit(id, |tx| tx.remove_triangles(&[0])).unwrap();
            assert_eq!(s.dab(id, &zero_snap_probe).unwrap(), DabReport::default());
        }
        // (b) undo and redo invalidate stroke (with `curr != rest_pos` at probe time,
        // plus placement-only and cross-mesh undo steps)
        {
            let mut s = Scene::new(SceneOptions::default());
            let id = s
                .add_mesh(symmetric_terrain(5), MeshOptions::default())
                .unwrap();
            // Step 1: P0 -> P1.
            assert!(s.dab(id, &non_stroke_pull).unwrap().moved > 0);
            let p1 = bits(&s, id);
            // Step 2: P1 -> P2, capture stroke at P2, then P2 -> P3 (all in one step).
            s.begin_step();
            assert!(s.dab(id, &non_stroke_pull).unwrap().moved > 0);
            let p2 = bits(&s, id);
            assert_ne!(p1, p2);
            s.capture_stroke_state(id, false).unwrap();
            assert!(s.dab(id, &pull).unwrap().moved > 0);
            s.end_step().unwrap();
            // Undo Step 2 reverts mesh to P1 (`curr == P1 != P2 == rest_pos`):
            // if stroke were NOT invalidated, `zero_snap_probe` would snap P1 -> P2!
            assert!(s.undo().unwrap());
            assert_eq!(bits(&s, id), p1);
            assert_eq!(s.dab(id, &zero_snap_probe).unwrap(), DabReport::default());

            // Now arm stroke at P1 and populate `rest_pos = P1` via a 0-move dab
            // (which leaves the redo stack intact), then `redo()` Step 2 to P3 (`!= P1`):
            // `curr == P3 != P1 == rest_pos`, so if `redo()` failed to invalidate `stroke`,
            // `zero_snap_probe` would immediately snap P3 -> P1 (`moved > 0`)!
            s.capture_stroke_state(id, false).unwrap();
            assert_eq!(s.dab(id, &zero_snap_probe).unwrap(), DabReport::default());
            assert!(s.can_redo());
            assert!(s.redo().unwrap());
            assert_ne!(bits(&s, id), p1);
            assert_eq!(s.dab(id, &zero_snap_probe).unwrap(), DabReport::default());
            // Also verify `epoch == 0` after `redo()` even when `touched` was empty before `redo()`:
            assert!(s.undo().unwrap());
            s.capture_stroke_state(id, false).unwrap();
            assert!(s.redo().unwrap());
            assert!(s.dab(id, &non_stroke_pull).unwrap().moved > 0);
            s.capture_stroke_state(id, true).unwrap();
            assert_eq!(s.dab(id, &zero_snap_probe).unwrap(), DabReport::default());

            // Placement-only step undo and cross-mesh edit undo also invalidate `id`'s stroke
            // while `id`'s vertices remain at `curr != rest_pos`!
            s.capture_stroke_state(id, false).unwrap();
            assert!(s.dab(id, &pull).unwrap().moved > 0);
            s.set_placement(
                id,
                crate::Placement {
                    translation: [1.0, 0.0, 0.0],
                    ..crate::Placement::IDENTITY
                },
            )
            .unwrap();
            assert!(s.undo().unwrap()); // undoes placement (`done` has `(id, None)`)
            assert_eq!(s.dab(id, &zero_snap_probe).unwrap(), DabReport::default());

            let other = s
                .add_mesh(symmetric_terrain(3), MeshOptions::default())
                .unwrap();
            s.capture_stroke_state(id, false).unwrap();
            assert!(s.dab(id, &pull).unwrap().moved > 0);
            s.edit(other, |tx| tx.set_positions(&[0], &[[0.0, 0.0, 3.0]]))
                .unwrap();
            assert!(s.undo().unwrap()); // undoes `other` (`done` does not mention `id`)
            assert_eq!(s.dab(id, &zero_snap_probe).unwrap(), DabReport::default());
        }
        // (c) cancel_step invalidates stroke both when reverting to `curr != rest_pos`
        // AND when cancelling an empty open step (`done` is empty)
        {
            let mut s = Scene::new(SceneOptions::default());
            let id = s
                .add_mesh(symmetric_terrain(5), MeshOptions::default())
                .unwrap();
            // P0 -> P1 outside open step.
            assert!(s.dab(id, &non_stroke_pull).unwrap().moved > 0);
            let p1 = bits(&s, id);
            // Inside open step: P1 -> P2, capture stroke at P2, then P2 -> P3.
            s.begin_step();
            assert!(s.dab(id, &non_stroke_pull).unwrap().moved > 0);
            let p2 = bits(&s, id);
            assert_ne!(p1, p2);
            s.capture_stroke_state(id, false).unwrap();
            assert!(s.dab(id, &pull).unwrap().moved > 0);
            // `cancel_step` reverts to P1 (`!= P2 == rest_pos`): must move 0 on probe!
            s.cancel_step().unwrap();
            assert_eq!(bits(&s, id), p1);
            assert_eq!(s.dab(id, &zero_snap_probe).unwrap(), DabReport::default());

            // Empty open step: `begin_step` -> `capture_stroke_state(id, false)` -> `cancel_step`
            // (`done` is empty!) must also reset `epoch = 0`, so `non_stroke_pull` +
            // `capture_stroke_state(id, true)` captures post-move positions rather than P1!
            s.begin_step();
            s.capture_stroke_state(id, false).unwrap();
            s.cancel_step().unwrap();
            assert!(s.dab(id, &non_stroke_pull).unwrap().moved > 0);
            s.capture_stroke_state(id, true).unwrap();
            assert_eq!(s.dab(id, &zero_snap_probe).unwrap(), DabReport::default());
        }
        // (d) compact invalidates stroke
        {
            let mut s = Scene::new(SceneOptions::default());
            let id = s
                .add_mesh(symmetric_terrain(5), MeshOptions::default())
                .unwrap();
            s.capture_stroke_state(id, false).unwrap();
            assert!(s.dab(id, &pull).unwrap().moved > 0);
            s.compact(id).unwrap();
            assert_eq!(s.dab(id, &zero_snap_probe).unwrap(), DabReport::default());
        }
    }

    #[test]
    fn dab_protected_band_matches_whole_mesh_and_skips_empty_undo_steps() {
        use fast_mesh_ops::OpsError;
        use fast_mesh_ops::protect::with_protection_band;
        use fast_mesh_ops::sculpt::{ClayOptions, ClayProfile, Symmetry, clay, grab, inflate};

        let held_left = |mesh: &Mesh| -> Vec<VertexId> {
            mesh.positions()
                .iter()
                .enumerate()
                .filter(|(_, p)| p[0] <= -0.5)
                .map(|(v, _)| v as VertexId)
                .collect()
        };

        let band = 1.35f32;
        let sym_x = Symmetry {
            x: true,
            y: false,
            z: false,
        };
        let dabs = [
            Dab::Inflate(InflateOptions {
                brush: Brush {
                    center: [0.4, 0.2, 0.0],
                    radius: 2.8,
                    falloff: Falloff::Smooth,
                    strength: 0.5,
                },
                symmetry: sym_x,
                ..Default::default()
            }),
            Dab::Grab(GrabOptions {
                brush: Brush {
                    center: [-0.2, 0.0, 0.0],
                    radius: 1.2,
                    falloff: Falloff::Smooth,
                    strength: 0.8,
                },
                direction: [1.1, 0.2, 0.5],
                adaptive_reach: true,
                ..Default::default()
            }),
            Dab::Clay(ClayOptions {
                brush: Brush {
                    center: [0.0, 0.0, 0.0],
                    radius: 2.0,
                    falloff: Falloff::Smooth,
                    strength: 0.6,
                },
                profile: ClayProfile::Strips,
                tangent: Some([1.0, 0.2, 0.0]),
                ..Default::default()
            }),
        ];

        for mode in [UpdateMode::Eager, UpdateMode::Lazy, UpdateMode::Manual] {
            let mut sa = Scene::new(SceneOptions::default());
            let ia = sa
                .add_mesh(
                    symmetric_terrain(8),
                    MeshOptions {
                        mode,
                        ..MeshOptions::default()
                    },
                )
                .unwrap();
            let mut sb = Scene::new(SceneOptions::default());
            let ib = sb
                .add_mesh(
                    symmetric_terrain(8),
                    MeshOptions {
                        mode,
                        ..MeshOptions::default()
                    },
                )
                .unwrap();
            let held = held_left(sa.mesh(ia).unwrap());
            let zeros = vec![0.0f32; held.len()];

            for dab in &dabs {
                let ra = sa.dab_protected_band(ia, dab, &held, &zeros, band).unwrap();
                let rb = sb
                    .edit(ib, |tx| -> std::result::Result<DabReport, SceneError> {
                        with_protection_band(tx, None, &held, &zeros, band, |tx| {
                            Ok(match dab {
                                Dab::Inflate(o) => {
                                    let r = inflate(tx, o)?;
                                    DabReport {
                                        moved: r.moved,
                                        max_displacement: r.max_displacement,
                                    }
                                }
                                Dab::Grab(o) => {
                                    let r = grab(tx, o)?;
                                    DabReport {
                                        moved: r.moved,
                                        max_displacement: r.max_displacement,
                                    }
                                }
                                Dab::Clay(o) => {
                                    let r = clay(tx, o)?;
                                    DabReport {
                                        moved: r.moved,
                                        max_displacement: r.max_displacement,
                                    }
                                }
                                _ => unreachable!(),
                            })
                        })
                    })
                    .unwrap();
                assert!(ra.moved > 0);
                assert_eq!(ra, rb, "{mode:?} {dab:?}");
                assert_eq!(bits(&sa, ia), bits(&sb, ib), "{mode:?} {dab:?}");
            }
        }

        // Invalid band (< 0, NaN, inf) fails with OpsError::InvalidInput and creates no undo step.
        let mut s = Scene::new(SceneOptions::default());
        let id = s
            .add_mesh(symmetric_terrain(4), MeshOptions::default())
            .unwrap();
        for bad_band in [-0.01f32, f32::NAN, f32::INFINITY] {
            assert!(matches!(
                s.dab_protected_band(id, &dabs[0], &[0], &[0.0], bad_band),
                Err(SceneError::Ops(OpsError::InvalidInput(_)))
            ));
        }
        assert!(!s.can_undo());

        // Fully held brush (and zero-move brush) with band > 0 reports moved == 0 and leaves no undo step.
        let all: Vec<VertexId> = (0..s.mesh(id).unwrap().vertex_slots() as VertexId).collect();
        let all_zeros = vec![0.0f32; all.len()];
        assert_eq!(
            s.dab_protected_band(id, &dabs[0], &all, &all_zeros, 0.5)
                .unwrap(),
            DabReport::default()
        );
        assert!(!s.can_undo());

        let zero_strength = Dab::Inflate(InflateOptions {
            brush: Brush {
                center: [0.0; 3],
                radius: 2.0,
                falloff: Falloff::Smooth,
                strength: 0.0,
            },
            ..Default::default()
        });
        assert_eq!(
            s.dab_protected_band(id, &zero_strength, &[0], &[0.0], 0.5)
                .unwrap(),
            DabReport::default()
        );
        assert!(!s.can_undo());
    }

    #[test]
    fn vertices_in_sphere_queries_bvh_validates_and_respects_manual_mode() {
        use fast_mesh_ops::OpsError;

        let mut scene = Scene::new(SceneOptions::default());
        let make_mesh = || {
            let mut m = symmetric_terrain(6);
            // Add an isolated vertex inside the sphere (not used by any triangle).
            let mut tx = m.begin_edit();
            tx.add_vertices(&[[0.1, 0.1, 0.0]]).unwrap();
            tx.commit().unwrap();
            m
        };
        let lazy_id = scene
            .add_mesh(
                make_mesh(),
                MeshOptions {
                    mode: UpdateMode::Lazy,
                    defer_bvh: true,
                    ..MeshOptions::default()
                },
            )
            .unwrap();
        assert_eq!(scene.is_current(lazy_id), Ok(false));

        let center = [0.5f32, -0.25, 0.0];
        let radius = 1.6f32;
        let got = scene.vertices_in_sphere(lazy_id, center, radius).unwrap();
        assert_eq!(scene.is_current(lazy_id), Ok(true));
        assert!(got.windows(2).all(|w| w[0] < w[1]), "strictly ascending");

        // Brute-force expected live triangle-used vertices inside closed ball `d <= radius * radius`.
        let mesh_ref = scene.mesh(lazy_id).unwrap();
        let mut want: Vec<VertexId> = (0..mesh_ref.triangle_slots() as u32)
            .filter_map(|t| mesh_ref.triangle(t))
            .flatten()
            .filter(|&v| {
                let p = mesh_ref.positions()[v as usize];
                let d2 = (p[0] - center[0]).powi(2)
                    + (p[1] - center[1]).powi(2)
                    + (p[2] - center[2]).powi(2);
                mesh_ref.is_vertex_alive(v) && d2 <= radius * radius
            })
            .collect();
        want.sort_unstable();
        want.dedup();
        assert!(!got.is_empty());
        assert_eq!(got, want);

        // Validation: negative or non-finite radius/center rejected with OpsError::InvalidInput.
        for (bad_c, bad_r) in [
            ([0.0, 0.0, 0.0], -0.1f32),
            ([0.0, 0.0, 0.0], f32::NAN),
            ([0.0, 0.0, 0.0], f32::INFINITY),
            ([f32::NAN, 0.0, 0.0], 1.0),
            ([0.0, f32::INFINITY, 0.0], 1.0),
        ] {
            assert!(matches!(
                scene.vertices_in_sphere(lazy_id, bad_c, bad_r),
                Err(SceneError::Ops(OpsError::InvalidInput(_)))
            ));
        }

        // Manual stale BVH returns SceneError::StaleBvh(manual_id).
        let manual_id = scene
            .add_mesh(
                make_mesh(),
                MeshOptions {
                    mode: UpdateMode::Manual,
                    ..MeshOptions::default()
                },
            )
            .unwrap();
        scene
            .edit(manual_id, |tx| tx.set_positions(&[0], &[[0.0, 0.0, 2.0]]))
            .unwrap();
        assert_eq!(
            scene.vertices_in_sphere(manual_id, center, radius),
            Err(SceneError::StaleBvh(manual_id))
        );
        scene.update(manual_id).unwrap();
        assert!(scene.vertices_in_sphere(manual_id, center, radius).is_ok());

        // Unknown mesh returns UnknownMesh.
        scene.remove_mesh(manual_id).unwrap();
        assert_eq!(
            scene.vertices_in_sphere(manual_id, center, radius),
            Err(SceneError::UnknownMesh)
        );
    }
}
