//! Face and region selection, topological boundary operators, mesh extraction/deletion,
//! and vertex-mask interoperation.

use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_query::{Cull, Ray, RayOptions, Region, raycast, triangles_in};
use fast_mesh_store::{Mesh, Transaction, TriangleId, VertexId};

use crate::cleanup::unused_vertices;
use crate::error::{OpsError, Result};
use crate::topology::{EdgeMap, VertexFans, components, cross, dot, point, proper_triangle, sub};
use crate::{filled, push, reserve};

/// Boolean set combination mode for [`FaceSelection::apply_mode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SelectionCombineMode {
    /// Replace the target selection with the incoming selection.
    Replace,
    /// Union: keep all triangles in either selection.
    Add,
    /// Difference: remove all triangles present in the incoming selection.
    Subtract,
    /// Intersection: keep only triangles present in both selections.
    Intersect,
}

/// Adjacency relation used when expanding a [`FaceSelection`] across neighboring faces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdjacencyKind {
    /// Expand across any shared vertex (full 1-ring vertex star).
    Vertex,
    /// Expand strictly across shared edges.
    Edge,
}

/// 2D screen-space selection region expressed in Normalized Device Coordinates (`[-1, 1]^2`).
#[derive(Debug, Clone, PartialEq)]
pub enum ScreenSelectionShape {
    /// Axis-aligned rectangle in NDC `[-1, 1]^2`.
    Rect {
        /// Lower-left corner `(min_x, min_y)` in NDC.
        min_ndc: [f32; 2],
        /// Upper-right corner `(max_x, max_y)` in NDC.
        max_ndc: [f32; 2],
    },
    /// Closed 2D polygon lasso in NDC `[-1, 1]^2` (implicitly closed from last to first vertex).
    Lasso {
        /// Ordered polygon boundary vertices in NDC.
        polygon_ndc: Vec<[f32; 2]>,
    },
}

/// Camera, projection, and visibility options for [`select_faces_screen`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenSelectOptions {
    /// Column-major 4x4 view-projection matrix (`view_proj[col][row]`), mapping world-space
    /// `(x, y, z, 1.0)` to homogeneous clip space `(x, y, z, w)`.
    pub view_proj: [[f32; 4]; 4],
    /// World-space camera eye position (ray origin for perspective visibility/backface tests).
    pub camera_origin: [f32; 3],
    /// World-space forward view vector pointing from the camera into the scene.
    pub camera_forward: [f32; 3],
    /// `true` for orthographic projection (`view_proj` uses constant ray direction `camera_forward`),
    /// `false` for perspective projection originating at `camera_origin`.
    pub is_orthographic: bool,
    /// When `true`, reject triangles occluded by closer geometry using BVH raycasts.
    pub visible_only: bool,
    /// When `true`, reject triangles whose geometric normal faces away from the camera.
    pub reject_backfaces: bool,
    /// When `true`, select triangles touching or overlapping the screen region; when `false`,
    /// require the entire projected triangle to lie inside the selection region.
    pub include_touching: bool,
}

/// Summary report returned by [`delete_selected_faces`] and [`split_selected_faces`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FaceEditReport {
    /// Number of alive triangles removed from the mesh transaction.
    pub removed_triangles: usize,
    /// Number of newly orphaned vertices removed alongside the deleted triangles.
    pub removed_vertices: usize,
}

/// Compact slot-indexed face selection backed by a `u64` bitset indexed by [`TriangleId`].
#[derive(Debug, Clone, Default)]
pub struct FaceSelection {
    words: Vec<u64>,
    count: usize,
}

impl PartialEq for FaceSelection {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.trimmed_words() == other.trimmed_words()
    }
}

impl Eq for FaceSelection {}

impl FaceSelection {
    /// Creates an empty [`FaceSelection`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a [`FaceSelection`] pre-allocated to hold up to `slots` triangle slots.
    pub(crate) fn with_slots(slots: usize) -> Result<Self> {
        Ok(Self {
            words: filled(slots.div_ceil(64), 0u64)?,
            count: 0,
        })
    }

    /// Builds a [`FaceSelection`] from explicit triangle IDs, verifying each ID is alive in `mesh`.
    ///
    /// # Errors
    /// Returns [`OpsError::InvalidInput`] if any ID is dead or out of range, or propagates
    /// allocation errors under strict memory limits.
    pub fn from_triangles(mesh: &Mesh, triangles: &[TriangleId]) -> Result<Self> {
        for &tri in triangles {
            if !mesh.is_triangle_alive(tri) {
                return Err(OpsError::InvalidInput(
                    "triangle id is dead or out of range",
                ));
            }
        }
        let mut sel = Self::with_slots(mesh.triangle_slots())?;
        for &tri in triangles {
            sel.insert_within_capacity(tri);
        }
        Ok(sel)
    }

    /// Returns all currently selected triangles that are alive in `mesh`, sorted ascending.
    #[must_use]
    pub fn to_triangles(&self, mesh: &Mesh) -> Vec<TriangleId> {
        let mut out = Vec::with_capacity(self.count.min(mesh.triangle_count()));
        let slots = mesh.triangle_slots();
        for id in self.iter() {
            if (id as usize) < slots && mesh.is_triangle_alive(id) {
                out.push(id);
            }
        }
        out
    }

    /// Fallible version of [`Self::to_triangles`] for internal `Result`-returning operations.
    pub(crate) fn to_triangles_fallible(&self, mesh: &Mesh) -> Result<Vec<TriangleId>> {
        let mut out = Vec::new();
        reserve(&mut out, self.count.min(mesh.triangle_count()))?;
        let slots = mesh.triangle_slots();
        for id in self.iter() {
            if (id as usize) < slots && mesh.is_triangle_alive(id) {
                push(&mut out, id)?;
            }
        }
        Ok(out)
    }

    /// Number of selected triangle bits currently stored.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns `true` if no triangles are selected.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Clears all selected triangles.
    pub fn clear(&mut self) {
        self.words.clear();
        self.count = 0;
    }

    /// Returns `true` if `triangle` is currently selected.
    #[inline]
    #[must_use]
    pub fn contains(&self, triangle: TriangleId) -> bool {
        let idx = triangle as usize;
        let word = idx / 64;
        if word >= self.words.len() {
            return false;
        }
        (self.words[word] >> (idx % 64)) & 1 != 0
    }

    /// Inserts `triangle` into the selection, returning `true` if it was not previously present.
    pub fn insert(&mut self, triangle: TriangleId) -> bool {
        let idx = triangle as usize;
        let word = idx / 64;
        if word >= self.words.len() {
            self.words.resize(word + 1, 0);
        }
        self.insert_within_capacity(triangle)
    }

    /// Fallible variant of [`Self::insert`] using `crate::reserve`.
    pub(crate) fn try_insert(&mut self, triangle: TriangleId) -> Result<bool> {
        let idx = triangle as usize;
        let word = idx / 64;
        if word >= self.words.len() {
            let extra = word + 1 - self.words.len();
            reserve(&mut self.words, extra)?;
            self.words.resize(word + 1, 0);
        }
        Ok(self.insert_within_capacity(triangle))
    }

    #[inline]
    fn insert_within_capacity(&mut self, triangle: TriangleId) -> bool {
        let idx = triangle as usize;
        let word = idx / 64;
        let mask = 1u64 << (idx % 64);
        if self.words[word] & mask != 0 {
            return false;
        }
        self.words[word] |= mask;
        self.count += 1;
        true
    }

    /// Removes `triangle` from the selection, returning `true` if it was present.
    pub fn remove(&mut self, triangle: TriangleId) -> bool {
        let idx = triangle as usize;
        let word = idx / 64;
        if word >= self.words.len() {
            return false;
        }
        let mask = 1u64 << (idx % 64);
        if self.words[word] & mask == 0 {
            return false;
        }
        self.words[word] &= !mask;
        self.count -= 1;
        self.trim();
        true
    }

    /// Inverts the selection with respect to all alive triangles in `mesh`.
    ///
    /// Dead or out-of-range triangle bits are cleared so `len()` equals the selected alive count.
    pub fn invert(&mut self, mesh: &Mesh) {
        let slots = mesh.triangle_slots();
        let needed_words = slots.div_ceil(64);
        if self.words.len() < needed_words {
            self.words.resize(needed_words, 0);
        } else {
            self.words.truncate(needed_words);
        }

        let mut new_count = 0usize;
        for (w_idx, word) in self.words.iter_mut().enumerate() {
            let base = w_idx * 64;
            let mut next_word = 0u64;
            let limit = 64.min(slots.saturating_sub(base));
            for bit in 0..limit {
                let id = (base + bit) as TriangleId;
                if mesh.is_triangle_alive(id) && ((*word >> bit) & 1 == 0) {
                    next_word |= 1u64 << bit;
                    new_count += 1;
                }
            }
            *word = next_word;
        }
        self.count = new_count;
        self.trim();
    }

    /// Returns a newly inverted selection with respect to all alive triangles in `mesh`.
    #[must_use]
    pub fn inverted(&self, mesh: &Mesh) -> Self {
        let mut copy = self.clone();
        copy.invert(mesh);
        copy
    }

    /// Combines `self` with `other` according to `mode`.
    pub fn apply_mode(&mut self, other: &FaceSelection, mode: SelectionCombineMode) {
        match mode {
            SelectionCombineMode::Replace => {
                self.words.clear();
                self.words.extend_from_slice(other.trimmed_words());
                self.count = other.count;
            }
            SelectionCombineMode::Add => {
                let other_words = other.trimmed_words();
                if self.words.len() < other_words.len() {
                    self.words.resize(other_words.len(), 0);
                }
                for (dst, &src) in self.words.iter_mut().zip(other_words.iter()) {
                    *dst |= src;
                }
                self.recompute_meta();
            }
            SelectionCombineMode::Subtract => {
                for (dst, &src) in self.words.iter_mut().zip(other.words.iter()) {
                    *dst &= !src;
                }
                self.recompute_meta();
            }
            SelectionCombineMode::Intersect => {
                let keep = self.words.len().min(other.words.len());
                self.words.truncate(keep);
                for (dst, &src) in self.words.iter_mut().zip(other.words.iter()) {
                    *dst &= src;
                }
                self.recompute_meta();
            }
        }
    }

    /// Iterates over all selected triangle IDs in ascending order.
    pub fn iter(&self) -> impl Iterator<Item = TriangleId> + '_ {
        self.words
            .iter()
            .enumerate()
            .flat_map(|(w_idx, &word)| BitWordIter {
                word,
                base: (w_idx * 64) as u32,
            })
    }

    fn trimmed_words(&self) -> &[u64] {
        let mut end = self.words.len();
        while end > 0 && self.words[end - 1] == 0 {
            end -= 1;
        }
        &self.words[..end]
    }

    fn trim(&mut self) {
        while self.words.last().copied() == Some(0) {
            self.words.pop();
        }
    }

    fn recompute_meta(&mut self) {
        self.trim();
        self.count = self.words.iter().map(|w| w.count_ones() as usize).sum();
    }

    /// Copies only alive triangles of `mesh` that are marked in `self` into a fresh bitset.
    fn alive_copy(&self, mesh: &Mesh) -> Result<Self> {
        let mut out = Self::with_slots(mesh.triangle_slots())?;
        let slots = mesh.triangle_slots();
        for id in self.iter() {
            if (id as usize) < slots && mesh.is_triangle_alive(id) {
                out.insert_within_capacity(id);
            }
        }
        Ok(out)
    }
}

struct BitWordIter {
    word: u64,
    base: u32,
}

impl Iterator for BitWordIter {
    type Item = TriangleId;

    fn next(&mut self) -> Option<Self::Item> {
        if self.word == 0 {
            return None;
        }
        let tz = self.word.trailing_zeros();
        self.word &= self.word - 1;
        Some(self.base + tz)
    }
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

#[inline]
fn tri_normal_f64(mesh: &Mesh, tri: [VertexId; 3]) -> Option<[f64; 3]> {
    let p0 = point(mesh, tri[0]);
    let p1 = point(mesh, tri[1]);
    let p2 = point(mesh, tri[2]);
    let n = cross(sub(p1, p0), sub(p2, p0));
    let len2 = dot(n, n);
    if !len2.is_finite() || len2 <= 0.0 {
        return None;
    }
    let inv = len2.sqrt().recip();
    Some([n[0] * inv, n[1] * inv, n[2] * inv])
}

#[inline]
fn normalize_f64(v: [f64; 3]) -> Option<[f64; 3]> {
    let len2 = dot(v, v);
    if !len2.is_finite() || len2 <= 0.0 {
        return None;
    }
    let inv = len2.sqrt().recip();
    Some([v[0] * inv, v[1] * inv, v[2] * inv])
}

#[inline]
fn ordered_edge(a: VertexId, b: VertexId) -> (VertexId, VertexId) {
    if a <= b { (a, b) } else { (b, a) }
}

// ---------------------------------------------------------------------------
// Interactive Selection Algorithms
// ---------------------------------------------------------------------------

/// Selects connected faces by flood-filling from `seeds` across manifold interior edges
/// whose adjacent triangle normal dihedral angle does not exceed `max_crease_angle_deg`.
///
/// Degenerate or zero-area triangles halt propagation across themselves, though valid seed
/// IDs are always included in the output selection.
///
/// # Errors
/// Returns [`OpsError::InvalidInput`] if `max_crease_angle_deg` is outside `[0, 180]` or
/// any seed is dead/out of range.
pub fn select_by_crease_angle(
    mesh: &Mesh,
    seeds: &[TriangleId],
    max_crease_angle_deg: f32,
) -> Result<FaceSelection> {
    if !max_crease_angle_deg.is_finite() || !(0.0..=180.0).contains(&max_crease_angle_deg) {
        return Err(OpsError::InvalidInput(
            "max_crease_angle_deg must be finite and within [0, 180]",
        ));
    }
    for &seed in seeds {
        if !mesh.is_triangle_alive(seed) {
            return Err(OpsError::InvalidInput(
                "seed triangle is dead or out of range",
            ));
        }
    }
    if seeds.is_empty() || mesh.triangle_count() == 0 {
        return Ok(FaceSelection::new());
    }

    let edges = EdgeMap::build(mesh)?;
    let cos_limit = f64::from(max_crease_angle_deg).to_radians().cos();
    let mut sel = FaceSelection::with_slots(mesh.triangle_slots())?;
    let mut queue = Vec::new();
    reserve(&mut queue, seeds.len())?;

    for &seed in seeds {
        if sel.try_insert(seed)? {
            push(&mut queue, seed)?;
        }
    }

    let mut head = 0usize;
    while head < queue.len() {
        let t = queue[head];
        head += 1;
        let Some(tri) = proper_triangle(mesh, t) else {
            continue;
        };
        let Some(n_t) = tri_normal_f64(mesh, tri) else {
            continue;
        };

        for (u, v) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            let (ea, eb) = ordered_edge(u, v);
            let incident = edges.triangles(ea, eb);
            if incident.len() != 2 {
                continue;
            }
            let nb = if incident[0] == t {
                incident[1]
            } else {
                incident[0]
            };
            if sel.contains(nb) {
                continue;
            }
            let Some(nb_tri) = proper_triangle(mesh, nb) else {
                continue;
            };
            let Some(n_nb) = tri_normal_f64(mesh, nb_tri) else {
                continue;
            };
            if dot(n_t, n_nb).clamp(-1.0, 1.0) >= cos_limit - 1e-12 && sel.try_insert(nb)? {
                push(&mut queue, nb)?;
            }
        }
    }

    Ok(sel)
}

/// Selects all triangles edge-connected to any of `seeds`.
///
/// # Errors
/// Returns [`OpsError::InvalidInput`] if any seed triangle is dead or out of range.
pub fn select_connected_shell(mesh: &Mesh, seeds: &[TriangleId]) -> Result<FaceSelection> {
    for &seed in seeds {
        if !mesh.is_triangle_alive(seed) {
            return Err(OpsError::InvalidInput(
                "seed triangle is dead or out of range",
            ));
        }
    }
    if seeds.is_empty() || mesh.triangle_count() == 0 {
        return Ok(FaceSelection::new());
    }

    let edges = EdgeMap::build(mesh)?;
    let mut sel = FaceSelection::with_slots(mesh.triangle_slots())?;
    let mut queue = Vec::new();
    reserve(&mut queue, seeds.len())?;

    for &seed in seeds {
        if sel.try_insert(seed)? {
            push(&mut queue, seed)?;
        }
    }

    let mut head = 0usize;
    while head < queue.len() {
        let t = queue[head];
        head += 1;
        let Some(tri) = mesh.triangle(t) else {
            continue;
        };
        for (u, v) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            if u == v {
                continue;
            }
            let (ea, eb) = ordered_edge(u, v);
            for &nb in edges.triangles(ea, eb) {
                if nb != t && mesh.is_triangle_alive(nb) && sel.try_insert(nb)? {
                    push(&mut queue, nb)?;
                }
            }
        }
    }

    Ok(sel)
}

/// Selects every edge-connected shell whose triangle count is at most `max_triangles`
/// and whose total surface area is at most `max_area`.
///
/// Pass `usize::MAX` or `f32::MAX` to effectively disable either upper bound.
///
/// # Errors
/// Returns [`OpsError::InvalidInput`] if `max_area` is negative or non-finite.
pub fn select_small_face_islands(
    mesh: &Mesh,
    max_triangles: usize,
    max_area: f32,
) -> Result<FaceSelection> {
    if !max_area.is_finite() || max_area < 0.0 {
        return Err(OpsError::InvalidInput(
            "max_area must be finite and non-negative",
        ));
    }
    if mesh.triangle_count() == 0 || max_triangles == 0 {
        return Ok(FaceSelection::new());
    }

    let edges = EdgeMap::build(mesh)?;
    let comps = components(mesh, &edges)?;
    let max_area_f64 = f64::from(max_area);
    let mut sel = FaceSelection::with_slots(mesh.triangle_slots())?;

    for comp in comps {
        if comp.triangles.len() <= max_triangles && comp.area <= max_area_f64 {
            for t in comp.triangles {
                sel.try_insert(t)?;
            }
        }
    }

    Ok(sel)
}

/// Selects all triangles intersecting the closed sphere `(center, radius)`, optionally
/// restricted to triangles whose geometric normal lies within `max_angle_deg` of `axis`.
///
/// When `bvh` is `None`, a temporary [`Bvh`] is constructed internally.
///
/// # Errors
/// Returns [`OpsError::InvalidInput`] on non-finite coordinates, negative `radius`,
/// degenerate/non-finite normal cone parameters, or propagates BVH/query errors.
pub fn select_faces_in_sphere(
    mesh: &Mesh,
    bvh: Option<&Bvh>,
    center: [f32; 3],
    radius: f32,
    normal_cone: Option<([f32; 3], f32)>,
) -> Result<FaceSelection> {
    if !center.iter().all(|v| v.is_finite()) {
        return Err(OpsError::InvalidInput("sphere center must be finite"));
    }
    if !radius.is_finite() || radius < 0.0 {
        return Err(OpsError::InvalidInput(
            "sphere radius must be finite and non-negative",
        ));
    }

    let cone = match normal_cone {
        Some((axis, max_angle_deg)) => {
            if !axis.iter().all(|v| v.is_finite()) {
                return Err(OpsError::InvalidInput(
                    "normal_cone axis must have finite components",
                ));
            }
            let Some(unit_axis) =
                normalize_f64([f64::from(axis[0]), f64::from(axis[1]), f64::from(axis[2])])
            else {
                return Err(OpsError::InvalidInput("normal_cone axis must be non-zero"));
            };
            if !max_angle_deg.is_finite() || !(0.0..=180.0).contains(&max_angle_deg) {
                return Err(OpsError::InvalidInput(
                    "normal_cone angle must be finite and within [0, 180]",
                ));
            }
            Some((unit_axis, f64::from(max_angle_deg).to_radians().cos()))
        }
        None => None,
    };

    let owned_bvh = if bvh.is_none() {
        Some(Bvh::build(mesh, &BuildOptions::default())?)
    } else {
        None
    };
    let active_bvh = bvh.or(owned_bvh.as_ref()).expect("bvh available");

    let mut candidates = Vec::new();
    triangles_in(
        mesh,
        active_bvh,
        &Region::Sphere { center, radius },
        &mut candidates,
    )?;

    let mut sel = FaceSelection::with_slots(mesh.triangle_slots())?;
    for t in candidates {
        if !mesh.is_triangle_alive(t) {
            continue;
        }
        if let Some((axis, cos_limit)) = cone {
            let Some(tri) = proper_triangle(mesh, t) else {
                continue;
            };
            let Some(n) = tri_normal_f64(mesh, tri) else {
                continue;
            };
            if dot(n, axis).clamp(-1.0, 1.0) < cos_limit - 1e-12 {
                continue;
            }
        }
        sel.try_insert(t)?;
    }

    Ok(sel)
}

// ---------------------------------------------------------------------------
// Screen-Space Selection (Rect / Lasso + BVH Occlusion)
// ---------------------------------------------------------------------------

/// Selects triangles falling inside (or crossing, when `options.include_touching` is `true`)
/// a 2D NDC rectangle or lasso polygon.
///
/// `options.view_proj` is column-major (`view_proj[col][row]`).
/// When `options.visible_only` is `true`, visibility is tested via BVH raycasts from
/// `camera_origin` (perspective) or upstream along `-camera_forward` (orthographic);
/// a temporary [`Bvh`] is constructed only when `bvh` is `None`.
///
/// # Errors
/// Returns [`OpsError::InvalidInput`] on non-finite matrix/camera/shape values, an inverted
/// rectangle (`min > max`), zero `camera_forward`, or stale/mismatched `bvh`.
pub fn select_faces_screen(
    mesh: &Mesh,
    bvh: Option<&Bvh>,
    shape: &ScreenSelectionShape,
    options: &ScreenSelectOptions,
) -> Result<FaceSelection> {
    validate_screen_inputs(shape, options)?;

    if bvh.is_some_and(|provided| !provided.is_current(mesh)) {
        return Err(OpsError::InvalidInput(
            "provided BVH does not match the current mesh",
        ));
    }

    if mesh.triangle_count() == 0 {
        return Ok(FaceSelection::new());
    }
    if matches!(
        shape,
        ScreenSelectionShape::Lasso { polygon_ndc } if polygon_ndc.len() < 3
    ) {
        return Ok(FaceSelection::new());
    }

    let fwd = normalize_f64([
        f64::from(options.camera_forward[0]),
        f64::from(options.camera_forward[1]),
        f64::from(options.camera_forward[2]),
    ])
    .ok_or(OpsError::InvalidInput("camera_forward must be non-zero"))?;

    let origin64 = [
        f64::from(options.camera_origin[0]),
        f64::from(options.camera_origin[1]),
        f64::from(options.camera_origin[2]),
    ];

    let owned_bvh = if options.visible_only && bvh.is_none() {
        Some(Bvh::build(mesh, &BuildOptions::default())?)
    } else {
        None
    };
    let active_bvh = bvh.or(owned_bvh.as_ref());

    let ortho_bounds = if options.visible_only && options.is_orthographic {
        compute_ortho_ray_backplane(mesh, fwd)
    } else {
        None
    };

    let vp = mat4_to_f64(&options.view_proj);
    let shape64 = ShapeF64::from_shape(shape);
    let mut sel = FaceSelection::with_slots(mesh.triangle_slots())?;

    for slot in 0..mesh.triangle_slots() {
        let t = slot as TriangleId;
        let Some(tri) = proper_triangle(mesh, t) else {
            continue;
        };

        let p0 = point(mesh, tri[0]);
        let p1 = point(mesh, tri[1]);
        let p2 = point(mesh, tri[2]);

        let tri_normal = if options.reject_backfaces {
            let n = cross(sub(p1, p0), sub(p2, p0));
            let len2 = dot(n, n);
            if !len2.is_finite() || len2 <= 0.0 {
                continue;
            }
            let view_dir = if options.is_orthographic {
                fwd
            } else {
                sub(p0, origin64)
            };
            if dot(n, view_dir) >= -1e-12 * len2.sqrt() {
                continue;
            }
            Some(n)
        } else {
            None
        };
        let _ = tri_normal;

        let clip = [
            project_clip(&vp, p0),
            project_clip(&vp, p1),
            project_clip(&vp, p2),
        ];

        // Clip against w >= 1e-7 in homogeneous clip space so triangles behind the
        // perspective camera plane do not invert across w = 0 while orthographic
        // projections (w == 1.0) pass untouched.
        let mut poly_clip = [[0.0_f64; 4]; 4];
        let poly_len = clip_triangle_near_w(&clip, 1e-7, &mut poly_clip);
        if poly_len < 3 {
            continue;
        }
        if !options.include_touching && poly_len != 3 {
            continue;
        }

        let mut poly_ndc = [[0.0_f64; 2]; 4];
        for i in 0..poly_len {
            let inv_w = 1.0 / poly_clip[i][3];
            poly_ndc[i] = [poly_clip[i][0] * inv_w, poly_clip[i][1] * inv_w];
        }

        if !poly_matches_shape(&poly_ndc[..poly_len], &shape64, options.include_touching) {
            continue;
        }

        if options.visible_only {
            let Some(tree) = active_bvh else {
                continue;
            };
            if !is_triangle_visible(
                mesh,
                tree,
                t,
                [p0, p1, p2],
                &vp,
                &shape64,
                options,
                origin64,
                fwd,
                ortho_bounds,
            )? {
                continue;
            }
        }

        sel.try_insert(t)?;
    }

    Ok(sel)
}

fn validate_screen_inputs(
    shape: &ScreenSelectionShape,
    options: &ScreenSelectOptions,
) -> Result<()> {
    for col in &options.view_proj {
        if !col.iter().all(|v| v.is_finite()) {
            return Err(OpsError::InvalidInput(
                "view_proj must contain only finite floats",
            ));
        }
    }
    if !options.camera_origin.iter().all(|v| v.is_finite()) {
        return Err(OpsError::InvalidInput("camera_origin must be finite"));
    }
    if !options.camera_forward.iter().all(|v| v.is_finite()) {
        return Err(OpsError::InvalidInput("camera_forward must be finite"));
    }
    match shape {
        ScreenSelectionShape::Rect { min_ndc, max_ndc } => {
            if !min_ndc.iter().all(|v| v.is_finite()) || !max_ndc.iter().all(|v| v.is_finite()) {
                return Err(OpsError::InvalidInput("Rect NDC bounds must be finite"));
            }
            if min_ndc[0] > max_ndc[0] || min_ndc[1] > max_ndc[1] {
                return Err(OpsError::InvalidInput(
                    "Rect min_ndc must not exceed max_ndc",
                ));
            }
        }
        ScreenSelectionShape::Lasso { polygon_ndc } => {
            for pt in polygon_ndc {
                if !pt[0].is_finite() || !pt[1].is_finite() {
                    return Err(OpsError::InvalidInput(
                        "Lasso polygon vertices must be finite",
                    ));
                }
            }
        }
    }
    Ok(())
}

enum ShapeF64 {
    Rect {
        min: [f64; 2],
        max: [f64; 2],
    },
    Lasso {
        pts: Vec<[f64; 2]>,
        min: [f64; 2],
        max: [f64; 2],
    },
}

impl ShapeF64 {
    fn from_shape(shape: &ScreenSelectionShape) -> Self {
        match shape {
            ScreenSelectionShape::Rect { min_ndc, max_ndc } => Self::Rect {
                min: [f64::from(min_ndc[0]), f64::from(min_ndc[1])],
                max: [f64::from(max_ndc[0]), f64::from(max_ndc[1])],
            },
            ScreenSelectionShape::Lasso { polygon_ndc } => {
                let mut min = [f64::INFINITY; 2];
                let mut max = [f64::NEG_INFINITY; 2];
                let mut pts = Vec::with_capacity(polygon_ndc.len());
                for p in polygon_ndc {
                    let q = [f64::from(p[0]), f64::from(p[1])];
                    min[0] = min[0].min(q[0]);
                    min[1] = min[1].min(q[1]);
                    max[0] = max[0].max(q[0]);
                    max[1] = max[1].max(q[1]);
                    pts.push(q);
                }
                Self::Lasso { pts, min, max }
            }
        }
    }

    fn contains_point(&self, p: [f64; 2]) -> bool {
        match self {
            Self::Rect { min, max } => {
                p[0] >= min[0] - 1e-12
                    && p[0] <= max[0] + 1e-12
                    && p[1] >= min[1] - 1e-12
                    && p[1] <= max[1] + 1e-12
            }
            Self::Lasso { pts, min, max } => {
                if p[0] < min[0] - 1e-12
                    || p[0] > max[0] + 1e-12
                    || p[1] < min[1] - 1e-12
                    || p[1] > max[1] + 1e-12
                {
                    return false;
                }
                point_in_polygon_2d(p, pts)
            }
        }
    }
}

#[inline]
fn mat4_to_f64(m: &[[f32; 4]; 4]) -> [[f64; 4]; 4] {
    let mut out = [[0.0_f64; 4]; 4];
    for c in 0..4 {
        for r in 0..4 {
            out[c][r] = f64::from(m[c][r]);
        }
    }
    out
}

#[inline]
fn project_clip(vp: &[[f64; 4]; 4], p: [f64; 3]) -> [f64; 4] {
    [
        vp[0][0] * p[0] + vp[1][0] * p[1] + vp[2][0] * p[2] + vp[3][0],
        vp[0][1] * p[0] + vp[1][1] * p[1] + vp[2][1] * p[2] + vp[3][1],
        vp[0][2] * p[0] + vp[1][2] * p[1] + vp[2][2] * p[2] + vp[3][2],
        vp[0][3] * p[0] + vp[1][3] * p[1] + vp[2][3] * p[2] + vp[3][3],
    ]
}

fn clip_triangle_near_w(clip: &[[f64; 4]; 3], w_min: f64, out: &mut [[f64; 4]; 4]) -> usize {
    let mut count = 0usize;
    for i in 0..3 {
        let a = clip[i];
        let b = clip[(i + 1) % 3];
        let a_in = a[3] >= w_min;
        let b_in = b[3] >= w_min;
        if a_in {
            out[count] = a;
            count += 1;
            if !b_in {
                let t = (w_min - a[3]) / (b[3] - a[3]);
                out[count] = lerp4(a, b, t);
                count += 1;
            }
        } else if b_in {
            let t = (w_min - a[3]) / (b[3] - a[3]);
            out[count] = lerp4(a, b, t);
            count += 1;
        }
    }
    count
}

#[inline]
fn lerp4(a: [f64; 4], b: [f64; 4], t: f64) -> [f64; 4] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}

fn poly_matches_shape(poly: &[[f64; 2]], shape: &ShapeF64, include_touching: bool) -> bool {
    // Quick 2D AABB reject first.
    let mut p_min = [f64::INFINITY; 2];
    let mut p_max = [f64::NEG_INFINITY; 2];
    for &v in poly {
        p_min[0] = p_min[0].min(v[0]);
        p_min[1] = p_min[1].min(v[1]);
        p_max[0] = p_max[0].max(v[0]);
        p_max[1] = p_max[1].max(v[1]);
    }
    let (s_min, s_max) = match shape {
        ShapeF64::Rect { min, max } => (*min, *max),
        ShapeF64::Lasso { min, max, .. } => (*min, *max),
    };
    if p_max[0] < s_min[0] - 1e-12
        || p_min[0] > s_max[0] + 1e-12
        || p_max[1] < s_min[1] - 1e-12
        || p_min[1] > s_max[1] + 1e-12
    {
        return false;
    }

    if !include_touching {
        for &v in poly {
            if !shape.contains_point(v) {
                return false;
            }
        }
        // For possibly concave lasso, verify no triangle edge leaves through a concave notch.
        if let ShapeF64::Lasso { pts, .. } = shape {
            let m = poly.len();
            let k = pts.len();
            for i in 0..m {
                let a = poly[i];
                let b = poly[(i + 1) % m];
                if !shape.contains_point([(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]) {
                    return false;
                }
                for j in 0..k {
                    let c = pts[j];
                    let d = pts[(j + 1) % k];
                    if segments_properly_cross(a, b, c, d) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    // include_touching == true: check if any vertex or centroid is inside shape.
    let mut cx = 0.0;
    let mut cy = 0.0;
    for &v in poly {
        if shape.contains_point(v) {
            return true;
        }
        cx += v[0];
        cy += v[1];
    }
    let inv_n = 1.0 / (poly.len() as f64);
    if shape.contains_point([cx * inv_n, cy * inv_n]) {
        return true;
    }

    // Check shape edges vs triangle edges, and shape vertices inside 2D triangle fan.
    let rect_corners;
    let shape_pts: &[[f64; 2]] = match shape {
        ShapeF64::Rect { min, max } => {
            rect_corners = [
                [min[0], min[1]],
                [max[0], min[1]],
                [max[0], max[1]],
                [min[0], max[1]],
            ];
            &rect_corners
        }
        ShapeF64::Lasso { pts, .. } => pts.as_slice(),
    };

    for &sp in shape_pts {
        if point_in_convex_poly_2d(sp, poly) {
            return true;
        }
    }

    let m = poly.len();
    let k = shape_pts.len();
    for i in 0..m {
        let a = poly[i];
        let b = poly[(i + 1) % m];
        for j in 0..k {
            let c = shape_pts[j];
            let d = shape_pts[(j + 1) % k];
            if segments_intersect_2d(a, b, c, d) {
                return true;
            }
        }
    }

    false
}

fn point_in_polygon_2d(p: [f64; 2], poly: &[[f64; 2]]) -> bool {
    let n = poly.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let a = poly[i];
        let b = poly[j];
        if point_on_segment_2d(p, a, b) {
            return true;
        }
        let cross_y = (a[1] > p[1]) != (b[1] > p[1]);
        if cross_y {
            let x_int = a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]);
            if p[0] < x_int {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

fn point_in_convex_poly_2d(p: [f64; 2], poly: &[[f64; 2]]) -> bool {
    if poly.len() < 3 {
        return false;
    }
    let mut i = 1usize;
    while i + 1 < poly.len() {
        if point_in_triangle_2d(p, poly[0], poly[i], poly[i + 1]) {
            return true;
        }
        i += 1;
    }
    false
}

fn point_in_triangle_2d(p: [f64; 2], a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> bool {
    let d1 = orient_2d(a, b, p);
    let d2 = orient_2d(b, c, p);
    let d3 = orient_2d(c, a, p);
    let area = orient_2d(a, b, c).abs();
    let eps = 1e-12 * area.max(1e-12);
    let has_neg = d1 < -eps || d2 < -eps || d3 < -eps;
    let has_pos = d1 > eps || d2 > eps || d3 > eps;
    if area <= 1e-15 {
        return point_on_segment_2d(p, a, b)
            || point_on_segment_2d(p, b, c)
            || point_on_segment_2d(p, c, a);
    }
    !(has_neg && has_pos)
}

#[inline]
fn orient_2d(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

fn point_on_segment_2d(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> bool {
    let abx = b[0] - a[0];
    let aby = b[1] - a[1];
    let len = (abx * abx + aby * aby).sqrt();
    let cross_abs = orient_2d(a, b, p).abs();
    if cross_abs > 1e-11 * len.max(1.0) {
        return false;
    }
    p[0] >= a[0].min(b[0]) - 1e-11
        && p[0] <= a[0].max(b[0]) + 1e-11
        && p[1] >= a[1].min(b[1]) - 1e-11
        && p[1] <= a[1].max(b[1]) + 1e-11
}

fn segments_intersect_2d(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
    if a[0].max(b[0]) < c[0].min(d[0]) - 1e-12
        || c[0].max(d[0]) < a[0].min(b[0]) - 1e-12
        || a[1].max(b[1]) < c[1].min(d[1]) - 1e-12
        || c[1].max(d[1]) < a[1].min(b[1]) - 1e-12
    {
        return false;
    }
    let o1 = orient_2d(a, b, c);
    let o2 = orient_2d(a, b, d);
    let o3 = orient_2d(c, d, a);
    let o4 = orient_2d(c, d, b);
    if ((o1 > 1e-12 && o2 < -1e-12) || (o1 < -1e-12 && o2 > 1e-12))
        && ((o3 > 1e-12 && o4 < -1e-12) || (o3 < -1e-12 && o4 > 1e-12))
    {
        return true;
    }
    point_on_segment_2d(c, a, b)
        || point_on_segment_2d(d, a, b)
        || point_on_segment_2d(a, c, d)
        || point_on_segment_2d(b, c, d)
}

fn segments_properly_cross(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
    let o1 = orient_2d(a, b, c);
    let o2 = orient_2d(a, b, d);
    let o3 = orient_2d(c, d, a);
    let o4 = orient_2d(c, d, b);
    ((o1 > 1e-11 && o2 < -1e-11) || (o1 < -1e-11 && o2 > 1e-11))
        && ((o3 > 1e-11 && o4 < -1e-11) || (o3 < -1e-11 && o4 > 1e-11))
}

fn compute_ortho_ray_backplane(mesh: &Mesh, fwd: [f64; 3]) -> Option<(f64, f64)> {
    let mut min_proj = f64::INFINITY;
    let mut max_proj = f64::NEG_INFINITY;
    for v_idx in 0..mesh.vertex_slots() {
        let v = v_idx as VertexId;
        if !mesh.is_vertex_alive(v) {
            continue;
        }
        let p = point(mesh, v);
        let proj = dot(p, fwd);
        min_proj = min_proj.min(proj);
        max_proj = max_proj.max(proj);
    }
    if !min_proj.is_finite() {
        return None;
    }
    let padding = (max_proj - min_proj).max(1.0) * 0.5 + 1.0;
    Some((min_proj, padding))
}

#[allow(clippy::too_many_arguments)]
fn is_triangle_visible(
    mesh: &Mesh,
    bvh: &Bvh,
    target: TriangleId,
    pts: [[f64; 3]; 3],
    vp: &[[f64; 4]; 4],
    shape: &ShapeF64,
    options: &ScreenSelectOptions,
    origin64: [f64; 3],
    fwd: [f64; 3],
    ortho_bounds: Option<(f64, f64)>,
) -> Result<bool> {
    const SAMPLES: [[f64; 3]; 7] = [
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [0.7, 0.15, 0.15],
        [0.15, 0.7, 0.15],
        [0.15, 0.15, 0.7],
        [0.45, 0.45, 0.1],
        [0.45, 0.1, 0.45],
        [0.1, 0.45, 0.45],
    ];

    let mut tested_any_in_shape = false;
    for (idx, bary) in SAMPLES.iter().enumerate() {
        let sample = [
            pts[0][0] * bary[0] + pts[1][0] * bary[1] + pts[2][0] * bary[2],
            pts[0][1] * bary[0] + pts[1][1] * bary[1] + pts[2][1] * bary[2],
            pts[0][2] * bary[0] + pts[1][2] * bary[1] + pts[2][2] * bary[2],
        ];
        if options.include_touching {
            let c = project_clip(vp, sample);
            if c[3] < 1e-7 {
                continue;
            }
            let ndc = [c[0] / c[3], c[1] / c[3]];
            if !shape.contains_point(ndc) {
                continue;
            }
        } else if idx > 0 && tested_any_in_shape {
            // Centroid failed occlusion; still allow interior sample points in case
            // a smaller front occluder covers only the centroid.
        }
        tested_any_in_shape = true;

        if test_single_sample_visibility(
            mesh,
            bvh,
            target,
            sample,
            options.is_orthographic,
            origin64,
            fwd,
            ortho_bounds,
        )? {
            return Ok(true);
        }
    }

    if !tested_any_in_shape {
        // Fallback when touching edge falls outside every fixed barycentric sample: test centroid.
        let sample = [
            (pts[0][0] + pts[1][0] + pts[2][0]) / 3.0,
            (pts[0][1] + pts[1][1] + pts[2][1]) / 3.0,
            (pts[0][2] + pts[1][2] + pts[2][2]) / 3.0,
        ];
        return test_single_sample_visibility(
            mesh,
            bvh,
            target,
            sample,
            options.is_orthographic,
            origin64,
            fwd,
            ortho_bounds,
        );
    }

    Ok(false)
}

#[allow(clippy::too_many_arguments)]
fn test_single_sample_visibility(
    mesh: &Mesh,
    bvh: &Bvh,
    target: TriangleId,
    sample: [f64; 3],
    is_orthographic: bool,
    origin64: [f64; 3],
    fwd: [f64; 3],
    ortho_bounds: Option<(f64, f64)>,
) -> Result<bool> {
    let (ray_origin, ray_dir) = if is_orthographic {
        let Some((min_proj, padding)) = ortho_bounds else {
            return Ok(false);
        };
        let travel = (dot(sample, fwd) - min_proj) + padding;
        let ro = [
            sample[0] - fwd[0] * travel,
            sample[1] - fwd[1] * travel,
            sample[2] - fwd[2] * travel,
        ];
        (ro, fwd)
    } else {
        let delta = sub(sample, origin64);
        let Some(unit) = normalize_f64(delta) else {
            return Ok(false);
        };
        (origin64, unit)
    };

    let ray = Ray {
        origin: [
            ray_origin[0] as f32,
            ray_origin[1] as f32,
            ray_origin[2] as f32,
        ],
        dir: [ray_dir[0] as f32, ray_dir[1] as f32, ray_dir[2] as f32],
    };
    let opts = RayOptions {
        min_t: 0.0,
        max_t: f32::INFINITY,
        cull: Cull::None,
    };

    if let Some(hit) = raycast(mesh, bvh, &ray, &opts)? {
        Ok(hit.triangle == target)
    } else {
        Ok(false)
    }
}

// ---------------------------------------------------------------------------
// Topological Boundary Operations
// ---------------------------------------------------------------------------

/// Grows `sel` outward by `rings` 1-ring vertex-adjacency steps.
///
/// Equivalent to calling [`grow_face_selection_with`] with [`AdjacencyKind::Vertex`].
pub fn grow_face_selection(
    mesh: &Mesh,
    sel: &FaceSelection,
    rings: usize,
) -> Result<FaceSelection> {
    grow_face_selection_with(mesh, sel, rings, AdjacencyKind::Vertex)
}

/// Grows `sel` outward by `rings` topological steps using `adjacency` (`Vertex` or `Edge`).
pub fn grow_face_selection_with(
    mesh: &Mesh,
    sel: &FaceSelection,
    rings: usize,
    adjacency: AdjacencyKind,
) -> Result<FaceSelection> {
    let mut out = sel.alive_copy(mesh)?;
    if rings == 0 || out.is_empty() || mesh.triangle_count() == 0 {
        return Ok(out);
    }

    let mut frontier = out.to_triangles_fallible(mesh)?;
    match adjacency {
        AdjacencyKind::Vertex => {
            let fans = VertexFans::build(mesh)?;
            for _ in 0..rings {
                let mut next = Vec::new();
                for &t in &frontier {
                    let Some(tri) = mesh.triangle(t) else {
                        continue;
                    };
                    for v in tri {
                        for &nb in fans.fan(v) {
                            if mesh.is_triangle_alive(nb) && out.try_insert(nb)? {
                                push(&mut next, nb)?;
                            }
                        }
                    }
                }
                if next.is_empty() {
                    break;
                }
                frontier = next;
            }
        }
        AdjacencyKind::Edge => {
            let edges = EdgeMap::build(mesh)?;
            for _ in 0..rings {
                let mut next = Vec::new();
                for &t in &frontier {
                    let Some(tri) = mesh.triangle(t) else {
                        continue;
                    };
                    for (u, v) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                        if u == v {
                            continue;
                        }
                        let (ea, eb) = ordered_edge(u, v);
                        for &nb in edges.triangles(ea, eb) {
                            if nb != t && mesh.is_triangle_alive(nb) && out.try_insert(nb)? {
                                push(&mut next, nb)?;
                            }
                        }
                    }
                }
                if next.is_empty() {
                    break;
                }
                frontier = next;
            }
        }
    }

    Ok(out)
}

/// Shrinks `sel` inward by `rings` iterations, peeling selected boundary triangles that
/// border either an unselected triangle or an open mesh boundary edge.
pub fn shrink_face_selection(
    mesh: &Mesh,
    sel: &FaceSelection,
    rings: usize,
) -> Result<FaceSelection> {
    let mut out = sel.alive_copy(mesh)?;
    if rings == 0 || out.is_empty() || mesh.triangle_count() == 0 {
        return Ok(out);
    }

    let edges = EdgeMap::build(mesh)?;
    let mut candidates = out.to_triangles_fallible(mesh)?;
    let mut queued = filled(mesh.triangle_slots(), 0u32)?;
    let mut stamp = 1u32;

    for _ in 0..rings {
        if candidates.is_empty() || out.is_empty() {
            break;
        }
        let mut to_peel = Vec::new();
        for &t in &candidates {
            if out.contains(t) && is_selection_boundary_tri(mesh, &edges, &out, t) {
                push(&mut to_peel, t)?;
            }
        }
        if to_peel.is_empty() {
            break;
        }
        for &t in &to_peel {
            out.remove(t);
        }
        if out.is_empty() {
            break;
        }

        // Only surviving selected neighbors of peeled triangles can become new boundary faces.
        let mut next_candidates = Vec::new();
        stamp = stamp.wrapping_add(1).max(1);
        for &t in &to_peel {
            let Some(tri) = mesh.triangle(t) else {
                continue;
            };
            for (u, v) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                if u == v {
                    continue;
                }
                let (ea, eb) = ordered_edge(u, v);
                for &nb in edges.triangles(ea, eb) {
                    if nb != t && out.contains(nb) && queued[nb as usize] != stamp {
                        queued[nb as usize] = stamp;
                        push(&mut next_candidates, nb)?;
                    }
                }
            }
        }
        candidates = next_candidates;
    }

    Ok(out)
}

#[inline]
fn is_selection_boundary_tri(
    mesh: &Mesh,
    edges: &EdgeMap,
    sel: &FaceSelection,
    t: TriangleId,
) -> bool {
    let Some(tri) = mesh.triangle(t) else {
        return false;
    };
    for (u, v) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
        if u == v {
            continue;
        }
        let (ea, eb) = ordered_edge(u, v);
        let inc = edges.triangles(ea, eb);
        if inc.len() < 2 {
            return true;
        }
        for &nb in inc {
            if nb != t && (!mesh.is_triangle_alive(nb) || !sel.contains(nb)) {
                return true;
            }
        }
    }
    false
}

/// Absorbs small unselected connected triangle islands (at most `max_hole_tris` triangles)
/// that are completely enclosed by selected triangles without touching an open mesh boundary.
pub fn fill_selection_holes(
    mesh: &Mesh,
    sel: &FaceSelection,
    max_hole_tris: usize,
) -> Result<FaceSelection> {
    let mut out = sel.alive_copy(mesh)?;
    if max_hole_tris == 0 || mesh.triangle_count() == 0 {
        return Ok(out);
    }

    let edges = EdgeMap::build(mesh)?;
    let slots = mesh.triangle_slots();
    let mut visited = filled(slots, false)?;
    let mut comp = Vec::new();

    for slot in 0..slots {
        let seed = slot as TriangleId;
        if visited[slot] || !mesh.is_triangle_alive(seed) || out.contains(seed) {
            continue;
        }

        comp.clear();
        visited[slot] = true;
        push(&mut comp, seed)?;

        let mut touches_boundary = false;
        let mut touches_selected = false;
        let mut head = 0usize;

        while head < comp.len() {
            let t = comp[head];
            head += 1;
            let Some(tri) = mesh.triangle(t) else {
                continue;
            };
            for (u, v) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                if u == v {
                    continue;
                }
                let (ea, eb) = ordered_edge(u, v);
                let inc = edges.triangles(ea, eb);
                if inc.len() == 1 {
                    touches_boundary = true;
                }
                for &nb in inc {
                    if nb == t || !mesh.is_triangle_alive(nb) {
                        continue;
                    }
                    if out.contains(nb) {
                        touches_selected = true;
                    } else if !visited[nb as usize] {
                        visited[nb as usize] = true;
                        push(&mut comp, nb)?;
                    }
                }
            }
        }

        if !touches_boundary && touches_selected && comp.len() <= max_hole_tris {
            for &t in &comp {
                out.try_insert(t)?;
            }
        }
    }

    Ok(out)
}

/// Smooths jagged selection borders by applying synchronous edge-neighbor majority voting
/// for `iterations` passes (ties preserve each triangle's existing state).
pub fn smooth_selection_boundary(
    mesh: &Mesh,
    sel: &FaceSelection,
    iterations: usize,
) -> Result<FaceSelection> {
    let mut cur = sel.alive_copy(mesh)?;
    if iterations == 0 || mesh.triangle_count() == 0 {
        return Ok(cur);
    }

    let edges = EdgeMap::build(mesh)?;
    let slots = mesh.triangle_slots();

    for _ in 0..iterations {
        let mut next = FaceSelection::with_slots(slots)?;
        for slot in 0..slots {
            let t = slot as TriangleId;
            let Some(tri) = mesh.triangle(t) else {
                continue;
            };

            let was_selected = cur.contains(t);
            let mut nbs = [TriangleId::MAX; 16];
            let mut nbs_len = 0usize;
            let mut total_neighbors = 0usize;
            let mut selected_neighbors = 0usize;

            for (u, v) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                if u == v {
                    continue;
                }
                let (ea, eb) = ordered_edge(u, v);
                for &nb in edges.triangles(ea, eb) {
                    if nb == t || !mesh.is_triangle_alive(nb) {
                        continue;
                    }
                    // Deduplicate edge neighbors within small inline cache before falling back.
                    if nbs[..nbs_len].contains(&nb) {
                        continue;
                    }
                    if nbs_len < nbs.len() {
                        nbs[nbs_len] = nb;
                        nbs_len += 1;
                    }
                    total_neighbors += 1;
                    if cur.contains(nb) {
                        selected_neighbors += 1;
                    }
                }
            }

            let keep = if total_neighbors == 0 {
                was_selected
            } else {
                match (selected_neighbors * 2).cmp(&total_neighbors) {
                    std::cmp::Ordering::Greater => true,
                    std::cmp::Ordering::Less => false,
                    std::cmp::Ordering::Equal => was_selected,
                }
            };

            if keep {
                next.insert_within_capacity(t);
            }
        }
        cur = next;
    }

    Ok(cur)
}

// ---------------------------------------------------------------------------
// Mesh Editing & Vertex Mask Interop
// ---------------------------------------------------------------------------

/// Removes all alive triangles in `sel` from `tx` along with any vertices that become
/// unreferenced as a result, preserving `UV_CHANNEL` on surviving vertices and cleanly
/// supporting transaction rollback/undo.
pub fn delete_selected_faces(tx: &mut Transaction, sel: &FaceSelection) -> Result<FaceEditReport> {
    let to_remove = sel.to_triangles_fallible(tx.mesh())?;
    if to_remove.is_empty() {
        return Ok(FaceEditReport::default());
    }

    let unused = unused_vertices(tx.mesh(), &to_remove, &[], &[])?;
    tx.remove_triangles(&to_remove)?;
    if !unused.is_empty() {
        tx.remove_vertices(&unused)?;
    }

    Ok(FaceEditReport {
        removed_triangles: to_remove.len(),
        removed_vertices: unused.len(),
    })
}

/// Extracts the alive triangles selected in `sel` into a standalone compact [`Mesh`],
/// preserving `origin` and `UV_CHANNEL` (if present on `mesh`) without mutating `mesh`.
pub fn extract_selected_faces(mesh: &Mesh, sel: &FaceSelection) -> Result<Mesh> {
    let tris = sel.to_triangles_fallible(mesh)?;
    if tris.is_empty() {
        let mut empty = Mesh::from_buffers_f64_with_origin(&[], &[], mesh.origin())?;
        if mesh.uvs().is_some() {
            empty.add_uvs(&[])?;
        }
        return Ok(empty);
    }

    let v_slots = mesh.vertex_slots();
    let mut remap = filled(v_slots, u32::MAX)?;
    for &t in &tris {
        if let Some(tri) = mesh.triangle(t) {
            for v in tri {
                remap[v as usize] = 0;
            }
        }
    }

    let mut used_verts = 0u32;
    for entry in &mut remap {
        if *entry == 0 {
            *entry = used_verts;
            used_verts += 1;
        }
    }

    let mut flat_pos = Vec::new();
    reserve(&mut flat_pos, (used_verts as usize) * 3)?;
    let origin = mesh.origin();
    let mut flat_uvs = Vec::new();
    let src_uvs = mesh.uvs();
    if src_uvs.is_some() {
        reserve(&mut flat_uvs, (used_verts as usize) * 2)?;
    }

    for slot in 0..v_slots {
        if remap[slot] != u32::MAX {
            let p = point(mesh, slot as VertexId);
            push(&mut flat_pos, origin[0] + p[0])?;
            push(&mut flat_pos, origin[1] + p[1])?;
            push(&mut flat_pos, origin[2] + p[2])?;
            if let Some(uvs) = src_uvs {
                push(&mut flat_uvs, uvs[slot * 2])?;
                push(&mut flat_uvs, uvs[slot * 2 + 1])?;
            }
        }
    }

    let mut flat_tris = Vec::new();
    reserve(&mut flat_tris, tris.len() * 3)?;
    for &t in &tris {
        if let Some(tri) = mesh.triangle(t) {
            push(&mut flat_tris, remap[tri[0] as usize])?;
            push(&mut flat_tris, remap[tri[1] as usize])?;
            push(&mut flat_tris, remap[tri[2] as usize])?;
        }
    }

    let mut out = Mesh::from_buffers_f64_with_origin(&flat_pos, &flat_tris, origin)?;
    if src_uvs.is_some() {
        out.add_uvs(&flat_uvs)?;
    }
    Ok(out)
}

/// Atomically extracts the selected triangles into a standalone [`Mesh`] and deletes them
/// (along with newly orphaned vertices) from `tx`, returning `(extracted_mesh, report)`.
///
/// Extraction completes before mutating `tx`, so any allocation failure leaves `tx` untouched.
pub fn split_selected_faces(
    tx: &mut Transaction,
    sel: &FaceSelection,
) -> Result<(Mesh, FaceEditReport)> {
    let extracted = extract_selected_faces(tx.mesh(), sel)?;
    let report = delete_selected_faces(tx, sel)?;
    Ok((extracted, report))
}

/// Converts a [`FaceSelection`] into sorted `(VertexId, weight)` pairs in `(0.0, 1.0]`.
///
/// Vertices incident to selected faces receive weight `1.0`. When `feather_rings > 0`,
/// surrounding graph rings out to distance `feather_rings` receive linearly decaying weights
/// `(feather_rings + 1 - ring) / (feather_rings + 1)`; vertices beyond `feather_rings` are omitted.
pub fn face_selection_to_vertex_mask(
    mesh: &Mesh,
    sel: &FaceSelection,
    feather_rings: usize,
) -> Result<Vec<(VertexId, f32)>> {
    let v_slots = mesh.vertex_slots();
    if v_slots == 0 || sel.is_empty() || mesh.triangle_count() == 0 {
        return Ok(Vec::new());
    }

    let mut dist = filled(v_slots, u32::MAX)?;
    let mut frontier = Vec::new();
    let mut core_count = 0usize;

    for t in sel.iter() {
        if (t as usize) >= mesh.triangle_slots() || !mesh.is_triangle_alive(t) {
            continue;
        }
        let Some(tri) = mesh.triangle(t) else {
            continue;
        };
        for v in tri {
            if mesh.is_vertex_alive(v) && dist[v as usize] == u32::MAX {
                dist[v as usize] = 0;
                push(&mut frontier, v)?;
                core_count += 1;
            }
        }
    }

    if core_count == 0 {
        return Ok(Vec::new());
    }

    let mut emit_count = core_count;
    if feather_rings > 0 {
        let fans = VertexFans::build(mesh)?;
        for ring in 1..=feather_rings {
            let mut next = Vec::new();
            for &u in &frontier {
                for &t in fans.fan(u) {
                    let Some(tri) = mesh.triangle(t) else {
                        continue;
                    };
                    for w in tri {
                        if mesh.is_vertex_alive(w) && dist[w as usize] == u32::MAX {
                            dist[w as usize] = ring as u32;
                            push(&mut next, w)?;
                            emit_count += 1;
                        }
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }
    }

    let mut out = Vec::new();
    reserve(&mut out, emit_count)?;
    let denom = (feather_rings + 1) as f64;
    for (slot, &d) in dist.iter().enumerate() {
        let v = slot as VertexId;
        if !mesh.is_vertex_alive(v) {
            continue;
        }
        if (d as usize) <= feather_rings {
            let w = if d == 0 {
                1.0_f32
            } else {
                (((feather_rings + 1 - d as usize) as f64) / denom) as f32
            };
            push(&mut out, (v, w))?;
        }
    }

    Ok(out)
}

/// Converts per-vertex mask weights in `[0.0, 1.0]` back into a [`FaceSelection`].
///
/// Unlisted vertices default to weight `0.0`. A triangle is selected when all three of its
/// vertices meet `threshold` (`weight >= threshold`, or `weight > 0.0` when `threshold == 0.0`),
/// ensuring exact round-trip preservation with [`face_selection_to_vertex_mask`].
///
/// # Errors
/// Returns [`OpsError::InvalidInput`] if `threshold` or any weight is non-finite or outside
/// `[0.0, 1.0]`, or if any vertex ID is dead or out of range.
pub fn vertex_mask_to_face_selection(
    mesh: &Mesh,
    weights: &[(VertexId, f32)],
    threshold: f32,
) -> Result<FaceSelection> {
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(OpsError::InvalidInput(
            "threshold must be finite and within [0, 1]",
        ));
    }

    let mut vert_w = filled(mesh.vertex_slots(), 0.0_f32)?;
    for &(v, w) in weights {
        if !mesh.is_vertex_alive(v) {
            return Err(OpsError::InvalidInput(
                "weight references a dead or out-of-range vertex",
            ));
        }
        if !w.is_finite() || !(0.0..=1.0).contains(&w) {
            return Err(OpsError::InvalidInput(
                "vertex weight must be finite and within [0, 1]",
            ));
        }
        vert_w[v as usize] = w;
    }

    let mut sel = FaceSelection::with_slots(mesh.triangle_slots())?;
    for slot in 0..mesh.triangle_slots() {
        let t = slot as TriangleId;
        let Some(tri) = mesh.triangle(t) else {
            continue;
        };
        let min_w = vert_w[tri[0] as usize]
            .min(vert_w[tri[1] as usize])
            .min(vert_w[tri[2] as usize]);
        let selected = if threshold == 0.0 {
            min_w > 0.0
        } else {
            min_w >= threshold
        };
        if selected {
            sel.insert_within_capacity(t);
        }
    }

    Ok(sel)
}
