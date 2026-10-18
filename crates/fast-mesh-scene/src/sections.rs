//! Plane sections of meshes of the scene (spec §6, §11).
//!
//! A section cuts the meshes it selects with one world [`Plane`]. For each
//! selected mesh it keeps a [`SectionResult`]: the contours of the cut in
//! world and plane `(u, v)` coordinates, their lengths and signed areas, and
//! optionally the filled cross-section.
//!
//! **What an update recomputes.** Each section has a change accumulator on
//! every mesh it selects (a [`SectionHook`], fed like a subscriber's report).
//! An entry is stale when the hook holds a non-empty change set, "everything"
//! or a placement change, or when the section's plane or fill changed, or the
//! mesh was just selected. A stale entry is recomputed in full: the world
//! plane is mapped into the mesh's local frame, cut with [`slice()`], and the
//! points are mapped back.
//!
//! **Edits that cannot reach the plane** (spec §6.3, §11.5). Each entry
//! keeps the sorted vertex ids of its crossed edges and its crossed
//! triangles. An edit, undo or redo leaves the entry current, with nothing
//! recomputed or reported, unless one of these holds for the changes merged
//! since the last computation:
//! 1. a moved or removed vertex is a crossed-edge vertex;
//! 2. a removed or rewritten triangle is a crossed triangle;
//! 3. a moved or added vertex that is not a crossed-edge vertex has an alive
//!    incident triangle with vertices on both sides;
//! 4. an added or rewritten triangle is alive and has vertices on both sides.
//!
//! Sides use the cut's own rule (`s ≥ 0` is `+`, from the stored `f32`
//! positions) against the same local plane. Before the edit no edge at a
//! vertex outside the crossed set crossed the plane, so these cover every
//! segment and crossing point the cut depends on. A placement, plane or fill
//! change, compaction and "everything" always recompute.

use fast_mesh_query::{Fill, Gap2, Plane, Slice, contour_gap, contour_length, signed_area};
use fast_mesh_store::{ChangeSet, Mesh, StoreError, TriangleId, VertexId, par};

use crate::placement::{Placement, local_to_world, world_to_local};
use crate::scene::Entry;
use crate::{MeshId, Result, Scene, SceneError, SectionId, UpdateMode};

/// A plane section to register with
/// [`Scene::add_section`](crate::Scene::add_section).
#[derive(Debug, Clone, PartialEq)]
pub struct SectionDesc {
    /// The cutting plane, in world coordinates. Its `(u, v)` frame is the
    /// frame of every result's 2D points.
    pub plane: Plane,
    /// The meshes cut, each once.
    pub meshes: Vec<MeshId>,
    /// Whether results carry the filled cross-section.
    pub fill: bool,
}

/// Whether a section's result for a mesh matches the scene as it is now,
/// from [`Scene::section_status`](crate::Scene::section_status).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionStatus {
    /// Computed, and nothing it depends on changed since.
    Current,
    /// Not computed yet, or something it depends on changed since.
    Stale,
}

/// One chain of a section: a closed loop or an open polyline.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SectionContour {
    /// The crossing point of each edge in `edges`, in world coordinates.
    pub points: Vec<[f64; 3]>,
    /// The same points in the section plane's `(u, v)` frame.
    pub uv: Vec<[f64; 2]>,
    /// The crossed edge `(min id, max id)` of each point.
    pub edges: Vec<(VertexId, VertexId)>,
    /// The triangle of each segment: `triangles[i]` joins point `i` to point
    /// `i + 1`, and the last one of a closed contour joins the last point to
    /// the first.
    pub triangles: Vec<TriangleId>,
    /// Whether the last point joins back to the first.
    pub closed: bool,
    /// Whether every segment runs the same way along the contour. On a
    /// closed, outward-wound mesh outer loops run counter-clockwise in
    /// `(u, v)` and holes clockwise.
    pub oriented: bool,
    /// Whether no end of the contour is a split at a non-manifold edge.
    pub manifold: bool,
    /// The length of the polyline (with the closing segment when closed).
    pub length: f64,
    /// The signed area in `(u, v)`, positive counter-clockwise; 0 when open.
    pub area: f64,
}

/// A section's result for one mesh, from
/// [`Scene::section`](crate::Scene::section).
#[derive(Debug, Clone, PartialEq)]
pub struct SectionResult {
    /// The contours, sorted by their first edge key.
    pub contours: Vec<SectionContour>,
    /// The summed length of every contour.
    pub length: f64,
    /// The summed signed area of the closed, oriented contours.
    pub area: f64,
    /// The filled cross-section over the closed contours, when the section
    /// has fill on and the result was computed. Its points are `(u, v)`;
    /// [`Plane::from_uv`] of the section plane maps them to the world.
    pub fill: Option<Fill>,
    /// Whether the fill looks good (see [`fast_mesh_query::fill`]): best
    /// effort, a bad fill can go unreported. `true` when `fill` is `None`.
    pub fill_ok: bool,
}

impl Default for SectionResult {
    /// The result before the first computation: no contours, zero length
    /// and area, no fill.
    fn default() -> Self {
        SectionResult {
            contours: Vec::new(),
            length: 0.0,
            area: 0.0,
            fill: None,
            fill_ok: true,
        }
    }
}

/// The closest pair between two meshes' contours in a section, from
/// [`Scene::section_gap`](crate::Scene::section_gap).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gap {
    /// The distance between `a` and `b` in the plane; 0 when the contours
    /// cross or touch.
    pub distance: f64,
    /// The point on the first mesh's contours, in world coordinates.
    pub a: [f64; 3],
    /// The point on the second mesh's contours, in world coordinates.
    pub b: [f64; 3],
}

#[cfg(test)]
thread_local! {
    /// Test-only: batches of entries computed in parallel on this thread.
    static PARALLEL_SECTION_BATCHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Triangles, summed over the meshes of the section entries due in one
/// update, from which the entries are cut in parallel, one entry per task,
/// with the `parallel` feature.
///
/// Measured on the i5-13600K with one section over three spheres, dragging
/// its plane (three entries due per plane): at 3 x 960 triangles parallel is
/// 1.0 to 1.7 times slower, at 3 x 2,208 even to 1.4 times faster, at
/// 3 x 3,968 1.45 times faster and at 3 x 36k 2.6 times.
const PAR_MIN_SECTION_WORK: usize = 8 * 1024;

fn oom() -> SceneError {
    SceneError::Store(StoreError::OutOfMemory)
}

/// A `Vec` with room for `n` items, or `OutOfMemory`.
fn try_vec<T>(n: usize) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| oom())?;
    Ok(v)
}

/// What happened to one mesh of a section since its entry was last
/// computed.
#[derive(Debug, Clone)]
pub(crate) struct SectionHook {
    pub(crate) section: SectionId,
    /// Merged changes; `None` means "everything".
    pub(crate) changes: Option<ChangeSet>,
    pub(crate) placement_changed: bool,
}

impl SectionHook {
    fn new(section: SectionId) -> SectionHook {
        SectionHook {
            section,
            changes: Some(ChangeSet::default()),
            placement_changed: false,
        }
    }

    fn is_clean(&self) -> bool {
        !self.placement_changed && self.changes.as_ref().is_some_and(ChangeSet::is_empty)
    }
}

/// A section's state for one selected mesh.
struct SectionEntry {
    mesh: MeshId,
    result: SectionResult,
    /// The plane or fill changed, or the mesh was just selected.
    stale: bool,
    computed_once: bool,
    /// The vertices of the crossed edges, ascending without duplicates.
    crossed_vertices: Vec<VertexId>,
    /// The crossed triangles, ascending without duplicates.
    crossed_triangles: Vec<TriangleId>,
    /// The world plane of the last computation; `None` before the first.
    plane: Option<Plane>,
}

impl SectionEntry {
    fn new(mesh: MeshId) -> SectionEntry {
        SectionEntry {
            mesh,
            result: SectionResult::default(),
            stale: true,
            computed_once: false,
            crossed_vertices: Vec::new(),
            crossed_triangles: Vec::new(),
            plane: None,
        }
    }
}

/// A registered section.
pub(crate) struct Section {
    id: SectionId,
    desc: SectionDesc,
    /// One per mesh of `desc.meshes`, in that order.
    entries: Vec<SectionEntry>,
}

/// A freshly computed entry.
struct Computed {
    result: SectionResult,
    crossed_vertices: Vec<VertexId>,
    crossed_triangles: Vec<TriangleId>,
}

impl Entry {
    fn section_hook(&self, id: SectionId) -> Option<&SectionHook> {
        self.section_hooks.iter().find(|h| h.section == id)
    }

    fn reset_section_hook(&mut self, id: SectionId) {
        if let Some(h) = self.section_hooks.iter_mut().find(|h| h.section == id) {
            *h = SectionHook::new(id);
        }
    }

    fn remove_section_hook(&mut self, id: SectionId) {
        self.section_hooks.retain(|h| h.section != id);
    }
}

/// `Rᵀ d`: a world direction in the mesh's local frame.
fn rotate_back(rotation: &[[f64; 3]; 3], d: [f64; 3]) -> [f64; 3] {
    let r = rotation;
    std::array::from_fn(|i| r[0][i] * d[0] + r[1][i] * d[1] + r[2][i] * d[2])
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn unit(a: [f64; 3]) -> [f64; 3] {
    let n = dot(a, a).sqrt();
    a.map(|c| c / n)
}

/// The world plane in the mesh's stored local frame (origin excluded), with
/// axes `Rᵀu` and `Rᵀv`, so that local `(u, v)` equals world `(u, v)`. The
/// axes are re-orthonormalised: a placement's rotation is only orthonormal
/// within `1e-6`.
fn local_plane(p: &Placement, mesh_origin: [f64; 3], plane: &Plane) -> Result<Plane> {
    let origin = world_to_local(p, mesh_origin, plane.origin());
    let u = unit(rotate_back(&p.rotation, plane.u()));
    let v = rotate_back(&p.rotation, plane.v());
    let d = dot(v, u);
    let v = unit(std::array::from_fn(|k| v[k] - d * u[k]));
    Ok(Plane::with_axes(origin, u, v)?)
}

/// Cuts the entry's mesh with `desc`'s plane; its BVH must be current.
fn compute(entry: &Entry, desc: &SectionDesc) -> Result<Computed> {
    let plane = &desc.plane;
    let (p, o) = (&entry.placement, entry.mesh.origin());
    let mut cut = Slice::default();
    fast_mesh_query::slice(
        &entry.mesh,
        &entry.bvh,
        &local_plane(p, o, plane)?,
        &mut cut,
    )?;

    let mut contours = try_vec(cut.contours.len())?;
    let (mut length, mut area) = (0.0, 0.0);
    let (mut nv, mut nt) = (0, 0);
    for c in cut.contours {
        let mut points = try_vec(c.points.len())?;
        let mut uv = try_vec(c.points.len())?;
        for &q in &c.points {
            let w = local_to_world(p, o, q);
            points.push(w);
            uv.push(plane.to_uv(w));
        }
        let l = contour_length(&uv, c.closed);
        let a = if c.closed { signed_area(&uv) } else { 0.0 };
        length += l;
        if c.closed && c.oriented {
            area += a;
        }
        nv += 2 * c.edges.len();
        nt += c.triangles.len();
        contours.push(SectionContour {
            points,
            uv,
            edges: c.edges,
            triangles: c.triangles,
            closed: c.closed,
            oriented: c.oriented,
            manifold: c.manifold,
            length: l,
            area: a,
        });
    }

    let mut crossed_vertices = try_vec(nv)?;
    let mut crossed_triangles = try_vec(nt)?;
    for c in &contours {
        crossed_vertices.extend(c.edges.iter().flat_map(|&(a, b)| [a, b]));
        crossed_triangles.extend_from_slice(&c.triangles);
    }
    crossed_vertices.sort_unstable();
    crossed_vertices.dedup();
    crossed_triangles.sort_unstable();
    crossed_triangles.dedup();

    let (fill, fill_ok) = if desc.fill {
        let mut closed = try_vec(contours.len())?;
        closed.extend(
            contours
                .iter()
                .filter(|c| c.closed)
                .map(|c| (c.uv.as_slice(), c.oriented)),
        );
        let mut f = Fill::default();
        let ok = fast_mesh_query::fill(&closed, &mut f)?;
        (Some(f), ok)
    } else {
        (None, true)
    };

    Ok(Computed {
        result: SectionResult {
            contours,
            length,
            area,
            fill,
            fill_ok,
        },
        crossed_vertices,
        crossed_triangles,
    })
}

/// Whether the changes `c`, made since `se` was computed with the local
/// plane `plane`, can alter its cut (see the module docs). `mesh` is the
/// current mesh; its incidence index is built if needed, and failing that
/// the answer is `true`.
fn reaches_plane(mesh: &mut Mesh, plane: &Plane, c: &ChangeSet, se: &SectionEntry) -> bool {
    let crossed_vertex = |v: &VertexId| se.crossed_vertices.binary_search(v).is_ok();
    let crossed_triangle = |t: &TriangleId| se.crossed_triangles.binary_search(t).is_ok();
    // 1 and 2: something the cut used moved or went.
    if c.vertices_moved
        .iter()
        .chain(&c.vertices_removed)
        .any(crossed_vertex)
        || c.triangles_removed
            .iter()
            .chain(&c.triangles_changed)
            .any(crossed_triangle)
    {
        return true;
    }
    // The side rule of the cut: `+` iff `s ≥ 0`, from the stored position.
    let straddles = |mesh: &Mesh, t: TriangleId| {
        mesh.triangle(t).is_some_and(|tri| {
            let plus = tri.map(|v| {
                let p = mesh.positions()[v as usize].map(f64::from);
                plane.signed_distance(p) >= 0.0
            });
            plus[0] != plus[1] || plus[1] != plus[2]
        })
    };
    // 4: a new or rewritten triangle crosses.
    if c.triangles_added
        .iter()
        .chain(&c.triangles_changed)
        .any(|&t| straddles(mesh, t))
    {
        return true;
    }
    // 3: a triangle around a moved or added vertex off the cut now crosses.
    let mut off_cut = c
        .vertices_moved
        .iter()
        .chain(&c.vertices_added)
        .filter(|v| !crossed_vertex(v))
        .peekable();
    if off_cut.peek().is_none() {
        return false;
    }
    if mesh.ensure_incidence().is_err() {
        return true;
    }
    let mesh = &*mesh;
    let Some(incidence) = mesh.incidence() else {
        return true;
    };
    off_cut.any(|&v| incidence.triangles(v).iter().any(|&t| straddles(mesh, t)))
}

/// A polyline's `(u, v)` points and whether it is closed.
type Polyline<'a> = (&'a [[f64; 2]], bool);

/// The polylines of a result's contours, for [`contour_gap`].
fn polylines(r: &SectionResult) -> Result<Vec<Polyline<'_>>> {
    let mut out = try_vec(r.contours.len())?;
    out.extend(r.contours.iter().map(|c| (c.uv.as_slice(), c.closed)));
    Ok(out)
}

impl Scene {
    /// Registers a plane section of the meshes `desc.meshes` and returns its
    /// id. Sections are not in the history: undo does not move planes.
    ///
    /// Each selected mesh gets its own [`SectionResult`], which follows the
    /// [`UpdateMode`] of that mesh: an Eager mesh's is computed right away
    /// and after every edit, placement change, undo, redo or cancel of the
    /// mesh, and after every plane or fill change; a Lazy one's by
    /// [`Scene::flush`], [`Scene::update`] or [`Scene::update_sections`]; a
    /// Manual one's by [`Scene::update`] or [`Scene::update_sections`]. A
    /// failed Eager computation is swallowed and leaves the entry stale.
    /// Until its first computation an entry reads as an empty result with
    /// status [`SectionStatus::Stale`].
    ///
    /// Results that change are reported in the mesh's
    /// [`MeshChanges::sections_changed`](crate::MeshChanges::sections_changed).
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale mesh id;
    /// [`SceneError::InvalidInput`] for a mesh listed twice; a memory error.
    /// Never [`SceneError::StaleBvh`]. The scene is unchanged on error.
    pub fn add_section(&mut self, desc: SectionDesc) -> Result<SectionId> {
        self.check_selection(&desc.meshes)?;
        let next = self
            .next_section
            .checked_add(1)
            .ok_or(SceneError::InvalidInput("too many sections"))?;
        let mut entries = try_vec(desc.meshes.len())?;
        for &m in &desc.meshes {
            self.entries
                .get_mut(m)
                .ok_or(SceneError::UnknownMesh)?
                .section_hooks
                .try_reserve(1)
                .map_err(|_| oom())?;
            entries.push(SectionEntry::new(m));
        }
        self.sections.try_reserve(1).map_err(|_| oom())?;
        // Nothing can fail from here on.
        let id = SectionId(self.next_section);
        self.next_section = next;
        for &m in &desc.meshes {
            if let Some(e) = self.entries.get_mut(m) {
                e.section_hooks.push(SectionHook::new(id));
            }
        }
        self.sections.push(Section { id, desc, entries });
        self.run_section_eager(self.sections.len() - 1);
        Ok(id)
    }

    /// Removes a section. The removal is reported in the
    /// [`MeshChanges::sections_changed`](crate::MeshChanges::sections_changed)
    /// of every mesh it selected, and the id is rejected from then on.
    ///
    /// # Errors
    /// [`SceneError::UnknownSection`] for an unknown or removed section.
    pub fn remove_section(&mut self, id: SectionId) -> Result<()> {
        let i = self.section_index(id)?;
        let s = self.sections.remove(i);
        for se in &s.entries {
            if let Some(e) = self.entries.get_mut(se.mesh) {
                e.remove_section_hook(id);
                e.report_section(id);
            }
        }
        Ok(())
    }

    /// Moves a section's plane. Every entry of the section becomes stale
    /// (Eager ones are recomputed right away); nothing else is touched.
    /// Computing an Eager entry needs a current BVH, so an Eager mesh whose
    /// earlier BVH update failed has that update retried here.
    ///
    /// # Errors
    /// [`SceneError::UnknownSection`] for an unknown or removed section.
    pub fn set_section_plane(&mut self, id: SectionId, plane: Plane) -> Result<()> {
        let i = self.section_index(id)?;
        self.sections[i].desc.plane = plane;
        self.restale_section(i);
        Ok(())
    }

    /// Changes the meshes a section cuts. Meshes it keeps keep their entries;
    /// added meshes start stale (Eager ones are computed right away); removed
    /// meshes drop their entries, which is reported in their
    /// [`MeshChanges::sections_changed`](crate::MeshChanges::sections_changed).
    ///
    /// # Errors
    /// [`SceneError::UnknownSection`]; [`SceneError::UnknownMesh`] for a
    /// stale mesh id; [`SceneError::InvalidInput`] for a mesh listed twice; a
    /// memory error. Never [`SceneError::StaleBvh`]. The scene is unchanged
    /// on error.
    pub fn set_section_meshes(&mut self, id: SectionId, meshes: Vec<MeshId>) -> Result<()> {
        let i = self.section_index(id)?;
        self.check_selection(&meshes)?;
        let mut entries = try_vec(meshes.len())?;
        let old = &self.sections[i].entries;
        for &m in &meshes {
            if !old.iter().any(|se| se.mesh == m) {
                self.entries
                    .get_mut(m)
                    .ok_or(SceneError::UnknownMesh)?
                    .section_hooks
                    .try_reserve(1)
                    .map_err(|_| oom())?;
            }
        }
        // Nothing can fail from here on.
        let mut old = std::mem::take(&mut self.sections[i].entries);
        for &m in &meshes {
            match old.iter().position(|se| se.mesh == m) {
                Some(k) => entries.push(old.swap_remove(k)),
                None => {
                    if let Some(e) = self.entries.get_mut(m) {
                        e.section_hooks.push(SectionHook::new(id));
                    }
                    entries.push(SectionEntry::new(m));
                }
            }
        }
        for se in &old {
            if let Some(e) = self.entries.get_mut(se.mesh) {
                e.remove_section_hook(id);
                e.report_section(id);
            }
        }
        let s = &mut self.sections[i];
        s.entries = entries;
        s.desc.meshes = meshes;
        self.run_section_eager(i);
        Ok(())
    }

    /// Turns a section's fill on or off. Every entry of the section becomes
    /// stale (Eager ones are recomputed right away); nothing else is touched.
    /// As with [`Scene::set_section_plane`], an Eager mesh whose earlier BVH
    /// update failed has that update retried.
    ///
    /// # Errors
    /// [`SceneError::UnknownSection`] for an unknown or removed section.
    pub fn set_section_fill(&mut self, id: SectionId, fill: bool) -> Result<()> {
        let i = self.section_index(id)?;
        self.sections[i].desc.fill = fill;
        self.restale_section(i);
        Ok(())
    }

    /// Computes every stale section entry, whatever the mode of its mesh,
    /// bringing the BVHs it needs up to date under the query rules (Lazy
    /// ones are updated). Renderers call this (or [`Scene::flush`]), then
    /// read any number of results.
    ///
    /// # Errors
    /// The first error, for example [`SceneError::StaleBvh`] for an entry
    /// whose mesh has a stale Manual BVH: such an entry stays stale, and
    /// makes every call return `StaleBvh` until that BVH is brought up to
    /// date with [`Scene::update`]. Every entry is still attempted.
    pub fn update_sections(&mut self) -> Result<()> {
        self.run_sections(|_| true, None)
    }

    /// The section's result for `mesh` as of its last computation (see
    /// [`Scene::section_status`]); empty before the first one. Never
    /// updates.
    ///
    /// # Errors
    /// [`SceneError::UnknownSection`]; [`SceneError::UnknownMesh`] for a
    /// stale mesh id; [`SceneError::InvalidInput`] for a live mesh the
    /// section does not select.
    pub fn section(&self, id: SectionId, mesh: MeshId) -> Result<&SectionResult> {
        let (i, k) = self.section_entry(id, mesh)?;
        Ok(&self.sections[i].entries[k].result)
    }

    /// The section's description as last set: its plane (the one
    /// [`Scene::set_section_plane`] gave, which results not yet recomputed
    /// may lag behind; see [`Scene::section_plane`]), meshes and fill.
    ///
    /// # Errors
    /// [`SceneError::UnknownSection`] for an unknown or removed section.
    pub fn section_desc(&self, id: SectionId) -> Result<&SectionDesc> {
        let i = self.section_index(id)?;
        Ok(&self.sections[i].desc)
    }

    /// Whether the section's result for `mesh` is current. Never updates.
    ///
    /// # Errors
    /// As for [`Scene::section`].
    pub fn section_status(&self, id: SectionId, mesh: MeshId) -> Result<SectionStatus> {
        let (i, k) = self.section_entry(id, mesh)?;
        Ok(if self.entry_current(i, k) {
            SectionStatus::Current
        } else {
            SectionStatus::Stale
        })
    }

    /// The world plane the section's result for `mesh` was computed with,
    /// whose `(u, v)` frame its 2D points and fill are in; `None` before the
    /// first computation. It differs from the section's plane while a plane
    /// move is not yet computed. Never updates.
    ///
    /// # Errors
    /// As for [`Scene::section`].
    pub fn section_plane(&self, id: SectionId, mesh: MeshId) -> Result<Option<Plane>> {
        let (i, k) = self.section_entry(id, mesh)?;
        Ok(self.sections[i].entries[k].plane)
    }

    /// The closest pair between the contours of meshes `a` and `b` in the
    /// section, measured in the plane, with its points in the world; `None`
    /// when either has no contour segments. Never updates: it reads the last
    /// computed results, so for a stale entry (see [`Scene::section_status`])
    /// it describes the scene as of that computation. A gap is only defined
    /// within one plane: when the two results were computed with different
    /// planes (one was updated after a plane move, the other not yet), it is
    /// `None`; call [`Scene::update_sections`] first.
    ///
    /// # Errors
    /// As for [`Scene::section`], for either mesh;
    /// [`SceneError::InvalidInput`] when `a` and `b` are the same mesh; a
    /// memory error.
    pub fn section_gap(&self, id: SectionId, a: MeshId, b: MeshId) -> Result<Option<Gap>> {
        let (i, ka) = self.section_entry(id, a)?;
        let (_, kb) = self.section_entry(id, b)?;
        if a == b {
            return Err(SceneError::InvalidInput(
                "a section gap needs two different meshes",
            ));
        }
        let s = &self.sections[i];
        let (ea, eb) = (&s.entries[ka], &s.entries[kb]);
        // Each side's points are in the frame of the plane it was cut with.
        let (Some(plane_a), Some(plane_b)) = (ea.plane, eb.plane) else {
            return Ok(None);
        };
        if plane_a != plane_b {
            return Ok(None);
        }
        let pa = polylines(&ea.result)?;
        let pb = polylines(&eb.result)?;
        Ok(contour_gap(&pa, &pb)?.map(|g: Gap2| Gap {
            distance: g.distance,
            a: plane_a.from_uv(g.a),
            b: plane_b.from_uv(g.b),
        }))
    }

    /// The id of the section with this [`SectionId::raw`] value, or `None`
    /// when there is none (never added, or removed).
    pub fn section_id(&self, raw: u32) -> Option<SectionId> {
        self.sections.iter().map(|s| s.id).find(|id| id.0 == raw)
    }

    fn section_index(&self, id: SectionId) -> Result<usize> {
        self.sections
            .iter()
            .position(|s| s.id == id)
            .ok_or(SceneError::UnknownSection)
    }

    /// The section's index and the index of `mesh`'s entry in it.
    fn section_entry(&self, id: SectionId, mesh: MeshId) -> Result<(usize, usize)> {
        let i = self.section_index(id)?;
        self.entry(mesh)?;
        let k = self.sections[i]
            .entries
            .iter()
            .position(|se| se.mesh == mesh)
            .ok_or(SceneError::InvalidInput(
                "the section does not select this mesh",
            ))?;
        Ok((i, k))
    }

    /// Every mesh live and listed once.
    fn check_selection(&self, meshes: &[MeshId]) -> Result<()> {
        for (j, &m) in meshes.iter().enumerate() {
            self.entry(m)?;
            if meshes[..j].contains(&m) {
                return Err(SceneError::InvalidInput(
                    "a section lists a mesh more than once",
                ));
            }
        }
        Ok(())
    }

    /// Computed at least once, and nothing changed since.
    fn entry_current(&self, i: usize, k: usize) -> bool {
        let s = &self.sections[i];
        let se = &s.entries[k];
        se.computed_once
            && !se.stale
            && self
                .entry(se.mesh)
                .ok()
                .and_then(|e| e.section_hook(s.id))
                .is_some_and(SectionHook::is_clean)
    }

    /// Clears the pending changes of entry `k` of section `i` when they
    /// cannot reach its plane (see the module docs), which leaves the entry
    /// current with nothing recomputed or reported. Returns whether it did.
    fn settle_entry(&mut self, i: usize, k: usize) -> bool {
        let s = &self.sections[i];
        let se = &s.entries[k];
        if se.stale || !se.computed_once {
            return false;
        }
        let Some(Entry {
            mesh,
            placement,
            section_hooks,
            ..
        }) = self.entries.get_mut(se.mesh)
        else {
            return false;
        };
        let Some(h) = section_hooks.iter_mut().find(|h| h.section == s.id) else {
            return false;
        };
        if h.placement_changed {
            return false;
        }
        let Some(c) = &h.changes else {
            return false;
        };
        if !c.is_empty() {
            // The local plane the computation used: the plane, the placement
            // and the mesh's origin are all unchanged since.
            let Ok(plane) = local_plane(placement, mesh.origin(), &s.desc.plane) else {
                return false;
            };
            if reaches_plane(mesh, &plane, c, se) {
                return false;
            }
        }
        *h = SectionHook::new(s.id);
        true
    }

    /// Stales every entry of section `i`, then computes the Eager ones.
    fn restale_section(&mut self, i: usize) {
        for se in &mut self.sections[i].entries {
            se.stale = true;
        }
        self.run_section_eager(i);
    }

    /// Computes the stale entries of section `i` whose mesh is Eager,
    /// swallowing failures.
    fn run_section_eager(&mut self, i: usize) {
        let mut due = Vec::new();
        for k in 0..self.sections[i].entries.len() {
            let eager = self
                .entry(self.sections[i].entries[k].mesh)
                .is_ok_and(|e| e.mode == UpdateMode::Eager);
            if eager && !self.entry_current(i, k) {
                if due.try_reserve(1).is_err() {
                    return;
                }
                due.push((i, k));
            }
        }
        let _ = self.compute_entries(&due);
    }

    /// Computes the stale entries whose mesh's mode passes `want` and (with
    /// `touched`) is one of those meshes. Entries of those meshes whose
    /// changes cannot reach the plane are settled first, whatever the mode.
    /// Returns the first error; every entry is attempted.
    pub(crate) fn run_sections(
        &mut self,
        want: impl Fn(UpdateMode) -> bool,
        touched: Option<&[MeshId]>,
    ) -> Result<()> {
        let mut due = Vec::new();
        for i in 0..self.sections.len() {
            for k in 0..self.sections[i].entries.len() {
                let mesh = self.sections[i].entries[k].mesh;
                if touched.is_some_and(|t| !t.contains(&mesh)) {
                    continue;
                }
                if self.entry_current(i, k) || self.settle_entry(i, k) {
                    continue;
                }
                let Ok(e) = self.entry(mesh) else { continue };
                if !want(e.mode) {
                    continue;
                }
                due.try_reserve(1).map_err(|_| oom())?;
                due.push((i, k));
            }
        }
        self.compute_entries(&due)
    }

    /// Computes the Eager entries of `touched` (all with `None`). Failures
    /// are swallowed: the entry stays stale (§6.3).
    pub(crate) fn run_eager_sections(&mut self, touched: Option<&[MeshId]>) {
        let _ = self.run_sections(|m| m == UpdateMode::Eager, touched);
    }

    /// Drops `mesh` from every section selecting it, reporting it on the
    /// mesh (before its removal).
    pub(crate) fn drop_sections_of(&mut self, mesh: MeshId) {
        for s in &mut self.sections {
            if let Some(k) = s.entries.iter().position(|se| se.mesh == mesh) {
                s.entries.remove(k);
                s.desc.meshes.retain(|&m| m != mesh);
                if let Some(e) = self.entries.get_mut(mesh) {
                    e.remove_section_hook(s.id);
                    e.report_section(s.id);
                }
            }
        }
    }

    /// Recomputes the entries `due` (`(section, entry)` index pairs, in
    /// section order, then entry order) in full, each reported when its
    /// result changed or was never computed, in three phases: each entry's
    /// mesh BVH is brought up to date in turn under the query rules (a
    /// non-Manual one is updated, a stale Manual one fails with
    /// [`SceneError::StaleBvh`]); then every entry is cut without writing
    /// (in parallel, one entry per task, when the work is large enough;
    /// see [`PAR_MIN_SECTION_WORK`]); then each is written and reported in
    /// order, as if the entries were recomputed one after the other.
    ///
    /// All or nothing per entry: on error the entry and its pending changes
    /// stay as they were. Returns the first error in order; every entry is
    /// attempted.
    ///
    /// Bringing a BVH up to date before the earlier entries are written
    /// changes nothing: it writes no section state and reports nothing, and
    /// an entry's cut reads only its mesh, BVH, placement and plane.
    fn compute_entries(&mut self, due: &[(usize, usize)]) -> Result<()> {
        let mut outcomes: Vec<Result<()>> = try_vec(due.len())?;
        let mut jobs: Vec<(usize, usize, usize)> = try_vec(due.len())?;
        for (n, &(i, k)) in due.iter().enumerate() {
            let r = self.prepare_entry(i, k);
            if r.is_ok() {
                jobs.push((n, i, k));
            }
            outcomes.push(r);
        }
        match self.cut_entries(&jobs) {
            Ok(cuts) => {
                for ((n, i, k), c) in jobs.into_iter().zip(cuts) {
                    outcomes[n] = c.map(|c| self.apply_entry(i, k, c));
                }
            }
            Err(e) => {
                for (n, _, _) in jobs {
                    outcomes[n] = Err(e.clone());
                }
            }
        }
        outcomes.into_iter().find(Result::is_err).unwrap_or(Ok(()))
    }

    /// Brings the BVH of entry `k` of section `i` up to date under the query
    /// rules (see [`Scene::compute_entries`]).
    fn prepare_entry(&mut self, i: usize, k: usize) -> Result<()> {
        let mesh = self.sections[i].entries[k].mesh;
        let e = self.entries.get_mut(mesh).ok_or(SceneError::UnknownMesh)?;
        if e.mode != UpdateMode::Manual {
            e.update_bvh()?;
        } else if !e.is_current() {
            return Err(SceneError::StaleBvh(mesh));
        }
        Ok(())
    }

    /// Cuts the prepared entries `jobs` (`(_, section, entry)`) without
    /// writing anything: one result per job, in job order. With the
    /// `parallel` feature, when their meshes hold [`PAR_MIN_SECTION_WORK`]
    /// triangles or more in all, the entries are cut in parallel, one job
    /// per task.
    ///
    /// # Errors
    /// `OutOfMemory` when the per-job results cannot be allocated.
    fn cut_entries(&self, jobs: &[(usize, usize, usize)]) -> Result<Vec<Result<Computed>>> {
        let mesh_of = |i: usize, k: usize| self.entry(self.sections[i].entries[k].mesh);
        let work: usize = jobs
            .iter()
            .map(|&(_, i, k)| mesh_of(i, k).map_or(0, |e| e.mesh.triangle_count()))
            .sum();
        let parallel = work >= PAR_MIN_SECTION_WORK
            && jobs.len() > 1
            && par::enabled()
            && !par::force_sequential();
        #[cfg(test)]
        if parallel {
            PARALLEL_SECTION_BATCHES.with(|c| c.set(c.get() + 1));
        }
        let threshold = if parallel { 0 } else { usize::MAX };
        // Chunks of one job; an entry's error is its own result.
        par::map_chunks(jobs, 1, threshold, oom, |_, chunk| {
            let (_, i, k) = chunk[0];
            Ok(mesh_of(i, k).and_then(|e| compute(e, &self.sections[i].desc)))
        })
    }

    /// Writes the fresh cut `c` into entry `k` of section `i`, and reports
    /// it when the result changed or was never computed.
    fn apply_entry(&mut self, i: usize, k: usize, c: Computed) {
        #[cfg(test)]
        {
            self.section_recomputes += 1;
        }
        let s = &mut self.sections[i];
        let id = s.id;
        let se = &mut s.entries[k];
        let changed = !se.computed_once || se.result != c.result;
        se.result = c.result;
        se.crossed_vertices = c.crossed_vertices;
        se.crossed_triangles = c.crossed_triangles;
        se.plane = Some(s.desc.plane);
        se.stale = false;
        se.computed_once = true;
        if let Some(e) = self.entries.get_mut(se.mesh) {
            e.reset_section_hook(id);
            if changed {
                e.report_section(id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::f64::consts::PI;

    use fast_mesh_store::Mesh;

    use super::*;
    use crate::{MeshChanges, MeshOptions, Placement, SceneOptions, SubscriberId};

    /// A closed UV sphere of radius `r` around `centre`, outward wound: one
    /// vertex at each pole and `stacks - 1` rings of `slices` vertices.
    /// Built from `f64` positions, so its origin is the centre of its box.
    fn sphere(stacks: u32, slices: u32, r: f64, centre: [f64; 3]) -> Mesh {
        let mut pos = vec![centre[0], centre[1], centre[2] + r];
        for i in 1..stacks {
            let theta = PI * f64::from(i) / f64::from(stacks);
            for j in 0..slices {
                let phi = 2.0 * PI * f64::from(j) / f64::from(slices);
                let (rs, rc) = (r * theta.sin(), r * theta.cos());
                pos.extend([
                    centre[0] + rs * phi.cos(),
                    centre[1] + rs * phi.sin(),
                    centre[2] + rc,
                ]);
            }
        }
        pos.extend([centre[0], centre[1], centre[2] - r]);
        let south = 1 + (stacks - 1) * slices;
        let ring = |i: u32, j: u32| 1 + (i - 1) * slices + j % slices;
        let mut idx = Vec::new();
        for j in 0..slices {
            idx.extend([0, ring(1, j), ring(1, j + 1)]);
            for i in 1..stacks - 1 {
                let (a, b, c, d) = (
                    ring(i, j),
                    ring(i + 1, j),
                    ring(i + 1, j + 1),
                    ring(i, j + 1),
                );
                idx.extend([a, b, c, a, c, d]);
            }
            idx.extend([ring(stacks - 1, j), south, ring(stacks - 1, j + 1)]);
        }
        Mesh::from_buffers_f64(&pos, &idx).unwrap()
    }

    fn unit_sphere() -> Mesh {
        sphere(32, 64, 1.0, [0.0; 3])
    }

    fn with_mode(mode: UpdateMode) -> MeshOptions {
        MeshOptions {
            mode,
            ..MeshOptions::default()
        }
    }

    fn at(x: f64, y: f64, z: f64) -> Placement {
        Placement {
            translation: [x, y, z],
            ..Placement::IDENTITY
        }
    }

    fn z_plane(h: f64) -> Plane {
        Plane::new([0.0, 0.0, h], [0.0, 0.0, 1.0]).unwrap()
    }

    fn desc(plane: Plane, meshes: &[MeshId], fill: bool) -> SectionDesc {
        SectionDesc {
            plane,
            meshes: meshes.to_vec(),
            fill,
        }
    }

    fn close(value: f64, expected: f64, rel: f64) -> bool {
        (value - expected).abs() <= rel * expected.abs()
    }

    /// The circumference of the cut of a unit sphere at height `h`.
    fn circle(h: f64) -> f64 {
        2.0 * PI * (1.0 - h * h).sqrt()
    }

    /// A scene with one unit sphere in `mode`, its add already taken.
    fn one_sphere(mode: UpdateMode) -> (Scene, MeshId) {
        let mut scene = Scene::new(SceneOptions::default());
        let m = scene.add_mesh(unit_sphere(), with_mode(mode)).unwrap();
        scene.take_changes();
        (scene, m)
    }

    /// The sections reported on `mesh` in `ch`.
    fn of(ch: &[MeshChanges], mesh: MeshId) -> Vec<SectionId> {
        ch.iter()
            .find(|c| c.id == mesh)
            .map_or(Vec::new(), |c| c.sections_changed.clone())
    }

    /// The section's result for `mesh` is current and equals one computed
    /// from scratch by a new section with the same plane and fill.
    fn assert_fresh(scene: &mut Scene, id: SectionId, mesh: MeshId, at: &str) {
        scene.update_sections().unwrap();
        assert_eq!(
            scene.section_status(id, mesh),
            Ok(SectionStatus::Current),
            "{at}"
        );
        let s = &scene.sections[scene.section_index(id).unwrap()];
        let d = desc(s.desc.plane, &[mesh], s.desc.fill);
        let fresh = scene.add_section(d).unwrap();
        scene.update_sections().unwrap();
        let want = scene.section(fresh, mesh).unwrap().clone();
        assert_eq!(scene.section(id, mesh).unwrap(), &want, "{at}");
        scene.remove_section(fresh).unwrap();
    }

    fn fill_area(f: &Fill) -> f64 {
        f.triangles
            .iter()
            .map(|t| {
                let [a, b, c] = t.map(|i| f.points[i as usize]);
                0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
            })
            .sum()
    }

    /// The rotation of the quaternion `q` (normalised first).
    fn rotation(q: [f64; 4]) -> [[f64; 3]; 3] {
        let n = q.iter().map(|c| c * c).sum::<f64>().sqrt();
        let [w, x, y, z] = q.map(|c| c / n);
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ]
    }

    fn remove_triangles_using(scene: &mut Scene, m: MeshId, v: VertexId) -> Vec<TriangleId> {
        let tris: Vec<TriangleId> = scene
            .mesh(m)
            .unwrap()
            .triangles()
            .iter()
            .enumerate()
            .filter(|(t, tri)| {
                scene.mesh(m).unwrap().is_triangle_alive(*t as TriangleId) && tri.contains(&v)
            })
            .map(|(t, _)| t as TriangleId)
            .collect();
        scene.edit(m, |tx| tx.remove_triangles(&tris)).unwrap();
        tris
    }

    #[test]
    fn add_section_eager_computes_immediately() {
        let (mut scene, m) = one_sphere(UpdateMode::Eager);
        let id = scene.add_section(desc(z_plane(0.3), &[m], false)).unwrap();
        assert_eq!(scene.section_status(id, m), Ok(SectionStatus::Current));
        let r = scene.section(id, m).unwrap().clone();
        assert_eq!(r.contours.len(), 1);
        let c = &r.contours[0];
        assert!(c.closed && c.oriented && c.manifold);
        assert!(close(r.length, 2.0 * PI * 0.954, 1e-2), "{}", r.length);
        assert!(close(r.length, circle(0.3), 1e-2));
        assert_eq!(r.length, c.length);
        assert!(r.area > 0.0 && r.area == c.area);
        assert!(close(r.area, PI * (1.0 - 0.09), 1e-2), "{}", r.area);
        assert!(r.fill.is_none() && r.fill_ok);
        assert_eq!(c.points.len(), c.uv.len());
        assert_eq!(c.points.len(), c.edges.len());
        assert_eq!(c.triangles.len(), c.points.len());
        assert!(c.points.iter().all(|p| (p[2] - 0.3).abs() < 1e-6));
        let ch = scene.take_changes();
        assert_eq!(ch.len(), 1);
        assert_eq!(ch[0].sections_changed, [id]);
        assert_eq!(scene.section_recomputes, 1);
        // The crossed sets cover the contour.
        let se = &scene.sections[0].entries[0];
        assert_eq!(se.crossed_triangles.len(), c.triangles.len());
        assert!(se.crossed_vertices.windows(2).all(|w| w[0] < w[1]));
        assert!(
            c.edges
                .iter()
                .all(|&(a, b)| se.crossed_vertices.binary_search(&a).is_ok()
                    && se.crossed_vertices.binary_search(&b).is_ok())
        );
    }

    #[test]
    fn lazy_waits_for_update_sections() {
        let (mut scene, m) = one_sphere(UpdateMode::Lazy);
        let id = scene.add_section(desc(z_plane(0.3), &[m], true)).unwrap();
        assert_eq!(scene.section_status(id, m), Ok(SectionStatus::Stale));
        assert_eq!(scene.section(id, m), Ok(&SectionResult::default()));
        assert!(scene.take_changes().is_empty());
        scene.update_sections().unwrap();
        assert_eq!(scene.section_status(id, m), Ok(SectionStatus::Current));
        let r = scene.section(id, m).unwrap();
        assert_eq!(r.contours.len(), 1);
        assert!(r.fill.is_some() && r.fill_ok);
        assert_eq!(scene.take_changes()[0].sections_changed, [id]);
        // Nothing stale: nothing recomputed or reported.
        scene.update_sections().unwrap();
        scene.flush().unwrap();
        assert_eq!(scene.section_recomputes, 1);
        assert!(scene.take_changes().is_empty());
        // An edit stales it until a flush.
        scene
            .edit(m, |tx| tx.set_positions(&[40], &[[0.0, 0.0, 0.0]]))
            .unwrap();
        assert_eq!(scene.section_status(id, m), Ok(SectionStatus::Stale));
        scene.flush().unwrap();
        assert_eq!(scene.section_status(id, m), Ok(SectionStatus::Current));
        assert_fresh(&mut scene, id, m, "after flush");
    }

    #[test]
    fn plane_move_restales_and_recomputes_eager() {
        let (mut scene, m) = one_sphere(UpdateMode::Eager);
        let id = scene.add_section(desc(z_plane(0.3), &[m], false)).unwrap();
        scene.set_section_plane(id, z_plane(-0.5)).unwrap();
        assert_eq!(scene.section_status(id, m), Ok(SectionStatus::Current));
        let r = scene.section(id, m).unwrap();
        assert!(close(r.length, circle(0.5), 1e-2), "{}", r.length);
        assert!(
            r.contours[0]
                .points
                .iter()
                .all(|p| (p[2] + 0.5).abs() < 1e-6)
        );
        assert_eq!(scene.section_recomputes, 2);
        assert_fresh(&mut scene, id, m, "eager");

        // A Lazy mesh keeps the old result, Stale, until updated.
        let (mut scene, m) = one_sphere(UpdateMode::Lazy);
        let id = scene.add_section(desc(z_plane(0.3), &[m], false)).unwrap();
        scene.update_sections().unwrap();
        let old = scene.section(id, m).unwrap().clone();
        scene.set_section_plane(id, z_plane(-0.5)).unwrap();
        assert_eq!(scene.section_status(id, m), Ok(SectionStatus::Stale));
        assert_eq!(scene.section(id, m), Ok(&old));
        // The description has the new plane while the result lags.
        assert_eq!(scene.section_desc(id).unwrap().plane, z_plane(-0.5));
        assert_eq!(scene.section_plane(id, m), Ok(Some(z_plane(0.3))));
        scene.update_sections().unwrap();
        let r = scene.section(id, m).unwrap();
        assert!(close(r.length, circle(0.5), 1e-2));
        // A fill change stales it too.
        scene.set_section_fill(id, true).unwrap();
        assert_eq!(scene.section_status(id, m), Ok(SectionStatus::Stale));
        assert!(scene.section(id, m).unwrap().fill.is_none());
        scene.update_sections().unwrap();
        assert!(scene.section(id, m).unwrap().fill.is_some());
    }

    #[test]
    fn world_points_match_local_slice_with_origin() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let centre = [1000.0, -1000.0, 500.0];
        for _ in 0..8 {
            let mut scene = Scene::new(SceneOptions::default());
            let placement = Placement {
                rotation: rotation([next(), next(), next(), next()]),
                translation: [next() * 50.0, next() * 50.0, next() * 50.0],
            };
            let opts = MeshOptions {
                placement,
                ..MeshOptions::default()
            };
            let m = scene.add_mesh(sphere(24, 48, 1.0, centre), opts).unwrap();
            let o = scene.mesh(m).unwrap().origin();
            assert!((0..3).all(|k| (o[k] - centre[k]).abs() < 1e-9), "{o:?}");
            // A plane through the sphere, in the world.
            let c = placement.apply(centre);
            let n = [next(), next(), next()];
            let off = next() * 1.2;
            let len = n.iter().map(|x| x * x).sum::<f64>().sqrt();
            let origin: [f64; 3] = std::array::from_fn(|k| c[k] + off * n[k] / len);
            let plane = Plane::new(origin, n).unwrap();
            let id = scene.add_section(desc(plane, &[m], false)).unwrap();
            let r = scene.section(id, m).unwrap().clone();
            assert!(!r.contours.is_empty());

            // The same cut made directly in local space.
            let r_t = |d: [f64; 3]| -> [f64; 3] {
                let r = &placement.rotation;
                std::array::from_fn(|i| r[0][i] * d[0] + r[1][i] * d[1] + r[2][i] * d[2])
            };
            let local = Plane::with_axes(
                scene.world_to_local(m, plane.origin()).unwrap(),
                r_t(plane.u()),
                r_t(plane.v()),
            )
            .unwrap();
            let mut cut = Slice::default();
            fast_mesh_query::slice(
                scene.mesh(m).unwrap(),
                scene.bvh(m).unwrap(),
                &local,
                &mut cut,
            )
            .unwrap();
            assert_eq!(cut.contours.len(), r.contours.len());
            let tol = 1e-9 * 1000.0;
            for (want, got) in cut.contours.iter().zip(&r.contours) {
                assert_eq!(want.edges, got.edges);
                assert_eq!(want.triangles, got.triangles);
                assert_eq!(
                    (want.closed, want.oriented, want.manifold),
                    (got.closed, got.oriented, got.manifold)
                );
                for ((&l, &w), &uv) in want.points.iter().zip(&got.points).zip(&got.uv) {
                    let p = scene.local_to_world(m, l).unwrap();
                    assert!((0..3).all(|k| (p[k] - w[k]).abs() <= tol), "{p:?} {w:?}");
                    let q = plane.to_uv(w);
                    assert!((0..2).all(|k| (q[k] - uv[k]).abs() <= tol), "{q:?} {uv:?}");
                    assert!(plane.signed_distance(w).abs() < 1e-6);
                }
            }
            // Outward winding stays counter-clockwise in the world frame.
            assert!(r.area > 0.0);
        }
    }

    #[test]
    fn plane_missing_mesh_gives_empty_current_result() {
        let (mut scene, m) = one_sphere(UpdateMode::Eager);
        let id = scene.add_section(desc(z_plane(5.0), &[m], true)).unwrap();
        assert_eq!(scene.section_status(id, m), Ok(SectionStatus::Current));
        let r = scene.section(id, m).unwrap();
        assert!(r.contours.is_empty());
        assert_eq!((r.length, r.area), (0.0, 0.0));
        assert_eq!(r.fill, Some(Fill::default()));
        assert!(r.fill_ok);
        // The first computation is reported, even when empty.
        assert_eq!(scene.take_changes()[0].sections_changed, [id]);
        // Still missing: recomputed to the same result, not reported.
        scene.set_section_plane(id, z_plane(6.0)).unwrap();
        assert_eq!(scene.section_recomputes, 2);
        assert!(scene.take_changes().is_empty());
    }

    #[test]
    fn edit_undo_redo_keep_result_equal_to_from_scratch() {
        type Op = Box<dyn Fn(&mut Scene, MeshId)>;
        // A vertex of the ring just above the plane (ring 16 is the equator).
        let v: VertexId = 1 + 15 * 64 + 3;
        let ops: Vec<(&str, Op)> = vec![
            (
                "move across",
                Box::new(move |s, m| {
                    s.edit(m, |tx| tx.set_positions(&[v], &[[0.9, 0.2, -0.2]]))
                        .unwrap();
                }),
            ),
            (
                "remove triangles",
                Box::new(move |s, m| {
                    remove_triangles_using(s, m, v + 1);
                }),
            ),
            ("undo", Box::new(|s, _| assert!(s.undo().unwrap()))),
            ("undo", Box::new(|s, _| assert!(s.undo().unwrap()))),
            ("redo", Box::new(|s, _| assert!(s.redo().unwrap()))),
            (
                "placement",
                Box::new(|s, m| s.set_placement(m, at(0.1, 0.0, 0.02)).unwrap()),
            ),
            (
                "cancelled drag",
                Box::new(move |s, m| {
                    s.begin_step();
                    s.set_placement(m, at(0.0, 0.0, 0.6)).unwrap();
                    s.edit(m, |tx| tx.set_positions(&[v], &[[0.0, 0.0, 0.0]]))
                        .unwrap();
                    s.cancel_step().unwrap();
                }),
            ),
            (
                "undo placement",
                Box::new(|s, _| assert!(s.undo().unwrap())),
            ),
            (
                "redo placement",
                Box::new(|s, _| assert!(s.redo().unwrap())),
            ),
        ];
        for mode in [UpdateMode::Eager, UpdateMode::Lazy, UpdateMode::Manual] {
            let (mut scene, m) = one_sphere(mode);
            let id = scene.add_section(desc(z_plane(0.05), &[m], true)).unwrap();
            assert_fresh(&mut scene, id, m, "start");
            for (name, op) in &ops {
                let before = scene.section(id, m).unwrap().clone();
                op(&mut scene, m);
                let expect = if mode == UpdateMode::Eager {
                    SectionStatus::Current
                } else {
                    SectionStatus::Stale
                };
                assert_eq!(scene.section_status(id, m), Ok(expect), "{mode:?} {name}");
                if mode != UpdateMode::Eager {
                    assert_eq!(scene.section(id, m), Ok(&before), "{mode:?} {name}");
                }
                if mode == UpdateMode::Manual {
                    scene.update(m).unwrap();
                }
                assert_fresh(&mut scene, id, m, &format!("{mode:?} {name}"));
            }
        }
    }

    #[test]
    fn placement_change_recomputes() {
        let (mut scene, m) = one_sphere(UpdateMode::Eager);
        let id = scene.add_section(desc(z_plane(0.3), &[m], false)).unwrap();
        scene.take_changes();
        scene.set_placement(m, at(0.0, 0.0, 0.2)).unwrap();
        assert_eq!(scene.section_status(id, m), Ok(SectionStatus::Current));
        let r = scene.section(id, m).unwrap();
        assert!(close(r.length, circle(0.1), 1e-2), "{}", r.length);
        assert!(
            r.contours[0]
                .points
                .iter()
                .all(|p| (p[2] - 0.3).abs() < 1e-6)
        );
        let ch = scene.take_changes();
        assert!(ch[0].placement_changed);
        assert_eq!(ch[0].sections_changed, [id]);
        assert_fresh(&mut scene, id, m, "moved");
    }

    #[test]
    fn fill_on_gives_area_matching_contours() {
        let (mut scene, m) = one_sphere(UpdateMode::Eager);
        let id = scene.add_section(desc(z_plane(0.3), &[m], true)).unwrap();
        let r = scene.section(id, m).unwrap();
        assert!(r.fill_ok);
        let f = r.fill.as_ref().unwrap();
        assert!(!f.triangles.is_empty());
        let a = fill_area(f);
        assert!(close(a, r.area, 1e-6), "{a} {}", r.area);
        assert!(close(r.area, PI * (1.0 - 0.09), 1e-2));
        // Turning fill off drops it.
        scene.set_section_fill(id, false).unwrap();
        let r = scene.section(id, m).unwrap();
        assert!(r.fill.is_none() && r.fill_ok);
    }

    #[test]
    fn gap_between_two_spheres() {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene
            .add_mesh(unit_sphere(), MeshOptions::default())
            .unwrap();
        let opts = MeshOptions {
            placement: at(3.0, 0.0, 0.0),
            ..MeshOptions::default()
        };
        let b = scene.add_mesh(unit_sphere(), opts).unwrap();
        let id = scene
            .add_section(desc(z_plane(0.0), &[a, b], false))
            .unwrap();
        let g = scene.section_gap(id, a, b).unwrap().unwrap();
        assert!((g.distance - 1.0).abs() < 1e-3, "{}", g.distance);
        assert!((g.a[0] - 1.0).abs() < 1e-3 && (g.b[0] - 2.0).abs() < 1e-3);
        for p in [g.a, g.b] {
            assert!(p[1].abs() < 1e-3 && p[2].abs() < 1e-9, "{p:?}");
        }
        let d = (0..3)
            .map(|k| (g.a[k] - g.b[k]).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!((d - g.distance).abs() < 1e-9);
        // Symmetric, and None when one side has no contours.
        let h = scene.section_gap(id, b, a).unwrap().unwrap();
        assert!((h.distance - g.distance).abs() < 1e-12);
        scene.set_placement(b, at(3.0, 0.0, 5.0)).unwrap();
        assert_eq!(scene.section_gap(id, a, b), Ok(None));
    }

    #[test]
    fn error_paths() {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene
            .add_mesh(unit_sphere(), MeshOptions::default())
            .unwrap();
        let b = scene
            .add_mesh(unit_sphere(), MeshOptions::default())
            .unwrap();
        let gone = scene
            .add_mesh(unit_sphere(), MeshOptions::default())
            .unwrap();
        scene.remove_mesh(gone).unwrap();
        let p = z_plane(0.0);
        let invalid = |r: &Result<SectionId>| matches!(r, Err(SceneError::InvalidInput(_)));
        assert_eq!(
            scene.add_section(desc(p, &[a, gone], false)),
            Err(SceneError::UnknownMesh)
        );
        assert!(invalid(&scene.add_section(desc(p, &[a, b, a], false))));
        assert!(scene.sections.is_empty());
        let no_hooks = |s: &Scene| {
            s.ids()
                .all(|m| s.entry(m).unwrap().section_hooks.is_empty())
        };
        assert!(no_hooks(&scene));

        let id = scene.add_section(desc(p, &[a], false)).unwrap();
        assert_eq!(
            scene.set_section_meshes(id, vec![a, gone]),
            Err(SceneError::UnknownMesh)
        );
        assert!(matches!(
            scene.set_section_meshes(id, vec![b, b]),
            Err(SceneError::InvalidInput(_))
        ));
        assert_eq!(scene.sections[0].desc.meshes, [a]);
        assert!(scene.entry(b).unwrap().section_hooks.is_empty());
        assert!(matches!(
            scene.section(id, b),
            Err(SceneError::InvalidInput(_))
        ));
        assert!(matches!(
            scene.section_status(id, b),
            Err(SceneError::InvalidInput(_))
        ));
        assert!(matches!(
            scene.section_gap(id, a, b),
            Err(SceneError::InvalidInput(_))
        ));
        assert_eq!(scene.section(id, gone), Err(SceneError::UnknownMesh));
        assert_eq!(scene.section_gap(id, gone, a), Err(SceneError::UnknownMesh));

        assert_eq!(scene.section_id(id.raw()), Some(id));
        assert_eq!(scene.section_id(id.raw() + 1), None);
        scene.remove_section(id).unwrap();
        assert_eq!(scene.section_id(id.raw()), None);
        assert!(no_hooks(&scene));
        let u = SceneError::UnknownSection;
        assert_eq!(scene.remove_section(id), Err(u.clone()));
        assert_eq!(scene.set_section_plane(id, p), Err(u.clone()));
        assert_eq!(scene.set_section_meshes(id, vec![a]), Err(u.clone()));
        assert_eq!(scene.set_section_fill(id, true), Err(u.clone()));
        assert_eq!(scene.section(id, a), Err(u.clone()));
        assert_eq!(scene.section_status(id, a), Err(u.clone()));
        assert_eq!(scene.section_gap(id, a, a), Err(u.clone()));
        assert_eq!(u.to_string(), "unknown or removed section id");
        // Ids are not reused.
        let again = scene.add_section(desc(p, &[a], false)).unwrap();
        assert_ne!(again, id);
        assert_eq!(scene.section(id, a), Err(u));
    }

    #[test]
    fn remove_section_and_mesh_report_once() {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene
            .add_mesh(unit_sphere(), MeshOptions::default())
            .unwrap();
        let b = scene
            .add_mesh(unit_sphere(), with_mode(UpdateMode::Lazy))
            .unwrap();
        let s2 = scene.subscribe();
        let id = scene
            .add_section(desc(z_plane(0.2), &[a, b], false))
            .unwrap();
        scene.update_sections().unwrap();
        scene.take_changes();
        scene.take_changes_for(s2).unwrap();
        scene.remove_section(id).unwrap();
        for s in [SubscriberId::DEFAULT, s2] {
            let ch = scene.take_changes_for(s).unwrap();
            assert_eq!(ch.len(), 2);
            for c in &ch {
                assert_eq!(c.sections_changed, [id], "{:?}", c.id);
                assert!(c.changes.as_ref().unwrap().is_empty());
            }
        }

        let id2 = scene
            .add_section(desc(z_plane(0.2), &[a, b], false))
            .unwrap();
        scene.take_changes();
        scene.take_changes_for(s2).unwrap();
        scene.remove_mesh(b).unwrap();
        assert_eq!(scene.sections[0].desc.meshes, [a]);
        assert_eq!(scene.section(id2, b), Err(SceneError::UnknownMesh));
        assert_eq!(scene.section_status(id2, b), Err(SceneError::UnknownMesh));
        assert_eq!(scene.section_status(id2, a), Ok(SectionStatus::Current));
        for s in [SubscriberId::DEFAULT, s2] {
            let ch = scene.take_changes_for(s).unwrap();
            assert_eq!(ch.len(), 1);
            assert!(ch[0].removed && ch[0].id == b);
            assert_eq!(ch[0].sections_changed, [id2]);
        }
        scene.update_sections().unwrap();
        scene.remove_section(id2).unwrap();
        assert_eq!(of(&scene.take_changes_for(s2).unwrap(), a), [id2]);
        assert_eq!(scene.section(id2, a), Err(SceneError::UnknownSection));
        assert_eq!(scene.section_status(id, a), Err(SceneError::UnknownSection));
    }

    #[test]
    fn selection_change_keeps_and_drops_entries() {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene
            .add_mesh(unit_sphere(), MeshOptions::default())
            .unwrap();
        let b = scene
            .add_mesh(unit_sphere(), with_mode(UpdateMode::Lazy))
            .unwrap();
        let c = scene
            .add_mesh(unit_sphere(), MeshOptions::default())
            .unwrap();
        let id = scene
            .add_section(desc(z_plane(0.2), &[a, b], false))
            .unwrap();
        scene.take_changes();
        let n = scene.section_recomputes;
        scene.set_section_meshes(id, vec![c, a]).unwrap();
        // c (Eager) is computed, a is kept as it is, b is dropped.
        assert_eq!(scene.section_recomputes, n + 1);
        assert_eq!(scene.section_status(id, a), Ok(SectionStatus::Current));
        assert_eq!(scene.section_status(id, c), Ok(SectionStatus::Current));
        assert!(matches!(
            scene.section(id, b),
            Err(SceneError::InvalidInput(_))
        ));
        let ch = scene.take_changes();
        assert_eq!(of(&ch, a), []);
        assert_eq!(of(&ch, b), [id]);
        assert_eq!(of(&ch, c), [id]);
        assert!(scene.entry(b).unwrap().section_hooks.is_empty());
        // A selected Lazy mesh is stale until updated.
        scene.set_section_meshes(id, vec![c, a, b]).unwrap();
        assert_eq!(scene.section_status(id, b), Ok(SectionStatus::Stale));
        assert_fresh(&mut scene, id, b, "reselected");
    }

    #[test]
    fn compact_recomputes_and_reports() {
        let (mut scene, m) = one_sphere(UpdateMode::Eager);
        let id = scene.add_section(desc(z_plane(0.3), &[m], false)).unwrap();
        // Remove the north pole, so that compaction renumbers every vertex.
        remove_triangles_using(&mut scene, m, 0);
        scene.edit(m, |tx| tx.remove_vertices(&[0])).unwrap();
        assert_fresh(&mut scene, id, m, "holed");
        scene.take_changes();
        let before = scene.section(id, m).unwrap().clone();
        let n = scene.section_recomputes;
        scene.compact(m).unwrap();
        assert_eq!(scene.section_recomputes, n + 1);
        assert_eq!(scene.section_status(id, m), Ok(SectionStatus::Current));
        let after = scene.section(id, m).unwrap();
        assert_ne!(after.contours[0].edges, before.contours[0].edges);
        assert!(close(after.length, before.length, 1e-12));
        let ch = scene.take_changes();
        assert!(ch[0].changes.is_none());
        assert_eq!(ch[0].sections_changed, [id]);
        assert_fresh(&mut scene, id, m, "compacted");
    }

    #[test]
    fn manual_stale_bvh_returns_error_and_stays_stale() {
        let (mut scene, m) = one_sphere(UpdateMode::Manual);
        let id = scene.add_section(desc(z_plane(0.3), &[m], false)).unwrap();
        // Manual waits, even with a current BVH; flush leaves it alone.
        assert_eq!(scene.section_status(id, m), Ok(SectionStatus::Stale));
        scene.flush().unwrap();
        assert_eq!(scene.section_status(id, m), Ok(SectionStatus::Stale));
        scene.update_sections().unwrap();
        assert_eq!(scene.section_status(id, m), Ok(SectionStatus::Current));
        scene
            .edit(m, |tx| tx.set_positions(&[40], &[[0.0, 0.0, 0.0]]))
            .unwrap();
        assert_eq!(scene.section_status(id, m), Ok(SectionStatus::Stale));
        let old = scene.section(id, m).unwrap().clone();
        assert_eq!(scene.update_sections(), Err(SceneError::StaleBvh(m)));
        assert_eq!(scene.update_sections(), Err(SceneError::StaleBvh(m)));
        assert_eq!(scene.section_status(id, m), Ok(SectionStatus::Stale));
        assert_eq!(scene.section(id, m), Ok(&old));
        scene.flush().unwrap();
        assert_eq!(scene.section_status(id, m), Ok(SectionStatus::Stale));
        scene.update(m).unwrap();
        assert_eq!(scene.section_status(id, m), Ok(SectionStatus::Current));
        assert_fresh(&mut scene, id, m, "updated");
        // add_section never fails with StaleBvh.
        scene
            .edit(m, |tx| tx.set_positions(&[41], &[[0.0, 0.0, 0.0]]))
            .unwrap();
        let other = scene.add_section(desc(z_plane(0.0), &[m], false)).unwrap();
        assert_eq!(scene.section_status(other, m), Ok(SectionStatus::Stale));
        assert_eq!(scene.update_sections(), Err(SceneError::StaleBvh(m)));
    }

    #[test]
    fn failed_eager_bvh_update_is_retried_by_the_section() {
        let (mut scene, m) = one_sphere(UpdateMode::Eager);
        let id = scene.add_section(desc(z_plane(0.3), &[m], false)).unwrap();
        // The edit's own BVH update fails; the edit still succeeds, and the
        // section brings the BVH up to date before cutting.
        crate::scene::fault::fail_bvh_update_on(1);
        scene
            .edit(m, |tx| tx.set_positions(&[40], &[[0.0, 0.0, 0.0]]))
            .unwrap();
        assert_eq!(scene.is_current(m), Ok(true));
        assert_eq!(scene.section_status(id, m), Ok(SectionStatus::Current));
        assert_fresh(&mut scene, id, m, "retried");
    }

    /// The id of vertex `j` of ring `i` of a UV sphere with `slices` slices.
    fn ring(slices: u32, i: u32, j: u32) -> VertexId {
        1 + (i - 1) * slices + j % slices
    }

    /// A 64 x 128 unit sphere in `mode`, cut at z = 0.3, its add and first
    /// computation taken.
    fn fine_sphere_cut(mode: UpdateMode) -> (Scene, MeshId, SectionId) {
        let mut scene = Scene::new(SceneOptions::default());
        let m = scene
            .add_mesh(sphere(64, 128, 1.0, [0.0; 3]), with_mode(mode))
            .unwrap();
        let id = scene.add_section(desc(z_plane(0.3), &[m], true)).unwrap();
        scene.update_sections().unwrap();
        scene.take_changes();
        (scene, m, id)
    }

    fn crossed(scene: &Scene, id: SectionId) -> (Vec<VertexId>, Vec<TriangleId>) {
        let se = &scene.sections[scene.section_index(id).unwrap()].entries[0];
        (se.crossed_vertices.clone(), se.crossed_triangles.clone())
    }

    fn nudge(scene: &mut Scene, m: MeshId, v: VertexId, d: [f32; 3]) {
        scene
            .edit(m, |tx| {
                let p = tx.mesh().position(v).unwrap();
                tx.set_positions(&[v], &[std::array::from_fn(|k| p[k] + d[k])])
            })
            .unwrap();
    }

    /// The triangle with exactly the vertices `tri`, in that winding.
    fn find_triangle(scene: &Scene, m: MeshId, tri: [VertexId; 3]) -> TriangleId {
        let mesh = scene.mesh(m).unwrap();
        (0..mesh.triangle_slots() as TriangleId)
            .find(|&t| mesh.triangle(t) == Some(tri))
            .unwrap()
    }

    /// Flips the diagonal of the quad of rings `i`, `i + 1` at slice `j`.
    fn flip_quad(scene: &mut Scene, m: MeshId, slices: u32, i: u32, j: u32) -> [TriangleId; 2] {
        let (a, b, c, d) = (
            ring(slices, i, j),
            ring(slices, i + 1, j),
            ring(slices, i + 1, j + 1),
            ring(slices, i, j + 1),
        );
        let t = [
            find_triangle(scene, m, [a, b, c]),
            find_triangle(scene, m, [a, c, d]),
        ];
        scene
            .edit(m, |tx| tx.set_triangles(&t, &[[a, b, d], [b, c, d]]))
            .unwrap();
        t
    }

    #[test]
    fn far_edit_does_not_recompute() {
        for mode in [UpdateMode::Eager, UpdateMode::Lazy, UpdateMode::Manual] {
            let (mut scene, m, id) = fine_sphere_cut(mode);
            let before = scene.section(id, m).unwrap().clone();
            let n = scene.section_recomputes;
            // Ring 55 sits at z = -0.904.
            let v = ring(128, 55, 7);
            assert!(scene.mesh(m).unwrap().position(v).unwrap()[2] < -0.9);
            nudge(&mut scene, m, v, [0.0, 0.0, 0.01]);
            // A far edge flip and a far triangle removal.
            flip_quad(&mut scene, m, 128, 50, 3);
            let t = find_triangle(&scene, m, [0, ring(128, 1, 9), ring(128, 1, 10)]);
            scene.edit(m, |tx| tx.remove_triangles(&[t])).unwrap();
            assert_eq!(
                scene.section_status(id, m),
                Ok(SectionStatus::Current),
                "{mode:?}"
            );
            scene.update_sections().unwrap();
            scene.flush().unwrap();
            assert_eq!(scene.section_recomputes, n, "{mode:?}");
            assert_eq!(scene.section(id, m), Ok(&before));
            let ch = scene.take_changes();
            assert_eq!(of(&ch, m), [], "{mode:?}");
            if mode == UpdateMode::Manual {
                scene.update(m).unwrap();
            }
            assert_fresh(&mut scene, id, m, &format!("{mode:?} far"));
        }
    }

    #[test]
    fn edit_touching_plane_recomputes() {
        let (mut scene, m, id) = fine_sphere_cut(UpdateMode::Eager);
        let n = scene.section_recomputes;
        let (cv, _) = crossed(&scene, id);
        nudge(&mut scene, m, cv[cv.len() / 2], [0.0, 0.0, 0.004]);
        assert_eq!(scene.section_recomputes, n + 1);
        assert_eq!(of(&scene.take_changes(), m), [id]);
        assert_fresh(&mut scene, id, m, "touching");
    }

    #[test]
    fn vertex_moved_across_plane_recomputes() {
        let (mut scene, m, id) = fine_sphere_cut(UpdateMode::Eager);
        let n = scene.section_recomputes;
        // Ring 27 sits at z = 0.243, below the crossed rings 25 and 26.
        let v = ring(128, 27, 40);
        let (cv, _) = crossed(&scene, id);
        assert!(cv.binary_search(&v).is_err());
        let p = scene.mesh(m).unwrap().position(v).unwrap();
        assert!((p[2] - 0.25).abs() < 0.01, "{p:?}");
        scene
            .edit(m, |tx| tx.set_positions(&[v], &[[p[0], p[1], 0.35]]))
            .unwrap();
        assert_eq!(scene.section_recomputes, n + 1);
        assert_eq!(of(&scene.take_changes(), m), [id]);
        assert_fresh(&mut scene, id, m, "across");
    }

    #[test]
    fn edge_flip_near_plane_recomputes() {
        let (mut scene, m, id) = fine_sphere_cut(UpdateMode::Lazy);
        let n = scene.section_recomputes;
        let before = scene.section(id, m).unwrap().clone();
        let (_, ct) = crossed(&scene, id);
        // The quad between the crossed rings 25 and 26.
        let t = flip_quad(&mut scene, m, 128, 25, 11);
        assert!(t.iter().all(|t| ct.binary_search(t).is_ok()));
        assert_eq!(scene.section_status(id, m), Ok(SectionStatus::Stale));
        scene.update_sections().unwrap();
        assert_eq!(scene.section_recomputes, n + 1);
        assert_ne!(scene.section(id, m), Ok(&before));
        assert_eq!(of(&scene.take_changes(), m), [id]);
        assert_fresh(&mut scene, id, m, "flip");
    }

    #[test]
    fn far_triangles_added_and_vertices_added_do_not_recompute() {
        let (mut scene, m, id) = fine_sphere_cut(UpdateMode::Eager);
        let n = scene.section_recomputes;
        let (p, q) = (ring(128, 60, 0), ring(128, 60, 1));
        scene
            .edit(m, |tx| {
                let v = tx.add_vertices(&[[0.1, 0.1, -0.95]])?[0];
                tx.add_triangles(&[[v, p, q]])
            })
            .unwrap();
        assert_eq!(scene.section_recomputes, n);
        // A new triangle crossing the plane does recompute.
        scene
            .edit(m, |tx| {
                let v = tx.add_vertices(&[[0.0, 0.0, 0.9]])?[0];
                tx.add_triangles(&[[v, p, q]])
            })
            .unwrap();
        assert_eq!(scene.section_recomputes, n + 1);
        assert_fresh(&mut scene, id, m, "added");
    }

    #[test]
    fn gap_needs_two_meshes_and_one_plane() {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene
            .add_mesh(unit_sphere(), MeshOptions::default())
            .unwrap();
        let opts = MeshOptions {
            placement: at(3.0, 0.0, 0.0),
            mode: UpdateMode::Lazy,
            ..MeshOptions::default()
        };
        let b = scene.add_mesh(unit_sphere(), opts).unwrap();
        let id = scene
            .add_section(desc(z_plane(0.0), &[a, b], false))
            .unwrap();
        assert!(matches!(
            scene.section_gap(id, a, a),
            Err(SceneError::InvalidInput(_))
        ));
        scene.update_sections().unwrap();
        assert!(scene.section_gap(id, a, b).unwrap().is_some());
        // a (Eager) is cut at the new plane, b (Lazy) still at the old one.
        scene.set_section_plane(id, z_plane(0.1)).unwrap();
        assert_eq!(scene.section_gap(id, a, b), Ok(None));
        scene.update_sections().unwrap();
        let g = scene.section_gap(id, a, b).unwrap().unwrap();
        assert!((g.a[2] - 0.1).abs() < 1e-9 && (g.b[2] - 0.1).abs() < 1e-9);
    }

    #[test]
    fn section_plane_is_the_plane_computed_with() {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene
            .add_mesh(unit_sphere(), MeshOptions::default())
            .unwrap();
        let b = scene
            .add_mesh(unit_sphere(), with_mode(UpdateMode::Lazy))
            .unwrap();
        let c = scene
            .add_mesh(unit_sphere(), MeshOptions::default())
            .unwrap();
        let id = scene
            .add_section(desc(z_plane(0.0), &[a, b], true))
            .unwrap();
        // a (Eager) is computed, b (Lazy) not yet.
        assert_eq!(scene.section_plane(id, a), Ok(Some(z_plane(0.0))));
        assert_eq!(scene.section_plane(id, b), Ok(None));
        scene.set_section_plane(id, z_plane(0.1)).unwrap();
        scene.update_sections().unwrap();
        scene.set_section_plane(id, z_plane(0.2)).unwrap();
        assert_eq!(scene.section_plane(id, a), Ok(Some(z_plane(0.2))));
        assert_eq!(scene.section_plane(id, b), Ok(Some(z_plane(0.1))));
        // The fill maps to the world through that plane.
        let p = scene.section_plane(id, a).unwrap().unwrap();
        let fill = scene.section(id, a).unwrap().fill.clone().unwrap();
        assert!(fill.points.iter().all(|&q| p.from_uv(q)[2] == 0.2));
        assert!(matches!(
            scene.section_plane(id, c),
            Err(SceneError::InvalidInput(_))
        ));
        scene.remove_section(id).unwrap();
        assert_eq!(scene.section_plane(id, a), Err(SceneError::UnknownSection));
    }

    #[test]
    fn sections_changed_exact() {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene
            .add_mesh(unit_sphere(), MeshOptions::default())
            .unwrap();
        let b = scene
            .add_mesh(unit_sphere(), MeshOptions::default())
            .unwrap();
        let c = scene
            .add_mesh(unit_sphere(), MeshOptions::default())
            .unwrap();
        let s1 = scene
            .add_section(desc(z_plane(0.2), &[a, b], false))
            .unwrap();
        let s2 = scene
            .add_section(desc(z_plane(0.4), &[b, c], false))
            .unwrap();
        let ch = scene.take_changes();
        assert_eq!(of(&ch, a), [s1]);
        assert_eq!(of(&ch, b), [s1, s2]);
        assert_eq!(of(&ch, c), [s2]);
        // Two plane moves: s1 once, on its meshes only.
        scene.set_section_plane(s1, z_plane(0.1)).unwrap();
        scene.set_section_plane(s1, z_plane(0.0)).unwrap();
        let ch = scene.take_changes();
        assert_eq!(ch.len(), 2);
        assert_eq!(of(&ch, a), [s1]);
        assert_eq!(of(&ch, b), [s1]);
        assert_eq!(of(&ch, c), []);
        // A placement change of b: both its sections, on b only.
        scene.set_placement(b, at(0.0, 0.0, 0.05)).unwrap();
        let ch = scene.take_changes();
        assert_eq!(ch.len(), 1);
        assert_eq!(of(&ch, b), [s1, s2]);
    }

    /// Four sections over three overlapping spheres of `stacks` x `2 stacks`
    /// cells in `mode`, the plane of each dragged through four poses; per
    /// pose, the reports and every result and gap, printed in full (so equal
    /// strings mean equal bits).
    fn section_drag(stacks: u32, mode: UpdateMode) -> Vec<String> {
        let mut scene = Scene::new(SceneOptions::default());
        let meshes: Vec<MeshId> = (0..3)
            .map(|k| {
                let c = [0.6 * k as f64, 0.1 * k as f64, 0.0];
                let mesh = sphere(stacks, 2 * stacks, 1.0, c);
                scene.add_mesh(mesh, with_mode(mode)).unwrap()
            })
            .collect();
        let normals = [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.3, 1.0, 0.2],
            [1.0, 1.0, 1.0],
        ];
        let plane = |s: usize, pose: usize| {
            let t = 0.05 * pose as f64 - 0.1 + 0.02 * s as f64;
            Plane::new([0.6 + t, t, t], normals[s]).unwrap()
        };
        let sections: Vec<SectionId> = (0..4)
            .map(|s| {
                let fill = s % 2 == 0;
                scene.add_section(desc(plane(s, 0), &meshes, fill)).unwrap()
            })
            .collect();
        scene.update_sections().unwrap();
        scene.take_changes();
        let mut out = Vec::new();
        for pose in 1..5 {
            for (s, &id) in sections.iter().enumerate() {
                scene.set_section_plane(id, plane(s, pose)).unwrap();
            }
            scene.update_sections().unwrap();
            let mut seen = format!("{:?}", scene.take_changes());
            for &id in &sections {
                for &m in &meshes {
                    let r = scene.section(id, m).unwrap();
                    assert!(!r.contours.is_empty());
                    seen += &format!("{r:?}");
                }
                seen += &format!("{:?}", scene.section_gap(id, meshes[0], meshes[2]));
            }
            out.push(seen);
        }
        out
    }

    #[test]
    fn sections_bit_identical() {
        use fast_mesh_store::par;
        for mode in [UpdateMode::Eager, UpdateMode::Lazy] {
            // The forced-sequential run is the code a feature-off build runs.
            par::set_force_sequential(true);
            let reference = section_drag(40, mode);
            par::set_force_sequential(false);
            if !par::enabled() {
                // Every drag below would be this one again.
                continue;
            }
            for k in [1, 2, 8] {
                let got = par::with_threads(k, || section_drag(40, mode));
                assert!(got == reference, "{mode:?}: {k} threads");
            }
            let batches = PARALLEL_SECTION_BATCHES.with(Cell::get);
            assert!(section_drag(40, mode) == reference, "{mode:?}: global pool");
            let ran = PARALLEL_SECTION_BATCHES.with(Cell::get) > batches;
            assert!(ran, "{mode:?}: ran sequentially");
        }
    }

    #[test]
    fn small_inputs_stay_sequential() {
        use fast_mesh_store::par;
        let before = par::parallel_entries();
        let batches = PARALLEL_SECTION_BATCHES.with(Cell::get);
        for mode in [UpdateMode::Eager, UpdateMode::Lazy] {
            section_drag(8, mode);
        }
        assert_eq!(par::parallel_entries(), before);
        assert_eq!(PARALLEL_SECTION_BATCHES.with(Cell::get), batches);
    }
}
