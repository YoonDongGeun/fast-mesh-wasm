//! World-space queries: raycasts, closest points and mesh-vs-mesh queries.
//!
//! Every query maps its world `f64` input into each mesh's local space (in
//! `f64`, then rounded to `f32`), runs the Phase 3 query there and maps the
//! result back. World bounds are rounded outward to `f32` and the results
//! filtered again against the exact `f64` bound (spec §15.9).
//!
//! Queries over all meshes cull meshes by their world boxes ([`world_box`]).
//! The culling test uses the exact world input, while the Phase 3 query sees
//! it rounded to `f32` and mapped with a rotation that is orthonormal only
//! within `1e-6`. So each test grows the box by a pad that covers both (see
//! [`query_pad`]): culling never drops a mesh the local query would find
//! something in.

use fast_mesh_bvh::{Aabb, Bvh};
use fast_mesh_query::{ClosestHit, ClosestOptions, Cull, Ray, RayHit, RayOptions, Rigid};
use fast_mesh_store::{StoreError, TriangleId};

use crate::placement::{local_to_world, relative_rigid, world_to_local};
use crate::scene::Entry;
use crate::{MeshId, Placement, Result, Scene, SceneError};

/// Relative slack for a placement's rotation being orthonormal only within
/// `1e-6` per entry, and for the re-orthonormalised relative rotation of
/// mesh-vs-mesh queries (§15.2), with room to spare.
pub(crate) const PLACEMENT_SLACK: f64 = 2e-5;

/// Relative rounding slack of the `f32` arithmetic inside Phase 3 queries.
const F32_SLACK: f64 = 16.0 * f32::EPSILON as f64;

/// The smallest `f32` not below `x` (`x` not NaN).
pub(crate) fn round_up(x: f64) -> f32 {
    let r = x as f32;
    if f64::from(r) < x { r.next_up() } else { r }
}

/// The largest `f32` not above `x` (`x` not NaN).
pub(crate) fn round_down(x: f64) -> f32 {
    let r = x as f32;
    if f64::from(r) > x { r.next_down() } else { r }
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// The largest absolute coordinate of an `f32` point.
fn max_abs(v: [f32; 3]) -> f64 {
    v.iter().fold(0.0, |m, c| m.max(f64::from(c.abs())))
}

/// `Rᵀ d`: a world direction in the mesh's local frame.
fn rotate_back(p: &Placement, d: [f64; 3]) -> [f64; 3] {
    let r = &p.rotation;
    std::array::from_fn(|i| r[0][i] * d[0] + r[1][i] * d[1] + r[2][i] * d[2])
}

fn oom() -> SceneError {
    SceneError::Store(StoreError::OutOfMemory)
}

/// An empty vector with room for `n` elements.
pub(crate) fn try_vec<T>(n: usize) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve(n).map_err(|_| oom())?;
    Ok(v)
}

fn check_finite(p: [f64; 3], what: &'static str) -> Result<()> {
    if p.iter().all(|c| c.is_finite()) {
        Ok(())
    } else {
        Err(SceneError::InvalidInput(what))
    }
}

/// Checks a distance-like bound: zero or greater, infinity allowed.
pub(crate) fn check_bound(x: f64, what: &'static str) -> Result<()> {
    if x.is_nan() || x < 0.0 {
        Err(SceneError::InvalidInput(what))
    } else {
        Ok(())
    }
}

/// `d` scaled to unit length.
fn unit_dir(d: [f64; 3]) -> Result<[f64; 3]> {
    let bad = SceneError::InvalidInput("ray direction must be finite and non-zero");
    if !d.iter().all(|c| c.is_finite()) || d == [0.0; 3] {
        return Err(bad);
    }
    let n = norm(d);
    // Scale first when squaring would overflow or underflow.
    if n.is_finite() && n > 1e-150 {
        return Ok(d.map(|c| c / n));
    }
    let m = d.iter().fold(0.0f64, |m, c| m.max(c.abs()));
    let s = d.map(|c| c / m);
    let n = norm(s);
    Ok(s.map(|c| c / n))
}

/// The mesh's local box: its BVH root box, `None` when it has no triangle.
fn local_box(entry: &Entry) -> Option<Aabb> {
    entry.bvh.root_bounds().filter(|b| !b.is_empty())
}

/// The largest absolute local coordinate of the mesh's box (0 when empty).
pub(crate) fn local_mag(entry: &Entry) -> f64 {
    local_box(entry).map_or(0.0, |b| max_abs(b.min).max(max_abs(b.max)))
}

/// A world point in a mesh's local `f32` space.
struct LocalPoint {
    p: [f32; 3],
    /// How far rounding to `f32` moved it.
    err: f64,
}

fn local_point(entry: &Entry, q: [f64; 3]) -> Result<LocalPoint> {
    let l = world_to_local(&entry.placement, entry.mesh.origin(), q);
    let p = l.map(|c| c as f32);
    if !p.iter().all(|c| c.is_finite()) {
        return Err(SceneError::InvalidInput(
            "the query point is out of f32 range in a mesh's local space",
        ));
    }
    let err = norm(std::array::from_fn(|k| f64::from(p[k]) - l[k]));
    Ok(LocalPoint { p, err })
}

/// How much to grow the world box `wbox` of `entry` so that a culling test
/// with the exact world point `q` keeps every mesh in which the local query
/// (from the `f32` point `local`, `err` from the exact local point, and for a
/// ray an `f32` direction `err_dir` from the exact one) can find something.
///
/// Anything found lies in the local box, so within `far` of `q`. Mapped back
/// to the world, the local query's point or ray strays from the exact one by
/// at most its rounding (`err`, plus `err_dir` per unit of `t`), plus the
/// rotation's distance from orthonormal times `|q - t|` and `t`, plus the
/// `f32` rounding inside the query.
fn query_pad(
    entry: &Entry,
    wbox: &Aabb64,
    q: [f64; 3],
    local: [f32; 3],
    err: f64,
    err_dir: f64,
) -> f64 {
    let (c, h) = wbox.centre_half();
    let far = norm(sub(q, c)) + norm(h);
    let from_t = norm(sub(q, entry.placement.translation));
    let mag = max_abs(local) + local_mag(entry) + far;
    err + 2.0 * (far + err) * err_dir + PLACEMENT_SLACK * (from_t + far) + F32_SLACK * mag
}

/// How much to grow the distance test between the world boxes `wa`, `wb` of
/// `a`, `b` so that it keeps every pair that Phase 3b, run in A's space with
/// the re-orthonormalised relative transform, finds within the margin.
pub(crate) fn pair_pad(a: &Entry, wa: &Aabb64, b: &Entry, wb: &Aabb64) -> f64 {
    let (ca, ha) = wa.centre_half();
    let (cb, hb) = wb.centre_half();
    let spread = norm(sub(cb, a.placement.translation))
        + norm(sub(ca, b.placement.translation))
        + norm(ha)
        + norm(hb)
        + local_mag(a)
        + local_mag(b);
    PLACEMENT_SLACK * (wa.distance_to_box(wb) + spread)
}

/// The world point of a mesh-local `f32` point.
fn to_world(entry: &Entry, p: [f32; 3]) -> [f64; 3] {
    local_to_world(&entry.placement, entry.mesh.origin(), p.map(f64::from))
}

fn closest_hit(entry: &Entry, mesh: MeshId, hit: ClosestHit) -> SceneClosestHit {
    SceneClosestHit {
        mesh,
        hit,
        point: to_world(entry, hit.point),
        distance: f64::from(hit.distance),
    }
}

/// Options for [`Scene::raycast`], in world units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneRayOptions {
    /// Smallest world distance that counts as a hit (inclusive). Default `0.0`.
    pub min_t: f64,
    /// Largest world distance that counts as a hit (inclusive). Default
    /// `f64::INFINITY`.
    pub max_t: f64,
    /// Which faces to ignore. Default [`Cull::None`].
    pub cull: Cull,
}

impl Default for SceneRayOptions {
    fn default() -> Self {
        SceneRayOptions {
            min_t: 0.0,
            max_t: f64::INFINITY,
            cull: Cull::None,
        }
    }
}

/// Options for [`Scene::closest_point`] and [`Scene::closest_point_on`], in
/// world units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneClosestOptions {
    /// Largest world distance that counts as found (inclusive). Default
    /// `f64::INFINITY`.
    pub max_distance: f64,
}

impl Default for SceneClosestOptions {
    fn default() -> Self {
        SceneClosestOptions {
            max_distance: f64::INFINITY,
        }
    }
}

/// Where a world ray hit a mesh of the scene.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneRayHit {
    /// The mesh that was hit.
    pub mesh: MeshId,
    /// The hit as the Phase 3 raycast reported it, in the mesh's local space.
    pub hit: RayHit,
    /// The hit point in the world: `hit.point` placed by the mesh's origin
    /// and placement.
    pub point: [f64; 3],
    /// The world distance from the ray's origin: `hit.t`, which is a distance
    /// because the local direction is the rotated unit world direction.
    pub t: f64,
}

/// The point of the scene nearest to a world point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneClosestHit {
    /// The mesh the point lies on.
    pub mesh: MeshId,
    /// The hit as the Phase 3 query reported it, in the mesh's local space.
    pub hit: ClosestHit,
    /// The nearest point in the world: `hit.point` placed by the mesh's
    /// origin and placement.
    pub point: [f64; 3],
    /// The distance: `hit.distance`, which carries over to the world because
    /// the placement is rigid.
    pub distance: f64,
}

/// The closest pair of points between two meshes, in the world.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneMeshDistance {
    /// The distance between the two surfaces (0 when they touch or
    /// intersect), as Phase 3's `mesh_distance` reports it.
    pub distance: f64,
    /// The closest point on A's triangle `triangle_a`, in the world.
    pub point_a: [f64; 3],
    /// The closest point on B's triangle `triangle_b`, in the world.
    pub point_b: [f64; 3],
    /// The triangle of A that `point_a` lies on.
    pub triangle_a: TriangleId,
    /// The triangle of B that `point_b` lies on.
    pub triangle_b: TriangleId,
}

impl Scene {
    /// The nearest hit of a world ray over every mesh, with
    /// `opts.min_t <= t <= opts.max_t` in world distance. Equal `t` goes to
    /// the lower [`MeshId`]; within a mesh, Phase 3's rule (smallest
    /// triangle id) applies.
    ///
    /// `dir` is normalised in `f64`, so `t` is a world distance. For each
    /// mesh not culled by its world box, the ray is mapped into the mesh's
    /// local space (`Rᵀ (origin - t) - o` and `Rᵀ dir`, in `f64`, then
    /// rounded to `f32`) and Phase 3's [`fast_mesh_query::raycast`] runs
    /// there, with the bounds rounded outward to `f32`. The local `t` is the
    /// world `t` up to that rounding; hits are kept only when it lies within
    /// the exact `f64` bounds, and a hit dropped just below `min_t` does not
    /// hide one beyond it.
    ///
    /// Every Lazy BVH is brought up to date first (§15.4).
    ///
    /// # Errors
    /// [`SceneError::InvalidInput`] for a non-finite origin, a zero or
    /// non-finite direction, NaN bounds or `min_t > max_t`; then a BVH or
    /// memory error from a Lazy update, or [`SceneError::StaleBvh`] for the
    /// lowest stale Manual mesh; then [`SceneError::InvalidInput`] for an
    /// origin out of `f32` range in some mesh's local space, or a memory
    /// error.
    pub fn raycast(
        &mut self,
        origin: [f64; 3],
        dir: [f64; 3],
        opts: &SceneRayOptions,
    ) -> Result<Option<SceneRayHit>> {
        self.drop_field_unit();
        check_finite(origin, "ray origin must be finite")?;
        let d = unit_dir(dir)?;
        if opts.min_t.is_nan() || opts.max_t.is_nan() || opts.min_t > opts.max_t {
            return Err(SceneError::InvalidInput(
                "min_t and max_t must not be NaN, and min_t must not exceed max_t",
            ));
        }
        self.prepare(None)?;
        // Broadphase: the meshes whose padded world box the ray enters within
        // the bounds, nearest entry first.
        let mut order = Vec::new();
        for (id, e) in self.entries.iter() {
            // Empty meshes hold nothing: skip them before mapping the
            // point, which may be out of f32 range in their local space.
            let wbox = world_box(e);
            if wbox.is_empty() {
                continue;
            }
            let lo = local_point(e, origin)?;
            let dl = rotate_back(&e.placement, d);
            let dl32 = dl.map(|c| c as f32);
            let err_dir = norm(std::array::from_fn(|k| f64::from(dl32[k]) - dl[k]));
            let pad = query_pad(e, &wbox, origin, lo.p, lo.err, err_dir);
            if let Some(enter) = wbox.ray_enter(origin, d, pad, opts.min_t, opts.max_t) {
                order.try_reserve(1).map_err(|_| oom())?;
                let ray = Ray {
                    origin: lo.p,
                    dir: dl32,
                };
                order.push((enter, id, ray));
            }
        }
        order.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
        let (min32, max32) = (round_down(opts.min_t), round_up(opts.max_t));
        let mut best: Option<SceneRayHit> = None;
        for (enter, id, ray) in order {
            if best.is_some_and(|b| enter > b.t) {
                break;
            }
            let e = self.entry(id)?;
            // Nothing beyond the best hit so far can win.
            let mut local = RayOptions {
                min_t: min32,
                max_t: best.map_or(max32, |b| b.hit.t),
                cull: opts.cull,
            };
            let hit = loop {
                let Some(h) = fast_mesh_query::raycast(&e.mesh, &e.bvh, &ray, &local)? else {
                    break None;
                };
                if f64::from(h.t) >= opts.min_t {
                    break Some(h);
                }
                // Only `min32` itself can fall below `min_t`: look past it.
                local.min_t = h.t.next_up();
                if local.min_t > local.max_t {
                    break None;
                }
            };
            let Some(h) = hit else { continue };
            let t = f64::from(h.t);
            if t > opts.max_t {
                continue;
            }
            if best.is_none_or(|b| t < b.t || (t == b.t && id < b.mesh)) {
                best = Some(SceneRayHit {
                    mesh: id,
                    hit: h,
                    point: to_world(e, h.point),
                    t,
                });
            }
        }
        Ok(best)
    }

    /// The point of the scene nearest to `p`, within `opts.max_distance`.
    /// Equal distances go to the lower [`MeshId`]; within a mesh, Phase 3's
    /// rule (smallest triangle id) applies.
    ///
    /// Meshes are visited by the distance from `p` to their world box and
    /// pruned by the best distance so far. In each, `p` is mapped into local
    /// space (in `f64`, then rounded to `f32`) and Phase 3's
    /// [`fast_mesh_query::closest_point`] runs with `max_distance` rounded up
    /// to `f32`; the local distance, which carries over to the world because
    /// the placement is rigid, must then be within the exact bound.
    ///
    /// Every Lazy BVH is brought up to date first (§15.4).
    ///
    /// # Errors
    /// [`SceneError::InvalidInput`] for a non-finite `p` or a negative or
    /// NaN `max_distance`; then a BVH or memory error from a Lazy update, or
    /// [`SceneError::StaleBvh`] for the lowest stale Manual mesh; then
    /// [`SceneError::InvalidInput`] for a `p` out of `f32` range in some
    /// mesh's local space, or a memory error.
    pub fn closest_point(
        &mut self,
        p: [f64; 3],
        opts: &SceneClosestOptions,
    ) -> Result<Option<SceneClosestHit>> {
        self.drop_field_unit();
        check_finite(p, "query point must be finite")?;
        check_bound(opts.max_distance, MAX_DISTANCE)?;
        self.prepare(None)?;
        let mut order = Vec::new();
        for (id, e) in self.entries.iter() {
            // Empty meshes hold nothing: skip them before mapping the
            // point, which may be out of f32 range in their local space.
            let wbox = world_box(e);
            if wbox.is_empty() {
                continue;
            }
            let lp = local_point(e, p)?;
            let pad = query_pad(e, &wbox, p, lp.p, lp.err, 0.0);
            let lower = (wbox.distance_to(p) - pad).max(0.0);
            if lower <= opts.max_distance {
                order.try_reserve(1).map_err(|_| oom())?;
                order.push((lower, id, lp.p));
            }
        }
        order.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
        let max32 = round_up(opts.max_distance);
        let mut best: Option<SceneClosestHit> = None;
        for (lower, id, q) in order {
            if best.is_some_and(|b| lower > b.distance) {
                break;
            }
            let e = self.entry(id)?;
            let local = ClosestOptions {
                max_distance: best.map_or(max32, |b| b.hit.distance),
            };
            let Some(h) = fast_mesh_query::closest_point(&e.mesh, &e.bvh, q, &local)? else {
                continue;
            };
            let d = f64::from(h.distance);
            if d <= opts.max_distance
                && best.is_none_or(|b| d < b.distance || (d == b.distance && id < b.mesh))
            {
                best = Some(closest_hit(e, id, h));
            }
        }
        Ok(best)
    }

    /// [`Scene::closest_point`] on mesh `id` alone. Only that mesh's BVH is
    /// brought up to date (§15.4).
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale id; then as for
    /// [`Scene::closest_point`], with [`SceneError::StaleBvh`] only when this
    /// mesh is a stale Manual one.
    pub fn closest_point_on(
        &mut self,
        id: MeshId,
        p: [f64; 3],
        opts: &SceneClosestOptions,
    ) -> Result<Option<SceneClosestHit>> {
        self.drop_field_unit();
        self.entry(id)?;
        check_finite(p, "query point must be finite")?;
        check_bound(opts.max_distance, MAX_DISTANCE)?;
        self.prepare(Some(&[id]))?;
        let e = self.entry(id)?;
        let lp = local_point(e, p)?;
        let local = ClosestOptions {
            max_distance: round_up(opts.max_distance),
        };
        let hit = fast_mesh_query::closest_point(&e.mesh, &e.bvh, lp.p, &local)?;
        Ok(hit
            .filter(|h| f64::from(h.distance) <= opts.max_distance)
            .map(|h| closest_hit(e, id, h)))
    }

    /// Checks both ids, then brings both BVHs up to date (§15.4) and returns
    /// the entries with the transform from B's local space to A's (§5,
    /// re-orthonormalised per §15.2).
    fn pair(&mut self, a: MeshId, b: MeshId) -> Result<(&Entry, &Entry, Rigid)> {
        self.prepare(Some(&[a, b]))?;
        let (ea, eb) = (self.entry(a)?, self.entry(b)?);
        let xf = relative_rigid(
            &ea.placement,
            ea.mesh.origin(),
            &eb.placement,
            eb.mesh.origin(),
        );
        Ok((ea, eb, xf))
    }

    /// The mesh's BVH, in mesh-local `f32` space, for running Phase 3
    /// queries directly. Never updates it.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale id, or
    /// [`SceneError::StaleBvh`] when the BVH is not current with the mesh
    /// (see [`Scene::is_current`]); bring it up to date with
    /// [`Scene::update`] or [`Scene::flush`] first.
    pub fn bvh(&self, id: MeshId) -> Result<&Bvh> {
        let e = self.entry(id)?;
        if !e.is_current() {
            return Err(SceneError::StaleBvh(id));
        }
        Ok(&e.bvh)
    }

    /// The transform from mesh `b`'s local space to mesh `a`'s local space,
    /// origins included: the one the pair queries ([`Scene::intersects`],
    /// [`Scene::mesh_distance`], [`Scene::triangles_near`]) pass to Phase 3,
    /// in the convention of [`Rigid`]. Its rotation is re-orthonormalised,
    /// so it always passes `Rigid::validate`.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale id.
    pub fn relative_transform(&self, a: MeshId, b: MeshId) -> Result<Rigid> {
        let (ea, eb) = (self.entry(a)?, self.entry(b)?);
        Ok(relative_rigid(
            &ea.placement,
            ea.mesh.origin(),
            &eb.placement,
            eb.mesh.origin(),
        ))
    }

    /// A world point in mesh `id`'s local coordinates, origin included:
    /// `Rᵀ (q - t) - o`, in `f64`.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale id.
    pub fn world_to_local(&self, id: MeshId, q: [f64; 3]) -> Result<[f64; 3]> {
        let e = self.entry(id)?;
        Ok(world_to_local(&e.placement, e.mesh.origin(), q))
    }

    /// A point in mesh `id`'s local coordinates in the world, origin
    /// included: `R (o + p) + t`, in `f64`.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale id.
    pub fn local_to_world(&self, id: MeshId, p: [f64; 3]) -> Result<[f64; 3]> {
        let e = self.entry(id)?;
        Ok(local_to_world(&e.placement, e.mesh.origin(), p))
    }

    fn check_ids(&self, a: MeshId, b: MeshId) -> Result<()> {
        self.entry(a)?;
        self.entry(b)?;
        Ok(())
    }

    /// Whether any triangle of mesh `a` intersects or touches any triangle of
    /// mesh `b`, as placed in the world: Phase 3's
    /// [`fast_mesh_query::intersects`] with the relative transform of B's
    /// local space into A's. Passing the same id twice means two coincident
    /// copies, as in Phase 3. Only these two BVHs are brought up to date.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`]; then a BVH or memory error from a Lazy
    /// update, or [`SceneError::StaleBvh`] when either is a stale Manual
    /// mesh (the lower id first).
    pub fn intersects(&mut self, a: MeshId, b: MeshId) -> Result<bool> {
        self.drop_field_unit();
        self.check_ids(a, b)?;
        let (ea, eb, xf) = self.pair(a, b)?;
        Ok(fast_mesh_query::intersects(
            &ea.mesh, &ea.bvh, &eb.mesh, &eb.bvh, &xf,
        )?)
    }

    /// The distance between the surfaces of meshes `a` and `b` as placed in
    /// the world, with a closest pair of points in the world, or `None` when
    /// it is greater than `max_distance` (inclusive, may be infinite) or a
    /// mesh has no triangle.
    ///
    /// Phase 3's [`fast_mesh_query::mesh_distance`] runs in A's space with
    /// `max_distance` rounded up to `f32`; the result is kept only when its
    /// distance is within the exact bound. Distances carry over unchanged
    /// because placements are rigid. Only these two BVHs are brought up to
    /// date.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`]; [`SceneError::InvalidInput`] for a
    /// negative or NaN `max_distance`; then as for [`Scene::intersects`].
    pub fn mesh_distance(
        &mut self,
        a: MeshId,
        b: MeshId,
        max_distance: f64,
    ) -> Result<Option<SceneMeshDistance>> {
        self.drop_field_unit();
        self.check_ids(a, b)?;
        check_bound(max_distance, MAX_DISTANCE)?;
        let (ea, eb, xf) = self.pair(a, b)?;
        let found = fast_mesh_query::mesh_distance(
            &ea.mesh,
            &ea.bvh,
            &eb.mesh,
            &eb.bvh,
            &xf,
            round_up(max_distance),
        )?;
        Ok(found
            .filter(|d| f64::from(d.distance) <= max_distance)
            .map(|d| SceneMeshDistance {
                distance: f64::from(d.distance),
                // Both points are in A's space.
                point_a: to_world(ea, d.point_a),
                point_b: to_world(ea, d.point_b),
                triangle_a: d.triangle_a,
                triangle_b: d.triangle_b,
            }))
    }

    /// Appends to `out_a` the triangles of mesh `a` within `margin` of mesh
    /// `b`, and to `out_b` those of `b` within `margin` of `a`, as placed in
    /// the world: Phase 3's [`fast_mesh_query::triangles_near`] in A's space.
    /// Elements already in the vectors stay in front; each appended part is
    /// ascending without duplicates. Only these two BVHs are brought up to
    /// date.
    ///
    /// `margin` is rounded up to `f32`. The lists cannot be filtered again
    /// per triangle, so when `margin` is not an `f32` value they may hold
    /// triangles up to one `f32` step (relative `6e-8`) beyond it.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`]; [`SceneError::InvalidInput`] for a
    /// negative or NaN `margin`; then as for [`Scene::intersects`], or
    /// [`SceneError::Query`] with `OutOfMemory` when a vector cannot grow. On
    /// any error both vectors keep their lengths.
    pub fn triangles_near(
        &mut self,
        a: MeshId,
        b: MeshId,
        margin: f64,
        out_a: &mut Vec<TriangleId>,
        out_b: &mut Vec<TriangleId>,
    ) -> Result<()> {
        self.drop_field_unit();
        self.check_ids(a, b)?;
        check_bound(margin, MARGIN)?;
        let (ea, eb, xf) = self.pair(a, b)?;
        Ok(fast_mesh_query::triangles_near(
            &ea.mesh,
            &ea.bvh,
            &eb.mesh,
            &eb.bvh,
            &xf,
            round_up(margin),
            out_a,
            out_b,
        )?)
    }
}

const MAX_DISTANCE: &str = "max_distance must be zero or greater, and not NaN";
pub(crate) const MARGIN: &str = "margin must be zero or greater, and not NaN";

/// An axis-aligned box in world `f64` coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Aabb64 {
    pub(crate) min: [f64; 3],
    pub(crate) max: [f64; 3],
}

impl Aabb64 {
    /// The box holding nothing.
    pub(crate) const EMPTY: Aabb64 = Aabb64 {
        min: [f64::INFINITY; 3],
        max: [f64::NEG_INFINITY; 3],
    };

    /// True when the box holds no point.
    pub(crate) fn is_empty(&self) -> bool {
        (0..3).any(|k| self.min[k] > self.max[k])
    }

    /// The centre and half extents (for a non-empty box).
    fn centre_half(&self) -> ([f64; 3], [f64; 3]) {
        let c = std::array::from_fn(|k| 0.5 * self.min[k] + 0.5 * self.max[k]);
        let h = std::array::from_fn(|k| 0.5 * self.max[k] - 0.5 * self.min[k]);
        (c, h)
    }

    /// Distance from `p` to the box; 0 inside.
    fn distance_to(&self, p: [f64; 3]) -> f64 {
        norm(std::array::from_fn(|k| {
            (self.min[k] - p[k]).max(p[k] - self.max[k]).max(0.0)
        }))
    }

    /// Distance between two boxes; 0 when they overlap.
    pub(crate) fn distance_to_box(&self, other: &Aabb64) -> f64 {
        norm(std::array::from_fn(|k| {
            (self.min[k] - other.max[k])
                .max(other.min[k] - self.max[k])
                .max(0.0)
        }))
    }

    /// The smallest `t` in `[t0, t1]` at which the ray `o + t d` is inside
    /// the box grown by `pad`, or `None`.
    fn ray_enter(&self, o: [f64; 3], d: [f64; 3], pad: f64, t0: f64, t1: f64) -> Option<f64> {
        let (mut lo, mut hi) = (t0, t1);
        for k in 0..3 {
            let (bmin, bmax) = (self.min[k] - pad, self.max[k] + pad);
            if d[k] == 0.0 {
                if o[k] < bmin || o[k] > bmax {
                    return None;
                }
                continue;
            }
            let (a, b) = ((bmin - o[k]) / d[k], (bmax - o[k]) / d[k]);
            // `max`/`min` skip a NaN (0 / tiny overflow): no constraint.
            lo = lo.max(a.min(b));
            hi = hi.min(a.max(b));
        }
        (lo <= hi).then_some(lo)
    }
}

/// A world box holding every triangle of the mesh: its BVH root box (local
/// `f32`, plus the origin) placed by the rigid transform, as centre ± `|R|`
/// half extents, grown by an `f64` rounding bound so it stays conservative.
/// Empty for a mesh with no triangle.
pub(crate) fn world_box(entry: &Entry) -> Aabb64 {
    let Some(b) = local_box(entry) else {
        return Aabb64::EMPTY;
    };
    let o = entry.mesh.origin();
    let lo: [f64; 3] = std::array::from_fn(|k| o[k] + f64::from(b.min[k]));
    let hi: [f64; 3] = std::array::from_fn(|k| o[k] + f64::from(b.max[k]));
    let c: [f64; 3] = std::array::from_fn(|k| 0.5 * lo[k] + 0.5 * hi[k]);
    let h: [f64; 3] = std::array::from_fn(|k| 0.5 * hi[k] - 0.5 * lo[k]);
    let p = &entry.placement;
    let wc = p.apply(c);
    let r = &p.rotation;
    let mut out = Aabb64::EMPTY;
    for i in 0..3 {
        let a = r[i].map(f64::abs);
        let wh = a[0] * h[0] + a[1] * h[1] + a[2] * h[2];
        // Every rounding above is relative to at most this size.
        let size = p.translation[i].abs()
            + a[0] * (c[0].abs() + h[0])
            + a[1] * (c[1].abs() + h[1])
            + a[2] * (c[2].abs() + h[2]);
        let pad = 16.0 * f64::EPSILON * size + f64::MIN_POSITIVE;
        out.min[i] = wc[i] - wh - pad;
        out.max[i] = wc[i] + wh + pad;
    }
    out
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{MeshOptions, Placement, SceneOptions, UpdateMode};
    use fast_mesh_query::{ClosestOptions, Ray, RayOptions, Rigid};
    use fast_mesh_store::Mesh;

    pub(crate) const RZ90: [[f64; 3]; 3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    pub(crate) const RX90: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]];

    /// A rotation by `deg` degrees about the axis `(1, 2, 2) / 3`.
    fn tilted(deg: f64) -> [[f64; 3]; 3] {
        let (s, c) = deg.to_radians().sin_cos();
        let [x, y, z] = [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0];
        let k = 1.0 - c;
        [
            [c + x * x * k, x * y * k - z * s, x * z * k + y * s],
            [y * x * k + z * s, c + y * y * k, y * z * k - x * s],
            [z * x * k - y * s, z * y * k + x * s, c + z * z * k],
        ]
    }

    pub(crate) fn place(rotation: [[f64; 3]; 3], translation: [f64; 3]) -> MeshOptions {
        MeshOptions {
            placement: Placement {
                rotation,
                translation,
            },
            ..MeshOptions::default()
        }
    }

    /// The closed box `[min, min + size]`, 12 triangles.
    pub(crate) fn cube(min: [f32; 3], size: f32) -> Mesh {
        let mut pos = Vec::new();
        for i in 0..8u32 {
            let b = |bit: u32| if i & bit != 0 { size } else { 0.0 };
            pos.extend_from_slice(&[min[0] + b(1), min[1] + b(2), min[2] + b(4)]);
        }
        let idx = [
            0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4, 2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4,
            6, 1, 3, 5, 3, 7, 5,
        ];
        Mesh::from_buffers(&pos, &idx).unwrap()
    }

    /// A unit square in the plane `x = x0` spanning `y, z` in `[-s, s]`.
    pub(crate) fn wall(x0: f32, s: f32) -> Mesh {
        let pos = [x0, -s, -s, x0, s, -s, x0, -s, s, x0, s, s];
        Mesh::from_buffers(&pos, &[0, 1, 2, 1, 3, 2]).unwrap()
    }

    /// A cube built from `f64` positions far from zero, so its origin is not
    /// zero.
    fn far_cube(min: [f64; 3], size: f64) -> Mesh {
        let m = cube([0.0; 3], 1.0);
        let pos: Vec<f64> = m
            .positions()
            .iter()
            .flat_map(|p| (0..3).map(move |k| min[k] + size * f64::from(p[k])))
            .collect();
        let idx: Vec<u32> = m.triangles().iter().flatten().copied().collect();
        Mesh::from_buffers_f64(&pos, &idx).unwrap()
    }

    /// `Rᵀ (q - t) - o`, written out independently of the crate's helpers.
    fn to_local(p: &Placement, origin: [f64; 3], q: [f64; 3]) -> [f64; 3] {
        let d = [0, 1, 2].map(|i| q[i] - p.translation[i]);
        let r = &p.rotation;
        [0, 1, 2].map(|i| r[0][i] * d[0] + r[1][i] * d[1] + r[2][i] * d[2] - origin[i])
    }

    fn rotate_back(p: &Placement, d: [f64; 3]) -> [f64; 3] {
        let r = &p.rotation;
        [0, 1, 2].map(|i| r[0][i] * d[0] + r[1][i] * d[1] + r[2][i] * d[2])
    }

    fn to_world(p: &Placement, origin: [f64; 3], l: [f32; 3]) -> [f64; 3] {
        let q = [0, 1, 2].map(|i| origin[i] + f64::from(l[i]));
        let r = &p.rotation;
        [0, 1, 2].map(|i| r[i][0] * q[0] + r[i][1] * q[1] + r[i][2] * q[2] + p.translation[i])
    }

    fn unit(d: [f64; 3]) -> [f64; 3] {
        let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        d.map(|c| c / n)
    }

    /// A scene of three placed cubes: integer, quarter-turn and tilted.
    fn three_cubes() -> (Scene, Vec<MeshId>) {
        let mut scene = Scene::new(SceneOptions::default());
        let ids = vec![
            scene
                .add_mesh(
                    cube([0.0; 3], 2.0),
                    place(Placement::IDENTITY.rotation, [5.0, 0.0, 0.0]),
                )
                .unwrap(),
            scene
                .add_mesh(cube([1.0, 0.0, -1.0], 1.0), place(RZ90, [9.0, 3.0, 0.5]))
                .unwrap(),
            scene
                .add_mesh(
                    far_cube([1e5, -2e5, 3e5], 1.5),
                    place(tilted(30.0), [-2.0, 1.0, 4.0]),
                )
                .unwrap(),
        ];
        (scene, ids)
    }

    /// The nearest hit by calling Phase 3 on every mesh with hand-computed
    /// local rays (default bounds).
    fn brute_raycast(scene: &Scene, o: [f64; 3], d: [f64; 3]) -> Option<SceneRayHit> {
        let d = unit(d);
        let mut best: Option<SceneRayHit> = None;
        for id in scene.ids() {
            let e = scene.entries.get(id).unwrap();
            let origin = e.mesh.origin();
            let ray = Ray {
                origin: to_local(&e.placement, origin, o).map(|c| c as f32),
                dir: rotate_back(&e.placement, d).map(|c| c as f32),
            };
            let Some(hit) =
                fast_mesh_query::raycast(&e.mesh, &e.bvh, &ray, &RayOptions::default()).unwrap()
            else {
                continue;
            };
            let t = f64::from(hit.t);
            if best.is_none_or(|b| t < b.t) {
                best = Some(SceneRayHit {
                    mesh: id,
                    hit,
                    point: to_world(&e.placement, origin, hit.point),
                    t,
                });
            }
        }
        best
    }

    fn brute_closest(scene: &Scene, only: Option<MeshId>, p: [f64; 3]) -> Option<SceneClosestHit> {
        let mut best: Option<SceneClosestHit> = None;
        for id in scene.ids().filter(|i| only.is_none_or(|o| o == *i)) {
            let e = scene.entries.get(id).unwrap();
            let origin = e.mesh.origin();
            let q = to_local(&e.placement, origin, p).map(|c| c as f32);
            let Some(hit) =
                fast_mesh_query::closest_point(&e.mesh, &e.bvh, q, &ClosestOptions::default())
                    .unwrap()
            else {
                continue;
            };
            let distance = f64::from(hit.distance);
            if best.is_none_or(|b| distance < b.distance) {
                best = Some(SceneClosestHit {
                    mesh: id,
                    hit,
                    point: to_world(&e.placement, origin, hit.point),
                    distance,
                });
            }
        }
        best
    }

    #[test]
    fn raycast_matches_phase3() {
        let (mut scene, ids) = three_cubes();
        let c2 = to_world(
            &scene.placement(ids[2]).unwrap(),
            scene.mesh(ids[2]).unwrap().origin(),
            [0.0; 3],
        );
        let rays = [
            ([0.0, 1.0, 1.0], [1.0, 0.0, 0.0]),
            ([0.0, 0.7, 1.3], [3.0, 0.1, -0.05]),
            ([20.0, 3.5, 0.2], [-1.0, 0.0, 0.0]),
            ([9.0, 10.0, 0.0], [0.0, -1.0, 0.0]),
            ([8.5, 4.5, -5.0], [0.0, 0.0, 2.0]),
            ([0.0, 0.0, 0.0], [c2[0], c2[1], c2[2]]),
            ([0.0, 50.0, 0.0], [0.0, 1.0, 0.0]),
        ];
        let mut hits = 0;
        for (o, d) in rays {
            let got = scene.raycast(o, d, &SceneRayOptions::default()).unwrap();
            let want = brute_raycast(&scene, o, d);
            assert_eq!(got, want, "ray {o:?} {d:?}");
            hits += usize::from(got.is_some());
        }
        assert!(hits >= 5, "only {hits} rays hit");
        // A direction of any length gives the same world distances.
        let a = scene.raycast(rays[0].0, [7.0, 0.0, 0.0], &SceneRayOptions::default());
        let b = scene.raycast(rays[0].0, [1.0, 0.0, 0.0], &SceneRayOptions::default());
        assert_eq!(a, b);
        assert_eq!(a.unwrap().unwrap().t, 5.0);
    }

    #[test]
    fn closest_point_matches_phase3() {
        let (mut scene, ids) = three_cubes();
        let c2 = to_world(
            &scene.placement(ids[2]).unwrap(),
            scene.mesh(ids[2]).unwrap().origin(),
            [0.3, -0.2, 0.1],
        );
        let points = [
            [0.0, 0.0, 0.0],
            [6.0, 1.0, 1.0],
            [8.6, 4.4, 0.0],
            [9.0, 9.0, 9.0],
            c2,
            [c2[0] + 3.0, c2[1], c2[2] - 1.0],
        ];
        for p in points {
            let got = scene
                .closest_point(p, &SceneClosestOptions::default())
                .unwrap();
            assert_eq!(got, brute_closest(&scene, None, p), "point {p:?}");
            for &id in &ids {
                let got = scene
                    .closest_point_on(id, p, &SceneClosestOptions::default())
                    .unwrap();
                assert_eq!(got, brute_closest(&scene, Some(id), p), "{id:?} {p:?}");
            }
        }
    }

    /// The relative transform from B's local space to A's, by hand, for
    /// quarter-turn placements (where Gram-Schmidt changes nothing).
    pub(crate) fn hand_rigid(scene: &Scene, a: MeshId, b: MeshId) -> Rigid {
        let (pa, pb) = (scene.placement(a).unwrap(), scene.placement(b).unwrap());
        let (oa, ob) = (
            scene.mesh(a).unwrap().origin(),
            scene.mesh(b).unwrap().origin(),
        );
        let (ra, rb) = (pa.rotation, pb.rotation);
        let rotation =
            [0, 1, 2].map(|i| [0, 1, 2].map(|j| (0..3).map(|k| ra[k][i] * rb[k][j]).sum::<f64>()));
        let wb = [0, 1, 2]
            .map(|i| rb[i][0] * ob[0] + rb[i][1] * ob[1] + rb[i][2] * ob[2] + pb.translation[i]);
        Rigid {
            rotation,
            translation: to_local(&pa, oa, wb),
        }
    }

    #[test]
    fn pair_queries_match_phase3() {
        let mut scene = Scene::new(SceneOptions::default());
        let ids = [
            scene
                .add_mesh(
                    cube([0.0; 3], 2.0),
                    place(Placement::IDENTITY.rotation, [5.0, 0.0, 0.0]),
                )
                .unwrap(),
            // Overlaps the first.
            scene
                .add_mesh(cube([0.0; 3], 1.0), place(RZ90, [7.5, 0.5, 0.5]))
                .unwrap(),
            // 0.75 beyond the first along +y.
            scene
                .add_mesh(cube([0.0; 3], 1.0), place(RX90, [5.0, 3.75, 1.0]))
                .unwrap(),
        ];
        for &a in &ids {
            for &b in &ids {
                let xf = hand_rigid(&scene, a, b);
                let (ea, eb) = (scene.entries.get(a).unwrap(), scene.entries.get(b).unwrap());
                let (ma, ba, mb, bb) = (&ea.mesh, &ea.bvh, &eb.mesh, &eb.bvh);
                let want_i = fast_mesh_query::intersects(ma, ba, mb, bb, &xf).unwrap();
                let want_d = fast_mesh_query::mesh_distance(ma, ba, mb, bb, &xf, 10.0).unwrap();
                let (mut wa, mut wb) = (Vec::new(), Vec::new());
                fast_mesh_query::triangles_near(ma, ba, mb, bb, &xf, 1.0, &mut wa, &mut wb)
                    .unwrap();
                let (pa, oa) = (ea.placement, ma.origin());
                let want_d = want_d.map(|d| SceneMeshDistance {
                    distance: f64::from(d.distance),
                    point_a: to_world(&pa, oa, d.point_a),
                    point_b: to_world(&pa, oa, d.point_b),
                    triangle_a: d.triangle_a,
                    triangle_b: d.triangle_b,
                });
                assert_eq!(scene.intersects(a, b), Ok(want_i), "{a:?} {b:?}");
                assert_eq!(scene.mesh_distance(a, b, 10.0), Ok(want_d), "{a:?} {b:?}");
                let (mut ga, mut gb) = (vec![TriangleId::MAX], Vec::new());
                scene.triangles_near(a, b, 1.0, &mut ga, &mut gb).unwrap();
                assert_eq!(ga[0], TriangleId::MAX, "existing elements stay in front");
                assert_eq!((&ga[1..], &gb[..]), (&wa[..], &wb[..]), "{a:?} {b:?}");
            }
        }
        assert_eq!(scene.intersects(ids[0], ids[1]), Ok(true));
        assert_eq!(scene.intersects(ids[0], ids[2]), Ok(false));
        let d = scene.mesh_distance(ids[0], ids[2], 10.0).unwrap().unwrap();
        assert_eq!(d.distance, 0.75);
        assert_eq!(d.point_a[1], 2.0);
        assert_eq!(d.point_b[1], 2.75);
        assert_eq!(scene.mesh_distance(ids[0], ids[2], 0.5), Ok(None));
    }

    #[test]
    fn raycast_picks_nearest_mesh_and_breaks_ties_by_id() {
        let mut scene = Scene::new(SceneOptions::default());
        let identity = Placement::IDENTITY.rotation;
        // `low` (index 0) and `high` (index 1) both have a wall at x = 5, but
        // `high`'s box starts at x = 1, so it is visited first.
        let low = scene
            .add_mesh(wall(5.0, 1.0), MeshOptions::default())
            .unwrap();
        let high_mesh = {
            let pos = [
                5.0, -1.0, -1.0, 5.0, 1.0, -1.0, 5.0, -1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 5.0, 5.0, 1.0,
                6.0, 5.0, 1.0, 5.0, 6.0,
            ];
            Mesh::from_buffers(&pos, &[0, 1, 2, 1, 3, 2, 4, 5, 6]).unwrap()
        };
        let high = scene.add_mesh(high_mesh, MeshOptions::default()).unwrap();
        assert!(low < high);
        let o = [0.0, 0.0, 0.0];
        let x = [1.0, 0.0, 0.0];
        let hit = scene
            .raycast(o, x, &SceneRayOptions::default())
            .unwrap()
            .unwrap();
        assert_eq!((hit.mesh, hit.t), (low, 5.0));
        // A nearer mesh wins whatever its id: a wall turned about x (which
        // keeps it in its plane) and moved to x = 3.
        let near = scene
            .add_mesh(wall(0.0, 1.0), place(RX90, [3.0, 0.0, 0.0]))
            .unwrap();
        let hit = scene
            .raycast(o, x, &SceneRayOptions::default())
            .unwrap()
            .unwrap();
        assert_eq!((hit.mesh, hit.t, hit.point), (near, 3.0, [3.0, 0.0, 0.0]));
        // Two walls at x = 2, one unplaced and one turned and moved there:
        // the lower id wins the tie.
        let flat = scene
            .add_mesh(wall(2.0, 1.0), MeshOptions::default())
            .unwrap();
        let turned = scene
            .add_mesh(wall(1.0, 1.0), place(RX90, [1.0, 0.0, 0.0]))
            .unwrap();
        assert!(flat < turned);
        let hit = scene
            .raycast(o, x, &SceneRayOptions::default())
            .unwrap()
            .unwrap();
        assert_eq!((hit.mesh, hit.t), (flat, 2.0));
        scene
            .set_placement(
                flat,
                Placement {
                    rotation: identity,
                    translation: [0.5, 0.0, 0.0],
                },
            )
            .unwrap();
        let hit = scene
            .raycast(o, x, &SceneRayOptions::default())
            .unwrap()
            .unwrap();
        assert_eq!((hit.mesh, hit.t), (turned, 2.0));
        // Bounds pick among meshes too: `turned` (at 2) is skipped from
        // 2.5 on, and `flat` (now at 2.5) from 2.75 on.
        let opts = SceneRayOptions {
            min_t: 2.5,
            ..SceneRayOptions::default()
        };
        let hit = scene.raycast(o, x, &opts).unwrap().unwrap();
        assert_eq!((hit.mesh, hit.t), (flat, 2.5));
        let opts = SceneRayOptions {
            min_t: 2.75,
            ..SceneRayOptions::default()
        };
        let hit = scene.raycast(o, x, &opts).unwrap().unwrap();
        assert_eq!((hit.mesh, hit.t), (near, 3.0));
    }

    #[test]
    fn closest_point_breaks_ties_by_id() {
        let mut scene = Scene::new(SceneOptions::default());
        let identity = Placement::IDENTITY.rotation;
        // Two walls at distance 2 from the origin: `low` (index 0) at x = 2,
        // `high` (index 1) moved to x = -2, whose larger pad sorts it first.
        let low = scene
            .add_mesh(wall(2.0, 1.0), MeshOptions::default())
            .unwrap();
        let high = scene
            .add_mesh(wall(0.0, 1.0), place(identity, [-2.0, 0.0, 0.0]))
            .unwrap();
        assert!(low < high);
        let opts = SceneClosestOptions::default();
        let o = [0.0, 0.0, 0.0];
        // Two coincident walls tie exactly too.
        let twin = scene
            .add_mesh(wall(2.0, 1.0), MeshOptions::default())
            .unwrap();
        let hit = scene.closest_point(o, &opts).unwrap().unwrap();
        assert_eq!((hit.mesh, hit.distance), (low, 2.0));
        // A nearer mesh wins whatever its id.
        scene
            .set_placement(
                twin,
                Placement {
                    rotation: identity,
                    translation: [-1.0, 0.0, 0.0],
                },
            )
            .unwrap();
        let hit = scene.closest_point(o, &opts).unwrap().unwrap();
        assert_eq!((hit.mesh, hit.distance), (twin, 1.0));
        scene
            .set_placement(
                low,
                Placement {
                    rotation: identity,
                    translation: [1.0, 0.0, 0.0],
                },
            )
            .unwrap();
        let hit = scene.closest_point(o, &opts).unwrap().unwrap();
        assert_eq!((hit.mesh, hit.distance), (twin, 1.0));
    }

    #[test]
    fn queries_update_lazy_and_reject_stale_manual() {
        let mut scene = Scene::new(SceneOptions::default());
        let lazy = scene
            .add_mesh(
                wall(5.0, 1.0),
                MeshOptions {
                    mode: UpdateMode::Lazy,
                    ..MeshOptions::default()
                },
            )
            .unwrap();
        let manual = scene
            .add_mesh(
                wall(9.0, 1.0),
                MeshOptions {
                    mode: UpdateMode::Manual,
                    ..MeshOptions::default()
                },
            )
            .unwrap();
        let o = [0.0, 0.0, 0.0];
        let x = [1.0, 0.0, 0.0];
        let ray = SceneRayOptions::default();
        let near = SceneClosestOptions::default();
        // Move the Lazy wall to x = 4: the next query sees the move.
        scene
            .edit(lazy, |tx| {
                let ids: Vec<u32> = (0..4).collect();
                let pos: Vec<[f32; 3]> = ids
                    .iter()
                    .map(|&v| {
                        let p = tx.mesh().position(v).unwrap();
                        [4.0, p[1], p[2]]
                    })
                    .collect();
                tx.set_positions(&ids, &pos)
            })
            .unwrap();
        assert_eq!(scene.is_current(lazy), Ok(false));
        let hit = scene.raycast(o, x, &ray).unwrap().unwrap();
        assert_eq!((hit.mesh, hit.t), (lazy, 4.0));
        assert_eq!(scene.is_current(lazy), Ok(true));
        // A stale Manual mesh fails every query over all meshes, and the
        // named queries that need it.
        scene
            .edit(manual, |tx| tx.set_positions(&[0], &[[9.0, -1.0, -2.0]]))
            .unwrap();
        scene
            .edit(lazy, |tx| tx.set_positions(&[0], &[[4.0, -1.0, -2.0]]))
            .unwrap();
        let stale = Some(SceneError::StaleBvh(manual));
        assert_eq!(scene.raycast(o, x, &ray).err(), stale);
        // The Lazy mesh was still updated.
        assert_eq!(scene.is_current(lazy), Ok(true));
        assert_eq!(scene.closest_point(o, &near).err(), stale);
        assert_eq!(scene.closest_point_on(manual, o, &near).err(), stale);
        assert_eq!(scene.intersects(lazy, manual).err(), stale);
        assert_eq!(scene.mesh_distance(manual, lazy, 1.0).err(), stale);
        let (mut a, mut b) = (Vec::new(), Vec::new());
        assert_eq!(
            scene
                .triangles_near(lazy, manual, 1.0, &mut a, &mut b)
                .err(),
            stale
        );
        // Queries naming only other meshes still run.
        assert!(scene.closest_point_on(lazy, o, &near).unwrap().is_some());
        assert_eq!(scene.intersects(lazy, lazy), Ok(true));
        // Once updated, the Manual mesh takes part again.
        scene.update(manual).unwrap();
        assert!(scene.raycast(o, x, &ray).unwrap().is_some());
        // A Manual mesh flagged for a rebuild is stale even though its BVH
        // matches the mesh's revision.
        scene.entries.get_mut(manual).unwrap().rebuild = true;
        assert!(
            scene
                .entries
                .get(manual)
                .unwrap()
                .bvh
                .is_current(scene.mesh(manual).unwrap())
        );
        assert_eq!(scene.raycast(o, x, &ray).err(), stale);
        assert_eq!(scene.closest_point_on(manual, o, &near).err(), stale);
        scene.update(manual).unwrap();
        assert_eq!(
            scene
                .closest_point_on(manual, o, &near)
                .unwrap()
                .unwrap()
                .mesh,
            manual
        );
        // Unknown ids.
        scene.remove_mesh(manual).unwrap();
        let unknown = Some(SceneError::UnknownMesh);
        assert_eq!(scene.closest_point_on(manual, o, &near).err(), unknown);
        assert_eq!(scene.intersects(lazy, manual).err(), unknown);
        assert_eq!(scene.mesh_distance(manual, lazy, 1.0).err(), unknown);
        assert_eq!(
            scene
                .triangles_near(manual, lazy, 1.0, &mut a, &mut b)
                .err(),
            unknown
        );
    }

    #[test]
    fn bounds_round_outward() {
        // 1e-12 is far below an f32 ulp at 3, so each bound below rounds to
        // 3.0 in f32 one way or the other.
        let tiny = 1e-12;
        let mut scene = Scene::new(SceneOptions::default());
        let identity = Placement::IDENTITY.rotation;
        let a = scene
            .add_mesh(wall(1.0, 1.0), place(identity, [2.0, 0.0, 0.0]))
            .unwrap();
        let o = [0.0, 0.0, 0.0];
        let x = [1.0, 0.0, 0.0];
        let ray = |min_t: f64, max_t: f64| SceneRayOptions {
            min_t,
            max_t,
            cull: Cull::None,
        };
        let t = |scene: &mut Scene, opts| scene.raycast(o, x, &opts).unwrap().map(|h| h.t);
        // The hit is at exactly t = 3.
        assert_eq!(t(&mut scene, ray(0.0, 3.0)), Some(3.0));
        assert_eq!(t(&mut scene, ray(0.0, 3.0 - tiny)), None);
        assert_eq!(t(&mut scene, ray(0.0, 3.0 + tiny)), Some(3.0));
        assert_eq!(t(&mut scene, ray(3.0, 3.0)), Some(3.0));
        assert_eq!(t(&mut scene, ray(3.0 - tiny, 4.0)), Some(3.0));
        assert_eq!(t(&mut scene, ray(3.0 + tiny, 10.0)), None);
        // A second wall at 4: a hit dropped below `min_t` does not hide it.
        let b = scene
            .add_mesh(wall(4.0, 1.0), MeshOptions::default())
            .unwrap();
        assert_eq!(t(&mut scene, ray(3.0 + tiny, 10.0)), Some(4.0));
        scene.remove_mesh(b).unwrap();
        // Also within one mesh: both walls in the same mesh.
        let two = {
            let w = [
                3.0, -1.0, -1.0, 3.0, 1.0, -1.0, 3.0, -1.0, 1.0, 4.0, -1.0, -1.0, 4.0, 1.0, -1.0,
                4.0, -1.0, 1.0,
            ];
            Mesh::from_buffers(&w, &[0, 1, 2, 3, 4, 5]).unwrap()
        };
        scene.remove_mesh(a).unwrap();
        scene.add_mesh(two, MeshOptions::default()).unwrap();
        assert_eq!(t(&mut scene, ray(3.0 + tiny, 10.0)), Some(4.0));
        assert_eq!(t(&mut scene, ray(3.0 - tiny, 10.0)), Some(3.0));

        // Closest point: the wall is at distance exactly 3 from the origin.
        let mut scene = Scene::new(SceneOptions::default());
        let w = scene
            .add_mesh(wall(1.0, 1.0), place(identity, [2.0, 0.0, 0.0]))
            .unwrap();
        let near = |max_distance| SceneClosestOptions { max_distance };
        let d =
            |scene: &mut Scene, opts| scene.closest_point(o, &opts).unwrap().map(|h| h.distance);
        assert_eq!(d(&mut scene, near(3.0)), Some(3.0));
        assert_eq!(d(&mut scene, near(3.0 - tiny)), None);
        assert_eq!(d(&mut scene, near(3.0 + tiny)), Some(3.0));
        let on = |scene: &mut Scene, opts| {
            scene
                .closest_point_on(w, o, &opts)
                .unwrap()
                .map(|h| h.distance)
        };
        assert_eq!(on(&mut scene, near(3.0)), Some(3.0));
        assert_eq!(on(&mut scene, near(3.0 - tiny)), None);

        // Mesh distance: a second wall at x = 6, 3 away.
        let v = scene
            .add_mesh(wall(4.0, 1.0), place(identity, [2.0, 0.0, 0.0]))
            .unwrap();
        let md = |scene: &mut Scene, m| scene.mesh_distance(w, v, m).unwrap().map(|r| r.distance);
        assert_eq!(md(&mut scene, 3.0), Some(3.0));
        assert_eq!(md(&mut scene, 3.0 - tiny), None);
        assert_eq!(md(&mut scene, 3.0 + tiny), Some(3.0));
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        let (mut scene, ids) = three_cubes();
        let o = [0.0, 0.0, 0.0];
        let ray = SceneRayOptions::default();
        let invalid =
            |r: Result<Option<SceneRayHit>>| matches!(r, Err(SceneError::InvalidInput(_)));
        assert!(invalid(scene.raycast(o, [0.0; 3], &ray)));
        assert!(invalid(scene.raycast(o, [f64::NAN, 1.0, 0.0], &ray)));
        assert!(invalid(scene.raycast(o, [f64::INFINITY, 0.0, 0.0], &ray)));
        assert!(invalid(scene.raycast(
            [f64::NAN, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            &ray
        )));
        let bad = SceneRayOptions {
            min_t: 2.0,
            max_t: 1.0,
            cull: Cull::None,
        };
        assert!(invalid(scene.raycast(o, [1.0, 0.0, 0.0], &bad)));
        let bad = SceneRayOptions {
            max_t: f64::NAN,
            ..SceneRayOptions::default()
        };
        assert!(invalid(scene.raycast(o, [1.0, 0.0, 0.0], &bad)));
        // A huge but finite direction is fine.
        let big = [f64::MAX, f64::MAX, 0.0];
        assert!(scene.raycast(o, big, &ray).is_ok());
        for max_distance in [-1.0, f64::NAN] {
            let opts = SceneClosestOptions { max_distance };
            assert!(matches!(
                scene.closest_point(o, &opts),
                Err(SceneError::InvalidInput(_))
            ));
            assert!(matches!(
                scene.closest_point_on(ids[0], o, &opts),
                Err(SceneError::InvalidInput(_))
            ));
            assert!(matches!(
                scene.mesh_distance(ids[0], ids[1], max_distance),
                Err(SceneError::InvalidInput(_))
            ));
            let (mut a, mut b) = (Vec::new(), Vec::new());
            assert!(matches!(
                scene.triangles_near(ids[0], ids[1], max_distance, &mut a, &mut b),
                Err(SceneError::InvalidInput(_))
            ));
        }
        assert!(matches!(
            scene.closest_point([0.0, f64::INFINITY, 0.0], &SceneClosestOptions::default()),
            Err(SceneError::InvalidInput(_))
        ));
    }

    #[test]
    fn world_box_holds_every_placed_vertex() {
        let (scene, ids) = three_cubes();
        for id in ids {
            let e = scene.entries.get(id).unwrap();
            let b = world_box(e);
            assert!(!b.is_empty());
            for &p in e.mesh.positions() {
                let w = to_world(&e.placement, e.mesh.origin(), p);
                for k in 0..3 {
                    assert!(b.min[k] <= w[k] && w[k] <= b.max[k], "{w:?} outside {b:?}");
                }
            }
        }
        let mut scene = Scene::new(SceneOptions::default());
        let empty = scene.add_mesh(Mesh::new(), MeshOptions::default()).unwrap();
        assert!(world_box(scene.entries.get(empty).unwrap()).is_empty());
        let o = [0.0, 0.0, 0.0];
        assert_eq!(
            scene.raycast(o, [1.0, 0.0, 0.0], &SceneRayOptions::default()),
            Ok(None)
        );
        assert_eq!(
            scene.closest_point(o, &SceneClosestOptions::default()),
            Ok(None)
        );
    }

    #[test]
    fn queries_skip_empty_meshes_before_mapping_the_point() {
        let mut scene = Scene::new(SceneOptions::default());
        let wall_id = scene
            .add_mesh(wall(5.0, 1.0), MeshOptions::default())
            .unwrap();
        // An empty mesh far beyond f32 range: mapping the query point into
        // its local space would fail, but it holds nothing and must be
        // skipped first.
        let far = place(Placement::IDENTITY.rotation, [1e39, 0.0, 0.0]);
        scene.add_mesh(Mesh::new(), far).unwrap();
        let o = [0.0, 0.0, 0.0];
        let hit = scene
            .raycast(o, [1.0, 0.0, 0.0], &SceneRayOptions::default())
            .unwrap()
            .unwrap();
        assert_eq!((hit.mesh, hit.t), (wall_id, 5.0));
        let near = scene
            .closest_point(o, &SceneClosestOptions::default())
            .unwrap()
            .unwrap();
        assert_eq!((near.mesh, near.distance), (wall_id, 5.0));
    }

    #[test]
    fn bvh_accessor_rejects_stale() {
        let mut scene = Scene::new(SceneOptions::default());
        let manual = MeshOptions {
            mode: UpdateMode::Manual,
            ..MeshOptions::default()
        };
        let id = scene.add_mesh(cube([0.0; 3], 1.0), manual).unwrap();
        assert_eq!(scene.bvh(id).unwrap().generation(), {
            scene.entries.get(id).unwrap().bvh.generation()
        });
        scene
            .edit(id, |tx| tx.set_positions(&[0], &[[-1.0, -1.0, -1.0]]))
            .unwrap();
        assert_eq!(scene.bvh(id).err(), Some(SceneError::StaleBvh(id)));
        scene.update(id).unwrap();
        let bvh = scene.bvh(id).unwrap();
        assert!(bvh.is_current(scene.mesh(id).unwrap()));
        scene.remove_mesh(id).unwrap();
        assert_eq!(scene.bvh(id).err(), Some(SceneError::UnknownMesh));
    }

    #[test]
    fn relative_transform_matches_pair_queries() {
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene
            .add_mesh(
                far_cube([1000.5, -20.0, 3.0], 2.0),
                place(RZ90, [5.0, 1.0, 0.0]),
            )
            .unwrap();
        let b = scene
            .add_mesh(cube([0.0; 3], 1.0), place(RX90, [7.5, 0.5, 0.5]))
            .unwrap();
        assert_eq!(scene.relative_transform(a, b), Ok(hand_rigid(&scene, a, b)));
        assert_eq!(scene.relative_transform(b, a), Ok(hand_rigid(&scene, b, a)));
        // The transform is the one the pair queries run with.
        let xf = scene.relative_transform(a, b).unwrap();
        let (ea, eb) = (scene.entries.get(a).unwrap(), scene.entries.get(b).unwrap());
        let want = fast_mesh_query::mesh_distance(&ea.mesh, &ea.bvh, &eb.mesh, &eb.bvh, &xf, 100.0)
            .unwrap()
            .map(|d| f64::from(d.distance));
        let got = scene
            .mesh_distance(a, b, 100.0)
            .unwrap()
            .map(|d| d.distance);
        assert_eq!(got, want);

        // Local and world points round-trip through the placement and origin.
        let pa = scene.placement(a).unwrap();
        let oa = scene.mesh(a).unwrap().origin();
        assert_ne!(oa, [0.0; 3]);
        let q = [3.0, -2.0, 7.0];
        let l = scene.world_to_local(a, q).unwrap();
        assert_eq!(l, to_local(&pa, oa, q));
        let back = scene.local_to_world(a, l).unwrap();
        for k in 0..3 {
            assert!((back[k] - q[k]).abs() < 1e-9, "{back:?} {q:?}");
        }
        // B-local to A-local agrees with going through the world.
        let lb = [0.25, 0.5, 0.75];
        let via_world = scene
            .world_to_local(a, scene.local_to_world(b, lb).unwrap())
            .unwrap();
        let direct = xf.apply(lb.map(|c| c as f32));
        for k in 0..3 {
            assert!((via_world[k] - direct[k]).abs() < 1e-9);
        }
        scene.remove_mesh(b).unwrap();
        assert_eq!(scene.relative_transform(a, b), Err(SceneError::UnknownMesh));
        assert_eq!(scene.world_to_local(b, q), Err(SceneError::UnknownMesh));
        assert_eq!(scene.local_to_world(b, q), Err(SceneError::UnknownMesh));
    }
}
