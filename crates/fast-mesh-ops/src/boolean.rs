//! Boolean operations (union, subtract, intersect) between two closed meshes.
//!
//! [`boolean_plan`] builds a canonical simplex-keyed intersection arrangement
//! between mesh A and the placed tool B, sub-triangulates cut faces via
//! constrained 2D triangulation, classifies connected seam-bounded patches
//! using exact coplanar predicates and generalized winding numbers, cancels
//! opposite duplicate interfaces, de-pinches balanced non-manifold edges and
//! multi-fan vertices into a strict 2-manifold, and verifies watertightness
//! before returning [`BooleanPlan`]. [`apply_boolean`] writes the plan
//! atomically into A's transaction. The result stays on A with A's placement:
//! for world-space booleans the caller maps B into A's frame first (see
//! `fast-mesh-scene`).
//!
//! Inputs must be closed, 2-manifold, outward wound (`signed_volume > 0`), and
//! free of degenerate, duplicate, and self-intersecting triangles; anything
//! else returns [`OpsError::InvalidInput`]. Point touches, edge touches, and
//! coplanar area overlaps (flush or opposing faces) are handled directly by the
//! arrangement and de-pinching pipeline, producing closed, consistently wound,
//! 2-manifold outputs free of pinch vertices. Disjoint and nested inputs skip
//! sub-triangulation and classify whole connected components.
//!
//! New vertices take channel defaults; only positions and triangles cross over,
//! and A's UV channel is dropped when present ([`BooleanReport::uvs_dropped`]).

mod arrangement;
mod cdt;
mod classify;
mod winding;

use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_query::Rigid;
use fast_mesh_store::{Mesh, StoreError, Transaction, TriangleId, UV_CHANNEL, VertexId};

use crate::analyze::analyze;
use crate::{OpsError, Result, filled, push, reserve};

/// Which boolean to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BooleanOp {
    /// Whatever is in either mesh.
    Union,
    /// Whatever is in A but not in B.
    Subtract,
    /// Whatever is in both meshes.
    Intersect,
}

/// What [`apply_boolean`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BooleanReport {
    /// Alive triangles of the result.
    pub kept_triangles: usize,
    /// Triangles added (split pieces and the tool's kept pieces).
    pub triangles_added: usize,
    /// Mesh A's UVs were dropped: the result mixes split pieces and the tool's
    /// surface, which have no UVs in A's space. Undo restores them.
    pub uvs_dropped: bool,
}

/// The plan of [`boolean_plan`]: what to drop, rewrite and add on mesh A.
/// Built read-only from both meshes; [`apply_boolean`] writes it.
#[derive(Debug, Clone, Default)]
pub struct BooleanPlan {
    pub(super) drop_a: Vec<TriangleId>,
    pub(super) rewrite: Vec<(TriangleId, Vec<[PRef; 3]>)>,
    pub(super) fresh: Vec<[f32; 3]>,
    pub(super) tool: Vec<[PRef; 3]>,
}

/// A vertex of a planned triangle: a live vertex of A or a fresh position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PRef {
    Live(VertexId),
    Fresh(u32),
}

/// Plans the boolean `op` of A with B (B placed in A's frame by `xf`): the
/// result replaces A. Reads only; [`apply_boolean`] writes the plan.
///
/// `bvh_a` and `bvh_b` must be current for `a` and `b`.
///
/// # Errors
/// - [`OpsError::InvalidInput`] when an input is not closed, manifold, outward
///   wound, or free of degenerate, duplicate, and self-intersecting triangles,
///   when a mapped tool position leaves `f32` range, or when an arrangement
///   cannot form a valid 2-manifold.
/// - [`OpsError::Query`] or [`OpsError::Store`] from BVH/transform checks or
///   fallible allocations. Nothing is written: the plan owns every allocation.
pub fn boolean_plan(
    a: &Mesh,
    bvh_a: &Bvh,
    b: &Mesh,
    bvh_b: &Bvh,
    xf: &Rigid,
    op: BooleanOp,
) -> Result<BooleanPlan> {
    check_input(a, bvh_a)?;
    check_input(b, bvh_b)?;
    if std::ptr::eq(a, b) {
        return Err(OpsError::InvalidInput("cannot combine a mesh with itself"));
    }
    xf.validate()?;

    // The tool in A's frame, slot for slot: positions mapped through `xf`,
    // triangles untouched.
    let tool = map_tool(b, xf)?;
    let tool_bvh = Bvh::build(&tool, &BuildOptions::default()).map_err(|e| match e {
        fast_mesh_bvh::BvhError::OutOfMemory => OpsError::Store(StoreError::OutOfMemory),
        _ => OpsError::InvalidInput("boolean tool BVH failed to build"),
    })?;
    if *xf != Rigid::IDENTITY {
        check_input(&tool, &tool_bvh)?;
    }

    let arr = arrangement::build_arrangement(a, bvh_a, &tool, &tool_bvh)?;
    classify::classify_and_build_plan(a, bvh_a, &tool, &tool_bvh, arr, op)
}

/// Applies `plan` to A's transaction: drops, rewrites, and adds triangles and
/// removes orphaned vertices. A failure rolls the transaction back, so the call
/// stays atomic through [`Scene::edit`](https://docs.rs/fast-mesh-scene)-style
/// callers.
pub fn apply_boolean(tx: &mut Transaction, plan: &BooleanPlan) -> Result<BooleanReport> {
    // Pre-compute orphaned live vertices and reserve all temporary buffers
    // BEFORE issuing any write to `tx`, keeping OOM failures strictly atomic.
    let (unused, mut set_ids, mut set_tris, mut add_tris, total_added) = {
        let mesh = tx.mesh();
        let v_slots = mesh.vertex_slots();
        let t_slots = mesh.triangle_slots();
        let mut used = filled(v_slots, false)?;
        let mut touched_t = filled(t_slots, false)?;

        for &t in &plan.drop_a {
            if (t as usize) < t_slots {
                touched_t[t as usize] = true;
            }
        }
        for &(t, ref pieces) in &plan.rewrite {
            if (t as usize) < t_slots {
                touched_t[t as usize] = true;
            }
            for tri in pieces {
                for &r in tri {
                    if let PRef::Live(v) = r
                        && (v as usize) < v_slots
                    {
                        used[v as usize] = true;
                    }
                }
            }
        }
        for tri in &plan.tool {
            for &r in tri {
                if let PRef::Live(v) = r
                    && (v as usize) < v_slots
                {
                    used[v as usize] = true;
                }
            }
        }
        for t in 0..t_slots as TriangleId {
            if !touched_t[t as usize]
                && let Some(tri) = mesh.triangle(t)
            {
                for v in tri {
                    used[v as usize] = true;
                }
            }
        }

        let mut unused: Vec<VertexId> = Vec::new();
        for v in 0..v_slots as VertexId {
            if mesh.is_vertex_alive(v) && !used[v as usize] {
                push(&mut unused, v)?;
            }
        }

        let extra_rewrite: usize = plan
            .rewrite
            .iter()
            .map(|(_, pieces)| pieces.len().saturating_sub(1))
            .sum();
        let total_added = extra_rewrite.saturating_add(plan.tool.len());

        let mut set_ids: Vec<TriangleId> = Vec::new();
        let mut set_tris: Vec<[VertexId; 3]> = Vec::new();
        let mut add_tris: Vec<[VertexId; 3]> = Vec::new();
        reserve(&mut set_ids, plan.rewrite.len())?;
        reserve(&mut set_tris, plan.rewrite.len())?;
        reserve(&mut add_tris, total_added)?;

        (unused, set_ids, set_tris, add_tris, total_added)
    };

    let uvs_dropped = tx.mesh().uvs().is_some();
    if uvs_dropped {
        tx.drop_channel(UV_CHANNEL)?;
    }

    let fresh = if plan.fresh.is_empty() {
        Vec::new()
    } else {
        tx.add_vertices(&plan.fresh)?
    };
    let at = |r: PRef| match r {
        PRef::Live(v) => v,
        PRef::Fresh(i) => fresh[i as usize],
    };

    for (t, pieces) in &plan.rewrite {
        set_ids.push(*t);
        set_tris.push(pieces[0].map(at));
        for piece in &pieces[1..] {
            add_tris.push(piece.map(at));
        }
    }
    for tri in &plan.tool {
        add_tris.push(tri.map(at));
    }

    if !set_ids.is_empty() {
        tx.set_triangles(&set_ids, &set_tris)?;
    }
    if !plan.drop_a.is_empty() {
        tx.remove_triangles(&plan.drop_a)?;
    }
    if !add_tris.is_empty() {
        tx.add_triangles(&add_tris)?;
    }
    if !unused.is_empty() {
        tx.remove_vertices(&unused)?;
    }

    Ok(BooleanReport {
        kept_triangles: tx.mesh().triangle_count(),
        triangles_added: total_added,
        uvs_dropped,
    })
}

/// Rejects inputs no boolean can consume.
fn check_input(mesh: &Mesh, bvh: &Bvh) -> Result<()> {
    let options = crate::AnalyzeOptions {
        weld_tolerance: 0.0,
        self_intersection: Some(bvh),
        max_listed: 1,
    };
    let r = analyze(mesh, &options)?;
    if !r.loops.is_empty() {
        return Err(OpsError::InvalidInput("boolean needs closed meshes"));
    }
    if r.non_manifold_edges > 0 || !r.pinch_vertices.is_empty() {
        return Err(OpsError::InvalidInput("boolean needs manifold meshes"));
    }
    if r.degenerate > 0 {
        return Err(OpsError::InvalidInput(
            "boolean input has degenerate triangles",
        ));
    }
    if r.duplicate > 0 {
        return Err(OpsError::InvalidInput(
            "boolean input has duplicate triangles",
        ));
    }
    if r.orientation_conflicts > 0 {
        return Err(OpsError::InvalidInput(
            "boolean input is inconsistently wound",
        ));
    }
    if r.self_intersects == Some(true) {
        return Err(OpsError::InvalidInput("boolean input self-intersects"));
    }
    if mesh.triangle_count() > 0 && winding::signed_volume6(mesh) <= 0.0 {
        return Err(OpsError::InvalidInput("boolean input is wound inward"));
    }
    Ok(())
}

/// Mesh B mapped into A's frame, slot for slot.
fn map_tool(b: &Mesh, xf: &Rigid) -> Result<Mesh> {
    let mut positions: Vec<f32> = Vec::new();
    reserve(&mut positions, b.vertex_slots() * 3)?;
    for p in b.positions() {
        let q = xf.apply(*p);
        let f = [q[0] as f32, q[1] as f32, q[2] as f32];
        if !f.iter().all(|c| c.is_finite()) {
            return Err(OpsError::InvalidInput("boolean tool leaves f32 range"));
        }
        positions.extend_from_slice(&f);
    }
    let mut indices: Vec<u32> = Vec::new();
    reserve(&mut indices, b.triangle_slots() * 3)?;
    for t in b.triangles() {
        indices.extend_from_slice(t);
    }
    let mut mapped = Mesh::from_buffers(&positions, &indices)?;
    // `from_buffers` wakes every slot; put the dead ones back to sleep so
    // queries see the same triangles as `b`.
    let mut dead_tris: Vec<TriangleId> = Vec::new();
    for t in 0..b.triangle_slots() as TriangleId {
        if b.triangle(t).is_none() {
            push(&mut dead_tris, t)?;
        }
    }
    let mut dead_verts: Vec<VertexId> = Vec::new();
    for v in 0..b.vertex_slots() as VertexId {
        if !b.is_vertex_alive(v) {
            push(&mut dead_verts, v)?;
        }
    }
    if !dead_tris.is_empty() || !dead_verts.is_empty() {
        let mut tx = mapped.begin_edit();
        if !dead_tris.is_empty() {
            tx.remove_triangles(&dead_tris)?;
        }
        if !dead_verts.is_empty() {
            tx.remove_vertices(&dead_verts)?;
        }
        let _ = tx.commit()?;
    }
    Ok(mapped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{box_mesh, edit, grid, uv_sphere, with_tetra};
    use fast_mesh_query::Rigid;

    /// A closed unit cube, outward wound.
    fn cube() -> Mesh {
        box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    }

    /// Six times the signed volume: positive for outward-wound meshes.
    fn volume6(mesh: &Mesh) -> f64 {
        let mut v = 0.0;
        for t in 0..mesh.triangle_slots() as TriangleId {
            let Some(tri) = mesh.triangle(t) else {
                continue;
            };
            let [a, b, c] = tri.map(|x| mesh.position(x).unwrap().map(f64::from));
            v += a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]);
        }
        v
    }

    fn closed(mesh: &Mesh) -> bool {
        crate::boundary_loops(mesh).unwrap().is_empty()
    }

    fn run(a: &mut Mesh, b: &Mesh, op: BooleanOp) -> BooleanReport {
        let bvh_a = Bvh::build(a, &BuildOptions::default()).unwrap();
        let bvh_b = Bvh::build(b, &BuildOptions::default()).unwrap();
        let (report, _) = edit(a, |tx| {
            let plan = boolean_plan(tx.mesh(), &bvh_a, b, &bvh_b, &Rigid::IDENTITY, op)?;
            apply_boolean(tx, &plan)
        });
        report
    }

    #[test]
    fn union_overlapping_cubes() {
        // B shifted half a unit in x: overlap 0.5 x 1 x 1.
        let mut a = cube();
        let b = box_mesh([0.5, 0.0, 0.0], [1.5, 1.0, 1.0]);
        let report = run(&mut a, &b, BooleanOp::Union);
        assert!(closed(&a), "union must be closed");
        assert!(
            (volume6(&a) - 6.0 * 1.5).abs() < 1e-3,
            "vol {}",
            volume6(&a)
        );
        assert_eq!(report.kept_triangles, a.triangle_count());
    }

    #[test]
    fn subtract_overlapping_cubes() {
        let mut a = cube();
        let b = box_mesh([0.5, 0.0, 0.0], [1.5, 1.0, 1.0]);
        run(&mut a, &b, BooleanOp::Subtract);
        assert!(closed(&a), "subtract must be closed");
        assert!(
            (volume6(&a) - 6.0 * 0.5).abs() < 1e-3,
            "vol {}",
            volume6(&a)
        );
    }

    #[test]
    fn intersect_overlapping_cubes() {
        let mut a = cube();
        let b = box_mesh([0.5, 0.0, 0.0], [1.5, 1.0, 1.0]);
        run(&mut a, &b, BooleanOp::Intersect);
        assert!(closed(&a), "intersect must be closed");
        assert!(
            (volume6(&a) - 6.0 * 0.5).abs() < 1e-3,
            "vol {}",
            volume6(&a)
        );
    }

    #[test]
    fn asymmetric_overlap_needs_interior_points() {
        // B offset in every axis: B's edges pierce A's faces away from
        // their diagonals, so triangles gain interior segment endpoints.
        // Overlap 0.5 x 0.7 x 0.3.
        let mut a = cube();
        let b = box_mesh([0.5, 0.3, 0.7], [1.5, 1.3, 1.7]);
        run(&mut a, &b, BooleanOp::Subtract);
        assert!(closed(&a), "subtract must be closed");
        assert!(
            (volume6(&a) - 6.0 * 0.895).abs() < 1e-3,
            "vol {}",
            volume6(&a)
        );
        let mut u = cube();
        run(&mut u, &b, BooleanOp::Union);
        assert!(closed(&u), "union must be closed");
        assert!(
            (volume6(&u) - 6.0 * 1.895).abs() < 1e-3,
            "vol {}",
            volume6(&u)
        );
    }

    #[test]
    fn stacked_cubes_share_a_face() {
        // Coplanar area overlap: the shared face drops out of the union.
        let mut a = cube();
        let b = box_mesh([0.0, 0.0, 1.0], [1.0, 1.0, 2.0]);
        run(&mut a, &b, BooleanOp::Union);
        assert!(closed(&a), "union must be closed");
        assert!((volume6(&a) - 12.0).abs() < 1e-3, "vol {}", volume6(&a));
        assert_eq!(a.triangle_count(), 20);
    }

    #[test]
    fn nested_spheres_classify_whole() {
        let outer = uv_sphere(8, 12, 1.0);
        let inner = uv_sphere(6, 10, 0.5);
        let (vo, vi) = (volume6(&outer), volume6(&inner));
        let mut u = uv_sphere(8, 12, 1.0);
        run(&mut u, &inner, BooleanOp::Union);
        assert!(closed(&u));
        assert!((volume6(&u) - vo).abs() < 1e-3 * vo.abs());
        let mut i = uv_sphere(8, 12, 1.0);
        run(&mut i, &inner, BooleanOp::Intersect);
        assert!(closed(&i));
        assert!((volume6(&i) - vi).abs() < 1e-3 * vi.abs());
        let mut s = uv_sphere(8, 12, 1.0);
        run(&mut s, &inner, BooleanOp::Subtract);
        assert!(closed(&s));
        assert!((volume6(&s) - (vo - vi)).abs() < 1e-3 * vo.abs());
    }

    #[test]
    fn overlapping_spheres() {
        // Two unit spheres a unit apart: every triangle near the intersection
        // is split, on both sides.
        let sphere_at = |x: f32| {
            let (mut positions, triangles) = crate::test_util::parts(&uv_sphere(8, 12, 1.0));
            for p in positions.as_chunks_mut::<3>().0 {
                p[0] += x;
            }
            let indices: Vec<u32> = triangles.iter().flatten().copied().collect();
            Mesh::from_buffers(&positions, &indices).unwrap()
        };
        // Analytic lens volume for r = 1, d = 1: 5π/12 ≈ 1.3090 (×6 below).
        let lens = 6.0 * 5.0 * std::f64::consts::PI / 12.0;
        let single = volume6(&sphere_at(0.0));
        let mut u = sphere_at(0.0);
        run(&mut u, &sphere_at(1.0), BooleanOp::Union);
        assert!(closed(&u), "union must be closed");
        assert!(
            (volume6(&u) - (2.0 * single - lens)).abs() < 0.05 * single.abs(),
            "union vol {}",
            volume6(&u)
        );
        assert!((volume6(&u) - (2.0 * single - lens)).abs() < 0.05 * single.abs());
        let mut i = sphere_at(0.0);
        run(&mut i, &sphere_at(1.0), BooleanOp::Intersect);
        assert!(closed(&i), "intersect must be closed");
        assert!((volume6(&i) - lens).abs() < 0.05 * single.abs());
        let mut s = sphere_at(0.0);
        run(&mut s, &sphere_at(1.0), BooleanOp::Subtract);
        assert!(closed(&s), "subtract must be closed");
        assert!((volume6(&s) - (single - lens)).abs() < 0.05 * single.abs());
    }

    #[test]
    fn disjoint_cubes() {
        let b = box_mesh([5.0, 0.0, 0.0], [6.0, 1.0, 1.0]);
        let mut u = cube();
        run(&mut u, &b, BooleanOp::Union);
        assert!(closed(&u));
        assert_eq!(u.triangle_count(), 24);
        assert!((volume6(&u) - 12.0).abs() < 1e-9);
        let mut i = cube();
        let report = run(&mut i, &b, BooleanOp::Intersect);
        assert_eq!(i.triangle_count(), 0);
        assert_eq!(report.kept_triangles, 0);
        let mut s = cube();
        run(&mut s, &b, BooleanOp::Subtract);
        assert_eq!(s.triangle_count(), 12);
        assert!((volume6(&s) - 6.0).abs() < 1e-9);
    }

    #[test]
    fn mesh_with_itself_is_rejected() {
        let mut a = cube();
        let bvh_a = Bvh::build(&a, &BuildOptions::default()).unwrap();
        // The scene short-circuits self-booleans; the op itself refuses: a
        // mesh against itself is all interface, which no rule resolves.
        assert_eq!(
            boolean_plan(&a, &bvh_a, &a, &bvh_a, &Rigid::IDENTITY, BooleanOp::Union).unwrap_err(),
            OpsError::InvalidInput("cannot combine a mesh with itself")
        );
        let _ = &mut a;
    }

    #[test]
    fn rejects_bad_inputs() {
        // Open.
        let open = grid(2);
        let a = cube();
        let bvh_a = Bvh::build(&a, &BuildOptions::default()).unwrap();
        let bvh_o = Bvh::build(&open, &BuildOptions::default()).unwrap();
        assert_eq!(
            boolean_plan(
                &a,
                &bvh_a,
                &open,
                &bvh_o,
                &Rigid::IDENTITY,
                BooleanOp::Union
            )
            .unwrap_err(),
            OpsError::InvalidInput("boolean needs closed meshes")
        );
        // Non-manifold edge: a tetrahedron on a cube edge gives edge
        // (0, 1) four triangles, without opening the mesh.
        let spike = with_tetra(&a, 0, 1, [[2.0, 0.0, 0.0], [2.0, 1.0, 0.0]]);
        let bvh_s = Bvh::build(&spike, &BuildOptions::default()).unwrap();
        assert_eq!(
            boolean_plan(
                &spike,
                &bvh_s,
                &a,
                &bvh_a,
                &Rigid::IDENTITY,
                BooleanOp::Union
            )
            .unwrap_err(),
            OpsError::InvalidInput("boolean needs manifold meshes")
        );
        // A degenerate triangle with its own vertices trips neither the
        // boundary nor the manifold check, but the degenerate one.
        let mut needle = cube();
        let n = edit(&mut needle, |tx| {
            tx.add_vertices(&[[9.0, 9.0, 9.0], [9.0, 9.0, 9.0], [9.0, 9.0, 8.0]])?;
            Ok(())
        });
        let _ = n;
        let base = needle.vertex_slots() as u32 - 3;
        edit(&mut needle, |tx| {
            tx.add_triangles(&[[base, base, base + 1]])?;
            Ok(())
        });
        let bvh_n = Bvh::build(&needle, &BuildOptions::default()).unwrap();
        assert_eq!(
            boolean_plan(
                &needle,
                &bvh_n,
                &a,
                &bvh_a,
                &Rigid::IDENTITY,
                BooleanOp::Union
            )
            .unwrap_err(),
            OpsError::InvalidInput("boolean input has degenerate triangles")
        );
        // Two identical triangles with their own vertices: closed and
        // manifold, but duplicates.
        let dup = Mesh::from_buffers(
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            &[0, 1, 2, 0, 1, 2],
        )
        .unwrap();
        let bvh_d = Bvh::build(&dup, &BuildOptions::default()).unwrap();
        assert_eq!(
            boolean_plan(&dup, &bvh_d, &a, &bvh_a, &Rigid::IDENTITY, BooleanOp::Union).unwrap_err(),
            OpsError::InvalidInput("boolean input has duplicate triangles")
        );
    }

    #[test]
    fn rejects_inward_wound_cube() {
        let a = cube();
        let all: Vec<TriangleId> = (0..a.triangle_count() as TriangleId).collect();
        let inv = crate::test_util::flip(&a, &all);
        let b = box_mesh([0.5, 0.0, 0.0], [1.5, 1.0, 1.0]);
        let bvh_a = Bvh::build(&a, &BuildOptions::default()).unwrap();
        let bvh_inv = Bvh::build(&inv, &BuildOptions::default()).unwrap();
        let bvh_b = Bvh::build(&b, &BuildOptions::default()).unwrap();

        assert_eq!(
            boolean_plan(
                &inv,
                &bvh_inv,
                &b,
                &bvh_b,
                &Rigid::IDENTITY,
                BooleanOp::Union
            )
            .unwrap_err(),
            OpsError::InvalidInput("boolean input is wound inward")
        );
        assert_eq!(
            boolean_plan(
                &a,
                &bvh_a,
                &inv,
                &bvh_inv,
                &Rigid::IDENTITY,
                BooleanOp::Union
            )
            .unwrap_err(),
            OpsError::InvalidInput("boolean input is wound inward")
        );
    }

    #[test]
    fn stacked_cubes_intersect_is_empty_and_subtract_keeps_a() {
        let b = box_mesh([0.0, 0.0, 1.0], [1.0, 1.0, 2.0]);

        let mut inter = cube();
        run(&mut inter, &b, BooleanOp::Intersect);
        assert_eq!(
            inter.triangle_count(),
            0,
            "face-touching intersect must be empty, not an open sheet"
        );
        let m_inter = crate::measure(&inter).unwrap();
        assert!(m_inter.closed);
        assert_eq!(m_inter.orientation_conflicts, 0);

        let mut sub_mesh = cube();
        run(&mut sub_mesh, &b, BooleanOp::Subtract);
        let m_sub = crate::measure(&sub_mesh).unwrap();
        assert!(m_sub.closed, "subtract must be closed: {m_sub:?}");
        assert_eq!(m_sub.orientation_conflicts, 0);
        assert!(
            (volume6(&sub_mesh) - 6.0).abs() < 1e-4,
            "vol {}",
            volume6(&sub_mesh)
        );
    }

    #[test]
    fn edge_and_vertex_touching_cubes_are_strictly_manifold() {
        let edge_touch = box_mesh([1.0, 1.0, 0.0], [2.0, 2.0, 1.0]);
        let vert_touch = box_mesh([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]);

        for b in [&edge_touch, &vert_touch] {
            for op in [BooleanOp::Union, BooleanOp::Subtract, BooleanOp::Intersect] {
                let mut a = cube();
                run(&mut a, b, op);
                let m = crate::measure(&a).unwrap();
                if a.triangle_count() > 0 {
                    assert!(m.closed, "{op:?}: expected closed, got {m:?}");
                }
                assert_eq!(m.non_manifold_edges, 0, "{op:?}: non-manifold edges");
                assert_eq!(m.orientation_conflicts, 0, "{op:?}: orientation conflicts");
                let an = crate::analyze(&a, &crate::AnalyzeOptions::default()).unwrap();
                assert!(
                    an.pinch_vertices.is_empty(),
                    "{op:?}: expected zero pinch vertices, got {:?}",
                    an.pinch_vertices
                );
            }
        }
    }

    #[test]
    fn hollow_shell_boolean_succeeds() {
        let make_hollow = || {
            let mut hollow = uv_sphere(12, 16, 1.0);
            let _ = edit(&mut hollow, |tx| {
                crate::shell(tx, &crate::ShellOptions { thickness: 0.3 })
            });
            hollow
        };
        let cutter = box_mesh([0.0, -0.6, -0.6], [1.5, 0.6, 0.6]);

        for op in [BooleanOp::Intersect, BooleanOp::Subtract] {
            let mut a = make_hollow();
            run(&mut a, &cutter, op);
            assert!(a.triangle_count() > 0);
            let m = crate::measure(&a).unwrap();
            assert!(m.closed, "{op:?}: {m:?}");
            assert_eq!(m.non_manifold_edges, 0);
            assert_eq!(m.orientation_conflicts, 0);
            assert!(
                m.signed_volume > 0.0,
                "{op:?}: signed_volume {}",
                m.signed_volume
            );
        }
    }

    #[test]
    fn rotated_tool_boolean_matches_analytic_volume() {
        let b = cube();
        let bvh_b = Bvh::build(&b, &BuildOptions::default()).unwrap();

        let len = (1.0_f64 + 4.0 + 9.0).sqrt();
        let (ux, uy, uz) = (1.0 / len, 2.0 / len, 3.0 / len);
        let (s, c) = 37.0_f64.to_radians().sin_cos();
        let t = 1.0 - c;
        let xf = Rigid {
            rotation: [
                [t * ux * ux + c, t * ux * uy - s * uz, t * ux * uz + s * uy],
                [t * uy * ux + s * uz, t * uy * uy + c, t * uy * uz - s * ux],
                [t * uz * ux - s * uy, t * uz * uy + s * ux, t * uz * uz + c],
            ],
            translation: [0.4, 0.3, 0.2],
        };
        assert_eq!(xf.validate(), Ok(()));

        let run_xf = |op: BooleanOp| -> f64 {
            let mut a = cube();
            let bvh_a = Bvh::build(&a, &BuildOptions::default()).unwrap();
            let (_report, _) = edit(&mut a, |tx| {
                let plan = boolean_plan(tx.mesh(), &bvh_a, &b, &bvh_b, &xf, op)?;
                apply_boolean(tx, &plan)
            });
            let m = crate::measure(&a).unwrap();
            assert!(m.closed, "{op:?}: {m:?}");
            assert_eq!(m.non_manifold_edges, 0);
            assert_eq!(m.orientation_conflicts, 0);
            m.volume.unwrap()
        };

        let v_union = run_xf(BooleanOp::Union);
        let v_sub = run_xf(BooleanOp::Subtract);
        let v_inter = run_xf(BooleanOp::Intersect);
        assert!(
            v_inter > 0.05,
            "expected non-trivial overlap, got {v_inter}"
        );
        assert!(
            (v_union + v_inter - 2.0).abs() < 1e-4,
            "V(A U B)={v_union}, V(A n B)={v_inter}"
        );
        assert!(
            (v_sub + v_inter - 1.0).abs() < 1e-4,
            "V(A \\ B)={v_sub}, V(A n B)={v_inter}"
        );
    }

    #[test]
    fn boolean_oom_sweep_is_atomic() {
        let a = uv_sphere(6, 10, 1.0);
        let b = uv_sphere(6, 10, 1.0);
        let bvh_a = Bvh::build(&a, &BuildOptions::default()).unwrap();
        let bvh_b = Bvh::build(&b, &BuildOptions::default()).unwrap();
        let xf = Rigid {
            translation: [0.45, 0.2, 0.15],
            ..Rigid::IDENTITY
        };
        let steps = crate::test_util::sweep(a, |tx| {
            let plan = boolean_plan(tx.mesh(), &bvh_a, &b, &bvh_b, &xf, BooleanOp::Union)?;
            apply_boolean(tx, &plan)?;
            Ok(())
        });
        assert!(steps > 0);
    }

    #[test]
    fn flush_coplanar_boxes_classify_cleanly() {
        // Two boxes sharing flush coplanar faces on z = 0 and z = 1:
        // A = [0, 2] x [0, 2] x [0, 1], B = [0.5, 1.5] x [0.5, 1.5] x [0, 1].
        let make_a = || box_mesh([0.0, 0.0, 0.0], [2.0, 2.0, 1.0]);
        let b = box_mesh([0.5, 0.5, 0.0], [1.5, 1.5, 1.0]);

        for (op, want_vol) in [
            (BooleanOp::Union, 4.0),
            (BooleanOp::Subtract, 3.0),
            (BooleanOp::Intersect, 1.0),
        ] {
            let mut a = make_a();
            run(&mut a, &b, op);
            let m = crate::measure(&a).unwrap();
            assert!(m.closed, "{op:?}: {m:?}");
            assert_eq!(m.non_manifold_edges, 0);
            assert_eq!(m.orientation_conflicts, 0);
            let an = crate::analyze(&a, &crate::AnalyzeOptions::default()).unwrap();
            assert!(
                an.pinch_vertices.is_empty(),
                "{op:?}: {:?}",
                an.pinch_vertices
            );
            assert!(
                (m.volume.unwrap() - want_vol).abs() < 1e-4,
                "{op:?}: got {:?}, want {want_vol}",
                m.volume
            );
        }
    }

    #[test]
    fn kissing_blind_cavities_are_strictly_manifold() {
        // Subtract two edge-kissing inner cubes sequentially from a larger outer block:
        // (Big \ C1) then \ C2, exercising void-wedge dihedral edge de-pinching.
        let mut big = box_mesh([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]);
        let c1 = box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let c2 = box_mesh([-1.0, -1.0, 0.0], [0.0, 0.0, 1.0]);

        run(&mut big, &c1, BooleanOp::Subtract);
        run(&mut big, &c2, BooleanOp::Subtract);

        let m = crate::measure(&big).unwrap();
        assert!(m.closed, "expected closed: {m:?}");
        assert_eq!(m.non_manifold_edges, 0);
        assert_eq!(m.orientation_conflicts, 0);
        let an = crate::analyze(&big, &crate::AnalyzeOptions::default()).unwrap();
        assert!(
            an.pinch_vertices.is_empty(),
            "expected zero pinch vertices, got {:?}",
            an.pinch_vertices
        );
        assert!(
            (m.volume.unwrap() - 62.0).abs() < 1e-4,
            "vol {:?}",
            m.volume
        );
    }

    #[test]
    fn production_boolean_uses_no_untracked_allocs_or_hash_collections() {
        for (name, src) in [
            ("boolean.rs", include_str!("boolean.rs")),
            ("classify.rs", include_str!("boolean/classify.rs")),
        ] {
            let prod = src.split("#[cfg(test)]").next().unwrap();
            for forbidden in [
                ".sort(",
                ".sort_by(",
                ".sort_by_key(",
                ".collect",
                "HashMap",
                "HashSet",
            ] {
                assert!(
                    !prod.contains(forbidden),
                    "{name} production code must not contain `{forbidden}`"
                );
            }
        }
    }
}
