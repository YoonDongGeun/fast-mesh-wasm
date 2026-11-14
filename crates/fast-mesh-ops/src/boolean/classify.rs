//! Patch flood-fill, generalized winding / coplanar classification, opposite
//! duplicate cancellation, 2-manifold de-pinching, and [`BooleanPlan`] assembly.

use fast_mesh_bvh::{Aabb, Bvh, NodeKind};
use fast_mesh_store::{Mesh, StoreError, TriangleId, VertexId};
use robust::{Coord, orient2d};

use super::arrangement::{ArrOrigin, ArrVertexId, Arrangement, orient3d_exact};
use super::cdt::triangulate_constrained;
use super::perturbation::{PerturbKey, orient3d_sos, ray_winding_sos};
use super::winding::WindingTree;
use super::{BooleanOp, BooleanPlan, PRef};
use crate::topology::{EdgeMap, UnionFind, components};
use crate::{OpsError, Result, filled, push, reserve};

type D3 = [f64; 3];
type D2 = [f64; 2];

fn oom() -> OpsError {
    OpsError::Store(StoreError::OutOfMemory)
}

fn normalize_zero(p: [f32; 3]) -> [f32; 3] {
    p.map(|x| if x == 0.0 { 0.0 } else { x })
}

fn pos_bits(p: [f32; 3]) -> [u32; 3] {
    normalize_zero(p).map(f32::to_bits)
}

fn sub3(a: D3, b: D3) -> D3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot3(a: D3, b: D3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: D3, b: D3) -> D3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn tri_normal(p0: D3, p1: D3, p2: D3) -> D3 {
    cross3(sub3(p1, p0), sub3(p2, p0))
}

fn tri_normal_len2(p0: D3, p1: D3, p2: D3) -> f64 {
    let n = tri_normal(p0, p1, p2);
    dot3(n, n)
}

fn tri_centroid(p0: D3, p1: D3, p2: D3) -> D3 {
    [
        (p0[0] + p1[0] + p2[0]) / 3.0,
        (p0[1] + p1[1] + p2[1]) / 3.0,
        (p0[2] + p1[2] + p2[2]) / 3.0,
    ]
}

fn mesh_tri_f64(mesh: &Mesh, t: TriangleId) -> [D3; 3] {
    let tri = mesh.triangle(t).expect("live triangle");
    tri.map(|v| mesh.position(v).expect("live vertex").map(f64::from))
}

fn canon_edge(u: ArrVertexId, v: ArrVertexId) -> (ArrVertexId, ArrVertexId) {
    if u < v { (u, v) } else { (v, u) }
}

fn keep_non_coplanar(op: BooleanOp, is_a: bool, inside: bool) -> bool {
    match op {
        BooleanOp::Union => !inside,
        BooleanOp::Subtract => {
            if is_a {
                !inside
            } else {
                inside
            }
        }
        BooleanOp::Intersect => inside,
    }
}

/// Builds a verified, closed, 2-manifold [`BooleanPlan`] from `arr`.
pub(super) fn classify_and_build_plan(
    a: &Mesh,
    bvh_a: &Bvh,
    tool: &Mesh,
    tool_bvh: &Bvh,
    arr: Arrangement,
    op: BooleanOp,
) -> Result<BooleanPlan> {
    if a.triangle_count() == 0 || tool.triangle_count() == 0 {
        return plan_with_empty_side(a, tool, op);
    }

    // Disjoint, nested, or isolated single-corner touch fast path: no triangle
    // on either side has an edge split, interior vertex, or constraint segment.
    if arr.tri_a.is_empty() && arr.tri_b.is_empty() && arr.seam_edges.is_empty() {
        return classify_disjoint_components(a, bvh_a, tool, tool_bvh, op);
    }

    classify_general(a, bvh_a, tool, tool_bvh, arr, op)
}

/// Handles cases where `a` or `tool` has zero alive triangles.
fn plan_with_empty_side(a: &Mesh, tool: &Mesh, op: BooleanOp) -> Result<BooleanPlan> {
    let mut plan = BooleanPlan::default();
    if a.triangle_count() == 0 {
        if op == BooleanOp::Union && tool.triangle_count() > 0 {
            append_whole_tool_components(tool, false, None, &mut plan)?;
        }
        return Ok(plan);
    }
    // `tool` is empty and `a` is non-empty.
    if op == BooleanOp::Intersect {
        reserve(&mut plan.drop_a, a.triangle_count())?;
        for t in 0..a.triangle_slots() as TriangleId {
            if a.is_triangle_alive(t) {
                plan.drop_a.push(t);
            }
        }
    }
    Ok(plan)
}

/// Fast path when `a` and `tool` have no cut triangles or seam segments
/// (disjoint, nested, or isolated corner-corner touches).
fn classify_disjoint_components(
    a: &Mesh,
    bvh_a: &Bvh,
    tool: &Mesh,
    tool_bvh: &Bvh,
    op: BooleanOp,
) -> Result<BooleanPlan> {
    let mut plan = BooleanPlan::default();

    let edges_a = EdgeMap::build(a)?;
    let comps_a = components(a, &edges_a)?;
    let tree_tool = WindingTree::build(tool, tool_bvh)?;

    for comp in &comps_a {
        let rep = best_mesh_tri(a, &comp.triangles);
        let [p0, p1, p2] = mesh_tri_f64(a, rep);
        let w = tree_tool.winding_number(tool, tool_bvh, tri_centroid(p0, p1, p2));
        if w < -0.5 {
            return Err(OpsError::InvalidInput("boolean input is wound inward"));
        }
        let inside = w > 0.5;
        if !keep_non_coplanar(op, true, inside) {
            reserve(&mut plan.drop_a, comp.triangles.len())?;
            plan.drop_a.extend_from_slice(&comp.triangles);
        }
    }
    plan.drop_a.sort_unstable();

    let edges_b = EdgeMap::build(tool)?;
    let comps_b = components(tool, &edges_b)?;
    let tree_a = WindingTree::build(a, bvh_a)?;
    let mut keep_b_tri = filled(tool.triangle_slots(), false)?;
    let mut any_b = false;

    for comp in &comps_b {
        let rep = best_mesh_tri(tool, &comp.triangles);
        let [p0, p1, p2] = mesh_tri_f64(tool, rep);
        let w = tree_a.winding_number(a, bvh_a, tri_centroid(p0, p1, p2));
        if w < -0.5 {
            return Err(OpsError::InvalidInput("boolean input is wound inward"));
        }
        let inside = w > 0.5;
        if keep_non_coplanar(op, false, inside) {
            any_b = true;
            for &t in &comp.triangles {
                keep_b_tri[t as usize] = true;
            }
        }
    }

    if any_b {
        append_whole_tool_components(
            tool,
            op == BooleanOp::Subtract,
            Some(&keep_b_tri),
            &mut plan,
        )?;
    }

    Ok(plan)
}

fn best_mesh_tri(mesh: &Mesh, tris: &[TriangleId]) -> TriangleId {
    let mut best_t = tris[0];
    let [p0, p1, p2] = mesh_tri_f64(mesh, best_t);
    let mut best_n2 = tri_normal_len2(p0, p1, p2);
    for &t in &tris[1..] {
        let [q0, q1, q2] = mesh_tri_f64(mesh, t);
        let n2 = tri_normal_len2(q0, q1, q2);
        if n2 > best_n2 {
            best_n2 = n2;
            best_t = t;
        }
    }
    best_t
}

fn append_whole_tool_components(
    tool: &Mesh,
    reverse: bool,
    filter: Option<&[bool]>,
    plan: &mut BooleanPlan,
) -> Result<()> {
    let mut v_map = filled(tool.vertex_slots(), u32::MAX)?;
    for t in 0..tool.triangle_slots() as TriangleId {
        if let Some(mask) = filter
            && !mask[t as usize]
        {
            continue;
        }
        let Some(tri) = tool.triangle(t) else {
            continue;
        };
        let mut refs = [PRef::Fresh(0); 3];
        for (k, &vb) in tri.iter().enumerate() {
            let mut idx = v_map[vb as usize];
            if idx == u32::MAX {
                idx = u32::try_from(plan.fresh.len()).map_err(|_| oom())?;
                let pos = normalize_zero(tool.position(vb).expect("live tool vertex"));
                push(&mut plan.fresh, pos)?;
                v_map[vb as usize] = idx;
            }
            refs[k] = PRef::Fresh(idx);
        }
        if reverse {
            refs.swap(1, 2);
        }
        push(&mut plan.tool, refs)?;
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TriOrigin {
    UntouchedA(TriangleId),
    CutA(TriangleId),
    Tool,
}

#[derive(Clone, Copy)]
struct SideTri {
    verts: [ArrVertexId; 3],
    cut_parent: Option<TriangleId>,
    origin: TriOrigin,
}

#[derive(Clone, Copy)]
struct SurvTri {
    verts: [ArrVertexId; 3],
    origin: TriOrigin,
}

/// Full arrangement + CDT + patch classification + de-pinching pipeline.
fn classify_general(
    a: &Mesh,
    bvh_a: &Bvh,
    tool: &Mesh,
    tool_bvh: &Bvh,
    mut arr: Arrangement,
    op: BooleanOp,
) -> Result<BooleanPlan> {
    // 1. Constrained 2D sub-triangulation of all cut triangles on Side A and Side B.
    let mut cut_a_subs: Vec<(TriangleId, [ArrVertexId; 3])> = Vec::new();
    reserve(&mut cut_a_subs, arr.tri_a.len().saturating_mul(3))?;
    for idx in 0..arr.tri_a.len() {
        let t = arr.tri_a[idx].0;
        let corners = arr.corners_a(a, t);
        let subs = triangulate_constrained(
            corners,
            &arr.tri_a[idx].1,
            &mut arr.positions,
            &mut arr.origins,
            &mut arr.seam_edges,
        )?;
        reserve(&mut cut_a_subs, subs.len())?;
        for s in subs {
            cut_a_subs.push((t, s));
        }
    }

    let mut cut_b_subs: Vec<(TriangleId, [ArrVertexId; 3])> = Vec::new();
    reserve(&mut cut_b_subs, arr.tri_b.len().saturating_mul(3))?;
    for idx in 0..arr.tri_b.len() {
        let t = arr.tri_b[idx].0;
        let corners = arr.corners_b(tool, t);
        let subs = triangulate_constrained(
            corners,
            &arr.tri_b[idx].1,
            &mut arr.positions,
            &mut arr.origins,
            &mut arr.seam_edges,
        )?;
        reserve(&mut cut_b_subs, subs.len())?;
        for s in subs {
            cut_b_subs.push((t, s));
        }
    }

    // 2. Unify all arrangement + CDT crossing vertices by exact [f32; 3] bit pattern.
    let UnifiedVerts {
        mut positions,
        mut origins,
        remap,
        bit_table,
    } = unify_arrangement_vertices(a, &arr.positions, &arr.origins)?;

    for entry in &mut cut_a_subs {
        entry.1 = entry.1.map(|v| remap[v as usize]);
    }
    cut_a_subs.retain(|&(_, [x, y, z])| x != y && y != z && x != z);

    for entry in &mut cut_b_subs {
        entry.1 = entry.1.map(|v| remap[v as usize]);
    }
    cut_b_subs.retain(|&(_, [x, y, z])| x != y && y != z && x != z);

    let mut seam_edges: Vec<(ArrVertexId, ArrVertexId)> = Vec::new();
    reserve(&mut seam_edges, arr.seam_edges.len())?;
    for &(u, v) in &arr.seam_edges {
        let ru = remap[u as usize];
        let rv = remap[v as usize];
        if ru != rv {
            seam_edges.push(canon_edge(ru, rv));
        }
    }
    seam_edges.sort_unstable();
    seam_edges.dedup();

    // 3. Build unified candidate triangle lists for Side A and Side B.
    let side_a = build_side_a_triangles(
        a,
        &arr,
        &remap,
        &bit_table,
        &cut_a_subs,
        &mut positions,
        &mut origins,
    )?;
    let side_b = build_side_b_triangles(
        tool,
        &arr,
        &remap,
        &bit_table,
        &cut_b_subs,
        &mut positions,
        &mut origins,
    )?;

    // 4. Patch flood-fill & coplanar / WindingTree classification on both sides.
    let mut surviving: Vec<SurvTri> = Vec::new();
    classify_side_patches(
        &side_a,
        a,
        tool,
        tool_bvh,
        &positions,
        &seam_edges,
        true,
        op,
        &mut surviving,
    )?;
    classify_side_patches(
        &side_b,
        tool,
        a,
        bvh_a,
        &positions,
        &seam_edges,
        false,
        op,
        &mut surviving,
    )?;

    // 5. Cancel exact opposite duplicate triangles (e.g. opposite interior interfaces).
    cancel_opposite_duplicates(&mut surviving)?;

    // 6. 2-manifold de-pinching (cyclic dihedral edge separation + vertex fan splitting).
    if !surviving.is_empty() {
        depinch_manifold(&mut surviving, &mut positions, &mut origins)?;
        verify_manifold(&surviving, positions.len())?;
    }

    // 7. Assemble BooleanPlan for mesh A.
    assemble_plan(a, &surviving, &positions, &origins)
}

type BitTable = Vec<([u32; 3], ArrVertexId)>;

struct UnifiedVerts {
    positions: Vec<[f32; 3]>,
    origins: Vec<ArrOrigin>,
    remap: Vec<ArrVertexId>,
    bit_table: BitTable,
}

/// Deduplicates `positions` by normalized `[f32; 3]` bit pattern (preferring `LiveA`)
/// so any fresh crossing vertices added inside `cdt` unify across shared seams.
fn unify_arrangement_vertices(
    a: &Mesh,
    in_positions: &[[f32; 3]],
    in_origins: &[ArrOrigin],
) -> Result<UnifiedVerts> {
    let n = in_positions.len();
    let mut by_bits: Vec<([u32; 3], usize)> = Vec::new();
    reserve(&mut by_bits, n)?;
    for (idx, &p) in in_positions.iter().enumerate() {
        by_bits.push((pos_bits(p), idx));
    }
    by_bits.sort_unstable();

    // (first_idx, chosen_pos, chosen_orig, g_start, g_end)
    let mut groups: Vec<(usize, [f32; 3], ArrOrigin, usize, usize)> = Vec::new();
    let mut gs = 0usize;
    while gs < by_bits.len() {
        let bits = by_bits[gs].0;
        let mut ge = gs + 1;
        while ge < by_bits.len() && by_bits[ge].0 == bits {
            ge += 1;
        }
        let first_idx = by_bits[gs].1;
        let mut live_va: Option<VertexId> = None;
        for &(_, old_idx) in &by_bits[gs..ge] {
            if let ArrOrigin::LiveA(va) = in_origins[old_idx] {
                live_va = Some(live_va.map_or(va, |cur| cur.min(va)));
            }
        }
        let (chosen_pos, chosen_orig) = match live_va {
            Some(va) => (a.position(va).expect("live A vertex"), ArrOrigin::LiveA(va)),
            None => (normalize_zero(in_positions[first_idx]), ArrOrigin::Fresh),
        };
        push(&mut groups, (first_idx, chosen_pos, chosen_orig, gs, ge))?;
        gs = ge;
    }
    groups.sort_unstable_by_key(|&(first_idx, _, _, _, _)| first_idx);

    let num_uni = groups.len();
    let _ = ArrVertexId::try_from(num_uni).map_err(|_| oom())?;
    let mut positions = Vec::new();
    let mut origins = Vec::new();
    let mut remap = filled(n, 0)?;
    let mut bit_table: BitTable = Vec::new();
    reserve(&mut positions, num_uni)?;
    reserve(&mut origins, num_uni)?;
    reserve(&mut bit_table, num_uni)?;

    for (new_idx, &(_, pos, orig, g_start, g_end)) in groups.iter().enumerate() {
        let new_id = new_idx as ArrVertexId;
        positions.push(pos);
        origins.push(orig);
        bit_table.push((pos_bits(pos), new_id));
        for &(_, old_idx) in &by_bits[g_start..g_end] {
            remap[old_idx] = new_id;
        }
    }
    bit_table.sort_unstable_by_key(|&(bits, _)| bits);

    Ok(UnifiedVerts {
        positions,
        origins,
        remap,
        bit_table,
    })
}

fn build_side_a_triangles(
    a: &Mesh,
    arr: &Arrangement,
    remap: &[ArrVertexId],
    bit_table: &[([u32; 3], ArrVertexId)],
    cut_a_subs: &[(TriangleId, [ArrVertexId; 3])],
    positions: &mut Vec<[f32; 3]>,
    origins: &mut Vec<ArrOrigin>,
) -> Result<Vec<SideTri>> {
    let mut a_to_uni = filled(a.vertex_slots(), u32::MAX)?;
    for (idx, &orig) in origins.iter().enumerate() {
        if let ArrOrigin::LiveA(va) = orig {
            a_to_uni[va as usize] = idx as ArrVertexId;
        }
    }
    for &(t, _) in &arr.tri_a {
        let tri_v = a.triangle(t).expect("live cut A triangle");
        let c_arr = arr.corners_a(a, t);
        for k in 0..3 {
            a_to_uni[tri_v[k] as usize] = remap[c_arr[k] as usize];
        }
    }

    let mut is_cut = filled(a.triangle_slots(), false)?;
    for &(t, _) in &arr.tri_a {
        is_cut[t as usize] = true;
    }

    let untouched_count = a.triangle_count().saturating_sub(arr.tri_a.len());
    let mut out: Vec<SideTri> = Vec::new();
    reserve(&mut out, untouched_count.saturating_add(cut_a_subs.len()))?;

    for t in 0..a.triangle_slots() as TriangleId {
        if is_cut[t as usize] {
            continue;
        }
        let Some(tri) = a.triangle(t) else {
            continue;
        };
        let mut verts = [0 as ArrVertexId; 3];
        for (k, &va) in tri.iter().enumerate() {
            let mut u = a_to_uni[va as usize];
            if u == u32::MAX {
                let p = a.position(va).expect("live A vertex");
                let bits = pos_bits(p);
                if let Ok(bi) = bit_table.binary_search_by_key(&bits, |&(b, _)| b) {
                    u = bit_table[bi].1;
                } else {
                    u = ArrVertexId::try_from(positions.len()).map_err(|_| oom())?;
                    push(positions, p)?;
                    push(origins, ArrOrigin::LiveA(va))?;
                }
                a_to_uni[va as usize] = u;
            }
            verts[k] = u;
        }
        if verts[0] != verts[1] && verts[1] != verts[2] && verts[0] != verts[2] {
            out.push(SideTri {
                verts,
                cut_parent: None,
                origin: TriOrigin::UntouchedA(t),
            });
        }
    }

    for &(t, verts) in cut_a_subs {
        out.push(SideTri {
            verts,
            cut_parent: Some(t),
            origin: TriOrigin::CutA(t),
        });
    }

    Ok(out)
}

fn build_side_b_triangles(
    tool: &Mesh,
    arr: &Arrangement,
    remap: &[ArrVertexId],
    bit_table: &[([u32; 3], ArrVertexId)],
    cut_b_subs: &[(TriangleId, [ArrVertexId; 3])],
    positions: &mut Vec<[f32; 3]>,
    origins: &mut Vec<ArrOrigin>,
) -> Result<Vec<SideTri>> {
    let mut b_to_uni = filled(tool.vertex_slots(), u32::MAX)?;
    for &(t, _) in &arr.tri_b {
        let tri_v = tool.triangle(t).expect("live cut tool triangle");
        let c_arr = arr.corners_b(tool, t);
        for k in 0..3 {
            b_to_uni[tri_v[k] as usize] = remap[c_arr[k] as usize];
        }
    }

    let mut is_cut = filled(tool.triangle_slots(), false)?;
    for &(t, _) in &arr.tri_b {
        is_cut[t as usize] = true;
    }

    let untouched_count = tool.triangle_count().saturating_sub(arr.tri_b.len());
    let mut out: Vec<SideTri> = Vec::new();
    reserve(&mut out, untouched_count.saturating_add(cut_b_subs.len()))?;

    for t in 0..tool.triangle_slots() as TriangleId {
        if is_cut[t as usize] {
            continue;
        }
        let Some(tri) = tool.triangle(t) else {
            continue;
        };
        let mut verts = [0 as ArrVertexId; 3];
        for (k, &vb) in tri.iter().enumerate() {
            let mut u = b_to_uni[vb as usize];
            if u == u32::MAX {
                let p = normalize_zero(tool.position(vb).expect("live tool vertex"));
                let bits = pos_bits(p);
                if let Ok(bi) = bit_table.binary_search_by_key(&bits, |&(b, _)| b) {
                    u = bit_table[bi].1;
                } else {
                    u = ArrVertexId::try_from(positions.len()).map_err(|_| oom())?;
                    push(positions, p)?;
                    push(origins, ArrOrigin::Fresh)?;
                }
                b_to_uni[vb as usize] = u;
            }
            verts[k] = u;
        }
        if verts[0] != verts[1] && verts[1] != verts[2] && verts[0] != verts[2] {
            out.push(SideTri {
                verts,
                cut_parent: None,
                origin: TriOrigin::Tool,
            });
        }
    }

    for &(t, verts) in cut_b_subs {
        out.push(SideTri {
            verts,
            cut_parent: Some(t),
            origin: TriOrigin::Tool,
        });
    }

    Ok(out)
}

/// Flood-fills `seam_edges`-bounded patches on one side and classifies each
/// patch via exact parent coplanarity (§5.6.2) or `WindingTree` (§5.6.4).
#[allow(clippy::too_many_arguments)]
fn classify_side_patches(
    tris: &[SideTri],
    this_mesh: &Mesh,
    other_mesh: &Mesh,
    other_bvh: &Bvh,
    positions: &[[f32; 3]],
    seam_edges: &[(ArrVertexId, ArrVertexId)],
    is_a: bool,
    op: BooleanOp,
    surviving: &mut Vec<SurvTri>,
) -> Result<()> {
    if tris.is_empty() {
        return Ok(());
    }
    let n = tris.len();
    let _ = u32::try_from(n).map_err(|_| oom())?;

    // Connect triangles across shared undirected edges NOT in `seam_edges`.
    let mut non_seam_inc: Vec<((ArrVertexId, ArrVertexId), u32)> = Vec::new();
    reserve(&mut non_seam_inc, n.saturating_mul(3))?;
    for (idx, st) in tris.iter().enumerate() {
        let [a, b, c] = st.verts;
        for (u, v) in [(a, b), (b, c), (c, a)] {
            let e = canon_edge(u, v);
            if seam_edges.binary_search(&e).is_err() {
                non_seam_inc.push((e, idx as u32));
            }
        }
    }
    non_seam_inc.sort_unstable();

    let mut uf = UnionFind::new();
    uf.reset(n)?;
    let mut i = 0usize;
    while i < non_seam_inc.len() {
        let e = non_seam_inc[i].0;
        let first_t = non_seam_inc[i].1;
        let mut j = i + 1;
        while j < non_seam_inc.len() && non_seam_inc[j].0 == e {
            uf.union(first_t, non_seam_inc[j].1);
            j += 1;
        }
        i = j;
    }

    // Find each patch's best representative triangle (largest |normal|^2, tie-break lowest index).
    let mut root_to_patch = filled(n, u32::MAX)?;
    // (best_tri_idx, best_norm_len2)
    let mut patch_rep: Vec<(usize, f64)> = Vec::new();
    let mut tri_patch = filled(n, 0u32)?;

    for (idx, st) in tris.iter().enumerate() {
        let r = uf.find(idx as u32) as usize;
        let p0 = positions[st.verts[0] as usize].map(f64::from);
        let p1 = positions[st.verts[1] as usize].map(f64::from);
        let p2 = positions[st.verts[2] as usize].map(f64::from);
        let n2 = tri_normal_len2(p0, p1, p2);

        let pid = if root_to_patch[r] == u32::MAX {
            let new_pid = patch_rep.len() as u32;
            push(&mut patch_rep, (idx, n2))?;
            root_to_patch[r] = new_pid;
            new_pid
        } else {
            let existing = root_to_patch[r];
            let (_, cur_n2) = patch_rep[existing as usize];
            if n2 > cur_n2 {
                patch_rep[existing as usize] = (idx, n2);
            }
            existing
        };
        tri_patch[idx] = pid;
    }

    let mut patch_keep = filled(patch_rep.len(), false)?;
    let mut winding_tree: Option<WindingTree> = None;

    for (pid, &(rep_idx, _)) in patch_rep.iter().enumerate() {
        let rep = &tris[rep_idx];
        if let Some(t_parent) = rep.cut_parent
            && let Some(keep_cop) = classify_coplanar_rep(
                this_mesh, t_parent, rep.verts, positions, other_mesh, other_bvh, is_a, op,
            )
        {
            patch_keep[pid] = keep_cop;
            continue;
        }

        let tree = match &winding_tree {
            Some(wt) => wt,
            None => {
                winding_tree = Some(WindingTree::build(other_mesh, other_bvh)?);
                winding_tree.as_ref().expect("just initialized")
            }
        };
        let p0 = positions[rep.verts[0] as usize].map(f64::from);
        let p1 = positions[rep.verts[1] as usize].map(f64::from);
        let p2 = positions[rep.verts[2] as usize].map(f64::from);
        let cen = tri_centroid(p0, p1, p2);
        let w = tree.winding_number(other_mesh, other_bvh, cen);
        if w < -0.5 {
            return Err(OpsError::InvalidInput("boolean input is wound inward"));
        }
        let inside = if (0.25..=0.75).contains(&w) {
            let side_sign = if is_a { -1 } else { 1 };
            let rw = ray_winding_sos(other_mesh, other_bvh, cen, side_sign);
            if rw < 0 {
                return Err(OpsError::InvalidInput("boolean input is wound inward"));
            }
            rw > 0
        } else {
            w > 0.5
        };
        patch_keep[pid] = keep_non_coplanar(op, is_a, inside);
    }

    let reverse = !is_a && op == BooleanOp::Subtract;
    for (idx, st) in tris.iter().enumerate() {
        if patch_keep[tri_patch[idx] as usize] {
            let mut verts = st.verts;
            if reverse {
                verts.swap(1, 2);
            }
            push(
                surviving,
                SurvTri {
                    verts,
                    origin: st.origin,
                },
            )?;
        }
    }

    Ok(())
}

/// Checks whether cut sub-triangle `sub_verts` (inside `t_parent` of `this_mesh`)
/// lies on a coplanar overlapping face of `other_mesh`. Returns `Some(keep)` if
/// coplanar, or `None` if non-coplanar.
#[allow(clippy::too_many_arguments)]
fn classify_coplanar_rep(
    this_mesh: &Mesh,
    t_parent: TriangleId,
    sub_verts: [ArrVertexId; 3],
    positions: &[[f32; 3]],
    other_mesh: &Mesh,
    other_bvh: &Bvh,
    is_a: bool,
    op: BooleanOp,
) -> Option<bool> {
    let pa = mesh_tri_f64(this_mesh, t_parent);
    let s_f32 = sub_verts.map(|v| positions[v as usize]);
    let s_f64 = s_f32.map(|p| p.map(f64::from));

    let mut min_pt = s_f32[0];
    let mut max_pt = s_f32[0];
    let mut max_abs = 1.0f32;
    for p in s_f32 {
        for k in 0..3 {
            min_pt[k] = min_pt[k].min(p[k]);
            max_pt[k] = max_pt[k].max(p[k]);
            max_abs = max_abs.max(p[k].abs());
        }
    }
    // Generous f32 broadphase margin so 0.5-ULP rounded EdgeEdge vertices on
    // tilted coplanar faces never miss a candidate BVH leaf; exact orient3d_exact
    // filters parent planes below.
    let pad = 64.0 * f32::EPSILON * max_abs;
    let query = Aabb::new(min_pt.map(|c| c - pad), max_pt.map(|c| c + pad));

    let other_positions = other_mesh.positions();
    let other_triangles = other_mesh.triangles();

    let mut strict_match: Option<TriangleId> = None;
    let mut best_overlap_q: Option<(TriangleId, f64)> = None;
    let mut total_overlap_2d = 0.0f64;
    let mut ref_area_s = 0.0f64;
    let mut cen_on_q: Option<TriangleId> = None;
    let mut vert_covered = [false; 3];

    other_bvh.traverse(|_, node| {
        if strict_match.is_some() || !node.bounds.overlaps(&query) {
            return false;
        }
        if let NodeKind::Leaf { triangles: leaf } = node.kind {
            for &q in leaf {
                if !Aabb::of_triangle(other_positions, other_triangles[q as usize]).overlaps(&query)
                {
                    continue;
                }
                let qb = mesh_tri_f64(other_mesh, q);
                if !parents_exact_coplanar(&pa, &qb) {
                    continue;
                }
                let Some((u_ax, v_ax)) = dominant_ccw_axes(&qb) else {
                    continue;
                };
                let q2 = qb.map(|p| [p[u_ax], p[v_ax]]);
                let s2 = s_f64.map(|p| [p[u_ax], p[v_ax]]);
                let cen2 = [
                    (s2[0][0] + s2[1][0] + s2[2][0]) / 3.0,
                    (s2[0][1] + s2[1][1] + s2[2][1]) / 3.0,
                ];
                if point_strictly_in_ccw_tri(cen2, &q2) {
                    strict_match = Some(q);
                    return false;
                }

                let s_area = polygon_area_abs(&s2);
                if s_area > ref_area_s {
                    ref_area_s = s_area;
                }
                let ov = clip_tri_against_ccw_tri_area(&s2, &q2);
                if ov > 0.0 {
                    total_overlap_2d += ov;
                    if best_overlap_q.is_none_or(|(bq, bov)| ov > bov || (ov == bov && q < bq)) {
                        best_overlap_q = Some((q, ov));
                    }
                }
                if point_in_or_on_ccw_tri(cen2, &q2) && cen_on_q.is_none_or(|prev| q < prev) {
                    cen_on_q = Some(q);
                }
                for k in 0..3 {
                    if point_in_or_on_ccw_tri(s2[k], &q2) {
                        vert_covered[k] = true;
                    }
                }
            }
        }
        strict_match.is_none()
    });

    let matched_q = if let Some(q) = strict_match {
        Some(q)
    } else if ref_area_s > 0.0 && total_overlap_2d > 0.5 * ref_area_s {
        best_overlap_q.map(|(q, _)| q)
    } else if ref_area_s == 0.0 && vert_covered == [true, true, true] {
        cen_on_q
    } else {
        None
    }?;

    if !is_a {
        // Side B coplanar sub-triangles are always dropped; Side A owns the shared region.
        return Some(false);
    }

    let qb = mesh_tri_f64(other_mesh, matched_q);
    let na = tri_normal(pa[0], pa[1], pa[2]);
    let nb = tri_normal(qb[0], qb[1], qb[2]);
    let same_sense = dot3(na, nb) > 0.0;
    let keep = if same_sense {
        matches!(op, BooleanOp::Union | BooleanOp::Intersect)
    } else {
        matches!(op, BooleanOp::Subtract)
    };
    Some(keep)
}

fn parents_exact_coplanar(pa: &[D3; 3], qb: &[D3; 3]) -> bool {
    pa.iter()
        .all(|&p| orient3d_exact(qb[0], qb[1], qb[2], p) == 0)
        && qb
            .iter()
            .all(|&q| orient3d_exact(pa[0], pa[1], pa[2], q) == 0)
}

fn dominant_ccw_axes(tri: &[D3; 3]) -> Option<(usize, usize)> {
    let mut best: Option<(usize, f64)> = None;
    for axis in 0..3 {
        let u = (axis + 1) % 3;
        let v = (axis + 2) % 3;
        let det = orient2d(
            Coord {
                x: tri[0][u],
                y: tri[0][v],
            },
            Coord {
                x: tri[1][u],
                y: tri[1][v],
            },
            Coord {
                x: tri[2][u],
                y: tri[2][v],
            },
        );
        if det != 0.0 && best.is_none_or(|(_, b)| det.abs() > b.abs()) {
            best = Some((axis, det));
        }
    }
    let (axis, det) = best?;
    if det > 0.0 {
        Some(((axis + 1) % 3, (axis + 2) % 3))
    } else {
        Some(((axis + 2) % 3, (axis + 1) % 3))
    }
}

fn orient2d_f64(a: D2, b: D2, c: D2) -> f64 {
    orient2d(
        Coord { x: a[0], y: a[1] },
        Coord { x: b[0], y: b[1] },
        Coord { x: c[0], y: c[1] },
    )
}

fn point_strictly_in_ccw_tri(p: D2, tri: &[D2; 3]) -> bool {
    orient2d_f64(tri[0], tri[1], p) > 0.0
        && orient2d_f64(tri[1], tri[2], p) > 0.0
        && orient2d_f64(tri[2], tri[0], p) > 0.0
}

fn point_in_or_on_ccw_tri(p: D2, tri: &[D2; 3]) -> bool {
    orient2d_f64(tri[0], tri[1], p) >= 0.0
        && orient2d_f64(tri[1], tri[2], p) >= 0.0
        && orient2d_f64(tri[2], tri[0], p) >= 0.0
}

fn polygon_area_abs(poly: &[D2]) -> f64 {
    if poly.len() < 3 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    for i in 1..(poly.len() - 1) {
        let a = poly[0];
        let b = poly[i];
        let c = poly[i + 1];
        sum += (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    }
    0.5 * sum.abs()
}

/// Exact stack-only Sutherland-Hodgman 2D triangle-triangle overlap area.
fn clip_tri_against_ccw_tri_area(subj: &[D2; 3], clip: &[D2; 3]) -> f64 {
    let mut poly = [[0.0f64; 2]; 12];
    poly[0] = subj[0];
    poly[1] = subj[1];
    poly[2] = subj[2];
    let mut len = 3usize;

    for e in 0..3 {
        if len < 3 {
            return 0.0;
        }
        let c0 = clip[e];
        let c1 = clip[(e + 1) % 3];
        let mut next_poly = [[0.0f64; 2]; 12];
        let mut next_len = 0usize;

        let mut prev = poly[len - 1];
        let mut prev_d = orient2d_f64(c0, c1, prev);
        for &curr in &poly[..len] {
            let curr_d = orient2d_f64(c0, c1, curr);
            if curr_d >= 0.0 {
                if prev_d < 0.0 {
                    let t = prev_d / (prev_d - curr_d);
                    next_poly[next_len] = [
                        prev[0] + t * (curr[0] - prev[0]),
                        prev[1] + t * (curr[1] - prev[1]),
                    ];
                    next_len += 1;
                }
                next_poly[next_len] = curr;
                next_len += 1;
            } else if prev_d >= 0.0 {
                let t = prev_d / (prev_d - curr_d);
                next_poly[next_len] = [
                    prev[0] + t * (curr[0] - prev[0]),
                    prev[1] + t * (curr[1] - prev[1]),
                ];
                next_len += 1;
            }
            prev = curr;
            prev_d = curr_d;
        }
        poly = next_poly;
        len = next_len;
    }

    polygon_area_abs(&poly[..len])
}

/// Cancels exact opposite duplicate triangles `(a, b, c)` and `(a, c, b)` one-for-one.
fn cancel_opposite_duplicates(surviving: &mut Vec<SurvTri>) -> Result<()> {
    surviving.retain(|t| {
        let [a, b, c] = t.verts;
        a != b && b != c && a != c
    });
    let n = surviving.len();
    if n < 2 {
        return Ok(());
    }

    let mut entries: Vec<([ArrVertexId; 3], i8, usize)> = Vec::new();
    reserve(&mut entries, n)?;
    for (idx, t) in surviving.iter().enumerate() {
        let (sorted, sign) = canonical_tri_key(t.verts);
        entries.push((sorted, sign, idx));
    }
    entries.sort_unstable();

    let mut keep = filled(n, true)?;
    let mut i = 0usize;
    while i < entries.len() {
        let key = entries[i].0;
        let mut j = i + 1;
        while j < entries.len() && entries[j].0 == key {
            j += 1;
        }
        // `entries[i..j]` has `-1` signs first, then `+1` signs, each sorted by `idx`.
        let split = i + entries[i..j].partition_point(|&(_, sign, _)| sign < 0);
        let neg = &entries[i..split];
        let pos = &entries[split..j];
        let cancel = neg.len().min(pos.len());
        for k in 0..cancel {
            keep[neg[k].2] = false;
            keep[pos[k].2] = false;
        }
        i = j;
    }

    let mut w = 0usize;
    for r in 0..n {
        if keep[r] {
            surviving[w] = surviving[r];
            w += 1;
        }
    }
    surviving.truncate(w);
    Ok(())
}

/// Returns `(sorted_verts, +1 if even cyclic rotation of sorted_verts else -1)`.
fn canonical_tri_key([a, b, c]: [ArrVertexId; 3]) -> ([ArrVertexId; 3], i8) {
    let mut s = [a, b, c];
    s.sort_unstable();
    let even = [a, b, c] == [s[0], s[1], s[2]]
        || [a, b, c] == [s[1], s[2], s[0]]
        || [a, b, c] == [s[2], s[0], s[1]];
    (s, if even { 1 } else { -1 })
}

#[derive(Clone, Copy)]
struct HalfEdgeInc {
    edge: (ArrVertexId, ArrVertexId),
    /// `true` if triangle has directed half-edge `lo -> hi`, `false` if `hi -> lo`.
    is_fwd: bool,
    tri_idx: u32,
    /// Corner index `(0..3)` at `lo`, and corner index `(0..3)` at `hi`.
    c_lo: u8,
    c_hi: u8,
    opp_v: ArrVertexId,
}

#[derive(Clone, Copy)]
struct PinchWedgeItem {
    tri_idx: u32,
    /// `true` for `hi -> lo` (solid immediately in `+theta` around `e = hi - lo`).
    starts_solid: bool,
    c_lo: u8,
    c_hi: u8,
    opp_v: ArrVertexId,
    theta: f64,
}

/// Resolves all `k >= 2` balanced pinch edges and multi-fan pinch vertices so
/// the output surface is a strict 2-manifold.
fn depinch_manifold(
    surviving: &mut Vec<SurvTri>,
    positions: &mut Vec<[f32; 3]>,
    origins: &mut Vec<ArrOrigin>,
) -> Result<()> {
    let num_tris = surviving.len();
    if num_tris == 0 {
        return Ok(());
    }
    let _ = u32::try_from(num_tris.saturating_mul(3)).map_err(|_| oom())?;

    let incidences = collect_sorted_half_edges(surviving)?;
    let mut corner_uf = UnionFind::new();
    corner_uf.reset(num_tris * 3)?;

    // Ordered cyclic rings of `2k` items for every `k >= 2` balanced edge.
    let mut pinch_edges: Vec<Vec<PinchWedgeItem>> = Vec::new();

    let mut i = 0usize;
    while i < incidences.len() {
        let edge = incidences[i].edge;
        let mut j = i + 1;
        while j < incidences.len() && incidences[j].edge == edge {
            j += 1;
        }
        let group = &incidences[i..j];
        let fwd_count = group.iter().filter(|h| h.is_fwd).count();
        let rev_count = group.len() - fwd_count;
        if fwd_count == 0 || fwd_count != rev_count {
            return Err(OpsError::InvalidInput(
                "boolean arrangement failed manifold check",
            ));
        }
        if fwd_count == 1 {
            let h0 = group[0];
            let h1 = group[1];
            corner_uf.union(
                h0.tri_idx * 3 + u32::from(h0.c_lo),
                h1.tri_idx * 3 + u32::from(h1.c_lo),
            );
            corner_uf.union(
                h0.tri_idx * 3 + u32::from(h0.c_hi),
                h1.tri_idx * 3 + u32::from(h1.c_hi),
            );
        } else {
            let ordered = order_pinch_edge_dihedral(edge, group, positions)?;
            push(&mut pinch_edges, ordered)?;
        }
        i = j;
    }

    // Iteratively resolve shift (0 = solid-wedge pairing, 1 = void-wedge pairing)
    // across `k >= 2` pinch edges, propagating endpoint constraints along chains.
    if !pinch_edges.is_empty() {
        resolve_pinch_edge_pairings(&pinch_edges, &mut corner_uf)?;
    }

    // Split every vertex whose incident corners form `m >= 2` disconnected fans.
    split_pinch_vertex_fans(surviving, positions, origins, &mut corner_uf)?;

    // Rare slit-pinch fallback: if a `k >= 2` balanced edge had both endpoints
    // inside a single connected fan at `lo` and `hi`, split wedges `1..k` at the
    // edge midpoint so no parallel edges remain.
    split_remaining_slit_edges(surviving, positions, origins)?;

    Ok(())
}

fn collect_sorted_half_edges(surviving: &[SurvTri]) -> Result<Vec<HalfEdgeInc>> {
    let mut out: Vec<HalfEdgeInc> = Vec::new();
    reserve(&mut out, surviving.len().saturating_mul(3))?;
    for (ti, st) in surviving.iter().enumerate() {
        let v = st.verts;
        for c in 0..3usize {
            let cn = (c + 1) % 3;
            let co = (c + 2) % 3;
            let (u, w) = (v[c], v[cn]);
            let (lo, hi, is_fwd, c_lo, c_hi) = if u < w {
                (u, w, true, c as u8, cn as u8)
            } else {
                (w, u, false, cn as u8, c as u8)
            };
            out.push(HalfEdgeInc {
                edge: (lo, hi),
                is_fwd,
                tri_idx: ti as u32,
                c_lo,
                c_hi,
                opp_v: v[co],
            });
        }
    }
    out.sort_unstable_by(|a, b| {
        a.edge
            .cmp(&b.edge)
            .then_with(|| a.tri_idx.cmp(&b.tri_idx))
            .then_with(|| a.c_lo.cmp(&b.c_lo))
    });
    Ok(out)
}

/// Orders all `2k` half-edges around `e = pos(hi) - pos(lo)` by cyclic dihedral
/// angle `theta` so even indices have `starts_solid == true` (`hi -> lo`) and odd
/// indices have `starts_solid == false` (`lo -> hi`).
fn order_pinch_edge_dihedral(
    edge: (ArrVertexId, ArrVertexId),
    group: &[HalfEdgeInc],
    positions: &[[f32; 3]],
) -> Result<Vec<PinchWedgeItem>> {
    let p_lo = positions[edge.0 as usize].map(f64::from);
    let p_hi = positions[edge.1 as usize].map(f64::from);
    let d = sub3(p_hi, p_lo);
    let len = dot3(d, d).sqrt();
    if len == 0.0 {
        return Err(OpsError::InvalidInput(
            "boolean arrangement failed manifold check",
        ));
    }
    let e_hat = [d[0] / len, d[1] / len, d[2] / len];

    // Orthonormal right-handed basis `(x_hat, y_hat, e_hat)` (`x_hat × y_hat = e_hat`).
    let ref_axis = if e_hat[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let x_raw = cross3(ref_axis, e_hat);
    let x_len = dot3(x_raw, x_raw).sqrt();
    let x_hat = [x_raw[0] / x_len, x_raw[1] / x_len, x_raw[2] / x_len];
    let y_hat = cross3(e_hat, x_hat);

    let mut items: Vec<PinchWedgeItem> = Vec::new();
    reserve(&mut items, group.len())?;
    for h in group {
        let pw = positions[h.opp_v as usize].map(f64::from);
        let rel = sub3(pw, p_lo);
        let x = dot3(rel, x_hat) + 0.0;
        let y = dot3(rel, y_hat) + 0.0;
        let theta = y.atan2(x);
        items.push(PinchWedgeItem {
            tri_idx: h.tri_idx,
            starts_solid: !h.is_fwd,
            c_lo: h.c_lo,
            c_hi: h.c_hi,
            opp_v: h.opp_v,
            theta,
        });
    }

    items.sort_unstable_by(|a, b| {
        a.theta
            .total_cmp(&b.theta)
            .then_with(|| a.tri_idx.cmp(&b.tri_idx))
    });

    // Refine any near-tied dihedral angles (within 1e-9 rad) using exact
    // `orient3d_exact(p_lo, p_hi, wa, wb)`. Note `y + 0.0` maps `-0.0` to `+0.0`,
    // so `+pi` never splits to `-pi`. On exact coplanar ties (`o == 0`), separate
    // adjacent solid wedges cleanly (`false` before `true`) and break same-sense
    // ties via canonical vertex-ID-keyed Simulation-of-Simplicity (`orient3d_sos`).
    let m = items.len();
    for _pass in 0..m {
        let mut swapped = false;
        for idx in 0..m.saturating_sub(1) {
            if (items[idx + 1].theta - items[idx].theta).abs() < 1e-9 {
                let wa = positions[items[idx].opp_v as usize].map(f64::from);
                let wb = positions[items[idx + 1].opp_v as usize].map(f64::from);
                let o = orient3d_exact(p_lo, p_hi, wa, wb);
                let need_swap = if o < 0 {
                    true
                } else if o == 0 {
                    if items[idx].starts_solid != items[idx + 1].starts_solid {
                        items[idx].starts_solid && !items[idx + 1].starts_solid
                    } else {
                        orient3d_sos(
                            (p_lo, PerturbKey::arr(edge.0)),
                            (p_hi, PerturbKey::arr(edge.1)),
                            (wa, PerturbKey::arr(items[idx].opp_v)),
                            (wb, PerturbKey::arr(items[idx + 1].opp_v)),
                        ) > 0
                    }
                } else {
                    false
                };
                if need_swap {
                    items.swap(idx, idx + 1);
                    swapped = true;
                }
            }
        }
        if !swapped {
            break;
        }
    }

    // If two coplanar opposite half-edges tied at exact `orient3d == 0`, arrange ties
    // so `starts_solid` strictly alternates around the circle.
    for idx in 0..m {
        let prev = (idx + m - 1) % m;
        let next = (idx + 1) % m;
        if items[idx].starts_solid == items[prev].starts_solid
            && items[next].starts_solid != items[prev].starts_solid
        {
            let wa = positions[items[idx].opp_v as usize].map(f64::from);
            let wb = positions[items[next].opp_v as usize].map(f64::from);
            if orient3d_exact(p_lo, p_hi, wa, wb) == 0 {
                items.swap(idx, next);
            }
        }
    }

    // Rotate cyclically so `items[0].starts_solid == true` (`hi -> lo`).
    let Some(first_solid) = items.iter().position(|it| it.starts_solid) else {
        return Err(OpsError::InvalidInput(
            "boolean arrangement failed manifold check",
        ));
    };
    items.rotate_left(first_solid);

    for (idx, it) in items.iter().enumerate() {
        let want_solid = idx % 2 == 0;
        if it.starts_solid != want_solid {
            return Err(OpsError::InvalidInput(
                "boolean arrangement failed manifold check",
            ));
        }
    }

    Ok(items)
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct ShiftScore {
    edge_conflicts: usize,
    vert_merges: usize,
    already_matched: usize,
}

impl ShiftScore {
    fn better_than(self, other: Self) -> bool {
        self.edge_conflicts < other.edge_conflicts
            || (self.edge_conflicts == other.edge_conflicts
                && (self.vert_merges < other.vert_merges
                    || (self.vert_merges == other.vert_merges
                        && self.already_matched > other.already_matched)))
    }
}

fn evaluate_shift(
    ring: &[PinchWedgeItem],
    shift: usize,
    corner_uf: &mut UnionFind,
    roots_lo: &mut Vec<u32>,
    roots_hi: &mut Vec<u32>,
) -> ShiftScore {
    let m = ring.len();
    let k = m / 2;
    roots_lo.clear();
    roots_hi.clear();
    for it in ring {
        roots_lo.push(corner_uf.find(it.tri_idx * 3 + u32::from(it.c_lo)));
        roots_hi.push(corner_uf.find(it.tri_idx * 3 + u32::from(it.c_hi)));
    }

    let mut already_matched = 0usize;
    for p in 0..k {
        let a = (2 * p + shift) % m;
        let b = (a + 1) % m;
        if roots_lo[a] == roots_lo[b] {
            already_matched += 1;
        }
        if roots_hi[a] == roots_hi[b] {
            already_matched += 1;
        }
    }

    // Simulate unioning each pair's roots in-place on `roots_lo` and `roots_hi`.
    for p in 0..k {
        let a = (2 * p + shift) % m;
        let b = (a + 1) % m;
        let (rla, rlb) = (roots_lo[a], roots_lo[b]);
        if rla != rlb {
            for r in roots_lo.iter_mut() {
                if *r == rlb {
                    *r = rla;
                }
            }
        }
        let (rha, rhb) = (roots_hi[a], roots_hi[b]);
        if rha != rhb {
            for r in roots_hi.iter_mut() {
                if *r == rhb {
                    *r = rha;
                }
            }
        }
    }

    let mut edge_conflicts = 0usize;
    let mut vert_merges = 0usize;
    for p in 0..k {
        let a_p = (2 * p + shift) % m;
        for q in (p + 1)..k {
            let a_q = (2 * q + shift) % m;
            let same_lo = roots_lo[a_p] == roots_lo[a_q];
            let same_hi = roots_hi[a_p] == roots_hi[a_q];
            if same_lo && same_hi {
                edge_conflicts += 1;
            }
            if same_lo {
                vert_merges += 1;
            }
            if same_hi {
                vert_merges += 1;
            }
        }
    }

    ShiftScore {
        edge_conflicts,
        vert_merges,
        already_matched,
    }
}

fn apply_shift_unions(ring: &[PinchWedgeItem], shift: usize, corner_uf: &mut UnionFind) {
    let m = ring.len();
    let k = m / 2;
    for p in 0..k {
        let a = (2 * p + shift) % m;
        let b = (a + 1) % m;
        corner_uf.union(
            ring[a].tri_idx * 3 + u32::from(ring[a].c_lo),
            ring[b].tri_idx * 3 + u32::from(ring[b].c_lo),
        );
        corner_uf.union(
            ring[a].tri_idx * 3 + u32::from(ring[a].c_hi),
            ring[b].tri_idx * 3 + u32::from(ring[b].c_hi),
        );
    }
}

fn resolve_pinch_edge_pairings(
    pinch_edges: &[Vec<PinchWedgeItem>],
    corner_uf: &mut UnionFind,
) -> Result<()> {
    let max_ring = pinch_edges.iter().map(Vec::len).max().unwrap_or(0);
    let mut roots_lo = Vec::new();
    let mut roots_hi = Vec::new();
    reserve(&mut roots_lo, max_ring)?;
    reserve(&mut roots_hi, max_ring)?;

    let mut resolved = filled(pinch_edges.len(), false)?;
    let mut remaining = pinch_edges.len();

    while remaining > 0 {
        let mut progressed = false;
        for (idx, ring) in pinch_edges.iter().enumerate() {
            if resolved[idx] {
                continue;
            }
            let s0 = evaluate_shift(ring, 0, corner_uf, &mut roots_lo, &mut roots_hi);
            let s1 = evaluate_shift(ring, 1, corner_uf, &mut roots_lo, &mut roots_hi);
            if s0 != s1 {
                let chosen = if s1.better_than(s0) { 1 } else { 0 };
                apply_shift_unions(ring, chosen, corner_uf);
                resolved[idx] = true;
                remaining -= 1;
                progressed = true;
            }
        }
        if !progressed {
            // Unconstrained closed pinch loop: default to solid-wedge separation (`shift = 0`)
            // on the first unresolved edge and propagate around the loop.
            let first = resolved
                .iter()
                .position(|&done| !done)
                .expect("remaining > 0");
            apply_shift_unions(&pinch_edges[first], 0, corner_uf);
            resolved[first] = true;
            remaining -= 1;
        }
    }

    Ok(())
}

/// Splits every vertex whose incident corners belong to `m >= 2` components in `corner_uf`.
fn split_pinch_vertex_fans(
    surviving: &mut [SurvTri],
    positions: &mut Vec<[f32; 3]>,
    origins: &mut Vec<ArrOrigin>,
    corner_uf: &mut UnionFind,
) -> Result<()> {
    // `(vertex_id, uf_root, corner_global_idx, is_untouched_a)`
    let mut entries: Vec<(ArrVertexId, u32, u32, bool)> = Vec::new();
    reserve(&mut entries, surviving.len().saturating_mul(3))?;

    for (ti, st) in surviving.iter().enumerate() {
        let is_untouched = matches!(st.origin, TriOrigin::UntouchedA(_));
        for c in 0..3u32 {
            let cid = ti as u32 * 3 + c;
            let root = corner_uf.find(cid);
            entries.push((st.verts[c as usize], root, cid, is_untouched));
        }
    }
    entries.sort_unstable_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });

    let mut i = 0usize;
    while i < entries.len() {
        let v = entries[i].0;
        let mut j = i + 1;
        while j < entries.len() && entries[j].0 == v {
            j += 1;
        }

        // Count distinct `uf_root` groups at vertex `v`.
        let mut num_fans = 0usize;
        let mut primary_fan = 0usize;
        let mut found_untouched = false;
        let mut gs = i;
        while gs < j {
            let r = entries[gs].1;
            let mut ge = gs + 1;
            let mut has_untouched = entries[gs].3;
            while ge < j && entries[ge].1 == r {
                has_untouched |= entries[ge].3;
                ge += 1;
            }
            if matches!(origins[v as usize], ArrOrigin::LiveA(_))
                && has_untouched
                && !found_untouched
            {
                primary_fan = num_fans;
                found_untouched = true;
            }
            num_fans += 1;
            gs = ge;
        }

        if num_fans >= 2 {
            let pos = normalize_zero(positions[v as usize]);
            let mut fan_idx = 0usize;
            let mut gs = i;
            while gs < j {
                let r = entries[gs].1;
                let mut ge = gs + 1;
                while ge < j && entries[ge].1 == r {
                    ge += 1;
                }
                if fan_idx != primary_fan {
                    let fresh_v = ArrVertexId::try_from(positions.len()).map_err(|_| oom())?;
                    push(positions, pos)?;
                    push(origins, ArrOrigin::Fresh)?;
                    for &(_, _, cid, _) in &entries[gs..ge] {
                        surviving[(cid / 3) as usize].verts[(cid % 3) as usize] = fresh_v;
                    }
                }
                fan_idx += 1;
                gs = ge;
            }
        }

        i = j;
    }

    Ok(())
}

/// Splits any remaining balanced `k >= 2` edge where both endpoints belong to a
/// single vertex fan (slit pinch) by inserting a fresh midpoint on wedges `1..k`.
fn split_remaining_slit_edges(
    surviving: &mut Vec<SurvTri>,
    positions: &mut Vec<[f32; 3]>,
    origins: &mut Vec<ArrOrigin>,
) -> Result<()> {
    let incidences = collect_sorted_half_edges(surviving)?;
    let mut has_slit = false;
    let mut i = 0usize;
    while i < incidences.len() {
        let edge = incidences[i].edge;
        let mut j = i + 1;
        while j < incidences.len() && incidences[j].edge == edge {
            j += 1;
        }
        if j - i > 2 {
            has_slit = true;
            break;
        }
        i = j;
    }
    if !has_slit {
        return Ok(());
    }

    let mut extra: Vec<SurvTri> = Vec::new();
    let mut idx = 0usize;
    while idx < incidences.len() {
        let edge = incidences[idx].edge;
        let mut end = idx + 1;
        while end < incidences.len() && incidences[end].edge == edge {
            end += 1;
        }
        let group = &incidences[idx..end];
        let fwd_count = group.iter().filter(|h| h.is_fwd).count();
        let rev_count = group.len() - fwd_count;
        if fwd_count == 0 || fwd_count != rev_count {
            return Err(OpsError::InvalidInput(
                "boolean arrangement failed manifold check",
            ));
        }
        if fwd_count >= 2 {
            let ring = order_pinch_edge_dihedral(edge, group, positions)?;
            let p_lo = positions[edge.0 as usize].map(f64::from);
            let p_hi = positions[edge.1 as usize].map(f64::from);
            let mid = normalize_zero([
                (0.5 * (p_lo[0] + p_hi[0])) as f32,
                (0.5 * (p_lo[1] + p_hi[1])) as f32,
                (0.5 * (p_lo[2] + p_hi[2])) as f32,
            ]);
            let k = ring.len() / 2;
            for p in 1..k {
                let m_id = ArrVertexId::try_from(positions.len()).map_err(|_| oom())?;
                push(positions, mid)?;
                push(origins, ArrOrigin::Fresh)?;

                // `ring[2*p]` has `hi -> lo`; `ring[2*p + 1]` has `lo -> hi`.
                for item in [ring[2 * p], ring[2 * p + 1]] {
                    let ti = item.tri_idx as usize;
                    let orig = surviving[ti].origin;
                    let w = item.opp_v;
                    if item.starts_solid {
                        // Original cyclic order is `[hi, lo, w]`.
                        surviving[ti].verts = [m_id, edge.0, w];
                        push(
                            &mut extra,
                            SurvTri {
                                verts: [edge.1, m_id, w],
                                origin: orig,
                            },
                        )?;
                    } else {
                        // Original cyclic order is `[lo, hi, w]`.
                        surviving[ti].verts = [edge.0, m_id, w];
                        push(
                            &mut extra,
                            SurvTri {
                                verts: [m_id, edge.1, w],
                                origin: orig,
                            },
                        )?;
                    }
                }
            }
        }
        idx = end;
    }

    if !extra.is_empty() {
        reserve(surviving, extra.len())?;
        surviving.extend_from_slice(&extra);

        // Splitting a slit edge can separate two corner fans that previously met
        // only across that parallel edge; re-split any resulting multi-fan vertices.
        let post_inc = collect_sorted_half_edges(surviving)?;
        let mut post_uf = UnionFind::new();
        post_uf.reset(surviving.len() * 3)?;
        let mut k = 0usize;
        while k + 1 < post_inc.len() {
            let h0 = post_inc[k];
            let h1 = post_inc[k + 1];
            if h0.edge == h1.edge {
                post_uf.union(
                    h0.tri_idx * 3 + u32::from(h0.c_lo),
                    h1.tri_idx * 3 + u32::from(h1.c_lo),
                );
                post_uf.union(
                    h0.tri_idx * 3 + u32::from(h0.c_hi),
                    h1.tri_idx * 3 + u32::from(h1.c_hi),
                );
                k += 2;
            } else {
                k += 1;
            }
        }
        split_pinch_vertex_fans(surviving, positions, origins, &mut post_uf)?;
    }
    Ok(())
}

/// Strict verification gate (§5.7): confirms zero degenerate vertex repeats,
/// zero duplicate triangles, every undirected edge has `count(lo->hi) == 1 &&
/// count(hi->lo) == 1` (closed, manifold, consistently wound), and every used
/// vertex has a single edge-connected corner fan (zero pinch vertices).
fn verify_manifold(surviving: &[SurvTri], num_verts: usize) -> Result<()> {
    let err = || OpsError::InvalidInput("boolean arrangement failed manifold check");
    if surviving.is_empty() {
        return Ok(());
    }

    let mut tri_sets: Vec<[ArrVertexId; 3]> = Vec::new();
    reserve(&mut tri_sets, surviving.len())?;
    for st in surviving {
        let [a, b, c] = st.verts;
        if a == b || b == c || a == c {
            return Err(err());
        }
        if (a as usize) >= num_verts || (b as usize) >= num_verts || (c as usize) >= num_verts {
            return Err(err());
        }
        let mut s = [a, b, c];
        s.sort_unstable();
        tri_sets.push(s);
    }
    tri_sets.sort_unstable();
    if tri_sets.windows(2).any(|w| w[0] == w[1]) {
        return Err(err());
    }

    let incidences = collect_sorted_half_edges(surviving)?;
    if incidences.len() % 2 != 0 {
        return Err(err());
    }

    let mut corner_uf = UnionFind::new();
    corner_uf.reset(surviving.len() * 3)?;

    let mut i = 0usize;
    while i < incidences.len() {
        if i + 1 >= incidences.len() {
            return Err(err());
        }
        let h0 = incidences[i];
        let h1 = incidences[i + 1];
        if h0.edge != h1.edge || h0.is_fwd == h1.is_fwd {
            return Err(err());
        }
        if i + 2 < incidences.len() && incidences[i + 2].edge == h0.edge {
            return Err(err());
        }
        corner_uf.union(
            h0.tri_idx * 3 + u32::from(h0.c_lo),
            h1.tri_idx * 3 + u32::from(h1.c_lo),
        );
        corner_uf.union(
            h0.tri_idx * 3 + u32::from(h0.c_hi),
            h1.tri_idx * 3 + u32::from(h1.c_hi),
        );
        i += 2;
    }

    let mut seen_root = filled(num_verts, u32::MAX)?;
    for (ti, st) in surviving.iter().enumerate() {
        for c in 0..3u32 {
            let v = st.verts[c as usize] as usize;
            let r = corner_uf.find(ti as u32 * 3 + c);
            if seen_root[v] == u32::MAX {
                seen_root[v] = r;
            } else if seen_root[v] != r {
                return Err(err());
            }
        }
    }

    Ok(())
}

/// Converts verified surviving triangles and unified vertices into a [`BooleanPlan`].
fn assemble_plan(
    a: &Mesh,
    surviving: &[SurvTri],
    positions: &[[f32; 3]],
    origins: &[ArrOrigin],
) -> Result<BooleanPlan> {
    let mut plan = BooleanPlan::default();

    // Allocate `plan.fresh` slots ONLY for `ArrOrigin::Fresh` vertices referenced
    // by surviving triangles.
    let mut arr_to_pref: Vec<Option<PRef>> = filled(positions.len(), None)?;
    for st in surviving {
        for &u in &st.verts {
            let slot = &mut arr_to_pref[u as usize];
            if slot.is_none() {
                *slot = Some(match origins[u as usize] {
                    ArrOrigin::LiveA(va) => PRef::Live(va),
                    ArrOrigin::Fresh => {
                        let f_idx = u32::try_from(plan.fresh.len()).map_err(|_| oom())?;
                        push(&mut plan.fresh, normalize_zero(positions[u as usize]))?;
                        PRef::Fresh(f_idx)
                    }
                });
            }
        }
    }

    let to_pref = |verts: [ArrVertexId; 3]| -> [PRef; 3] {
        verts.map(|u| arr_to_pref[u as usize].expect("used vertex mapped"))
    };

    // Group surviving A triangles by parent `TriangleId`, and collect tool triangles.
    // `(parent_tid, is_untouched_origin, order_idx, prefs)`
    let mut a_pieces: Vec<(TriangleId, bool, usize, [PRef; 3])> = Vec::new();
    reserve(&mut a_pieces, surviving.len())?;

    for (order_idx, st) in surviving.iter().enumerate() {
        let prefs = to_pref(st.verts);
        match st.origin {
            TriOrigin::UntouchedA(t) => a_pieces.push((t, true, order_idx, prefs)),
            TriOrigin::CutA(t) => a_pieces.push((t, false, order_idx, prefs)),
            TriOrigin::Tool => push(&mut plan.tool, prefs)?,
        }
    }
    a_pieces.sort_unstable_by(|x, y| x.0.cmp(&y.0).then_with(|| x.2.cmp(&y.2)));

    let mut ap_idx = 0usize;
    for t in 0..a.triangle_slots() as TriangleId {
        let Some(orig_tri) = a.triangle(t) else {
            continue;
        };
        let start = ap_idx;
        while ap_idx < a_pieces.len() && a_pieces[ap_idx].0 == t {
            ap_idx += 1;
        }
        let slice = &a_pieces[start..ap_idx];
        if slice.is_empty() {
            push(&mut plan.drop_a, t)?;
        } else if slice.len() == 1 && slice[0].1 && slice[0].3 == orig_tri.map(PRef::Live) {
            // Untouched triangle of A with unchanged Live(va) corners: leave as-is.
        } else {
            let mut pieces: Vec<[PRef; 3]> = Vec::new();
            reserve(&mut pieces, slice.len())?;
            for &(_, _, _, prefs) in slice {
                pieces.push(prefs);
            }
            push(&mut plan.rewrite, (t, pieces))?;
        }
    }

    Ok(plan)
}
