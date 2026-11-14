//! Non-manifold topology repair: duplicates, boundary fins, non-manifold edges,
//! and pinch vertices.

use std::cmp::Ordering;

use fast_mesh_store::{Transaction, TriangleId, VertexId};

use super::norm64;
use crate::topology::{UnionFind, cross, dot, sub};
use crate::{Result, allocation_allowed, filled, push, reserve};

const MAX_FIN_TRIANGLES: usize = 2;
const MAX_FIN_PASSES: usize = 8;

/// Options for [`make_manifold()`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ManifoldOptions {
    /// Remove exact duplicate triangles sharing the same unordered vertex triple.
    /// Default `true`.
    pub remove_duplicates: bool,
    /// Prune small dangling boundary fins attached to non-manifold edges when
    /// larger sheets already meet along that edge. Default `true`.
    pub prune_fins: bool,
    /// Separate multi-sheet junctions (`> 2` incident faces on an edge) by
    /// duplicating shared vertices across sheets. Default `true`.
    pub split_non_manifold_edges: bool,
    /// Separate pinch (bow-tie) vertices whose 1-ring fan has more than one
    /// edge-connected component. Default `true`.
    pub split_pinch_vertices: bool,
}

impl Default for ManifoldOptions {
    fn default() -> Self {
        Self {
            remove_duplicates: true,
            prune_fins: true,
            split_non_manifold_edges: true,
            split_pinch_vertices: true,
        }
    }
}

impl ManifoldOptions {
    /// A builder for [`ManifoldOptions`].
    pub fn builder() -> ManifoldOptionsBuilder {
        ManifoldOptionsBuilder::default()
    }
}

/// Builder for [`ManifoldOptions`].
#[derive(Debug, Clone, Default)]
pub struct ManifoldOptionsBuilder {
    options: ManifoldOptions,
}

impl ManifoldOptionsBuilder {
    /// Remove exact duplicate triangles.
    pub fn remove_duplicates(mut self, remove_duplicates: bool) -> Self {
        self.options.remove_duplicates = remove_duplicates;
        self
    }

    /// Prune small dangling boundary fins on non-manifold edges.
    pub fn prune_fins(mut self, prune_fins: bool) -> Self {
        self.options.prune_fins = prune_fins;
        self
    }

    /// Split non-manifold edges (`> 2` incident faces).
    pub fn split_non_manifold_edges(mut self, split_non_manifold_edges: bool) -> Self {
        self.options.split_non_manifold_edges = split_non_manifold_edges;
        self
    }

    /// Split pinch (bow-tie) vertices.
    pub fn split_pinch_vertices(mut self, split_pinch_vertices: bool) -> Self {
        self.options.split_pinch_vertices = split_pinch_vertices;
        self
    }

    /// The options as set.
    pub fn build(self) -> ManifoldOptions {
        self.options
    }
}

/// What [`make_manifold()`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ManifoldReport {
    /// Exact duplicate triangles removed.
    pub removed_duplicate_triangles: usize,
    /// Small dangling boundary fin triangles pruned from non-manifold edges.
    pub pruned_fin_triangles: usize,
    /// Distinct non-manifold edges separated across sheets.
    pub split_non_manifold_edges: usize,
    /// Pinch vertices split into separate 1-ring fan components.
    pub split_pinch_vertices: usize,
    /// New vertices created by splitting non-manifold edges or pinch vertices.
    pub vertices_added: usize,
    /// Vertices orphaned by duplicate or fin removal and removed.
    pub removed_vertices: usize,
}

#[derive(Clone, Copy)]
struct TriEdgeRef {
    key: (VertexId, VertexId),
    tri: u32,
}

struct EdgeIndex {
    keys: Vec<(VertexId, VertexId)>,
    offsets: Vec<usize>,
    tris: Vec<u32>,
}

impl EdgeIndex {
    fn build(tris: &[[VertexId; 3]], t_alive: &[bool]) -> Result<Self> {
        let mut refs: Vec<TriEdgeRef> = Vec::new();
        let live_estimate = t_alive.iter().filter(|&&a| a).count();
        reserve(&mut refs, live_estimate * 3)?;
        for (idx, (&tri, &alive)) in tris.iter().zip(t_alive).enumerate() {
            if !alive || !is_proper(tri) {
                continue;
            }
            let [a, b, c] = tri;
            for (u, v) in [(a, b), (b, c), (c, a)] {
                push(
                    &mut refs,
                    TriEdgeRef {
                        key: edge_key(u, v),
                        tri: idx as u32,
                    },
                )?;
            }
        }
        refs.sort_unstable_by_key(|r| (r.key, r.tri));

        let mut keys = Vec::new();
        let mut offsets = Vec::new();
        let mut out_tris = Vec::new();
        reserve(&mut out_tris, refs.len())?;
        push(&mut offsets, 0)?;

        let mut i = 0;
        while i < refs.len() {
            let k = refs[i].key;
            let mut end = i + 1;
            while end < refs.len() && refs[end].key == k {
                end += 1;
            }
            push(&mut keys, k)?;
            for item in &refs[i..end] {
                push(&mut out_tris, item.tri)?;
            }
            push(&mut offsets, out_tris.len())?;
            i = end;
        }
        Ok(Self {
            keys,
            offsets,
            tris: out_tris,
        })
    }

    fn edge_count(&self) -> usize {
        self.keys.len()
    }

    fn find(&self, u: VertexId, v: VertexId) -> Option<usize> {
        let key = edge_key(u, v);
        self.keys.binary_search(&key).ok()
    }

    fn triangles(&self, edge_idx: usize) -> &[u32] {
        let start = self.offsets[edge_idx];
        let end = self.offsets[edge_idx + 1];
        &self.tris[start..end]
    }

    fn non_manifold_count(&self) -> usize {
        (0..self.edge_count())
            .filter(|&e| self.triangles(e).len() > 2)
            .count()
    }
}

struct VertexStar {
    offsets: Vec<usize>,
    tris: Vec<u32>,
}

impl VertexStar {
    fn build(num_verts: usize, tris: &[[VertexId; 3]], t_alive: &[bool]) -> Result<Self> {
        let mut counts = filled(num_verts, 0usize)?;
        for (&tri, &alive) in tris.iter().zip(t_alive) {
            if !alive {
                continue;
            }
            let [a, b, c] = tri;
            counts[a as usize] += 1;
            if b != a {
                counts[b as usize] += 1;
            }
            if c != a && c != b {
                counts[c as usize] += 1;
            }
        }
        let mut offsets = filled(num_verts + 1, 0usize)?;
        let mut sum = 0usize;
        for v in 0..num_verts {
            offsets[v] = sum;
            sum += counts[v];
        }
        offsets[num_verts] = sum;

        let mut cursor = filled(num_verts, 0usize)?;
        cursor.copy_from_slice(&offsets[..num_verts]);
        let mut flat = filled(sum, 0u32)?;
        for (idx, (&tri, &alive)) in tris.iter().zip(t_alive).enumerate() {
            if !alive {
                continue;
            }
            let [a, b, c] = tri;
            let t = idx as u32;
            flat[cursor[a as usize]] = t;
            cursor[a as usize] += 1;
            if b != a {
                flat[cursor[b as usize]] = t;
                cursor[b as usize] += 1;
            }
            if c != a && c != b {
                flat[cursor[c as usize]] = t;
                cursor[c as usize] += 1;
            }
        }
        Ok(Self {
            offsets,
            tris: flat,
        })
    }

    fn fan(&self, v: VertexId) -> &[u32] {
        let i = v as usize;
        &self.tris[self.offsets[i]..self.offsets[i + 1]]
    }
}

#[inline]
pub(super) fn is_proper(tri: [VertexId; 3]) -> bool {
    tri[0] != tri[1] && tri[1] != tri[2] && tri[2] != tri[0]
}

#[inline]
pub(super) fn edge_key(u: VertexId, v: VertexId) -> (VertexId, VertexId) {
    if u < v { (u, v) } else { (v, u) }
}

#[inline]
pub(super) fn sorted_triple(mut tri: [VertexId; 3]) -> [VertexId; 3] {
    tri.sort_unstable();
    tri
}

#[inline]
pub(super) fn runs_forward_tri(tri: [VertexId; 3], u: VertexId, v: VertexId) -> bool {
    (tri[0] == u && tri[1] == v) || (tri[1] == u && tri[2] == v) || (tri[2] == u && tri[0] == v)
}

pub(super) fn unit_normal_raw(pos: &[[f32; 3]], tri: [VertexId; 3]) -> Option<[f64; 3]> {
    let p = |v: VertexId| {
        let q = pos[v as usize];
        [f64::from(q[0]), f64::from(q[1]), f64::from(q[2])]
    };
    let a = p(tri[0]);
    let b = p(tri[1]);
    let c = p(tri[2]);
    let n = cross(sub(b, a), sub(c, a));
    let len = norm64(n);
    if len > 1e-18 && len.is_finite() {
        Some([n[0] / len, n[1] / len, n[2] / len])
    } else {
        None
    }
}

fn remove_duplicate_pass(tris: &[[VertexId; 3]], t_alive: &mut [bool]) -> Result<usize> {
    let mut keys: Vec<([VertexId; 3], u32)> = Vec::new();
    reserve(&mut keys, tris.len())?;
    for (idx, (&tri, &alive)) in tris.iter().zip(t_alive.iter()).enumerate() {
        if alive {
            push(&mut keys, (sorted_triple(tri), idx as u32))?;
        }
    }
    keys.sort_unstable();
    let mut removed = 0usize;
    let mut i = 0;
    while i < keys.len() {
        let target = keys[i].0;
        let mut j = i + 1;
        while j < keys.len() && keys[j].0 == target {
            t_alive[keys[j].1 as usize] = false;
            removed += 1;
            j += 1;
        }
        i = j;
    }
    Ok(removed)
}

fn prune_fins_pass(tris: &[[VertexId; 3]], t_alive: &mut [bool]) -> Result<usize> {
    let mut pruned_total = 0usize;
    for _ in 0..MAX_FIN_PASSES {
        let edges = EdgeIndex::build(tris, t_alive)?;
        if edges.non_manifold_count() == 0 {
            break;
        }
        let n_tris = tris.len();
        let mut uf = UnionFind::new();
        uf.reset(n_tris)?;

        let mut tri_has_boundary = filled(n_tris, false)?;
        let mut nm_edges: Vec<usize> = Vec::new();
        for e in 0..edges.edge_count() {
            let inc = edges.triangles(e);
            if inc.len() == 1 {
                tri_has_boundary[inc[0] as usize] = true;
            } else if inc.len() == 2 {
                uf.union(inc[0], inc[1]);
            } else if inc.len() > 2 {
                push(&mut nm_edges, e)?;
            }
        }
        if nm_edges.is_empty() {
            break;
        }

        let mut comp_size = filled(n_tris, 0usize)?;
        let mut comp_boundary = filled(n_tris, false)?;
        for t in 0..n_tris {
            if !t_alive[t] || !is_proper(tris[t]) {
                continue;
            }
            let r = uf.find(t as u32) as usize;
            comp_size[r] += 1;
            comp_boundary[r] |= tri_has_boundary[t];
        }

        let mut prune_comp = filled(n_tris, false)?;
        for &e in &nm_edges {
            let inc = edges.triangles(e);
            for &t in inc {
                let r = uf.find(t) as usize;
                let sz = comp_size[r];
                if !comp_boundary[r] || sz > MAX_FIN_TRIANGLES {
                    continue;
                }
                let larger_others = inc
                    .iter()
                    .filter(|&&o| o != t)
                    .filter(|&&o| {
                        let ro = uf.find(o) as usize;
                        ro != r && (comp_size[ro] > sz || !comp_boundary[ro])
                    })
                    .count();
                if larger_others >= 2 {
                    prune_comp[r] = true;
                }
            }
        }

        let mut pruned_round = 0usize;
        for t in 0..n_tris {
            if t_alive[t] && is_proper(tris[t]) {
                let r = uf.find(t as u32) as usize;
                if prune_comp[r] {
                    t_alive[t] = false;
                    pruned_round += 1;
                }
            }
        }
        if pruned_round == 0 {
            break;
        }
        pruned_total += pruned_round;
    }
    Ok(pruned_total)
}

fn assign_edge_pair_groups(
    pos: &[[f32; 3]],
    tris: &[[VertexId; 3]],
    edges: &EdgeIndex,
) -> Result<Vec<u32>> {
    let mut group_of = filled(edges.tris.len(), 0u32)?;
    let mut normals: Vec<Option<[f64; 3]>> = Vec::new();
    let mut used: Vec<bool> = Vec::new();

    for e in 0..edges.edge_count() {
        let start = edges.offsets[e];
        let end = edges.offsets[e + 1];
        let len = end - start;
        if len <= 2 {
            continue;
        }
        let (u, v) = edges.keys[e];
        normals.clear();
        reserve(&mut normals, len)?;
        for &t in &edges.tris[start..end] {
            push(&mut normals, unit_normal_raw(pos, tris[t as usize]))?;
        }
        used.clear();
        reserve(&mut used, len)?;
        used.resize(len, false);

        let mut next_group = 0u32;
        loop {
            let mut best: Option<(bool, f64, usize, usize)> = None;
            for i in 0..len {
                if used[i] {
                    continue;
                }
                let fwd_i = runs_forward_tri(tris[edges.tris[start + i] as usize], u, v);
                for j in (i + 1)..len {
                    if used[j] {
                        continue;
                    }
                    let fwd_j = runs_forward_tri(tris[edges.tris[start + j] as usize], u, v);
                    let opposite = fwd_i != fwd_j;
                    let align = match (normals[i], normals[j]) {
                        (Some(na), Some(nb)) => {
                            let d = dot(na, nb);
                            if opposite { d } else { -d }
                        }
                        _ => -2.0,
                    };
                    let take = match best {
                        None => true,
                        Some((b_opp, b_align, _, _)) => {
                            (opposite, align).partial_cmp(&(b_opp, b_align))
                                == Some(Ordering::Greater)
                        }
                    };
                    if take {
                        best = Some((opposite, align, i, j));
                    }
                }
            }
            let Some((_, _, i, j)) = best else {
                break;
            };
            used[i] = true;
            used[j] = true;
            group_of[start + i] = next_group;
            group_of[start + j] = next_group;
            next_group += 1;
        }
        for i in 0..len {
            if !used[i] {
                group_of[start + i] = next_group;
                next_group += 1;
            }
        }
    }
    Ok(group_of)
}

fn edge_pair_group(edges: &EdgeIndex, group_of: &[u32], e_idx: usize, tri: u32) -> Option<u32> {
    let start = edges.offsets[e_idx];
    let end = edges.offsets[e_idx + 1];
    let inc = &edges.tris[start..end];
    inc.binary_search(&tri)
        .ok()
        .map(|pos| group_of[start + pos])
}

fn split_non_manifold_edges_pass(
    pos: &mut Vec<[f32; 3]>,
    root_parent: &mut Vec<VertexId>,
    tris: &mut [[VertexId; 3]],
    t_alive: &[bool],
) -> Result<usize> {
    let mut edges = EdgeIndex::build(tris, t_alive)?;
    let initial_nm = edges.non_manifold_count();
    if initial_nm == 0 {
        return Ok(0);
    }

    let max_rounds = initial_nm.saturating_add(4);
    let mut uf = UnionFind::new();
    for _ in 0..max_rounds {
        if edges.non_manifold_count() == 0 {
            break;
        }
        let group_of = assign_edge_pair_groups(pos, tris, &edges)?;
        let num_verts = pos.len();
        let fans = VertexStar::build(num_verts, tris, t_alive)?;

        let mut is_nm_vertex = filled(num_verts, false)?;
        for e in 0..edges.edge_count() {
            if edges.triangles(e).len() > 2 {
                let (u, v) = edges.keys[e];
                is_nm_vertex[u as usize] = true;
                is_nm_vertex[v as usize] = true;
            }
        }

        let mut changed = false;
        let mut comp_id: Vec<u32> = Vec::new();
        let mut seen_edges: Vec<(usize, u32, u32)> = Vec::new();
        let mut sub_map: Vec<(u32, u32)> = Vec::new();
        let mut edge_counts: Vec<((u32, VertexId), u8)> = Vec::new();

        for v in 0..num_verts {
            if !is_nm_vertex[v] {
                continue;
            }
            let vv = v as VertexId;
            let star = fans.fan(vv);
            if star.len() < 2 {
                continue;
            }

            uf.reset(star.len())?;
            seen_edges.clear();
            for (local_i, &t) in star.iter().enumerate() {
                let tri = tris[t as usize];
                if !is_proper(tri) {
                    continue;
                }
                for w in tri {
                    if w == vv {
                        continue;
                    }
                    let Some(e_idx) = edges.find(vv, w) else {
                        continue;
                    };
                    let inc = edges.triangles(e_idx);
                    if inc.len() == 2 {
                        let other = if inc[0] == t { inc[1] } else { inc[0] };
                        if let Ok(local_j) = star.binary_search(&other) {
                            uf.union(local_i as u32, local_j as u32);
                        }
                    } else if inc.len() > 2
                        && let Some(gid) = edge_pair_group(&edges, &group_of, e_idx, t)
                    {
                        push(&mut seen_edges, (e_idx, gid, local_i as u32))?;
                    }
                }
            }
            seen_edges.sort_unstable();
            let mut s = 0;
            while s < seen_edges.len() {
                let (e_idx, gid, first_local) = seen_edges[s];
                let mut e = s + 1;
                while e < seen_edges.len() && seen_edges[e].0 == e_idx && seen_edges[e].1 == gid {
                    uf.union(first_local, seen_edges[e].2);
                    e += 1;
                }
                s = e;
            }

            // Guarantee no single component retains > 2 triangles on any incident edge.
            comp_id.clear();
            reserve(&mut comp_id, star.len())?;
            sub_map.clear();
            edge_counts.clear();
            let mut next_comp = 0u32;
            for (local_i, &t) in star.iter().enumerate() {
                let uf_root = uf.find(local_i as u32);
                let base_comp = match sub_map.iter().find(|&&(r, _)| r == uf_root) {
                    Some(&(_, cid)) => cid,
                    None => {
                        let cid = next_comp;
                        next_comp += 1;
                        push(&mut sub_map, (uf_root, cid))?;
                        cid
                    }
                };
                let tri = tris[t as usize];
                let mut target_comp = base_comp;
                let overflow = is_proper(tri)
                    && tri.iter().filter(|&&w| w != vv).any(|&w| {
                        edge_counts
                            .iter()
                            .find(|&&((cid, ew), _)| cid == base_comp && ew == w)
                            .is_some_and(|&(_, cnt)| cnt >= 2)
                    });
                if overflow {
                    target_comp = next_comp;
                    next_comp += 1;
                }
                if is_proper(tri) {
                    for &w in &tri {
                        if w == vv {
                            continue;
                        }
                        match edge_counts
                            .iter_mut()
                            .find(|((cid, ew), _)| *cid == target_comp && *ew == w)
                        {
                            Some(entry) => entry.1 = entry.1.saturating_add(1),
                            None => push(&mut edge_counts, ((target_comp, w), 1))?,
                        }
                    }
                }
                push(&mut comp_id, target_comp)?;
            }

            if next_comp <= 1 {
                continue;
            }
            changed = true;
            let mut clone_vid: Vec<VertexId> = Vec::new();
            reserve(&mut clone_vid, next_comp as usize)?;
            push(&mut clone_vid, vv)?;
            for _ in 1..next_comp {
                let new_v = u32::try_from(pos.len())
                    .map_err(|_| crate::OpsError::InvalidInput("vertex count exceeds u32::MAX"))?;
                push(pos, pos[v])?;
                push(root_parent, root_parent[v])?;
                push(&mut clone_vid, new_v)?;
            }
            for (&t, &cid) in star.iter().zip(&comp_id) {
                if cid == 0 {
                    continue;
                }
                let nv = clone_vid[cid as usize];
                for c in &mut tris[t as usize] {
                    if *c == vv {
                        *c = nv;
                    }
                }
            }
        }
        if !changed {
            break;
        }
        edges = EdgeIndex::build(tris, t_alive)?;
    }

    Ok(initial_nm)
}

fn split_pinch_vertices_pass(
    pos: &mut Vec<[f32; 3]>,
    root_parent: &mut Vec<VertexId>,
    tris: &mut [[VertexId; 3]],
    t_alive: &[bool],
) -> Result<usize> {
    let edges = EdgeIndex::build(tris, t_alive)?;
    let initial_verts = pos.len();
    let fans = VertexStar::build(initial_verts, tris, t_alive)?;
    let mut uf = UnionFind::new();
    let mut split_count = 0usize;
    let mut root_to_comp: Vec<(u32, u32)> = Vec::new();

    for v in 0..initial_verts {
        let vv = v as VertexId;
        let star = fans.fan(vv);
        if star.len() < 2 {
            continue;
        }
        uf.reset(star.len())?;
        let mut pieces = star.len();
        for (i, &t) in star.iter().enumerate() {
            let tri = tris[t as usize];
            if !is_proper(tri) {
                continue;
            }
            for w in tri {
                if w == vv {
                    continue;
                }
                let Some(e_idx) = edges.find(vv, w) else {
                    continue;
                };
                for &u in edges.triangles(e_idx) {
                    if u > t
                        && let Ok(j) = star.binary_search(&u)
                        && uf.union(i as u32, j as u32)
                    {
                        pieces -= 1;
                    }
                }
            }
        }
        if pieces <= 1 {
            continue;
        }
        split_count += 1;
        root_to_comp.clear();
        let mut next_comp = 0u32;
        let mut clone_vid: Vec<VertexId> = Vec::new();
        reserve(&mut clone_vid, pieces)?;
        push(&mut clone_vid, vv)?;

        for (i, &t) in star.iter().enumerate() {
            let r = uf.find(i as u32);
            let cid = match root_to_comp.iter().find(|&&(rr, _)| rr == r) {
                Some(&(_, c)) => c,
                None => {
                    let c = next_comp;
                    next_comp += 1;
                    push(&mut root_to_comp, (r, c))?;
                    if c > 0 {
                        let new_v = u32::try_from(pos.len()).map_err(|_| {
                            crate::OpsError::InvalidInput("vertex count exceeds u32::MAX")
                        })?;
                        push(pos, pos[v])?;
                        push(root_parent, root_parent[v])?;
                        push(&mut clone_vid, new_v)?;
                    }
                    c
                }
            };
            if cid > 0 {
                let nv = clone_vid[cid as usize];
                for c in &mut tris[t as usize] {
                    if *c == vv {
                        *c = nv;
                    }
                }
            }
        }
    }
    Ok(split_count)
}

/// Repairs non-manifold defects on the mesh inside `tx`: exact duplicate
/// triangles, small dangling boundary fins on non-manifold edges, remaining
/// non-manifold edges (`> 2` faces), and pinch vertices.
///
/// New vertices duplicate both position and source channel values (including
/// [`UV_CHANNEL`](fast_mesh_store::UV_CHANNEL)) from their parent vertex. All
/// plan buffers are computed and reserved fallibly before writing to `tx`, so
/// an out-of-memory error leaves `tx` untouched.
///
/// # Errors
/// - [`OpsError::InvalidInput`] if vertex capacity would overflow `u32`.
/// - [`OpsError::Store`] when memory runs out or a store write fails.
pub fn make_manifold(tx: &mut Transaction, options: &ManifoldOptions) -> Result<ManifoldReport> {
    allocation_allowed()?;
    let mesh = tx.mesh();
    let orig_v_slots = mesh.vertex_slots();
    let orig_t_slots = mesh.triangle_slots();

    if mesh.triangle_count() == 0 {
        return Ok(ManifoldReport::default());
    }

    let mut pos: Vec<[f32; 3]> = Vec::new();
    reserve(&mut pos, orig_v_slots)?;
    let mut root_parent: Vec<VertexId> = Vec::new();
    reserve(&mut root_parent, orig_v_slots)?;
    let raw_pos = mesh.positions();
    for (v, &p) in raw_pos.iter().take(orig_v_slots).enumerate() {
        push(&mut pos, p)?;
        push(&mut root_parent, v as VertexId)?;
    }

    let mut orig_tris = filled(orig_t_slots, [0; 3])?;
    let mut tris = filled(orig_t_slots, [0; 3])?;
    let mut t_alive = filled(orig_t_slots, false)?;
    let mut orig_used = filled(orig_v_slots, false)?;
    for t in 0..orig_t_slots {
        if let Some(tri) = mesh.triangle(t as TriangleId) {
            orig_tris[t] = tri;
            tris[t] = tri;
            t_alive[t] = true;
            for v in tri {
                orig_used[v as usize] = true;
            }
        }
    }

    let removed_duplicate_triangles = if options.remove_duplicates {
        remove_duplicate_pass(&tris, &mut t_alive)?
    } else {
        0
    };

    let pruned_fin_triangles = if options.prune_fins {
        prune_fins_pass(&tris, &mut t_alive)?
    } else {
        0
    };

    let split_non_manifold_edges = if options.split_non_manifold_edges {
        split_non_manifold_edges_pass(&mut pos, &mut root_parent, &mut tris, &t_alive)?
    } else {
        0
    };

    let split_pinch_vertices = if options.split_pinch_vertices {
        split_pinch_vertices_pass(&mut pos, &mut root_parent, &mut tris, &t_alive)?
    } else {
        0
    };

    let added_count = pos.len() - orig_v_slots;
    let mut removed_tris: Vec<TriangleId> = Vec::new();
    let mut rewired_ids: Vec<TriangleId> = Vec::new();
    let mut rewired_tris: Vec<[VertexId; 3]> = Vec::new();

    let mut final_used = filled(orig_v_slots, false)?;
    for t in 0..orig_t_slots {
        let was_alive = mesh.is_triangle_alive(t as TriangleId);
        if !was_alive {
            continue;
        }
        if !t_alive[t] {
            push(&mut removed_tris, t as TriangleId)?;
        } else {
            let tri = tris[t];
            for v in tri {
                if (v as usize) < orig_v_slots {
                    final_used[v as usize] = true;
                }
            }
            if tri != orig_tris[t] {
                push(&mut rewired_ids, t as TriangleId)?;
                push(&mut rewired_tris, tri)?;
            }
        }
    }

    let mut removed_verts: Vec<VertexId> = Vec::new();
    for v in 0..orig_v_slots {
        if orig_used[v] && !final_used[v] && mesh.is_vertex_alive(v as VertexId) {
            push(&mut removed_verts, v as VertexId)?;
        }
    }

    if removed_tris.is_empty() && rewired_ids.is_empty() && added_count == 0 {
        return Ok(ManifoldReport::default());
    }

    // Reserve all commit scratch buffers before any Transaction write.
    let mut actual_new_vids = filled(added_count, 0 as VertexId)?;
    let mut mapped_rewired = filled(rewired_tris.len(), [0; 3])?;

    for (i, new_slot) in (orig_v_slots..pos.len()).enumerate() {
        let parent = root_parent[new_slot];
        actual_new_vids[i] = tx.add_vertex_with_parents(pos[new_slot], &[(parent, 1.0)])?;
    }
    for (i, &tri) in rewired_tris.iter().enumerate() {
        mapped_rewired[i] = tri.map(|v| {
            let idx = v as usize;
            if idx >= orig_v_slots {
                actual_new_vids[idx - orig_v_slots]
            } else {
                v
            }
        });
    }
    if !rewired_ids.is_empty() {
        tx.set_triangles(&rewired_ids, &mapped_rewired)?;
    }
    if !removed_tris.is_empty() {
        tx.remove_triangles(&removed_tris)?;
    }
    if !removed_verts.is_empty() {
        tx.remove_vertices(&removed_verts)?;
    }

    Ok(ManifoldReport {
        removed_duplicate_triangles,
        pruned_fin_triangles,
        split_non_manifold_edges,
        split_pinch_vertices,
        vertices_added: added_count,
        removed_vertices: removed_verts.len(),
    })
}

#[cfg(test)]
mod tests {
    use fast_mesh_store::{ChannelDesc, UV_CHANNEL};

    use super::*;
    use crate::analyze::{AnalyzeOptions, analyze};
    use crate::test_util::{box_mesh, edit, mesh_of, parts, sweep};

    #[test]
    fn clean_box_is_noop() {
        let mut mesh = box_mesh([0.0; 3], [1.0; 3]);
        let (r, empty) = edit(&mut mesh, |tx| {
            make_manifold(tx, &ManifoldOptions::default())
        });
        assert!(empty);
        assert_eq!(r, ManifoldReport::default());
    }

    #[test]
    fn removes_exact_duplicate_triangles_keeping_first() {
        let mut mesh = mesh_of(
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            &[[0, 1, 2], [2, 1, 0], [0, 1, 2]],
        );
        let (r, _) = edit(&mut mesh, |tx| {
            make_manifold(tx, &ManifoldOptions::default())
        });
        assert_eq!(r.removed_duplicate_triangles, 2);
        assert_eq!(mesh.triangle_count(), 1);
        assert_eq!(mesh.triangle(0), Some([0, 1, 2]));
    }

    #[test]
    fn prunes_dangling_fin_on_closed_box_edge() {
        let mut mesh = box_mesh([0.0; 3], [1.0; 3]);
        // Add a dangling fin off edge (0, 1) to vertex 8.
        edit(&mut mesh, |tx| {
            let v = tx.add_vertices(&[[0.5, 0.5, 2.0]])?;
            tx.add_triangles(&[[0, 1, v[0]]])?;
            Ok(())
        });
        let a_before = analyze(&mesh, &AnalyzeOptions::default()).unwrap();
        assert_eq!(a_before.non_manifold_edges, 1);

        let (r, _) = edit(&mut mesh, |tx| {
            make_manifold(tx, &ManifoldOptions::default())
        });
        assert_eq!(r.pruned_fin_triangles, 1);
        assert_eq!(r.split_non_manifold_edges, 0);
        assert_eq!(r.removed_vertices, 1);

        let a_after = analyze(&mesh, &AnalyzeOptions::default()).unwrap();
        assert_eq!(a_after.non_manifold_edges, 0);
        assert!(a_after.pinch_vertices.is_empty());
        assert_eq!(mesh.triangle_count(), 12);
    }

    #[test]
    fn splits_three_sheet_y_junction_and_preserves_uvs() {
        let mut mesh = mesh_of(
            &[
                0.0, 0.0, 0.0, // 0
                1.0, 0.0, 0.0, // 1
                0.5, 1.0, 0.0, // 2
                0.5, 0.0, 1.0, // 3
                0.5, -1.0, 0.0, // 4
            ],
            &[[0, 1, 2], [0, 1, 3], [0, 1, 4]],
        );
        mesh.add_channel(ChannelDesc::uv()).unwrap();
        edit(&mut mesh, |tx| {
            tx.write_channel(
                UV_CHANNEL,
                &[0, 1, 2, 3, 4],
                &[0.1, 0.2, 0.8, 0.2, 0.5, 0.9, 0.5, 0.4, 0.5, 0.0],
            )?;
            Ok(())
        });

        let (r, _) = edit(&mut mesh, |tx| {
            make_manifold(tx, &ManifoldOptions::default())
        });
        assert_eq!(r.pruned_fin_triangles, 0);
        assert_eq!(r.split_non_manifold_edges, 1);
        assert!(r.vertices_added >= 2);
        assert_eq!(mesh.triangle_count(), 3);

        let a = analyze(&mesh, &AnalyzeOptions::default()).unwrap();
        assert_eq!(a.non_manifold_edges, 0);
        assert!(a.pinch_vertices.is_empty());

        let uvs = mesh.uvs().unwrap();
        // Every vertex positioned at (0,0,0) or (1,0,0) kept its parent UV.
        for v in 0..mesh.vertex_slots() {
            let p = mesh.positions()[v];
            let uv = [uvs[2 * v], uvs[2 * v + 1]];
            if p == [0.0, 0.0, 0.0] {
                assert_eq!(uv, [0.1, 0.2]);
            } else if p == [1.0, 0.0, 0.0] {
                assert_eq!(uv, [0.8, 0.2]);
            }
        }
    }

    #[test]
    fn splits_pinch_bowtie_vertex_and_preserves_uvs() {
        let mut mesh = mesh_of(
            &[
                0.0, 0.0, 0.0, // 0: pinch vertex
                -1.0, 1.0, 0.0, // 1
                1.0, 1.0, 0.0, // 2
                -1.0, -1.0, 0.0, // 3
                1.0, -1.0, 0.0, // 4
            ],
            &[[0, 2, 1], [0, 3, 4]],
        );
        mesh.add_channel(ChannelDesc::uv()).unwrap();
        edit(&mut mesh, |tx| {
            tx.write_channel(
                UV_CHANNEL,
                &[0, 1, 2, 3, 4],
                &[0.5, 0.5, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            )?;
            Ok(())
        });

        let before = parts(&mesh);
        let mut tx = mesh.begin_edit();
        let r = make_manifold(&mut tx, &ManifoldOptions::default()).unwrap();
        let patch = tx.commit().unwrap();
        assert_eq!(r.split_pinch_vertices, 1);
        assert_eq!(r.vertices_added, 1);
        assert_eq!(mesh.vertex_count(), 6);
        assert_eq!(&mesh.uvs().unwrap()[10..12], &[0.5, 0.5]);
        assert!(
            analyze(&mesh, &AnalyzeOptions::default())
                .unwrap()
                .pinch_vertices
                .is_empty()
        );

        // Undo restores exact original live bowtie topology and attributes.
        let mut hist = fast_mesh_store::History::new(usize::MAX);
        hist.push(&mut mesh, patch).unwrap();
        hist.undo(&mut mesh).unwrap();
        assert_eq!(mesh.vertex_count(), 5);
        assert_eq!(mesh.triangle_count(), 2);
        let after = parts(&mesh);
        assert_eq!(after.1, before.1);
        assert_eq!(&after.0[..before.0.len()], &before.0[..]);
        assert_eq!(
            &mesh.uvs().unwrap()[..10],
            &[0.5, 0.5, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
        );
    }

    #[test]
    fn oom_writes_nothing() {
        let pos = [
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.5, 0.0, 1.0, 0.5, -1.0, 0.0,
        ];
        let tris = [[0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 2]];
        let failure_points = sweep(mesh_of(&pos, &tris), |tx| {
            make_manifold(tx, &ManifoldOptions::default()).map(|_| ())
        });
        assert!(failure_points > 0);
        // Sanity check final run succeeds.
        let mut mesh = mesh_of(&pos, &tris);
        let (r, _) = edit(&mut mesh, |tx| {
            make_manifold(tx, &ManifoldOptions::default())
        });
        assert_eq!(r.removed_duplicate_triangles, 1);
    }
}
