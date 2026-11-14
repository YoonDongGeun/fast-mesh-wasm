//! Per-vertex protection around a position-moving operation.
//!
//! [`with_protection`] runs any operation in a transaction, then scales back
//! what it did to the listed vertices. It knows nothing of the operation's
//! insides, so brushes keep one code path whether or not anything is protected.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use fast_mesh_store::{Mesh, StoreError, Transaction, VertexId};

use crate::sculpt::collect_proper_neighbors;
use crate::topology::VertexFans;
use crate::{OpsError, Result, allocation_allowed, push, reserve};

/// Runs `op` in `tx`, then scales each listed vertex's move by its weight:
/// `0` leaves the vertex at the bits it had before `op`, `1` keeps the whole
/// move, and values in between land on the straight line between the two.
///
/// Vertices that end where they started are not written. `op` must leave the
/// listed vertices alive. What `op` reports describes its own move, before
/// the scaling: a neighbor it computed against a protected vertex's moved
/// position keeps that result.
///
/// # Errors
/// - [`OpsError::InvalidInput`] when the lists differ in length, a weight is
///   not in `[0, 1]`, or a listed vertex is not alive; `op` has not run then.
/// - Whatever `op` returns, and [`OpsError::Store`] when memory runs out. The
///   caller's transaction should then be rolled back.
pub fn with_protection<R>(
    tx: &mut Transaction,
    vertices: &[VertexId],
    weights: &[f32],
    op: impl FnOnce(&mut Transaction) -> Result<R>,
) -> Result<R> {
    with_protection_band(tx, None, vertices, weights, 0.0, op)
}

#[derive(Clone, Copy)]
struct BandEntry {
    dist: f64,
    v: VertexId,
    idx: u32,
}

impl PartialEq for BandEntry {
    fn eq(&self, other: &Self) -> bool {
        self.dist.to_bits() == other.dist.to_bits() && self.v == other.v
    }
}

impl Eq for BandEntry {}

impl PartialOrd for BandEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BandEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .dist
            .total_cmp(&self.dist)
            .then_with(|| other.v.cmp(&self.v))
    }
}

fn push_band_heap(heap: &mut BinaryHeap<BandEntry>, entry: BandEntry) -> Result<()> {
    allocation_allowed()?;
    heap.try_reserve(1)
        .map_err(|_| OpsError::Store(StoreError::OutOfMemory))?;
    heap.push(entry);
    Ok(())
}

/// Deterministic `HashMap`-free local distance directory (`O(|local_band|)` memory).
struct LocalDistTable {
    nodes: Vec<VertexId>,
    dist: Vec<f64>,
    sorted: Vec<(VertexId, u32)>,
    staging: Vec<(VertexId, u32)>,
}

impl LocalDistTable {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            dist: Vec::new(),
            sorted: Vec::new(),
            staging: Vec::new(),
        }
    }

    fn lookup(&self, v: VertexId) -> Option<u32> {
        if let Ok(pos) = self.sorted.binary_search_by_key(&v, |&(k, _)| k) {
            return Some(self.sorted[pos].1);
        }
        self.staging
            .iter()
            .find_map(|&(k, idx)| (k == v).then_some(idx))
    }

    fn insert(&mut self, v: VertexId, d: f64) -> Result<u32> {
        let idx = self.nodes.len() as u32;
        push(&mut self.nodes, v)?;
        push(&mut self.dist, d)?;
        push(&mut self.staging, (v, idx))?;
        if self.staging.len() >= 64 && self.staging.len() * 4 >= self.sorted.len() {
            reserve(&mut self.sorted, self.staging.len())?;
            self.sorted.extend_from_slice(&self.staging);
            self.staging.clear();
            self.sorted.sort_unstable_by_key(|&(k, _)| k);
        }
        Ok(idx)
    }
}

type AabbBounds = ([f64; 3], [f64; 3]);

fn in_aabb(p: [f32; 3], bounds: Option<AabbBounds>) -> bool {
    let Some((lo, hi)) = bounds else {
        return true;
    };
    let (px, py, pz) = (f64::from(p[0]), f64::from(p[1]), f64::from(p[2]));
    px >= lo[0] && px <= hi[0] && py >= lo[1] && py <= hi[1] && pz >= lo[2] && pz <= hi[2]
}

/// Runs `f` in `tx`, scaling moves on `vertices` by `weights` and smoothly
/// feathering `weight == 0.0` masked seeds across mesh edges out to geodesic
/// edge-Dijkstra distance `band` (`smoothstep(dist / band)`).
///
/// When `band == 0.0`, behaves identically to [`with_protection`]. When
/// `region` is `Some(reg)`, edge-Dijkstra expansion is pruned to the `band`-expanded
/// bounding box of `reg` so localized brush dabs run in `O(|brush|)` without
/// scanning or allocating over the whole mesh.
///
/// # Errors
/// - [`OpsError::InvalidInput`] (converted into `E`) when `vertices`/`weights`
///   fail [`check_protection`] or `band` is negative or non-finite; `f` has not
///   run then.
/// - Whatever `f` returns, and [`OpsError::Store`] when memory runs out.
pub fn with_protection_band<R, E: From<OpsError>>(
    tx: &mut Transaction,
    region: Option<&[VertexId]>,
    vertices: &[VertexId],
    weights: &[f32],
    band: f32,
    f: impl FnOnce(&mut Transaction) -> std::result::Result<R, E>,
) -> std::result::Result<R, E> {
    check_protection(tx.mesh(), vertices, weights).map_err(E::from)?;
    if !band.is_finite() || band < 0.0 {
        return Err(E::from(OpsError::InvalidInput(
            "protection band must be finite and non-negative",
        )));
    }
    if vertices.is_empty() {
        return f(tx);
    }

    if band == 0.0 || !weights.contains(&0.0) {
        let positions = tx.mesh().positions();
        let mut before: Vec<[f32; 3]> = Vec::new();
        reserve(&mut before, vertices.len()).map_err(E::from)?;
        before.extend(vertices.iter().map(|&v| positions[v as usize]));
        let mut ids: Vec<VertexId> = Vec::new();
        reserve(&mut ids, vertices.len()).map_err(E::from)?;
        let mut back: Vec<[f32; 3]> = Vec::new();
        reserve(&mut back, vertices.len()).map_err(E::from)?;
        let report = f(tx)?;
        let positions = tx.mesh().positions();
        for ((&v, &w), from) in vertices.iter().zip(weights).zip(&before) {
            let to = positions[v as usize];
            if w == 1.0 || same_bits(&to, from) {
                continue;
            }
            let at = if w == 0.0 {
                *from
            } else {
                [
                    from[0] + w * (to[0] - from[0]),
                    from[1] + w * (to[1] - from[1]),
                    from[2] + w * (to[2] - from[2]),
                ]
            };
            ids.push(v);
            back.push(at);
        }
        if !ids.is_empty() {
            tx.set_positions(&ids, &back)
                .map_err(|e| E::from(OpsError::from(e)))?;
        }
        return Ok(report);
    }

    // Build sorted deduplicated explicit protection list + localized Dijkstra band BEFORE `f(tx)`.
    let effective: Vec<(VertexId, f32)> = {
        let mesh = tx.mesh();
        let positions = mesh.positions();
        let band_f64 = f64::from(band);

        let mut explicit: Vec<(VertexId, f32)> = Vec::new();
        reserve(&mut explicit, vertices.len()).map_err(E::from)?;
        for (&v, &w) in vertices.iter().zip(weights) {
            explicit.push((v, w));
        }
        explicit.sort_unstable_by_key(|&(v, _)| v);
        let mut write = 0usize;
        for i in 0..explicit.len() {
            if write > 0 && explicit[write - 1].0 == explicit[i].0 {
                explicit[write - 1].1 = explicit[write - 1].1.min(explicit[i].1);
            } else {
                explicit[write] = explicit[i];
                write += 1;
            }
        }
        explicit.truncate(write);

        let bounds: Option<AabbBounds> = region.map(|reg| {
            let mut lo = [f64::INFINITY; 3];
            let mut hi = [f64::NEG_INFINITY; 3];
            let mut any = false;
            for &v in reg {
                if mesh.is_vertex_alive(v) {
                    let p = positions[v as usize];
                    for k in 0..3 {
                        let c = f64::from(p[k]);
                        lo[k] = lo[k].min(c);
                        hi[k] = hi[k].max(c);
                    }
                    any = true;
                }
            }
            if !any {
                return ([1.0; 3], [-1.0; 3]);
            }
            let pad = band_f64 * (1.0 + 1e-5) + 1e-5;
            (
                [lo[0] - pad, lo[1] - pad, lo[2] - pad],
                [hi[0] + pad, hi[1] + pad, hi[2] + pad],
            )
        });

        let fans = if mesh.incidence().is_none() {
            Some(VertexFans::build(mesh).map_err(E::from)?)
        } else {
            None
        };
        let inc = mesh.incidence();

        let mut table = LocalDistTable::new();
        let mut heap: BinaryHeap<BandEntry> = BinaryHeap::new();
        for &(v, w) in &explicit {
            if w == 0.0 && in_aabb(positions[v as usize], bounds) {
                let idx = table.insert(v, 0.0).map_err(E::from)?;
                push_band_heap(&mut heap, BandEntry { dist: 0.0, v, idx }).map_err(E::from)?;
            }
        }

        let mut adjacent: Vec<VertexId> = Vec::new();
        while let Some(BandEntry {
            dist: d_u,
            v: u,
            idx,
        }) = heap.pop()
        {
            if d_u > table.dist[idx as usize] {
                continue;
            }
            let p_u = positions[u as usize].map(f64::from);
            let tris: &[u32] = match inc {
                Some(index) => index.triangles(u),
                None => fans.as_ref().map_or(&[], |f| f.fan(u)),
            };
            collect_proper_neighbors(mesh, u, tris, &mut adjacent).map_err(E::from)?;
            for &nb in &adjacent {
                let p_nb_f32 = positions[nb as usize];
                if !in_aabb(p_nb_f32, bounds) {
                    continue;
                }
                let p_nb = p_nb_f32.map(f64::from);
                let dx = p_nb[0] - p_u[0];
                let dy = p_nb[1] - p_u[1];
                let dz = p_nb[2] - p_u[2];
                let cand = d_u + (dx * dx + dy * dy + dz * dz).sqrt();
                if !(cand.is_finite() && cand < band_f64) {
                    continue;
                }
                match table.lookup(nb) {
                    Some(nb_idx) => {
                        if cand < table.dist[nb_idx as usize] {
                            table.dist[nb_idx as usize] = cand;
                            push_band_heap(
                                &mut heap,
                                BandEntry {
                                    dist: cand,
                                    v: nb,
                                    idx: nb_idx,
                                },
                            )
                            .map_err(E::from)?;
                        }
                    }
                    None => {
                        let nb_idx = table.insert(nb, cand).map_err(E::from)?;
                        push_band_heap(
                            &mut heap,
                            BandEntry {
                                dist: cand,
                                v: nb,
                                idx: nb_idx,
                            },
                        )
                        .map_err(E::from)?;
                    }
                }
            }
        }

        let mut combined: Vec<(VertexId, f32)> = Vec::new();
        for &(v, w) in &explicit {
            if w < 1.0 {
                push(&mut combined, (v, w)).map_err(E::from)?;
            }
        }
        for (&v, &d) in table.nodes.iter().zip(&table.dist) {
            if d > 0.0 && d < band_f64 {
                let s = (d / band_f64) as f32;
                if s < 1.0 {
                    let w_band = s * s * (3.0 - 2.0 * s);
                    if w_band < 1.0 {
                        push(&mut combined, (v, w_band)).map_err(E::from)?;
                    }
                }
            }
        }
        combined.sort_unstable_by_key(|&(v, _)| v);
        let mut out_len = 0usize;
        for i in 0..combined.len() {
            if out_len > 0 && combined[out_len - 1].0 == combined[i].0 {
                combined[out_len - 1].1 = combined[out_len - 1].1.min(combined[i].1);
            } else {
                combined[out_len] = combined[i];
                out_len += 1;
            }
        }
        combined.truncate(out_len);
        combined
    };

    let mut before: Vec<[f32; 3]> = Vec::new();
    reserve(&mut before, effective.len()).map_err(E::from)?;
    {
        let positions = tx.mesh().positions();
        before.extend(effective.iter().map(|&(v, _)| positions[v as usize]));
    }
    let mut ids: Vec<VertexId> = Vec::new();
    reserve(&mut ids, effective.len()).map_err(E::from)?;
    let mut back: Vec<[f32; 3]> = Vec::new();
    reserve(&mut back, effective.len()).map_err(E::from)?;

    let report = f(tx)?;
    let positions = tx.mesh().positions();
    for (&(v, w), from) in effective.iter().zip(&before) {
        let to = positions[v as usize];
        if w == 1.0 || same_bits(&to, from) {
            continue;
        }
        let at = if w == 0.0 {
            *from
        } else {
            [
                from[0] + w * (to[0] - from[0]),
                from[1] + w * (to[1] - from[1]),
                from[2] + w * (to[2] - from[2]),
            ]
        };
        ids.push(v);
        back.push(at);
    }
    if !ids.is_empty() {
        tx.set_positions(&ids, &back)
            .map_err(|e| E::from(OpsError::from(e)))?;
    }
    Ok(report)
}

/// Checks a protection list the way [`with_protection`] does.
///
/// # Errors
/// [`OpsError::InvalidInput`] when the lists differ in length, a weight is not
/// in `[0, 1]`, or a listed vertex is not alive.
pub fn check_protection(mesh: &Mesh, vertices: &[VertexId], weights: &[f32]) -> Result<()> {
    if vertices.len() != weights.len() {
        return Err(OpsError::InvalidInput(
            "protection needs one weight per vertex",
        ));
    }
    if weights.iter().any(|w| !(0.0..=1.0).contains(w)) {
        return Err(OpsError::InvalidInput(
            "protection weights must be between 0 and 1",
        ));
    }
    if vertices.iter().any(|&v| !mesh.is_vertex_alive(v)) {
        return Err(OpsError::InvalidInput("a protected vertex is not alive"));
    }
    Ok(())
}

fn same_bits(a: &[f32; 3], b: &[f32; 3]) -> bool {
    a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deform::{self, Brush, Falloff, GrabOptions, InflateOptions, SmoothOptions};
    use crate::test_util::{mesh_of, sweep};

    /// A flat grid of `n x n` quads in the xy plane, over [0, 1]².
    fn grid(n: u32) -> Mesh {
        let mut positions = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                positions.extend([x as f32 / n as f32, y as f32 / n as f32, 0.0]);
            }
        }
        let at = |x: u32, y: u32| y * (n + 1) + x;
        let mut triangles = Vec::new();
        for y in 0..n {
            for x in 0..n {
                triangles.push([at(x, y), at(x + 1, y), at(x + 1, y + 1)]);
                triangles.push([at(x, y), at(x + 1, y + 1), at(x, y + 1)]);
            }
        }
        mesh_of(&positions, &triangles)
    }

    fn brush() -> Brush {
        Brush {
            center: [0.5, 0.5, 0.0],
            radius: 0.4,
            falloff: Falloff::Smooth,
            strength: 0.1,
        }
    }

    /// The left half of the grid: the vertices with `x < 0.5`.
    fn left_half(mesh: &Mesh) -> Vec<VertexId> {
        mesh.positions()
            .iter()
            .enumerate()
            .filter(|(_, p)| p[0] < 0.5)
            .map(|(v, _)| v as VertexId)
            .collect()
    }

    /// Runs `op` under a full hold on the left half; asserts the held
    /// vertices keep their bits and some others move.
    fn holds(op: impl FnOnce(&mut Transaction) -> Result<()>) {
        let mut mesh = grid(20);
        let held = left_half(&mesh);
        let weights = vec![0.0; held.len()];
        let before = mesh.positions().to_vec();
        let mut tx = mesh.begin_edit();
        with_protection(&mut tx, &held, &weights, op).unwrap();
        tx.commit().unwrap();
        for &v in &held {
            let (a, b) = (before[v as usize], mesh.positions()[v as usize]);
            assert!(same_bits(&a, &b), "held vertex {v} moved");
        }
        let moved = before
            .iter()
            .zip(mesh.positions())
            .filter(|(a, b)| !same_bits(a, b))
            .count();
        assert!(moved > 0, "nothing moved on the free side");
    }

    #[test]
    fn inflate_moves_only_free_vertices() {
        holds(|tx| {
            deform::inflate(
                tx,
                &InflateOptions {
                    brush: brush(),
                    ..Default::default()
                },
            )?;
            Ok(())
        });
    }

    #[test]
    fn grab_moves_only_free_vertices() {
        holds(|tx| {
            let options = GrabOptions {
                brush: brush(),
                direction: [0.0, 0.0, 0.2],
                ..Default::default()
            };
            deform::grab(tx, &options)?;
            Ok(())
        });
    }

    #[test]
    fn smooth_moves_only_free_vertices() {
        let mut mesh = grid(20);
        // A bump to melt: a flat grid would not move under smoothing.
        let mut tx = mesh.begin_edit();
        deform::grab(
            &mut tx,
            &GrabOptions {
                brush: brush(),
                direction: [0.0, 0.0, 0.3],
                ..Default::default()
            },
        )
        .unwrap();
        tx.commit().unwrap();
        let held = left_half(&mesh);
        let weights = vec![0.0; held.len()];
        let before = mesh.positions().to_vec();
        let mut tx = mesh.begin_edit();
        with_protection(&mut tx, &held, &weights, |tx| {
            let options = SmoothOptions {
                brush: Brush {
                    strength: 0.5,
                    ..brush()
                },
                cycles: 2,
                ..SmoothOptions::default()
            };
            deform::smooth(tx, &options)
        })
        .unwrap();
        tx.commit().unwrap();
        for &v in &held {
            assert!(same_bits(
                &before[v as usize],
                &mesh.positions()[v as usize]
            ));
        }
        assert!(
            before
                .iter()
                .zip(mesh.positions())
                .any(|(a, b)| !same_bits(a, b))
        );
    }

    #[test]
    fn a_weight_scales_the_move() {
        let run = |weight: Option<f32>| {
            let mut mesh = grid(10);
            let v = 5 * 11 + 5; // The center vertex.
            let (ids, weights) = match weight {
                Some(w) => (vec![v as VertexId], vec![w]),
                None => (vec![], vec![]),
            };
            let mut tx = mesh.begin_edit();
            with_protection(&mut tx, &ids, &weights, |tx| {
                deform::inflate(
                    tx,
                    &InflateOptions {
                        brush: brush(),
                        ..Default::default()
                    },
                )
            })
            .unwrap();
            tx.commit().unwrap();
            mesh.positions()[v][2]
        };
        let whole = run(None);
        assert!(whole != 0.0);
        assert_eq!(run(Some(1.0)).to_bits(), whole.to_bits());
        assert!((run(Some(0.25)) - 0.25 * whole).abs() < 1e-6);
        assert_eq!(run(Some(0.0)).to_bits(), 0f32.to_bits());
    }

    #[test]
    fn undo_restores_every_vertex() {
        let mut mesh = grid(10);
        let held = left_half(&mesh);
        let weights = vec![0.0; held.len()];
        let before = mesh.positions().to_vec();
        let mut tx = mesh.begin_edit();
        with_protection(&mut tx, &held, &weights, |tx| {
            deform::inflate(
                tx,
                &InflateOptions {
                    brush: brush(),
                    ..Default::default()
                },
            )
        })
        .unwrap();
        let patch = tx.commit().unwrap();
        assert!(mesh.positions() != before.as_slice());
        patch.revert(&mut mesh).unwrap();
        assert_eq!(mesh.positions(), before.as_slice());
    }

    #[test]
    fn running_out_of_memory_writes_nothing() {
        let mesh = grid(4);
        let held = left_half(&mesh);
        let weights = vec![0.0; held.len()];
        let failures = sweep(mesh, |tx| {
            with_protection(tx, &held, &weights, |tx| {
                deform::inflate(
                    tx,
                    &InflateOptions {
                        brush: brush(),
                        ..Default::default()
                    },
                )?;
                Ok(())
            })
        });
        assert!(failures >= 3);
    }

    #[test]
    fn rejects_bad_input_before_running() {
        let mut mesh = grid(2);
        let mut tx = mesh.begin_edit();
        let mut ran = false;
        let mut try_with = |ids: &[VertexId], weights: &[f32]| {
            with_protection(&mut tx, ids, weights, |_| {
                ran = true;
                Ok(())
            })
            .unwrap_err()
        };
        assert!(matches!(try_with(&[0], &[]), OpsError::InvalidInput(_)));
        assert!(matches!(try_with(&[0], &[1.5]), OpsError::InvalidInput(_)));
        assert!(matches!(
            try_with(&[0], &[f32::NAN]),
            OpsError::InvalidInput(_)
        ));
        assert!(matches!(
            try_with(&[9999], &[0.0]),
            OpsError::InvalidInput(_)
        ));
        assert!(!ran);
    }

    #[test]
    fn with_protection_band_feathers_dijkstra_distances_and_matches_region() {
        // 10x10 grid over [0, 1]^2: cell width = 0.1, mask x <= 0.3 (`ix <= 3`).
        let held_of = |mesh: &Mesh| -> Vec<VertexId> {
            mesh.positions()
                .iter()
                .enumerate()
                .filter(|(_, p)| p[0] <= 0.3001)
                .map(|(v, _)| v as VertexId)
                .collect()
        };
        let band = 0.25f32; // reaches ix = 4 (dist = 0.1) and ix = 5 (dist = 0.2), while ix >= 6 (dist >= 0.3) is free!
        let wide_grab = GrabOptions {
            brush: Brush {
                center: [0.5, 0.5, 0.0],
                radius: 5.0,
                falloff: Falloff::Linear,
                strength: 1.0,
            },
            direction: [0.0, 0.0, 1.0],
            ..Default::default()
        };

        // Free baseline (no protection) to measure exact 100% move per vertex.
        let mut free_mesh = grid(10);
        {
            let mut tx = free_mesh.begin_edit();
            deform::grab(&mut tx, &wide_grab).unwrap();
            tx.commit().unwrap();
        }

        // Whole-mesh (`region = None`, without incidence) vs localized (`region = Some`, with incidence).
        let mut whole_mesh = grid(10);
        let held = held_of(&whole_mesh);
        let zeros = vec![0.0f32; held.len()];
        let before = whole_mesh.positions().to_vec();
        {
            let mut tx = whole_mesh.begin_edit();
            with_protection_band(&mut tx, None, &held, &zeros, band, |tx| {
                deform::grab(tx, &wide_grab)
            })
            .unwrap();
            tx.commit().unwrap();
        }

        let mut reg_mesh = grid(10);
        reg_mesh.ensure_incidence().unwrap();
        let all_region: Vec<VertexId> = (0..reg_mesh.vertex_slots() as VertexId).collect();
        {
            let mut tx = reg_mesh.begin_edit();
            with_protection_band(&mut tx, Some(&all_region), &held, &zeros, band, |tx| {
                deform::grab_region(tx, &wide_grab, &all_region)
            })
            .unwrap();
            tx.commit().unwrap();
        }

        // Bit-for-bit parity between region = None and region = Some(...)!
        for v in 0..whole_mesh.vertex_slots() {
            assert_eq!(
                whole_mesh.positions()[v].map(f32::to_bits),
                reg_mesh.positions()[v].map(f32::to_bits),
                "region vs whole mismatch at vertex {v}"
            );
        }

        // Verify exact smoothstep weights along horizontal grid row iy = 5 (`y = 0.5`):
        // shortest path along horizontal grid edges from ix = 3 (`x = 0.3`) has exact length `(ix - 3) * 0.1`.
        for ix in 0..=10usize {
            let v = 5 * 11 + ix;
            let got_z = whole_mesh.positions()[v][2];
            let full_z = free_mesh.positions()[v][2];
            if ix <= 3 {
                assert!(
                    same_bits(&whole_mesh.positions()[v], &before[v]),
                    "masked vertex ix={ix} must remain byte-identical"
                );
            } else if ix <= 5 {
                let dist = (ix - 3) as f32 * 0.1;
                let s = dist / band;
                let w_want = s * s * (3.0 - 2.0 * s);
                assert!(
                    (got_z - w_want * full_z).abs() < 1e-6,
                    "band vertex ix={ix}: got {got_z}, want {}",
                    w_want * full_z
                );
            } else {
                assert_eq!(
                    got_z.to_bits(),
                    full_z.to_bits(),
                    "vertex beyond band ix={ix} must move 100%"
                );
            }
        }

        // Invalid band (< 0 or NaN) rejected before running closure; OOM sweep leaves mesh untouched.
        let mut check_mesh = grid(4);
        let mut tx = check_mesh.begin_edit();
        let mut ran = false;
        assert!(matches!(
            with_protection_band(&mut tx, None, &[0], &[0.0], -0.1, |_| -> Result<()> {
                ran = true;
                Ok(())
            }),
            Err(OpsError::InvalidInput(_))
        ));
        assert!(matches!(
            with_protection_band(&mut tx, None, &[0], &[0.0], f32::NAN, |_| -> Result<()> {
                ran = true;
                Ok(())
            }),
            Err(OpsError::InvalidInput(_))
        ));
        assert!(!ran);
        drop(tx);

        let mut oom_mesh = grid(6);
        oom_mesh.ensure_incidence().unwrap();
        let oom_held = held_of(&oom_mesh);
        let oom_zeros = vec![0.0; oom_held.len()];
        let oom_reg: Vec<VertexId> = (0..oom_mesh.vertex_slots() as VertexId).collect();
        let failures = sweep(oom_mesh, |tx| {
            with_protection_band(tx, Some(&oom_reg), &oom_held, &oom_zeros, 0.25, |tx| {
                deform::grab_region(tx, &wide_grab, &oom_reg).map(drop)
            })
        });
        assert!(failures >= 3);
    }
}
