//! Face-consistent 256-case Manifold Marching Cubes over regular 3D scalar lattices.
//!
//! [`marching_cubes`] and [`marching_cubes_raw`] extract the `iso_value` level-set
//! (`inside` = `value < iso_value`, `outside` = `value >= iso_value`) from a slice-major
//! `(iz * ny + iy) * nx + ix` grid.
//!
//! All 256 cube configurations are built at compile time from canonical 2D face-consistent
//! segments: every shared grid face orders its 4 corners in the same right-handed `(u, v)` frame
//! and separates diagonal `1` corners on `1010` / `0101` saddles, reversing segment direction
//! between `side = 0` and `side = 1`. Adjacent cubes therefore emit shared-face edges in exact
//! opposite half-edge directions with zero cracks, zero non-manifold edges, and zero pinch
//! vertices.

use fast_mesh_store::{StoreError, par};

use crate::{OpsError, Result, allocation_allowed, filled, reserve};

/// Minimum cube count (`(nx - 1) * (ny - 1) * (nz - 1)`) before parallel chunking engages.
const PAR_MIN_CUBES: usize = 16 * 1024;

/// Cubes scanned per parallel range in Stage 1.
const CUBE_CHUNK: usize = 4 * 1024;

/// Active cubes / vertices processed per parallel chunk in Stages 2 and 3.
const WORK_CHUNK: usize = 2 * 1024;

/// Clamped lower bound on edge interpolation parameter `t` so vertices stay strictly inside edges.
const T_MIN: f64 = 1e-4;

/// Clamped upper bound on edge interpolation parameter `t`.
const T_MAX: f64 = 1.0 - 1e-4;

/// Canonical cube edge descriptors `(dx0, dy0, dz0, axis)` for `e in 0..12`.
const EDGE_DESC: [(u8, u8, u8, u8); 12] = [
    // X-aligned (axis = 0)
    (0, 0, 0, 0), // 0: (0,0,0)-(1,0,0)
    (0, 1, 0, 0), // 1: (0,1,0)-(1,1,0)
    (0, 0, 1, 0), // 2: (0,0,1)-(1,0,1)
    (0, 1, 1, 0), // 3: (0,1,1)-(1,1,1)
    // Y-aligned (axis = 1)
    (0, 0, 0, 1), // 4: (0,0,0)-(0,1,0)
    (1, 0, 0, 1), // 5: (1,0,0)-(1,1,0)
    (0, 0, 1, 1), // 6: (0,0,1)-(0,1,1)
    (1, 0, 1, 1), // 7: (1,0,1)-(1,1,1)
    // Z-aligned (axis = 2)
    (0, 0, 0, 2), // 8:  (0,0,0)-(0,0,1)
    (1, 0, 0, 2), // 9:  (1,0,0)-(1,0,1)
    (0, 1, 0, 2), // 10: (0,1,0)-(0,1,1)
    (1, 1, 0, 2), // 11: (1,1,0)-(1,1,1)
];

/// Precomputed triangulation for one 8-corner inside/outside bitmask.
#[derive(Clone, Copy)]
struct CubeCase {
    tri_count: u8,
    edge_mask: u16,
    tris: [[u8; 3]; 10],
}

const fn corner_index(dx: u8, dy: u8, dz: u8) -> u8 {
    dx | (dy << 1) | (dz << 2)
}

const fn edge_endpoints(e: u8) -> (u8, u8) {
    let (dx0, dy0, dz0, axis) = EDGE_DESC[e as usize];
    let c0 = corner_index(dx0, dy0, dz0);
    let c1 = corner_index(
        dx0 + (axis == 0) as u8,
        dy0 + (axis == 1) as u8,
        dz0 + (axis == 2) as u8,
    );
    (c0, c1)
}

const fn find_cube_edge(ca: u8, cb: u8) -> u8 {
    let mut e = 0u8;
    while e < 12 {
        let (u, v) = edge_endpoints(e);
        if (u == ca && v == cb) || (u == cb && v == ca) {
            return e;
        }
        e += 1;
    }
    panic!("missing cube edge between face corners");
}

/// 6-bit mask of the two cube faces (`axis * 2 + side`) containing cube edge `e`.
const fn edge_face_mask(e: u8) -> u8 {
    let (dx0, dy0, dz0, axis) = EDGE_DESC[e as usize];
    let coords = [dx0, dy0, dz0];
    let mut mask = 0u8;
    let mut k = 0usize;
    while k < 3 {
        if k != axis as usize {
            mask |= 1u8 << (k as u8 * 2 + coords[k]);
        }
        k += 1;
    }
    mask
}

const fn build_case(mask: u8) -> CubeCase {
    let mut next_edge = [u8::MAX; 12];
    let mut prev_edge = [u8::MAX; 12];

    let mut axis = 0u8;
    while axis < 3 {
        let u_axis = (axis + 1) % 3;
        let v_axis = (axis + 2) % 3;
        let uv_corners: [(u8, u8); 4] = [(0, 0), (1, 0), (1, 1), (0, 1)];

        let mut side = 0u8;
        while side < 2 {
            let mut q = [0u8; 4];
            let mut b = [0u8; 4];
            let mut k = 0usize;
            while k < 4 {
                let (u, v) = uv_corners[k];
                let mut coords = [0u8; 3];
                coords[axis as usize] = side;
                coords[u_axis as usize] = u;
                coords[v_axis as usize] = v;
                let c = corner_index(coords[0], coords[1], coords[2]);
                q[k] = c;
                b[k] = (mask >> c) & 1;
                k += 1;
            }

            if !(b[0] == b[1] && b[1] == b[2] && b[2] == b[3]) {
                let mut s = [0u8; 4];
                let mut i = 0usize;
                while i < 4 {
                    s[i] = find_cube_edge(q[i], q[(i + 1) % 4]);
                    i += 1;
                }

                // Walk maximal contiguous runs of `1` corners around `[q0, q1, q2, q3]`.
                // On `1010` / `0101` saddles, each `1` corner forms its own length-1 run,
                // separating the two `1` corners consistently on both `side = 0` and `side = 1`.
                let mut start = 0usize;
                while start < 4 {
                    if b[(start + 3) % 4] == 0 && b[start] == 1 {
                        let e_in = s[(start + 3) % 4];
                        let mut end = start;
                        while b[end] == 1 {
                            end = (end + 1) % 4;
                        }
                        let e_out = s[(end + 3) % 4];
                        let (from, to) = if side == 1 {
                            (e_in, e_out)
                        } else {
                            (e_out, e_in)
                        };
                        assert!(next_edge[from as usize] == u8::MAX);
                        assert!(prev_edge[to as usize] == u8::MAX);
                        next_edge[from as usize] = to;
                        prev_edge[to as usize] = from;
                    }
                    start += 1;
                }
            }
            side += 1;
        }
        axis += 1;
    }

    let mut edge_mask = 0u16;
    let mut e = 0u8;
    while e < 12 {
        let (c0, c1) = edge_endpoints(e);
        let active = ((mask >> c0) & 1) != ((mask >> c1) & 1);
        let has_next = next_edge[e as usize] != u8::MAX;
        let has_prev = prev_edge[e as usize] != u8::MAX;
        assert!(active == has_next && active == has_prev);
        if active {
            edge_mask |= 1u16 << e;
        }
        e += 1;
    }

    let mut visited = [false; 12];
    let mut tris = [[0u8; 3]; 10];
    let mut tri_count = 0usize;

    let mut start_e = 0usize;
    while start_e < 12 {
        if !visited[start_e] && next_edge[start_e] != u8::MAX {
            let mut cycle = [0u8; 12];
            let mut len = 0usize;
            let mut cur = start_e;
            loop {
                visited[cur] = true;
                cycle[len] = cur as u8;
                len += 1;
                cur = next_edge[cur] as usize;
                if cur == start_e {
                    break;
                }
            }
            assert!(len >= 3);

            // Prefer a single fan root whose interior chords never lie on a cube face,
            // guaranteeing strict cube-interior chords and zero cross-cube chord collisions.
            let mut chosen_root = usize::MAX;
            let mut r = 0usize;
            while r < len {
                let root_faces = edge_face_mask(cycle[r]);
                let mut ok = true;
                let mut k = 2usize;
                while k + 1 < len {
                    if (root_faces & edge_face_mask(cycle[(r + k) % len])) != 0 {
                        ok = false;
                        break;
                    }
                    k += 1;
                }
                if ok {
                    chosen_root = r;
                    break;
                }
                r += 1;
            }

            assert!(
                chosen_root != usize::MAX,
                "every cube cycle must admit a face-free fan root"
            );
            let mut k = 1usize;
            while k + 1 < len {
                tris[tri_count] = [
                    cycle[chosen_root],
                    cycle[(chosen_root + k) % len],
                    cycle[(chosen_root + k + 1) % len],
                ];
                tri_count += 1;
                k += 1;
            }
        }
        start_e += 1;
    }

    CubeCase {
        tri_count: tri_count as u8,
        edge_mask,
        tris,
    }
}

const fn build_cases() -> [CubeCase; 256] {
    let empty = CubeCase {
        tri_count: 0,
        edge_mask: 0,
        tris: [[0; 3]; 10],
    };
    let mut table = [empty; 256];
    let mut mask = 0usize;
    while mask < 256 {
        table[mask] = build_case(mask as u8);
        mask += 1;
    }
    table
}

const CASES: [CubeCase; 256] = build_cases();

/// Triangle mesh extracted from a 3D scalar lattice by Marching Cubes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct IsoMesh {
    /// Vertex positions in local `f32` coordinates.
    pub positions: Vec<[f32; 3]>,
    /// Outward-wound triangles indexing into `positions`.
    pub triangles: Vec<[u32; 3]>,
}

/// Per-chunk active cubes and owned canonical lattice edge keys from Stage 1.
struct ChunkScan {
    cubes: Vec<(usize, u8)>,
    keys: Vec<u64>,
    tri_count: usize,
}

#[inline]
fn grow_push<T>(v: &mut Vec<T>, value: T) -> Result<()> {
    if v.len() == v.capacity() {
        let add = v.len().max(64);
        reserve(v, add)?;
    }
    v.push(value);
    Ok(())
}

#[inline]
fn interpolate_edge(
    key: u64,
    origin: [f32; 3],
    dims: [usize; 3],
    voxel_size: f32,
    values: &[f32],
    iso_value: f32,
) -> [f32; 3] {
    let [nx, ny, _] = dims;
    let nxy = nx * ny;
    let axis = (key & 3) as usize;
    let node_a = (key >> 2) as usize;
    let ix = node_a % nx;
    let rem = node_a / nx;
    let iy = rem % ny;
    let iz = rem / ny;

    let stride = match axis {
        0 => 1,
        1 => nx,
        _ => nxy,
    };
    let node_b = node_a + stride;

    let va = f64::from(values[node_a]);
    let vb = f64::from(values[node_b]);
    let iso = f64::from(iso_value);

    let t = if va.is_finite() && vb.is_finite() && iso.is_finite() {
        ((iso - va) / (vb - va)).clamp(T_MIN, T_MAX)
    } else if va.is_finite() && !vb.is_finite() && iso.is_finite() {
        T_MIN
    } else if !va.is_finite() && vb.is_finite() && iso.is_finite() {
        T_MAX
    } else {
        0.5
    };

    let h = f64::from(voxel_size);
    let coords = [ix, iy, iz];
    let mut pos = [0.0f32; 3];
    for k in 0..3 {
        let ok = f64::from(origin[k]);
        let pa = ok + (coords[k] as f64) * h;
        let pb = if k == axis {
            ok + ((coords[k] + 1) as f64) * h
        } else {
            pa
        };
        pos[k] = (pa + t * (pb - pa)) as f32;
    }
    pos
}

/// Extracts the `iso_value` level-set (`inside` = `value < iso_value`) from `grid`.
pub fn marching_cubes(grid: &fast_mesh_fields::ScalarGrid, iso_value: f32) -> Result<IsoMesh> {
    let spec = grid.spec();
    marching_cubes_raw(
        spec.origin,
        spec.dims,
        spec.voxel_size,
        grid.values(),
        iso_value,
    )
}

/// Extracts the `iso_value` level-set (`inside` = `value < iso_value`, `outside` = `value >= iso_value`)
/// from a slice-major `(iz * ny + iy) * nx + ix` regular grid.
///
/// # Errors
/// - `OpsError::InvalidInput` if `origin` is non-finite, `voxel_size` is non-finite or `<= 0.0`,
///   `iso_value` is `NaN`, any `dims[k] < 2`, `nx * ny * nz` overflows or `!= values.len()`,
///   the upper grid corner overflows `f32`, or any element of `values` is `NaN`.
/// - `OpsError::Store(StoreError::OutOfMemory)` on fallible reservation failure.
pub fn marching_cubes_raw(
    origin: [f32; 3],
    dims: [usize; 3],
    voxel_size: f32,
    values: &[f32],
    iso_value: f32,
) -> Result<IsoMesh> {
    if !origin.iter().all(|c| c.is_finite()) {
        return Err(OpsError::InvalidInput("grid origin must be finite"));
    }
    if !voxel_size.is_finite() || voxel_size <= 0.0 {
        return Err(OpsError::InvalidInput(
            "grid voxel_size must be finite and positive",
        ));
    }
    if iso_value.is_nan() {
        return Err(OpsError::InvalidInput("iso_value must not be NaN"));
    }
    if dims.iter().any(|&d| d < 2) {
        return Err(OpsError::InvalidInput(
            "grid dimensions must be at least 2 along every axis",
        ));
    }
    let Some(total_nodes) = dims[0]
        .checked_mul(dims[1])
        .and_then(|xy| xy.checked_mul(dims[2]))
    else {
        return Err(OpsError::InvalidInput("grid voxel count overflows usize"));
    };
    if values.len() != total_nodes {
        return Err(OpsError::InvalidInput(
            "grid values length does not match dimensions",
        ));
    }
    for k in 0..3 {
        let span32 = ((dims[k] - 1) as f32) * voxel_size;
        let upper32 = origin[k] + span32;
        let upper64 =
            (f64::from(origin[k]) + ((dims[k] - 1) as f64) * f64::from(voxel_size)) as f32;
        if !span32.is_finite() || !upper32.is_finite() || !upper64.is_finite() {
            return Err(OpsError::InvalidInput("grid upper corner overflows f32"));
        }
    }
    if values.iter().any(|v| v.is_nan()) {
        return Err(OpsError::InvalidInput("grid values must not contain NaN"));
    }

    let [nx, ny, nz] = dims;
    let nxy = nx * ny;
    let cx = nx - 1;
    let cy = ny - 1;
    let cz = nz - 1;
    let num_cubes = cx * cy * cz;

    allocation_allowed()?;
    let oom = || OpsError::Store(StoreError::OutOfMemory);

    // Stage 1: Scan cube chunks in slice-major order, collecting active cubes and owned edge keys.
    let scans = par::map_range(num_cubes, CUBE_CHUNK, PAR_MIN_CUBES, oom, |range| {
        let mut cubes = Vec::new();
        let mut keys = Vec::new();
        let mut tri_count = 0usize;

        let mut ix = range.start % cx;
        let rem = range.start / cx;
        let mut iy = rem % cy;
        let mut iz = rem / cy;

        for cube_idx in range {
            let base = (iz * ny + iy) * nx + ix;
            let mut mask = 0u8;
            if values[base] < iso_value {
                mask |= 1 << 0;
            }
            if values[base + 1] < iso_value {
                mask |= 1 << 1;
            }
            if values[base + nx] < iso_value {
                mask |= 1 << 2;
            }
            if values[base + nx + 1] < iso_value {
                mask |= 1 << 3;
            }
            if values[base + nxy] < iso_value {
                mask |= 1 << 4;
            }
            if values[base + nxy + 1] < iso_value {
                mask |= 1 << 5;
            }
            if values[base + nxy + nx] < iso_value {
                mask |= 1 << 6;
            }
            if values[base + nxy + nx + 1] < iso_value {
                mask |= 1 << 7;
            }

            let case = &CASES[mask as usize];
            if case.tri_count > 0 {
                grow_push(&mut cubes, (cube_idx, mask))?;
                tri_count += case.tri_count as usize;

                let mut em = case.edge_mask;
                while em != 0 {
                    let e = em.trailing_zeros() as usize;
                    em &= em - 1;
                    let (dx0, dy0, dz0, axis) = EDGE_DESC[e];
                    // Emit each global lattice edge once from its unique owner cube.
                    if (dx0 == 0 || ix + 1 == cx)
                        && (dy0 == 0 || iy + 1 == cy)
                        && (dz0 == 0 || iz + 1 == cz)
                    {
                        let gx = ix + dx0 as usize;
                        let gy = iy + dy0 as usize;
                        let gz = iz + dz0 as usize;
                        let node = (gz * ny + gy) * nx + gx;
                        let key = ((node as u64) << 2) | (axis as u64);
                        grow_push(&mut keys, key)?;
                    }
                }
            }

            ix += 1;
            if ix == cx {
                ix = 0;
                iy += 1;
                if iy == cy {
                    iy = 0;
                    iz += 1;
                }
            }
        }

        Ok(ChunkScan {
            cubes,
            keys,
            tri_count,
        })
    })?;

    let total_tris: usize = scans.iter().map(|s| s.tri_count).sum();
    if total_tris == 0 {
        return Ok(IsoMesh::default());
    }

    let total_active_cubes: usize = scans.iter().map(|s| s.cubes.len()).sum();
    let total_keys: usize = scans.iter().map(|s| s.keys.len()).sum();

    let mut active_cubes = Vec::new();
    reserve(&mut active_cubes, total_active_cubes)?;
    let mut unique_keys = Vec::new();
    reserve(&mut unique_keys, total_keys)?;
    for s in &scans {
        active_cubes.extend_from_slice(&s.cubes);
        unique_keys.extend_from_slice(&s.keys);
    }
    drop(scans);

    // Stage 2: Sort + deduplicate canonical lattice edge keys and interpolate vertex positions.
    unique_keys.sort_unstable();
    unique_keys.dedup();
    if u32::try_from(unique_keys.len()).is_err() {
        return Err(OpsError::InvalidInput(
            "extracted vertex count overflows u32",
        ));
    }

    let sub_threshold = if num_cubes >= PAR_MIN_CUBES {
        WORK_CHUNK
    } else {
        usize::MAX
    };

    let mut positions = filled(unique_keys.len(), [0.0f32; 3])?;
    par::for_each_chunk_mut(
        &mut positions,
        WORK_CHUNK,
        sub_threshold,
        |chunk_idx, out| {
            let base = chunk_idx * WORK_CHUNK;
            for (offset, slot) in out.iter_mut().enumerate() {
                *slot = interpolate_edge(
                    unique_keys[base + offset],
                    origin,
                    dims,
                    voxel_size,
                    values,
                    iso_value,
                );
            }
        },
    );

    // Stage 3: Resolve active cube triangles in deterministic slice-major chunk order.
    let tri_parts = par::map_chunks(&active_cubes, WORK_CHUNK, sub_threshold, oom, |_, slice| {
        let part_tris: usize = slice
            .iter()
            .map(|&(_, mask)| CASES[mask as usize].tri_count as usize)
            .sum();
        let mut out = Vec::new();
        reserve(&mut out, part_tris)?;

        for &(cube_idx, mask) in slice {
            let ix = cube_idx % cx;
            let rem = cube_idx / cx;
            let iy = rem % cy;
            let iz = rem / cy;

            let case = &CASES[mask as usize];
            let mut local_verts = [0u32; 12];
            let mut em = case.edge_mask;
            while em != 0 {
                let e = em.trailing_zeros() as usize;
                em &= em - 1;
                let (dx0, dy0, dz0, axis) = EDGE_DESC[e];
                let gx = ix + dx0 as usize;
                let gy = iy + dy0 as usize;
                let gz = iz + dz0 as usize;
                let node = (gz * ny + gy) * nx + gx;
                let key = ((node as u64) << 2) | (axis as u64);
                let vid = unique_keys
                    .binary_search(&key)
                    .expect("active edge key must be present") as u32;
                local_verts[e] = vid;
            }

            for tri in &case.tris[..case.tri_count as usize] {
                out.push([
                    local_verts[tri[0] as usize],
                    local_verts[tri[1] as usize],
                    local_verts[tri[2] as usize],
                ]);
            }
        }
        Ok(out)
    })?;

    let mut triangles = Vec::new();
    reserve(&mut triangles, total_tris)?;
    for part in &tri_parts {
        triangles.extend_from_slice(part);
    }

    Ok(IsoMesh {
        positions,
        triangles,
    })
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use fast_mesh_bvh::{BuildOptions, Bvh};
    use fast_mesh_store::{Mesh, StoreError, par};

    use super::*;
    use crate::test_util::{assert_parallel_identical, mesh_of};
    use crate::{AnalyzeOptions, OpsError, analyze, fail_allocations_after, measure};

    fn iso_to_store_mesh(iso: &IsoMesh) -> Mesh {
        let flat: Vec<f32> = iso.positions.iter().flatten().copied().collect();
        mesh_of(&flat, &iso.triangles)
    }

    fn xorshift64(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    #[test]
    fn marching_cubes_all_adjacent_face_pairs_and_exact_iso_ties_are_manifold() {
        // 1. Exhaustively test all 256 single-cube cases on a 2x2x2 lattice.
        for mask in 0..=255u8 {
            let mut values = [1.0f32; 8];
            for (c, slot) in values.iter_mut().enumerate() {
                if (mask >> c) & 1 == 1 {
                    *slot = -1.0;
                }
            }
            let iso = marching_cubes_raw([0.0; 3], [2, 2, 2], 1.0, &values, 0.0).unwrap();
            if mask == 0 || mask == 255 {
                assert!(iso.positions.is_empty(), "mask {mask} should have 0 verts");
                assert!(iso.triangles.is_empty(), "mask {mask} should have 0 tris");
                continue;
            }
            assert!(
                !iso.triangles.is_empty(),
                "active mask {mask} must emit at least one triangle"
            );
            let mesh = iso_to_store_mesh(&iso);
            let bvh = Bvh::build(&mesh, &BuildOptions::default()).unwrap();
            let a = analyze(
                &mesh,
                &AnalyzeOptions::builder().self_intersection(&bvh).build(),
            )
            .unwrap();
            assert_eq!(a.degenerate, 0, "mask {mask}: degenerate triangles");
            assert_eq!(a.duplicate, 0, "mask {mask}: duplicate triangles");
            assert_eq!(a.coincident_groups, 0, "mask {mask}: coincident vertices");
            assert_eq!(a.non_manifold_edges, 0, "mask {mask}: non-manifold edges");
            assert!(
                a.non_manifold_vertices.is_empty(),
                "mask {mask}: non-manifold vertices"
            );
            assert!(a.pinch_vertices.is_empty(), "mask {mask}: pinch vertices");
            assert_eq!(
                a.orientation_conflicts, 0,
                "mask {mask}: orientation conflicts"
            );
            assert_eq!(
                a.self_intersects,
                Some(false),
                "mask {mask}: single-cube self-intersection"
            );
        }

        // 2. Exhaustively test all 256 interior 2x2x2 node configurations inside a
        // padded 4x4x4 grid (3x3x3 = 27 cubes), verifying closed 2-manifold output
        // and shared-face half-edge cancellation across X, Y, and Z faces.
        let dims4 = [4, 4, 4];
        for mask in 1..255u8 {
            let mut values = vec![1.0f32; 64];
            for c in 0..8usize {
                if (mask >> c) & 1 == 1 {
                    let ix = 1 + (c & 1);
                    let iy = 1 + ((c >> 1) & 1);
                    let iz = 1 + ((c >> 2) & 1);
                    values[(iz * 4 + iy) * 4 + ix] = -1.0;
                }
            }
            let iso = marching_cubes_raw([-1.0; 3], dims4, 1.0, &values, 0.0).unwrap();
            assert!(!iso.triangles.is_empty());
            let mesh = iso_to_store_mesh(&iso);
            let bvh = Bvh::build(&mesh, &BuildOptions::default()).unwrap();
            let a = analyze(
                &mesh,
                &AnalyzeOptions::builder().self_intersection(&bvh).build(),
            )
            .unwrap();
            assert!(a.loops.is_empty(), "2x2x2 mask {mask}: boundary loops");
            assert_eq!(a.non_manifold_edges, 0, "2x2x2 mask {mask}: non-manifold");
            assert!(
                a.non_manifold_vertices.is_empty(),
                "2x2x2 mask {mask}: non-manifold vertices"
            );
            assert!(
                a.pinch_vertices.is_empty(),
                "2x2x2 mask {mask}: pinch vertices"
            );
            assert_eq!(a.degenerate, 0, "2x2x2 mask {mask}: degenerate");
            assert_eq!(a.duplicate, 0, "2x2x2 mask {mask}: duplicate");
            assert_eq!(
                a.orientation_conflicts, 0,
                "2x2x2 mask {mask}: orientation conflicts"
            );
            assert_eq!(
                a.self_intersects,
                Some(false),
                "2x2x2 mask {mask}: self-intersection"
            );
            let m = measure(&mesh).unwrap();
            assert!(m.closed, "2x2x2 mask {mask}: must be closed");
            assert!(
                m.signed_volume > 0.0,
                "2x2x2 mask {mask}: signed volume {} must be positive",
                m.signed_volume
            );
        }

        // 3. Test 64 random 6x6x6 ternary {-1.0, 0.0, +1.0} grids with exact iso_value = 0.0
        // ties on grid nodes (including outer boundary = 0.0 ties on half the runs).
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let dims6 = [6, 6, 6];
        let mut non_empty_runs = 0usize;
        for run in 0..64usize {
            let border = if run % 2 == 0 { 1.0f32 } else { 0.0f32 };
            let mut values = vec![border; 216];
            for iz in 1..5 {
                for iy in 1..5 {
                    for ix in 1..5 {
                        let pick = match xorshift64(&mut seed) % 3 {
                            0 => -1.0f32,
                            1 => 0.0f32,
                            _ => 1.0f32,
                        };
                        values[(iz * 6 + iy) * 6 + ix] = pick;
                    }
                }
            }
            let iso = marching_cubes_raw([0.0; 3], dims6, 0.5, &values, 0.0).unwrap();
            if iso.triangles.is_empty() {
                assert!(iso.positions.is_empty());
                continue;
            }
            non_empty_runs += 1;
            let mesh = iso_to_store_mesh(&iso);
            let bvh = Bvh::build(&mesh, &BuildOptions::default()).unwrap();
            let a = analyze(
                &mesh,
                &AnalyzeOptions::builder().self_intersection(&bvh).build(),
            )
            .unwrap();
            assert!(a.loops.is_empty(), "run {run}: open boundary loops");
            assert_eq!(a.non_manifold_edges, 0, "run {run}: non-manifold edges");
            assert!(
                a.non_manifold_vertices.is_empty(),
                "run {run}: non-manifold vertices"
            );
            assert!(a.pinch_vertices.is_empty(), "run {run}: pinch vertices");
            assert_eq!(a.degenerate, 0, "run {run}: degenerate triangles");
            assert_eq!(a.duplicate, 0, "run {run}: duplicate triangles");
            assert_eq!(a.coincident_groups, 0, "run {run}: coincident vertices");
            assert_eq!(
                a.orientation_conflicts, 0,
                "run {run}: orientation conflicts"
            );
            assert_eq!(
                a.self_intersects,
                Some(false),
                "run {run}: self-intersection"
            );
            let m = measure(&mesh).unwrap();
            assert!(m.closed, "run {run}: must be watertight");
            assert!(
                m.signed_volume > 0.0,
                "run {run}: outward signed volume expected, got {}",
                m.signed_volume
            );
        }
        assert!(non_empty_runs >= 60);

        // 4. Verify clamping t in [1e-4, 1.0 - 1e-4] when grid nodes hit or nearly hit iso_value.
        let mut clamp_vals = [1.0f32; 8];
        clamp_vals[0] = -1e-20; // near 0 from inside vs 1.0 outside -> t clamped up to 1e-4
        let clamp_lo = marching_cubes_raw([0.0; 3], [2, 2, 2], 10.0, &clamp_vals, 0.0).unwrap();
        assert_eq!(clamp_lo.positions.len(), 3);
        for p in &clamp_lo.positions {
            let max_c = p[0].max(p[1]).max(p[2]);
            assert!(
                (max_c - 1e-3).abs() < 1e-6,
                "expected clamped coordinate 1e-4 * 10.0 = 1e-3, got {p:?}"
            );
        }

        let mut clamp_hi_vals = [0.0f32; 8];
        clamp_hi_vals[0] = -1.0; // corner 0 = -1.0, neighbours = 0.0 == iso_value -> t = 1.0 clamped to 1 - 1e-4
        let clamp_hi = marching_cubes_raw([0.0; 3], [2, 2, 2], 10.0, &clamp_hi_vals, 0.0).unwrap();
        assert_eq!(clamp_hi.positions.len(), 3);
        for p in &clamp_hi.positions {
            let max_c = p[0].max(p[1]).max(p[2]);
            assert!(
                (max_c - (10.0 - 1e-3)).abs() < 1e-5,
                "expected clamped coordinate (1 - 1e-4) * 10.0, got {p:?}"
            );
        }
    }

    #[test]
    fn marching_cubes_sphere_volume_and_area_converge() {
        let radius = 1.0f64;
        let expected_vol = 4.0 / 3.0 * PI * radius.powi(3);
        let expected_area = 4.0 * PI * radius.powi(2);

        let sample_sphere = |n: usize| -> (IsoMesh, f64, f64) {
            let span = 2.8f32;
            let origin = [-1.4f32; 3];
            let h = span / ((n - 1) as f32);
            let mut values = Vec::with_capacity(n * n * n);
            for iz in 0..n {
                let z = f64::from(origin[2]) + (iz as f64) * f64::from(h);
                for iy in 0..n {
                    let y = f64::from(origin[1]) + (iy as f64) * f64::from(h);
                    for ix in 0..n {
                        let x = f64::from(origin[0]) + (ix as f64) * f64::from(h);
                        let dist = (x * x + y * y + z * z).sqrt();
                        values.push((dist - radius) as f32);
                    }
                }
            }
            let iso = marching_cubes_raw(origin, [n, n, n], h, &values, 0.0).unwrap();
            let mesh = iso_to_store_mesh(&iso);
            let m = measure(&mesh).unwrap();
            assert!(m.closed);
            assert!(m.signed_volume > 0.0);
            let vol_err = ((m.signed_volume - expected_vol) / expected_vol).abs();
            let area_err = ((m.area - expected_area) / expected_area).abs();
            (iso, vol_err, area_err)
        };

        let (_, coarse_vol_err, coarse_area_err) = sample_sphere(24);
        let (fine_iso, fine_vol_err, fine_area_err) = sample_sphere(48);

        assert!(
            fine_vol_err < 0.02,
            "fine sphere volume error {fine_vol_err} exceeds 2%"
        );
        assert!(
            fine_area_err < 0.03,
            "fine sphere area error {fine_area_err} exceeds 3%"
        );
        assert!(
            fine_vol_err < coarse_vol_err,
            "volume error did not converge: {fine_vol_err} vs {coarse_vol_err}"
        );
        assert!(
            fine_area_err < coarse_area_err,
            "area error did not converge: {fine_area_err} vs {coarse_area_err}"
        );

        let mesh = iso_to_store_mesh(&fine_iso);
        let bvh = Bvh::build(&mesh, &BuildOptions::default()).unwrap();
        let a = analyze(
            &mesh,
            &AnalyzeOptions::builder().self_intersection(&bvh).build(),
        )
        .unwrap();
        assert!(a.loops.is_empty(), "sphere must have 0 boundary edges");
        assert_eq!(a.non_manifold_edges, 0);
        assert!(a.non_manifold_vertices.is_empty());
        assert!(a.pinch_vertices.is_empty());
        assert_eq!(a.degenerate, 0);
        assert_eq!(a.duplicate, 0);
        assert_eq!(a.orientation_conflicts, 0);
        assert_eq!(a.self_intersects, Some(false));
        assert_eq!(a.components.len(), 1, "sphere must be 1 component");

        // Genus check: V - E + F = 2 - 2g, with 2E = 3F on a closed manifold triangle mesh.
        let v = fine_iso.positions.len() as isize;
        let f = fine_iso.triangles.len() as isize;
        assert_eq!((3 * f) % 2, 0);
        let e = (3 * f) / 2;
        let euler = v - e + f;
        assert_eq!(euler, 2, "sphere Euler characteristic must be 2 (genus 0)");

        // Asymmetric dims [18, 22, 26], non-zero origin, and non-zero iso_value = 0.25
        // around an off-center sphere at [2.0, -3.0, 5.0] of base radius 0.55 (effective R = 0.8).
        let asym_dims = [18, 22, 26];
        let asym_origin = [0.8f32, -4.4f32, 3.4f32];
        let asym_h = 0.14f32;
        let center = [2.0f64, -3.0f64, 5.0f64];
        let mut asym_vals = Vec::with_capacity(asym_dims[0] * asym_dims[1] * asym_dims[2]);
        for iz in 0..asym_dims[2] {
            let z = f64::from(asym_origin[2]) + (iz as f64) * f64::from(asym_h);
            for iy in 0..asym_dims[1] {
                let y = f64::from(asym_origin[1]) + (iy as f64) * f64::from(asym_h);
                for ix in 0..asym_dims[0] {
                    let x = f64::from(asym_origin[0]) + (ix as f64) * f64::from(asym_h);
                    let d = ((x - center[0]).powi(2)
                        + (y - center[1]).powi(2)
                        + (z - center[2]).powi(2))
                    .sqrt();
                    asym_vals.push((d - 0.55) as f32);
                }
            }
        }
        let asym_iso =
            marching_cubes_raw(asym_origin, asym_dims, asym_h, &asym_vals, 0.25).unwrap();
        let asym_mesh = iso_to_store_mesh(&asym_iso);
        let asym_m = measure(&asym_mesh).unwrap();
        assert!(asym_m.closed && asym_m.signed_volume > 0.0);
        let r_eff = 0.8f64;
        let want_vol = 4.0 / 3.0 * PI * r_eff.powi(3);
        assert!(((asym_m.signed_volume - want_vol) / want_vol).abs() < 0.03);
        let mean = asym_iso
            .positions
            .iter()
            .fold([0.0f64; 3], |acc, p| {
                [
                    acc[0] + f64::from(p[0]),
                    acc[1] + f64::from(p[1]),
                    acc[2] + f64::from(p[2]),
                ]
            })
            .map(|s| s / (asym_iso.positions.len() as f64));
        for k in 0..3 {
            assert!(
                (mean[k] - center[k]).abs() < 0.05,
                "axis {k} centroid {} far from {}",
                mean[k],
                center[k]
            );
        }
    }

    #[test]
    fn marching_cubes_rejects_bad_inputs_and_honours_oom_hook() {
        let valid_vals = [0.0f32; 8];

        // Non-finite origin.
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(
                marching_cubes_raw([bad, 0.0, 0.0], [2, 2, 2], 1.0, &valid_vals, 0.0),
                Err(OpsError::InvalidInput(_))
            ));
            assert!(matches!(
                marching_cubes_raw([0.0, bad, 0.0], [2, 2, 2], 1.0, &valid_vals, 0.0),
                Err(OpsError::InvalidInput(_))
            ));
            assert!(matches!(
                marching_cubes_raw([0.0, 0.0, bad], [2, 2, 2], 1.0, &valid_vals, 0.0),
                Err(OpsError::InvalidInput(_))
            ));
        }

        // Non-finite or non-positive voxel_size.
        for bad_h in [0.0, -0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(
                marching_cubes_raw([0.0; 3], [2, 2, 2], bad_h, &valid_vals, 0.0),
                Err(OpsError::InvalidInput(_))
            ));
        }

        // NaN iso_value.
        assert!(matches!(
            marching_cubes_raw([0.0; 3], [2, 2, 2], 1.0, &valid_vals, f32::NAN),
            Err(OpsError::InvalidInput(_))
        ));

        // dims[k] < 2.
        for bad_dims in [[1, 2, 2], [2, 1, 2], [2, 2, 1], [0, 2, 2]] {
            let len = bad_dims[0] * bad_dims[1] * bad_dims[2];
            let buf = vec![0.0f32; len];
            assert!(matches!(
                marching_cubes_raw([0.0; 3], bad_dims, 1.0, &buf, 0.0),
                Err(OpsError::InvalidInput(_))
            ));
        }

        // dims product overflow and length mismatch.
        assert!(matches!(
            marching_cubes_raw([0.0; 3], [usize::MAX, 2, 2], 1.0, &valid_vals, 0.0),
            Err(OpsError::InvalidInput(_))
        ));
        assert!(matches!(
            marching_cubes_raw([0.0; 3], [2, 2, 2], 1.0, &valid_vals[..7], 0.0),
            Err(OpsError::InvalidInput(_))
        ));
        assert!(matches!(
            marching_cubes_raw([0.0; 3], [2, 2, 2], 1.0, &[0.0f32; 9], 0.0),
            Err(OpsError::InvalidInput(_))
        ));

        // Upper grid corner overflowing f32.
        assert!(matches!(
            marching_cubes_raw(
                [f32::MAX * 0.75, 0.0, 0.0],
                [4, 2, 2],
                f32::MAX * 0.25,
                &[0.0f32; 16],
                0.0
            ),
            Err(OpsError::InvalidInput(_))
        ));

        // NaN element in values (even when all other nodes are outside).
        let mut with_nan = [10.0f32; 8];
        with_nan[7] = f32::NAN;
        assert!(matches!(
            marching_cubes_raw([0.0; 3], [2, 2, 2], 1.0, &with_nan, 0.0),
            Err(OpsError::InvalidInput(_))
        ));

        // Non-NaN infinities in values (e.g. unvisited EDT nodes) are valid.
        let all_inf = [f32::INFINITY; 8];
        let empty = marching_cubes_raw([0.0; 3], [2, 2, 2], 1.0, &all_inf, 1.0).unwrap();
        assert!(empty.positions.is_empty() && empty.triangles.is_empty());

        let mut one_neg_inf = [f32::INFINITY; 8];
        one_neg_inf[0] = f32::NEG_INFINITY;
        let inf_iso = marching_cubes_raw([0.0; 3], [2, 2, 2], 1.0, &one_neg_inf, 0.0).unwrap();
        assert_eq!(inf_iso.positions.len(), 3);
        assert_eq!(inf_iso.triangles.len(), 1);
        assert!(
            inf_iso
                .positions
                .iter()
                .flatten()
                .all(|coord| coord.is_finite())
        );

        // OOM sweep via fail_allocations_after.
        let n_grid = 12usize;
        let h = 2.4f32 / ((n_grid - 1) as f32);
        let mut sphere_vals = Vec::with_capacity(n_grid * n_grid * n_grid);
        for iz in 0..n_grid {
            let z = -1.2 + (iz as f32) * h;
            for iy in 0..n_grid {
                let y = -1.2 + (iy as f32) * h;
                for ix in 0..n_grid {
                    let x = -1.2 + (ix as f32) * h;
                    sphere_vals.push((x * x + y * y + z * z).sqrt() - 0.8);
                }
            }
        }
        let expected =
            marching_cubes_raw([-1.2; 3], [n_grid, n_grid, n_grid], h, &sphere_vals, 0.0).unwrap();
        assert!(!expected.triangles.is_empty());

        let mut step = 0usize;
        loop {
            fail_allocations_after(Some(step));
            let res = marching_cubes_raw([-1.2; 3], [n_grid, n_grid, n_grid], h, &sphere_vals, 0.0);
            fail_allocations_after(None);
            match res {
                Ok(got) => {
                    assert!(step > 0, "must perform at least one fallible reservation");
                    assert_eq!(got, expected);
                    break;
                }
                Err(e) => {
                    assert_eq!(
                        e,
                        OpsError::Store(StoreError::OutOfMemory),
                        "unexpected error at allocation {step}"
                    );
                    step += 1;
                    assert!(step < 10_000, "OOM sweep did not terminate");
                }
            }
        }
    }

    #[test]
    fn marching_cubes_parallel_bit_identical() {
        // Small grid (< 16_384 cubes) stays sequential.
        let small_n = 10usize;
        let mut small_vals = vec![1.0f32; small_n * small_n * small_n];
        small_vals[(5 * small_n + 5) * small_n + 5] = -1.0;
        let before = par::parallel_entries();
        let small_iso = marching_cubes_raw([0.0; 3], [small_n; 3], 0.25, &small_vals, 0.0).unwrap();
        assert!(!small_iso.triangles.is_empty());
        assert_eq!(
            par::parallel_entries(),
            before,
            "small grid (< 16_384 cubes) must stay sequential"
        );

        // Large grid (36^3 nodes -> 35^3 = 42_875 >= 16_384 cubes) takes parallel path
        // and produces bit-for-bit identical positions and triangles across 1, 2, 8 threads.
        let n = 36usize;
        let origin = [-1.5f32; 3];
        let h = 3.0f32 / ((n - 1) as f32);
        let mut values = Vec::with_capacity(n * n * n);
        for iz in 0..n {
            let z = f64::from(origin[2]) + (iz as f64) * f64::from(h);
            for iy in 0..n {
                let y = f64::from(origin[1]) + (iy as f64) * f64::from(h);
                for ix in 0..n {
                    let x = f64::from(origin[0]) + (ix as f64) * f64::from(h);
                    // Torus + sphere blend with occasional exact 0.0 ties.
                    let r_xy = (x * x + y * y).sqrt() - 0.75;
                    let d_torus = (r_xy * r_xy + z * z).sqrt() - 0.32;
                    let d_sphere = ((x - 0.3).powi(2) + y * y + z * z).sqrt() - 0.55;
                    let v = d_torus.min(d_sphere) as f32;
                    values.push(if v.abs() < 1e-4 { 0.0 } else { v });
                }
            }
        }

        let reference = assert_parallel_identical("marching_cubes_raw", || {
            let iso = marching_cubes_raw(origin, [n, n, n], h, &values, 0.0).unwrap();
            let pos_bits: Vec<[u32; 3]> = iso
                .positions
                .iter()
                .map(|p| [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()])
                .collect();
            (pos_bits, iso.triangles)
        });
        assert!(!reference.0.is_empty());
        assert!(!reference.1.is_empty());
    }

    #[test]
    fn marching_cubes_from_scalar_grid_matches_raw() {
        let spec = fast_mesh_fields::GridSpec {
            origin: [-1.0, -1.0, -1.0],
            dims: [12, 12, 12],
            voxel_size: 2.0 / 11.0,
        };
        let mut vals = Vec::with_capacity(12 * 12 * 12);
        for iz in 0..12 {
            for iy in 0..12 {
                for ix in 0..12 {
                    let p = spec.point(ix, iy, iz);
                    vals.push((p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() - 0.55);
                }
            }
        }
        let grid = fast_mesh_fields::ScalarGrid::new(
            spec,
            vals.clone(),
            fast_mesh_fields::DEFAULT_MAX_VOXELS,
        )
        .unwrap();
        let from_grid = marching_cubes(&grid, 0.0).unwrap();
        let from_raw =
            marching_cubes_raw(spec.origin, spec.dims, spec.voxel_size, &vals, 0.0).unwrap();
        assert_eq!(from_grid, from_raw);
        assert!(!from_grid.triangles.is_empty());
    }
}
