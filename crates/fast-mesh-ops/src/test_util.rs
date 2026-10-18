//! Fixtures shared by the repair tests.

use fast_mesh_store::{Mesh, StoreError, Transaction, TriangleId};

use crate::{OpsError, Result, fail_allocations_after};

/// Runs `f` in a transaction and commits; returns its result and whether
/// the patch was empty.
pub(crate) fn edit<R>(mesh: &mut Mesh, f: impl FnOnce(&mut Transaction) -> Result<R>) -> (R, bool) {
    let mut tx = mesh.begin_edit();
    let r = f(&mut tx).unwrap();
    (r, tx.commit().unwrap().is_empty())
}

/// Runs `op` on `mesh` with the `n`-th reservation failing, for every `n`
/// until it succeeds; each failure must leave the transaction without a
/// single write, and the success must write. Returns how many failure points
/// there were.
pub(crate) fn sweep(mut mesh: Mesh, mut op: impl FnMut(&mut Transaction) -> Result<()>) -> usize {
    let before = parts(&mesh);
    let mut n = 0;
    loop {
        let mut tx = mesh.begin_edit();
        fail_allocations_after(Some(n));
        let r = op(&mut tx);
        fail_allocations_after(None);
        let patch = tx.commit().unwrap();
        if r.is_ok() {
            assert!(!patch.is_empty());
            return n;
        }
        assert_eq!(r, Err(OpsError::Store(StoreError::OutOfMemory)));
        assert!(patch.is_empty(), "write before failure point {n}");
        assert_eq!(parts(&mesh), before);
        n += 1;
    }
}

/// `positions` (flat `xyz`) and `triangles` as a mesh.
pub(crate) fn mesh_of(positions: &[f32], triangles: &[[u32; 3]]) -> Mesh {
    let indices: Vec<u32> = triangles.iter().flatten().copied().collect();
    Mesh::from_buffers(positions, &indices).unwrap()
}

/// Every vertex slot's local position (flat `xyz`) and the alive triangles.
/// Vertex ids stay the same; the origin is dropped.
pub(crate) fn parts(mesh: &Mesh) -> (Vec<f32>, Vec<[u32; 3]>) {
    let positions = mesh.positions().iter().flatten().copied().collect();
    let triangles = (0..mesh.triangle_slots() as u32)
        .filter_map(|t| mesh.triangle(t))
        .collect();
    (positions, triangles)
}

/// The meshes side by side in one mesh, the vertex ids of each offset past
/// those of the meshes before it.
pub(crate) fn combine(meshes: &[&Mesh]) -> Mesh {
    let (mut positions, mut triangles) = (Vec::new(), Vec::new());
    for m in meshes {
        let (p, t) = parts(m);
        let base = (positions.len() / 3) as u32;
        positions.extend(p);
        triangles.extend(t.iter().map(|t| t.map(|v| v + base)));
    }
    mesh_of(&positions, &triangles)
}

/// `mesh` with `extra` triangles appended.
pub(crate) fn with_triangles(mesh: &Mesh, extra: &[[u32; 3]]) -> Mesh {
    let (positions, mut triangles) = parts(mesh);
    triangles.extend_from_slice(extra);
    mesh_of(&positions, &triangles)
}

/// Positions and triangles of a UV sphere cap from the north pole down to
/// polar angle `PI * last / stacks`, with outward winding: one vertex at the
/// pole and `last` rings of `slices` vertices.
fn cap(stacks: u32, last: u32, slices: u32, r: f32) -> (Vec<f32>, Vec<[u32; 3]>) {
    use std::f64::consts::PI;
    let mut positions = vec![0.0, 0.0, r];
    for i in 1..=last {
        let theta = PI * f64::from(i) / f64::from(stacks);
        for j in 0..slices {
            let phi = 2.0 * PI * f64::from(j) / f64::from(slices);
            let (rs, rc) = (f64::from(r) * theta.sin(), f64::from(r) * theta.cos());
            positions.extend([(rs * phi.cos()) as f32, (rs * phi.sin()) as f32, rc as f32]);
        }
    }
    let ring = |i: u32, j: u32| 1 + (i - 1) * slices + j % slices;
    let mut triangles = Vec::new();
    for j in 0..slices {
        triangles.push([0, ring(1, j), ring(1, j + 1)]);
        for i in 1..last {
            let (a, b, c, d) = (
                ring(i, j),
                ring(i + 1, j),
                ring(i + 1, j + 1),
                ring(i, j + 1),
            );
            triangles.extend([[a, b, c], [a, c, d]]);
        }
    }
    (positions, triangles)
}

/// A closed UV sphere around the origin with outward winding: one vertex at
/// each pole and `stacks - 1` rings of `slices` vertices.
pub(crate) fn uv_sphere(stacks: u32, slices: u32, r: f32) -> Mesh {
    let (mut positions, mut triangles) = cap(stacks, stacks - 1, slices, r);
    let south = (positions.len() / 3) as u32;
    positions.extend([0.0, 0.0, -r]);
    let ring = |j: u32| 1 + (stacks - 2) * slices + j % slices;
    for j in 0..slices {
        triangles.push([ring(j), south, ring(j + 1)]);
    }
    mesh_of(&positions, &triangles)
}

/// The open upper half of a UV sphere around the origin with outward
/// winding: `stacks` rings of `slices` vertices below the north pole, the
/// last on the equator `z = 0`, which is the one boundary loop.
pub(crate) fn hemisphere(stacks: u32, slices: u32, r: f32) -> Mesh {
    let (positions, triangles) = cap(2 * stacks, stacks, slices, r);
    mesh_of(&positions, &triangles)
}

/// The triangles of cell `(x, y)` of [`grid`]`(n)`, counter-clockwise from `+z`.
fn grid_cell(n: u32, x: u32, y: u32) -> [[u32; 3]; 2] {
    let a = y * (n + 1) + x;
    [[a, a + 1, a + n + 2], [a, a + n + 2, a + n + 1]]
}

/// The positions of a `(n + 1) x (n + 1)` grid of vertices at integer `x`, `y`
/// and `z = 0`; vertex `(x, y)` has id `y * (n + 1) + x`.
fn grid_positions(n: u32) -> Vec<f32> {
    let mut positions = Vec::new();
    for y in 0..=n {
        for x in 0..=n {
            positions.extend([x as f32, y as f32, 0.0]);
        }
    }
    positions
}

/// A flat `n x n` grid of unit cells at `z = 0` (see [`grid_positions`]), two
/// triangles per cell, counter-clockwise from `+z`. Cell `(x, y)` holds
/// triangles `2 * (y * n + x)` and the one after it.
pub(crate) fn grid(n: u32) -> Mesh {
    let mut triangles = Vec::new();
    for y in 0..n {
        for x in 0..n {
            triangles.extend(grid_cell(n, x, y));
        }
    }
    mesh_of(&grid_positions(n), &triangles)
}

/// [`grid`]`(4)` without cells `(1, 1)` and `(2, 2)`: two square holes that
/// share exactly one corner, vertex 12 at `(2, 2)`. All 25 vertices stay.
pub(crate) fn bowtie_holes() -> Mesh {
    grid_without(4, &[(1, 1), (2, 2)])
}

/// [`grid`]`(n)` without the cells in `missing`. All vertices stay.
pub(crate) fn grid_without(n: u32, missing: &[(u32, u32)]) -> Mesh {
    let mut triangles = Vec::new();
    for y in 0..n {
        for x in 0..n {
            if !missing.contains(&(x, y)) {
                triangles.extend(grid_cell(n, x, y));
            }
        }
    }
    mesh_of(&grid_positions(n), &triangles)
}

/// A closed band of `2n` triangles between two rings of `n` vertices:
/// vertex `i` is `inner(t_i)` and vertex `n + i` is `outer(t_i)`, with
/// `t_i = i / n`. Quad `i` is `[i, n + i, n + i + 1]`, `[i, n + i + 1, i + 1]`
/// (indices wrapping), counter-clockwise from `+z` for a flat ring with
/// `inner` inside `outer`; the inner loop then runs clockwise from `+z`.
pub(crate) fn band(
    n: u32,
    inner: impl Fn(f64) -> [f64; 3],
    outer: impl Fn(f64) -> [f64; 3],
) -> Mesh {
    let mut positions = Vec::new();
    for ring in [&inner as &dyn Fn(f64) -> [f64; 3], &outer] {
        for i in 0..n {
            positions.extend(ring(f64::from(i) / f64::from(n)).map(|c| c as f32));
        }
    }
    let mut triangles = Vec::new();
    for i in 0..n {
        let j = (i + 1) % n;
        triangles.extend([[i, n + i, n + j], [i, n + j, j]]);
    }
    mesh_of(&positions, &triangles)
}

/// `mesh` with a closed tetrahedron on its edge `a`-`b` and two new vertices
/// at `apex`, so that `a`-`b` gains two triangles. The tetrahedron is wound
/// consistently with itself.
pub(crate) fn with_tetra(mesh: &Mesh, a: u32, b: u32, apex: [[f32; 3]; 2]) -> Mesh {
    let (mut positions, mut triangles) = parts(mesh);
    let c = (positions.len() / 3) as u32;
    let d = c + 1;
    positions.extend(apex.iter().flatten());
    triangles.extend([[a, b, c], [a, c, d], [a, d, b], [b, d, c]]);
    mesh_of(&positions, &triangles)
}

/// A flat hexagonal patch around vertex 0 with three triangular holes that
/// all touch vertex 0: the fan triangles `[0, r_k, r_(k+1)]` for odd `k` are
/// missing. Ring 1 is `r_k = 1 + k` at radius 1 and angle `60k` degrees; ring
/// 2 is `7 + 2k` at radius 2 and angle `60k`, and `8 + 2k` at angle `60k + 30`.
pub(crate) fn tri_pinch() -> Mesh {
    tri_pinch_lifted(|_, _| 0.0)
}

/// [`tri_pinch`] with each vertex at radius `r` and angle `deg` (degrees)
/// lifted to `z = lift(r, deg)`.
pub(crate) fn tri_pinch_lifted(lift: fn(f64, f64) -> f64) -> Mesh {
    let mut positions = vec![0.0f32, 0.0, 0.0];
    let at = |r: f64, deg: f64| {
        [
            (r * deg.to_radians().cos()) as f32,
            (r * deg.to_radians().sin()) as f32,
            lift(r, deg) as f32,
        ]
    };
    for k in 0..6 {
        positions.extend(at(1.0, 60.0 * f64::from(k)));
    }
    for k in 0..6 {
        positions.extend(at(2.0, 60.0 * f64::from(k)));
        positions.extend(at(2.0, 60.0 * f64::from(k) + 30.0));
    }
    let r = |k: u32| 1 + k % 6;
    let (s, m) = (|k: u32| 7 + 2 * (k % 6), |k: u32| 8 + 2 * (k % 6));
    let mut triangles = Vec::new();
    for k in 0..6 {
        if k % 2 == 0 {
            triangles.push([0, r(k), r(k + 1)]);
        }
        triangles.extend([
            [r(k), s(k), m(k)],
            [r(k), m(k), r(k + 1)],
            [r(k + 1), m(k), s(k + 1)],
        ]);
    }
    mesh_of(&positions, &triangles)
}

/// Three quads joined along one edge, from vertex 0 at the origin to vertex 1
/// at `(0, 0, 1)`: a non-manifold fin. Quad `k` is `[0, p, q]`, `[0, q, 1]`
/// with `p = 2 + 2k` and `q = 3 + 2k`.
pub(crate) fn fin() -> Mesh {
    let mut positions = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let mut triangles = Vec::new();
    for k in 0..3u32 {
        let angle = 2.0 * std::f64::consts::PI * f64::from(k) / 3.0;
        let (x, y) = (angle.cos() as f32, angle.sin() as f32);
        positions.extend([x, y, 0.0, x, y, 1.0]);
        let (p, q) = (2 + 2 * k, 3 + 2 * k);
        triangles.extend([[0, p, q], [0, q, 1]]);
    }
    mesh_of(&positions, &triangles)
}

/// A Möbius strip of `n` quads around a circle of radius 2, half-width 0.5.
/// Vertices `2i` and `2i + 1` are the strip's two sides at step `i`; the last
/// quad joins step `n - 1` to step 0 with the sides swapped, so no consistent
/// orientation exists. Its boundary is one curve of `2n` edges.
pub(crate) fn mobius(n: u32) -> Mesh {
    use std::f64::consts::PI;
    let mut positions = Vec::new();
    for i in 0..n {
        let u = 2.0 * PI * f64::from(i) / f64::from(n);
        for w in [-0.5, 0.5] {
            let rr = 2.0 + w * (u / 2.0).cos();
            positions.extend([
                (rr * u.cos()) as f32,
                (rr * u.sin()) as f32,
                (w * (u / 2.0).sin()) as f32,
            ]);
        }
    }
    let mut triangles = Vec::new();
    for i in 0..n {
        let (a0, a1) = (2 * i, 2 * i + 1);
        let (b0, b1) = if i + 1 < n {
            (2 * i + 2, 2 * i + 3)
        } else {
            (1, 0)
        };
        triangles.extend([[a0, b0, b1], [a0, b1, a1]]);
    }
    mesh_of(&positions, &triangles)
}

/// `mesh` without the triangles whose centroid lies within `radius` of any of
/// `centers`. Vertex ids stay the same.
pub(crate) fn punch_holes(mesh: &Mesh, centers: &[[f32; 3]], radius: f32) -> Mesh {
    let (positions, triangles) = parts(mesh);
    let p = |v: u32| {
        let i = 3 * v as usize;
        [positions[i], positions[i + 1], positions[i + 2]]
    };
    let kept: Vec<[u32; 3]> = triangles
        .into_iter()
        .filter(|t| {
            let c: [f32; 3] =
                std::array::from_fn(|k| t.iter().map(|&v| p(v)[k]).sum::<f32>() / 3.0);
            centers
                .iter()
                .all(|o| (0..3).map(|k| (c[k] - o[k]).powi(2)).sum::<f32>() > radius * radius)
        })
        .collect();
    mesh_of(&positions, &kept)
}

/// `mesh` with triangles `tris` reversed (their last two vertices swapped).
pub(crate) fn flip(mesh: &Mesh, tris: &[TriangleId]) -> Mesh {
    let (positions, mut triangles) = parts(mesh);
    let alive: Vec<u32> = (0..mesh.triangle_slots() as u32)
        .filter(|&t| mesh.is_triangle_alive(t))
        .collect();
    for &t in tris {
        let i = alive.binary_search(&t).expect("flip: dead triangle");
        triangles[i].swap(1, 2);
    }
    mesh_of(&positions, &triangles)
}

/// The box `[lo, hi]` as 12 triangles with outward winding. Corner `i` takes
/// `hi` on axis `k` when bit `k` of `i` is set.
pub(crate) fn box_mesh(lo: [f32; 3], hi: [f32; 3]) -> Mesh {
    let mut positions = Vec::new();
    for i in 0..8 {
        positions.extend((0..3).map(|k| if i >> k & 1 == 1 { hi[k] } else { lo[k] }));
    }
    let triangles = [
        [0, 4, 6],
        [0, 6, 2], // -x
        [1, 3, 7],
        [1, 7, 5], // +x
        [0, 1, 5],
        [0, 5, 4], // -y
        [2, 6, 7],
        [2, 7, 3], // +y
        [0, 2, 3],
        [0, 3, 1], // -z
        [4, 5, 7],
        [4, 7, 6], // +z
    ];
    mesh_of(&positions, &triangles)
}

/// Two closed boxes in one mesh, `[0, 2]^3` and `[1, 3]^3`, that
/// interpenetrate without sharing a vertex.
pub(crate) fn crossed_boxes() -> Mesh {
    combine(&[&box_mesh([0.0; 3], [2.0; 3]), &box_mesh([1.0; 3], [3.0; 3])])
}

/// [`uv_sphere`]`(8, 12, 1.0)` (vertex ids `1 + 12 (i - 1) + j` for ring `i`,
/// column `j`; south pole 85) with known damage, every site sharing no vertex
/// with another. Quad `(i, j)` lies between rings `i`, `i + 1` and columns
/// `j`, `j + 1`; its first triangle is `[ring(i, j), ring(i + 1, j),
/// ring(i + 1, j + 1)]`, the one damaged.
/// - Holes: quads (1, 0), (3, 3) and (5, 6) lose their triangle, which
///   leaves three 3-vertex loops.
/// - Flips: quads (1, 3), (1, 6), (3, 9), (5, 0) and (5, 9) have their
///   triangle reversed, each against three neighbours.
/// - Duplicated vertices 86 to 89: copies of ring(i, j) for quads (1, 9),
///   (3, 0), (3, 6) and (5, 3), in that order, each taking that corner of
///   the quad's triangle. Each leaves a 4-vertex slit loop.
/// - A copy of south triangle `[77, 85, 78]`, corners rotated: three
///   non-manifold edges.
/// - Two triangles with a repeated vertex id, `[5, 5, 6]` and `[42, 45, 42]`.
/// - A floating unit square of two triangles on vertices 90 to 93.
///
/// The appended triangles end up as 165 (the copy), 166 and 167 (repeated
/// ids), and 168 and 169 (the square).
pub(crate) fn damaged_sphere() -> Mesh {
    let (mut positions, mut triangles) = parts(&uv_sphere(8, 12, 1.0));
    let quad = |i: usize, j: usize| j * 13 + 1 + 2 * (i - 1);
    for (i, j) in [(1, 3), (1, 6), (3, 9), (5, 0), (5, 9)] {
        triangles[quad(i, j)].swap(1, 2);
    }
    for (i, j) in [(1, 9), (3, 0), (3, 6), (5, 3)] {
        let t = &mut triangles[quad(i, j)];
        let v = 3 * t[0] as usize;
        t[0] = (positions.len() / 3) as u32;
        positions.extend_from_within(v..v + 3);
    }
    let [a, b, c] = triangles[156 + 4];
    triangles.push([b, c, a]);
    triangles.extend([[5, 5, 6], [42, 45, 42]]);
    let base = (positions.len() / 3) as u32;
    positions.extend([5.0, 0.0, 0.0, 6.0, 0.0, 0.0, 6.0, 1.0, 0.0, 5.0, 1.0, 0.0]);
    triangles.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]]);
    // Last, from the highest id down, so the other indices above stay valid.
    for (i, j) in [(5, 6), (3, 3), (1, 0)] {
        triangles.remove(quad(i, j));
    }
    mesh_of(&positions, &triangles)
}

/// Runs `run` with parallel work forced off (the code a feature-off build
/// runs), then on pools of 1, 2 and 8 threads and on the global pool,
/// asserting every run matches the first; with the `parallel` feature, also
/// that the global run took a parallel path. Returns the first run.
pub(crate) fn assert_parallel_identical<R: PartialEq + std::fmt::Debug + Send>(
    what: &str,
    run: impl Fn() -> R + Sync,
) -> R {
    use fast_mesh_store::par;
    par::set_force_sequential(true);
    let reference = run();
    par::set_force_sequential(false);
    if !par::enabled() {
        // Every run below would be this one again.
        return reference;
    }
    for k in [1, 2, 8] {
        let got = par::with_threads(k, &run);
        assert!(got == reference, "{what}: {k} threads");
    }
    let before = par::parallel_entries();
    assert!(run() == reference, "{what}: global pool");
    assert!(par::parallel_entries() > before, "{what}: ran sequentially");
    reference
}
