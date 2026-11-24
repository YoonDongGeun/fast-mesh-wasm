//! The voxel grid of a lattice: the clip distances, the field, its extraction
//! and the placing of the vertices on the clip surface.

use fast_mesh_bvh::Bvh;
use fast_mesh_fields::{GridSpec, SignData, signed_distances};
use fast_mesh_store::{Mesh, StoreError, par};

use super::surface::{LatticeKind, Trig};
use crate::topology::UnionFind;
use crate::{IsoMesh, OpsError, Result, allocation_allowed, filled, marching_cubes_raw, reserve};

/// Grid values closer than this many voxels to zero are moved off it, so that
/// every marching cubes vertex keeps away from the grid nodes and no triangle
/// collapses.
const NUDGE: f32 = 0.01;

/// Grid values are cut at this many voxels: past what any edge that changes
/// sign needs, so a vertex keeps away from the nodes.
const CUT: f32 = 2.0;

/// Nodes per side of the largest blocks whose centre decides for all of them.
const BLOCK: usize = 8;

/// Points per batch of signed distances.
const QUERY_BATCH: usize = 16_384;

/// Voxels from which a pass over the grid runs in parallel, slice by slice.
const PAR_MIN_VOXELS: usize = 16_384;

/// Rounds of placing a vertex on the clip surface.
const PLACE_ROUNDS: usize = 8;

/// How close to the clip surface a vertex is placed, in voxels.
const PLACED: f32 = 0.005;

/// A vertex this many voxels inside the clip surface is on a wall, where
/// interpolation is good enough.
const CLEAR: f32 = 0.25;

/// What is extracted of the region: its walls, or the rest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Extract {
    Walls,
    Void,
}

/// The walls: their surface and half their thickness.
#[derive(Debug, Clone, Copy)]
pub(super) struct Walls {
    pub kind: LatticeKind,
    pub cell_size: f64,
    pub thickness: f64,
}

/// The boundary of the region that is filled, wound outward.
pub(super) struct Boundary<'a> {
    pub mesh: &'a Mesh,
    pub bvh: &'a Bvh,
    pub sign: &'a SignData,
}

/// What to sample on which grid.
pub(super) struct Job<'a> {
    pub boundary: Boundary<'a>,
    pub spec: GridSpec,
    /// How far inside the boundary the lattice starts.
    pub depth: f32,
    pub walls: Walls,
    pub extract: Extract,
    /// Nodes within this many voxels of the clip surface get the exact
    /// distance; at least 1.
    pub band: f32,
    /// Cut the result at the grid's own faces, one voxel in.
    pub cropped: bool,
}

/// An extracted surface, and which of its vertices the clip surface placed.
pub(super) struct Extracted {
    pub iso: IsoMesh,
    /// The vertices next to the clip surface, ascending.
    pub at_clip: Vec<u32>,
}

impl Job<'_> {
    /// Samples the grid and extracts the surface of the result.
    pub(super) fn run(&self) -> Result<Extracted> {
        let total = self.spec.dims.iter().product();
        let mut grid: Vec<f32> = filled(total, 0.0)?;
        let mut near: Vec<u8> = filled(total, 0)?;
        self.clip(&mut grid, &mut near)?;
        self.cross(&mut grid)?;
        let spec = self.spec;
        let mut iso = marching_cubes_raw(spec.origin, spec.dims, spec.voxel_size, &grid, 0.0)?;
        let at_clip = self.place(&grid, &near, &mut iso)?;
        Ok(Extracted { iso, at_clip })
    }

    /// Fills `grid` with the depth minus the distance inside the boundary
    /// (negative where the lattice may be): exact within `band` voxels of
    /// zero, where `near` becomes 1, and [`CUT`] voxels of the right sign
    /// elsewhere.
    ///
    /// The distance changes no faster than the position, so a block whose
    /// centre is farther from the clip surface than its corners are from the
    /// centre lies on one side of it. Blocks that may reach the clip surface
    /// are halved, down to single nodes.
    fn clip(&self, grid: &mut [f32], near: &mut [u8]) -> Result<()> {
        let [nx, ny, nz] = self.spec.dims;
        let h = self.spec.voxel_size;
        let mut blocks: Vec<[u32; 3]> = Vec::new();
        for z in (0..nz).step_by(BLOCK) {
            for y in (0..ny).step_by(BLOCK) {
                for x in (0..nx).step_by(BLOCK) {
                    grow_push(&mut blocks, [x as u32, y as u32, z as u32])?;
                }
            }
        }
        let mut points: Vec<[f32; 3]> = Vec::new();
        reserve(&mut points, QUERY_BATCH)?;
        let mut values: Vec<f32> = filled(QUERY_BATCH, 0.0)?;
        let mut size = BLOCK;
        loop {
            // The centre's offset from the block's first node, and the
            // distance to its farthest one, with a margin for rounding.
            let half = 0.5 * (size - 1) as f32;
            let reach = (3f32.sqrt() * half + self.band) * h * 1.001;
            let mut next: Vec<[u32; 3]> = Vec::new();
            for batch in blocks.chunks(QUERY_BATCH) {
                points.clear();
                points.extend(batch.iter().map(|b| {
                    std::array::from_fn(|k| self.spec.origin[k] + (b[k] as f32 + half) * h)
                }));
                let values = &mut values[..batch.len()];
                self.clip_at(&points, values)?;
                for (&b, &c) in batch.iter().zip(values.iter()) {
                    let [x, y, z] = b.map(|i| i as usize);
                    if size == 1 {
                        let i = (z * ny + y) * nx + x;
                        grid[i] = c;
                        near[i] = 1;
                    } else if c.abs() > reach {
                        let end = |from: usize, n: usize| (from + size).min(n);
                        for z in z..end(z, nz) {
                            for y in y..end(y, ny) {
                                let row = (z * ny + y) * nx;
                                grid[row + x..row + end(x, nx)].fill((CUT * h).copysign(c));
                            }
                        }
                    } else {
                        let half = size / 2;
                        for child in 0..8 {
                            let at = [
                                x + half * (child & 1),
                                y + half * (child >> 1 & 1),
                                z + half * (child >> 2),
                            ];
                            if at[0] < nx && at[1] < ny && at[2] < nz {
                                grow_push(&mut next, at.map(|i| i as u32))?;
                            }
                        }
                    }
                }
            }
            if size == 1 {
                return Ok(());
            }
            blocks = next;
            size /= 2;
        }
    }

    /// The depth minus the distance inside the boundary, at `points`.
    fn clip_at(&self, points: &[[f32; 3]], out: &mut [f32]) -> Result<()> {
        let b = &self.boundary;
        signed_distances(b.mesh, b.bvh, b.sign, points, f32::INFINITY, out)?;
        for v in out {
            *v += self.depth;
        }
        Ok(())
    }

    /// The walls' field at a point with the given phase coordinates: negative
    /// in what is extracted, in the grid's units.
    fn walls_at(&self, p: [Trig; 3]) -> f64 {
        let w = &self.walls;
        let k = std::f64::consts::TAU / w.cell_size;
        let level = 0.5 * w.thickness * k;
        let band = 2.0 * f64::from(self.spec.voxel_size) * k + 0.3 * level;
        let d = (w.kind.distance(p, level, band) - level) / k;
        match self.extract {
            Extract::Walls => d,
            Extract::Void => -d,
        }
    }

    /// The position of a node coordinate along axis `k`.
    fn coordinate(&self, k: usize, i: f64) -> f64 {
        f64::from(self.spec.origin[k]) + i * f64::from(self.spec.voxel_size)
    }

    /// Intersects the clip distances in `grid` with the walls, or with what
    /// is between them.
    fn cross(&self, grid: &mut [f32]) -> Result<()> {
        let [nx, ny, nz] = self.spec.dims;
        let h = self.spec.voxel_size;
        let k = std::f64::consts::TAU / self.walls.cell_size;
        // One table per axis, so that a node costs no sine.
        let mut tables: [Vec<Trig>; 3] = Default::default();
        for (a, table) in tables.iter_mut().enumerate() {
            let n = self.spec.dims[a];
            reserve(table, n)?;
            table.extend((0..n).map(|i| Trig::new(k * self.coordinate(a, i as f64))));
        }
        // How far a node is outside the cropped grid's inner box.
        let rim = |i: usize, n: usize| 1.0 - i.min(n - 1 - i) as f32;
        let (nudge, cut) = (NUDGE * h, CUT * h);
        par::for_each_chunk_mut(grid, nx * ny, PAR_MIN_VOXELS, |z, slice| {
            for (y, row) in slice.chunks_exact_mut(nx).enumerate() {
                for (x, v) in row.iter_mut().enumerate() {
                    let mut clip = *v;
                    if self.cropped {
                        let r = rim(x, nx).max(rim(y, ny)).max(rim(z, nz)) - 0.5;
                        clip = clip.max(r * h);
                    }
                    if clip < 1.5 * h {
                        let w = self.walls_at([tables[0][x], tables[1][y], tables[2][z]]);
                        clip = clip.max(w as f32);
                    }
                    *v = if clip < 0.0 {
                        clip.clamp(-cut, -nudge)
                    } else {
                        clip.clamp(nudge, cut)
                    };
                }
            }
        });
        Ok(())
    }

    /// Moves the vertices of `iso` next to the clip surface along their
    /// lattice edges to where the field is zero. Marching cubes places them by
    /// interpolating a distance, which is off near sharp edges of the
    /// boundary, inside and out. A vertex stays on its edge, so the surface
    /// stays as it was connected: closed, manifold, without
    /// self-intersections. Returns the vertices looked at.
    fn place(&self, grid: &[f32], near: &[u8], iso: &mut IsoMesh) -> Result<Vec<u32>> {
        let [nx, ny, nz] = self.spec.dims;
        let steps = [1, nx, nx * ny];
        let h = self.spec.voxel_size;
        let k = std::f64::consts::TAU / self.walls.cell_size;
        allocation_allowed()?;
        let oom = || OpsError::Store(StoreError::OutOfMemory);
        // Marching cubes numbers its vertices by node, then axis: one per
        // lattice edge that changes sign.
        let threshold = if grid.len() >= PAR_MIN_VOXELS {
            0
        } else {
            usize::MAX
        };
        let parts = par::map_range(nz, 1, threshold, oom, |slices| {
            let mut count = 0u32;
            let mut edges: Vec<(u32, Edge)> = Vec::new();
            for z in slices {
                for y in 0..ny {
                    let row = (z * ny + y) * nx;
                    for x in 0..nx {
                        let node = row + x;
                        let last = [x + 1 == nx, y + 1 == ny, z + 1 == nz];
                        for axis in 0..3 {
                            if last[axis] {
                                continue;
                            }
                            let other = node + steps[axis];
                            let (a, b) = (grid[node], grid[other]);
                            if (a < 0.0) == (b < 0.0) {
                                continue;
                            }
                            if near[node] | near[other] != 0 {
                                grow_push(&mut edges, (count, Edge::new(node, axis, a, b)))?;
                            }
                            count += 1;
                        }
                    }
                }
            }
            Ok((count, edges))
        })?;
        let mut edges: Vec<(u32, Edge)> = Vec::new();
        reserve(&mut edges, parts.iter().map(|(_, e)| e.len()).sum())?;
        let mut first = 0u32;
        for (count, part) in parts {
            edges.extend(part.into_iter().map(|(v, e)| (first + v, e)));
            first += count;
        }
        debug_assert_eq!(first as usize, iso.positions.len());

        let mut open: Vec<u32> = Vec::new();
        reserve(&mut open, edges.len())?;
        open.extend(0..edges.len() as u32);
        let mut points: Vec<[f32; 3]> = Vec::new();
        reserve(&mut points, QUERY_BATCH.min(edges.len()))?;
        let mut values: Vec<f32> = filled(QUERY_BATCH.min(edges.len()), 0.0)?;
        for round in 0..PLACE_ROUNDS {
            let mut kept = 0;
            for start in (0..open.len()).step_by(QUERY_BATCH) {
                let range = start..(start + QUERY_BATCH).min(open.len());
                points.clear();
                for &e in &open[range.clone()] {
                    points.push(self.on_edge(&edges[e as usize].1).map(|c| c as f32));
                }
                let values = &mut values[..range.len()];
                self.clip_at(&points, values)?;
                for (i, at) in range.enumerate() {
                    let e = open[at];
                    let edge = &mut edges[e as usize].1;
                    let p = self.on_edge(edge);
                    let mut f = values[i];
                    if self.cropped {
                        f = f.max(self.crop_at(p));
                    }
                    if f < -CLEAR * h {
                        continue;
                    }
                    let w = self.walls_at(p.map(|c| Trig::new(k * c)));
                    if !edge.narrow(f.max(w as f32), h) && round + 1 < PLACE_ROUNDS {
                        open[kept] = e;
                        kept += 1;
                    }
                }
            }
            open.truncate(kept);
        }
        let mut at_clip: Vec<u32> = Vec::new();
        reserve(&mut at_clip, edges.len())?;
        for (v, edge) in &edges {
            iso.positions[*v as usize] = self.on_edge(edge).map(|c| c as f32);
            at_clip.push(*v);
        }
        Ok(at_clip)
    }

    /// Where an edge's vertex is.
    fn on_edge(&self, e: &Edge) -> [f64; 3] {
        let [nx, ny, _] = self.spec.dims;
        let row = e.node / nx;
        let at = [e.node % nx, row % ny, row / ny];
        std::array::from_fn(|k| {
            let t = if k == e.axis as usize { e.t } else { 0.0 };
            self.coordinate(k, at[k] as f64 + f64::from(t))
        })
    }

    /// How far outside the cropped grid's inner box a point is.
    fn crop_at(&self, p: [f64; 3]) -> f32 {
        let h = f64::from(self.spec.voxel_size);
        (0..3)
            .map(|k| {
                let last = (self.spec.dims[k] - 1) as f64;
                let i = (p[k] - f64::from(self.spec.origin[k])) / h;
                ((0.5 - i.min(last - i)) * h) as f32
            })
            .fold(f32::NEG_INFINITY, f32::max)
    }
}

/// A lattice edge that changes sign, and the bracket its zero is in.
#[derive(Debug, Clone, Copy)]
struct Edge {
    node: usize,
    /// The end of the bracket that is in the result, and the one that is not,
    /// from 0 at `node` to 1 at the next node along `axis`; the field there.
    inner: f32,
    outer: f32,
    f_inner: f32,
    f_outer: f32,
    /// Where the vertex is.
    t: f32,
    axis: u8,
    /// Which end moved last: 1 the inner one, 2 the outer one.
    moved: u8,
}

impl Edge {
    fn new(node: usize, axis: usize, a: f32, b: f32) -> Edge {
        let (inner, outer, f_inner, f_outer) = if a < 0.0 {
            (0.0, 1.0, a, b)
        } else {
            (1.0, 0.0, b, a)
        };
        let mut e = Edge {
            node,
            inner,
            outer,
            f_inner,
            f_outer,
            t: 0.0,
            axis: axis as u8,
            moved: 0,
        };
        e.guess();
        e
    }

    /// Puts the vertex where the bracket's straight line is zero.
    fn guess(&mut self) {
        let s = self.f_inner / (self.f_inner - self.f_outer);
        let t = self.inner + s * (self.outer - self.inner);
        self.t = t.clamp(0.005, 0.995);
    }

    /// Takes in the field `f` at the vertex; `true` once the vertex is close
    /// enough to the zero, and otherwise moves it (the Illinois method).
    fn narrow(&mut self, f: f32, h: f32) -> bool {
        if f.abs() <= PLACED * h {
            return true;
        }
        if f < 0.0 {
            (self.inner, self.f_inner) = (self.t, f);
            if self.moved == 1 {
                self.f_outer *= 0.5;
            }
            self.moved = 1;
        } else {
            (self.outer, self.f_outer) = (self.t, f);
            if self.moved == 2 {
                self.f_inner *= 0.5;
            }
            self.moved = 2;
        }
        self.guess();
        false
    }
}

/// Pushes with a reservation that doubles the capacity.
pub(super) fn grow_push<T>(v: &mut Vec<T>, value: T) -> Result<()> {
    if v.len() == v.capacity() {
        let grow = v.len().max(64);
        reserve(v, grow)?;
    }
    v.push(value);
    Ok(())
}

/// Six times the signed volume of a triangle with the origin.
pub(super) fn volume6(p: &[[f32; 3]], [a, b, c]: [u32; 3]) -> f64 {
    let [p, q, r] = [a, b, c].map(|v| p[v as usize].map(f64::from));
    p[0] * (q[1] * r[2] - q[2] * r[1]) - p[1] * (q[0] * r[2] - q[2] * r[0])
        + p[2] * (q[0] * r[1] - q[1] * r[0])
}

/// What [`drop_fragments`] left.
pub(super) struct Kept {
    /// The new number of each old vertex, `u32::MAX` when it is gone.
    pub vertices: Vec<u32>,
    /// The connected pieces that were left out.
    pub dropped: usize,
    /// The signed volume of what is left.
    pub volume: f64,
}

/// Leaves out the connected pieces of `iso` that enclose less than
/// `min_volume`, whichever way they are wound.
pub(super) fn drop_fragments(iso: &mut IsoMesh, min_volume: f64) -> Result<Kept> {
    let n = iso.positions.len();
    let mut sets = UnionFind::new();
    sets.reset(n)?;
    for &[a, b, c] in &iso.triangles {
        sets.union(a, b);
        sets.union(a, c);
    }
    let mut volume: Vec<f64> = filled(n, 0.0)?;
    for &t in &iso.triangles {
        volume[sets.find(t[0]) as usize] += volume6(&iso.positions, t) / 6.0;
    }
    let mut kept = Kept {
        vertices: filled(n, u32::MAX)?,
        dropped: 0,
        volume: 0.0,
    };
    let mut next = 0u32;
    for v in 0..n {
        let root = sets.find(v as u32) as usize;
        let keep = volume[root].abs() >= min_volume;
        if root == v {
            if keep {
                kept.volume += volume[root];
            } else {
                kept.dropped += 1;
            }
        }
        if keep {
            kept.vertices[v] = next;
            iso.positions[next as usize] = iso.positions[v];
            next += 1;
        }
    }
    if kept.dropped > 0 {
        iso.positions.truncate(next as usize);
        iso.triangles
            .retain(|t| kept.vertices[t[0] as usize] != u32::MAX);
        for t in &mut iso.triangles {
            *t = t.map(|v| kept.vertices[v as usize]);
        }
    }
    Ok(kept)
}
