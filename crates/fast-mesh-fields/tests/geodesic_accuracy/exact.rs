//! Exact geodesic distances on a triangle mesh, independent of the code under
//! test: windows of straight rays propagated across the faces (Chen & Han
//! 1990) in order of distance, with the "one angle, one split" rule and the
//! filter of Xin & Wang (2009).
//!
//! A shortest path unfolds to straight lines between vertices of a total
//! angle above 2π (and boundary vertices), where it can bend. A window is the
//! part of an edge that straight rays from one such vertex (or a source)
//! reach through one sequence of faces.
//!
//! For consistently oriented meshes with at most two faces per edge.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::f64::consts::TAU;

const NONE: usize = usize::MAX;

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn length(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// Rays from one image of a source through a part of an edge, about to enter
/// the face of `half`. The frame has its origin at the edge's first corner,
/// its x axis along the edge and the face above it; the image lies below.
#[derive(Debug, Clone, Copy)]
struct Window {
    half: usize,
    from: f64,
    to: f64,
    image: [f64; 2],
    /// The distance of the vertex the rays start from.
    before: f64,
}

impl Window {
    /// The distance along the rays to a point of the edge.
    fn at(&self, x: f64) -> f64 {
        self.before + (self.image[0] - x).hypot(self.image[1])
    }

    fn nearest(&self) -> f64 {
        self.at(self.image[0].clamp(self.from, self.to))
    }
}

enum Event {
    Window(Window),
    Vertex(usize),
}

struct Item {
    key: f64,
    /// Makes the order total.
    serial: usize,
    event: Event,
}

impl PartialEq for Item {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Item {}

impl PartialOrd for Item {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Item {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .key
            .total_cmp(&self.key)
            .then(other.serial.cmp(&self.serial))
    }
}

/// The half-edge after `h` in its face. Half-edge `3 f + k` runs from corner
/// `k` of face `f` to the next; the third corner is across from it.
fn next(h: usize) -> usize {
    3 * (h / 3) + (h % 3 + 1) % 3
}

struct Surface<'a> {
    triangles: &'a [[u32; 3]],
    length: Vec<f64>,
    /// Where the corner across is, in the half-edge's frame.
    apex: Vec<[f64; 2]>,
    twin: Vec<usize>,
    /// The half-edges that start at each vertex.
    around: Vec<Vec<usize>>,
    /// Whether shortest paths can bend at a vertex.
    bends: Vec<bool>,
}

impl<'a> Surface<'a> {
    fn new(positions: &[[f64; 3]], triangles: &'a [[u32; 3]]) -> Surface<'a> {
        let halves = 3 * triangles.len();
        let mut s = Surface {
            triangles,
            length: vec![0.0; halves],
            apex: vec![[0.0; 2]; halves],
            twin: vec![NONE; halves],
            around: vec![Vec::new(); positions.len()],
            bends: Vec::new(),
        };
        let mut angle = vec![0.0f64; positions.len()];
        let mut open: HashMap<(usize, usize), usize> = HashMap::new();
        for h in 0..halves {
            let (va, vb, vc) = (s.corner(h), s.corner(next(h)), s.corner(next(next(h))));
            let [a, b, c] = [va, vb, vc].map(|v| positions[v]);
            let (ab, ac) = (sub(b, a), sub(c, a));
            let l = length(ab);
            let cross = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ];
            s.length[h] = l;
            s.apex[h] = [dot(ac, ab) / l, length(cross) / l];
            s.around[va].push(h);
            angle[va] += s.apex[h][1].atan2(s.apex[h][0]);
            if let Some(other) = open.remove(&(vb, va)) {
                s.twin[h] = other;
                s.twin[other] = h;
            } else {
                let clash = open.insert((va, vb), h);
                assert!(clash.is_none(), "not an oriented manifold mesh");
            }
        }
        s.bends = angle.iter().map(|a| *a > TAU + 1e-12).collect();
        for &h in open.values() {
            for v in [s.corner(h), s.corner(next(h))] {
                s.bends[v] = true;
            }
        }
        s
    }

    fn corner(&self, h: usize) -> usize {
        self.triangles[h / 3][h % 3] as usize
    }
}

struct Search<'a> {
    surface: Surface<'a>,
    distance: Vec<f64>,
    queue: BinaryHeap<Item>,
    serial: usize,
    /// Per half-edge, the shortest distance a window on it gave the corner
    /// across, and where its ray to that corner entered the edge.
    split: Vec<(f64, f64)>,
    tolerance: f64,
}

impl Search<'_> {
    fn push(&mut self, key: f64, event: Event) {
        self.serial += 1;
        self.queue.push(Item {
            key,
            serial: self.serial,
            event,
        });
    }

    fn reach(&mut self, v: usize, d: f64) {
        if d < self.distance[v] {
            self.distance[v] = d;
            if self.surface.bends[v] {
                self.push(d, Event::Vertex(v));
            }
        }
    }

    fn add(&mut self, w: Window) {
        let l = self.surface.length[w.half];
        // Too narrow to matter, or seen along the edge (or not a number).
        if !(w.to - w.from > 1e-13 * l && w.image[1] < -1e-13 * l) {
            return;
        }
        let a = self.surface.corner(w.half);
        let b = self.surface.corner(next(w.half));
        // Every point of the window is nearer by way of a corner.
        if self.distance[a] + w.to < w.at(w.to) - self.tolerance
            || self.distance[b] + (l - w.from) < w.at(w.from) - self.tolerance
        {
            return;
        }
        self.push(w.nearest(), Event::Window(w));
    }

    /// Rays in every direction from a vertex.
    fn start_from(&mut self, v: usize) {
        let d = self.distance[v];
        for i in 0..self.surface.around[v].len() {
            let h = self.surface.around[v][i];
            let opposite = next(h);
            let last = next(opposite);
            let (b, c) = (self.surface.corner(opposite), self.surface.corner(last));
            self.reach(b, d + self.surface.length[h]);
            self.reach(c, d + self.surface.length[last]);
            let half = self.surface.twin[opposite];
            if half == NONE {
                continue;
            }
            // The vertex, in the frame of the edge seen from its other side.
            let [x, y] = self.surface.apex[opposite];
            let l = self.surface.length[opposite];
            self.add(Window {
                half,
                from: 0.0,
                to: l,
                image: [l - x, -y],
                before: d,
            });
        }
    }

    /// Sends a window's rays across its face.
    fn cross(&mut self, w: Window) {
        let p = self.surface.apex[w.half];
        let l = self.surface.length[w.half];
        let s = w.image;
        let (right, left) = (next(w.half), next(next(w.half)));
        let ray = |x: f64| [x - s[0], -s[1]];
        let to_apex = ray_to(s, p);
        // The sine from a ray to the corner across: positive left of it.
        let side = |x: f64| {
            let r = ray(x);
            let sizes = r[0].hypot(r[1]) * to_apex[0].hypot(to_apex[1]);
            (r[0] * to_apex[1] - r[1] * to_apex[0]) / sizes.max(f64::MIN_POSITIVE)
        };
        let (from, to) = (side(w.from), side(w.to));
        // Some rays pass the corner on its left, or right.
        let (mut go_left, mut go_right) = (from < 0.0, to > 0.0);
        let both = go_left && go_right;
        if from <= 1e-11 && to >= -1e-11 {
            let d = w.before + to_apex[0].hypot(to_apex[1]);
            self.reach(self.surface.corner(left), d);
            let entry = s[0] - s[1] * to_apex[0] / to_apex[1];
            let (best, at) = self.split[w.half];
            if d < best {
                self.split[w.half] = (d, entry);
            } else if both && d > best + self.tolerance {
                // On one side its rays cross those of the nearer window,
                // beyond which they are no use.
                go_right = entry > at;
                go_left = entry < at;
            }
        }
        // The line from `o` to `e`: its length, its direction, the image in
        // its frame, and where the ray through `x` crosses it.
        let line = |o: [f64; 2], e: [f64; 2], m: f64| {
            let e = [(e[0] - o[0]) / m, (e[1] - o[1]) / m];
            let q = ray_to(o, s);
            let image = [q[0] * e[0] + q[1] * e[1], q[1] * e[0] - q[0] * e[1]];
            let meet = move |x: f64| {
                let r = ray(x);
                ((q[1] * r[0] - q[0] * r[1]) / (e[1] * r[0] - e[0] * r[1])).clamp(0.0, m)
            };
            (image, meet)
        };
        if go_left && self.surface.twin[left] != NONE {
            let m = self.surface.length[left];
            let (image, meet) = line([0.0, 0.0], p, m);
            self.add(Window {
                half: self.surface.twin[left],
                from: meet(w.from),
                to: if to > 0.0 { m } else { meet(w.to) },
                image,
                before: w.before,
            });
        }
        if go_right && self.surface.twin[right] != NONE {
            let m = self.surface.length[right];
            let (image, meet) = line(p, [l, 0.0], m);
            self.add(Window {
                half: self.surface.twin[right],
                from: if from < 0.0 { 0.0 } else { meet(w.from) },
                to: meet(w.to),
                image,
                before: w.before,
            });
        }
    }
}

fn ray_to(from: [f64; 2], to: [f64; 2]) -> [f64; 2] {
    [to[0] - from[0], to[1] - from[1]]
}

/// The exact distances over the surface from the nearest of `sources` to
/// every vertex; infinite where there is no way.
pub fn exact_distances(
    positions: &[[f64; 3]],
    triangles: &[[u32; 3]],
    sources: &[u32],
) -> Vec<f64> {
    let surface = Surface::new(positions, triangles);
    let size = surface.length.iter().fold(0.0f64, |a, &b| a.max(b));
    let mut search = Search {
        split: vec![(f64::INFINITY, 0.0); surface.length.len()],
        distance: vec![f64::INFINITY; positions.len()],
        queue: BinaryHeap::new(),
        serial: 0,
        tolerance: 1e-12 * size,
        surface,
    };
    // Along the edges, for upper bounds to filter the windows by.
    let mut walk = BinaryHeap::new();
    for &s in sources {
        search.distance[s as usize] = 0.0;
        walk.push(Item {
            key: 0.0,
            serial: s as usize,
            event: Event::Vertex(s as usize),
        });
    }
    while let Some(Item { key, serial: v, .. }) = walk.pop() {
        if key > search.distance[v] {
            continue;
        }
        for &h in &search.surface.around[v] {
            let before = next(next(h));
            for (to, along) in [
                (search.surface.corner(next(h)), search.surface.length[h]),
                (search.surface.corner(before), search.surface.length[before]),
            ] {
                if key + along < search.distance[to] {
                    search.distance[to] = key + along;
                    walk.push(Item {
                        key: key + along,
                        serial: to,
                        event: Event::Vertex(to),
                    });
                }
            }
        }
    }
    for v in 0..positions.len() {
        let d = search.distance[v];
        if sources.contains(&(v as u32)) || (search.surface.bends[v] && d.is_finite()) {
            search.push(d, Event::Vertex(v));
        }
    }
    let mut started = vec![false; positions.len()];
    while let Some(Item { key, event, .. }) = search.queue.pop() {
        match event {
            Event::Vertex(v) => {
                if key <= search.distance[v] && !started[v] {
                    started[v] = true;
                    search.start_from(v);
                }
            }
            Event::Window(w) => search.cross(w),
        }
    }
    search.distance
}
