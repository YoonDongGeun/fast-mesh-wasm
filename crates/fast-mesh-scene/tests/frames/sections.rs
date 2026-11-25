//! Plane sections and contours.

use fast_mesh_scene::{MeshId, Plane, SectionDesc, SectionId};

use crate::*;

const TOL: f64 = 1e-5;

/// World planes through the shapes, none along an axis of any put.
const PLANES: [(V3, V3); 3] = [
    ([0.1, 0.05, -0.2], [0.3, -0.5, 0.8]),
    ([0.2, -0.1, 0.1], [0.0, 0.0, 1.0]),
    // Through the shapes too, its point 70 units off.
    ([60.0, 40.0, -2.5], [0.02, 0.03, 0.999]),
];

fn plane(put: Put, (origin, normal): (V3, V3)) -> Plane {
    Plane::new(put.world(origin), normal).unwrap()
}

fn desc(plane: Plane, meshes: Vec<MeshId>) -> SectionDesc {
    SectionDesc {
        plane,
        meshes,
        fill: true,
    }
}

/// What a section says of a mesh: lengths, areas, its points' distance to the
/// plane and to the surface, its fill.
fn numbers(s: &Scene, id: SectionId, mesh: MeshId, put: Put, plane: &Plane) -> Vec<f64> {
    let r = s.section(id, mesh).unwrap();
    let mut out = vec![r.length, r.area, r.contours.len() as f64];
    let mut sorted: Vec<f64> = r.contours.iter().map(|c| c.length).collect();
    sorted.sort_by(f64::total_cmp);
    out.extend(sorted);
    let (mut lo, mut hi, mut off_plane) = ([f64::INFINITY; 3], [-f64::INFINITY; 3], 0.0f64);
    for c in &r.contours {
        for (p, uv) in c.points.iter().zip(&c.uv) {
            off_plane = off_plane.max(plane.signed_distance(*p).abs());
            off_plane = off_plane.max(norm(sub(plane.from_uv(*uv), *p)));
            for k in 0..3 {
                lo[k] = lo[k].min(put.model(*p)[k]);
                hi[k] = hi[k].max(put.model(*p)[k]);
            }
        }
    }
    out.extend(lo.into_iter().chain(hi));
    // Zero, to the precision of the numbers of the world.
    out.push((off_plane * 1e-3).min(1.0));
    let fill = r.fill.as_ref().unwrap();
    let area = fill.triangles.iter().map(|t| {
        let [a, b, c] = t.map(|i| fill.points[i as usize]);
        0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    });
    out.push(area.sum());
    out.push(f64::from(u8::from(r.fill_ok)));
    out
}

#[test]
fn sections() {
    for p in PLANES {
        agree("section", TOL, |put| {
            let mut s = scene();
            let id = put.add(&mut s, &lump());
            let plane = plane(put, p);
            let section = s.add_section(desc(plane, vec![id])).unwrap();
            s.update_sections().unwrap();
            let mut out = numbers(&s, section, id, put, &plane);
            // After an edit of the world, and back.
            let mut q = s.mesh(id).unwrap().position(40).unwrap().map(f64::from);
            q = s.local_to_world(id, q).unwrap();
            q[1] += 0.25;
            q[2] += 0.1;
            let q = s.world_to_local(id, q).unwrap().map(|c| c as f32);
            s.edit(id, |tx| tx.set_positions(&[40], &[q])).unwrap();
            s.update_sections().unwrap();
            out.extend(numbers(&s, section, id, put, &plane));
            let mut there = s.placement(id).unwrap();
            there.translation[0] += 0.2;
            s.set_placement(id, there).unwrap();
            s.update_sections().unwrap();
            out.extend(numbers(&s, section, id, put, &plane));
            s.undo().unwrap();
            s.undo().unwrap();
            s.update_sections().unwrap();
            out.extend(numbers(&s, section, id, put, &plane));
            out
        });
    }
}

#[test]
fn section_gaps() {
    for p in PLANES {
        agree_pairs("section_gap", TOL, |pa, pb| {
            let mut s = scene();
            let a = pa.add(&mut s, &lump());
            let b = pb.add(&mut s, &lump().moved([3.4, 0.37, -0.21]));
            let id = s.add_section(desc(plane(pa, p), vec![a, b])).unwrap();
            s.update_sections().unwrap();
            let gap = s.section_gap(id, a, b).unwrap().unwrap();
            let mut out = vec![gap.distance];
            out.extend(pa.model(gap.a));
            out.extend(pa.model(gap.b));
            out
        });
    }
}

#[test]
fn contours() {
    for (origin, normal) in PLANES {
        agree("slice_contours_world", TOL, |put| {
            let mut s = scene();
            let id = put.add(&mut s, &lump());
            let one = s
                .slice_contours_world(id, put.world(origin), normal)
                .unwrap();
            let stack = s
                .slice_contour_stack_world(id, put.world(origin), normal, 0.13, 3)
                .unwrap();
            let mut out = Vec::new();
            for slice in std::iter::once(&one).chain(&stack) {
                out.push(slice.loops.len() as f64);
                let (mut lo, mut hi) = ([f64::INFINITY; 3], [-f64::INFINITY; 3]);
                for l in &slice.loops {
                    out.extend([l.perimeter, l.enclosed_area, l.max_caliper_diameter]);
                    out.extend(put.model(l.centroid_3d));
                    for &p in &l.points_3d {
                        for k in 0..3 {
                            lo[k] = lo[k].min(put.model(p)[k]);
                            hi[k] = hi[k].max(put.model(p)[k]);
                        }
                    }
                }
                out.extend(lo.into_iter().chain(hi));
            }
            out
        });
    }
}
