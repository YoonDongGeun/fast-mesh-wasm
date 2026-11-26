//! Booleans, cuts, trims, joins, holes, placed meshes, extrusions, hulls,
//! wraps and lattices.

use fast_mesh_ops::boolean::{
    DrainHoleOptions, DrainHoleSpec, PlaceMeshParams, PlaceMode, PlaceUpAxis,
};
use fast_mesh_ops::lattice::LatticeOptions;
use fast_mesh_ops::wrap::WrapOptions;
use fast_mesh_ops::{BaseLoops, ExtrudeDirection, boundary_loops};
use fast_mesh_scene::{
    BasePlane, BooleanMethod, BooleanOp, BooleanOptions, CsgNode, CutOptions, ExtrudeBaseOptions,
    ExtrudeFacesOptions, JoinOptions, MeshId, Plane, TrimCurveOptions,
};

use crate::*;

const TOL: f64 = 2e-5;

fn crossing() -> Shape {
    block([0.6, -0.31, -0.42], [2.1, 0.53, 0.37])
}

/// What an edit of `id` gives, what undo and redo do to it.
fn with_history(s: &mut Scene, id: MeshId, put: Put) -> Vec<f64> {
    let mut out = solid(s, id, put);
    s.undo().unwrap();
    out.extend(solid(s, id, put));
    s.redo().unwrap();
    out.extend(solid(s, id, put));
    out
}

#[test]
fn booleans() {
    for op in [BooleanOp::Union, BooleanOp::Subtract, BooleanOp::Intersect] {
        agree_pairs("boolean", TOL, |pa, pb| {
            let mut s = scene();
            let (a, b) = (pa.add(&mut s, &lump()), pb.add(&mut s, &crossing()));
            s.boolean(a, b, op).unwrap();
            with_history(&mut s, a, pa)
        });
    }
}

#[test]
fn booleans_with_options() {
    let options = BooleanOptions {
        op: fast_mesh_ops::BooleanOp::Subtract,
        ..BooleanOptions::default()
    };
    agree_pairs("boolean_with_options", TOL, |pa, pb| {
        let mut s = scene();
        let (a, b) = (pa.add(&mut s, &lump()), pb.add(&mut s, &crossing()));
        s.boolean_with_options(a, b, &options).unwrap();
        with_history(&mut s, a, pa)
    });
}

/// On voxels, which lie along A's own axes and whose pitch is in A's own
/// units: only about the same.
#[test]
fn voxel_booleans() {
    for (method, pitch) in [
        (BooleanMethod::BvhSdf, Some(0.18f32)),
        (BooleanMethod::FhEdt, Some(0.18)),
        (BooleanMethod::FhEdt, None),
    ] {
        agree_few_pairs("voxel boolean", 0.25, |pa, pb| {
            let options = BooleanOptions {
                op: fast_mesh_ops::BooleanOp::Subtract,
                method,
                voxel_pitch: pitch.map(|p| p / pa.scale[0] as f32),
                ..BooleanOptions::default()
            };
            let mut s = scene();
            let (a, b) = (pa.add(&mut s, &lump()), pb.add(&mut s, &crossing()));
            s.boolean_with_options(a, b, &options).unwrap();
            let whole = solid(&s, a, pa);
            [&whole[..1], &whole[2..8]].concat()
        });
    }
}

#[test]
fn batches() {
    use fast_mesh_ops::BooleanOp as Op;
    let leaf = |i| Box::new(CsgNode::Leaf(i));
    let tree = CsgNode::Op {
        op: Op::Subtract,
        left: Box::new(CsgNode::Op {
            op: Op::Union,
            left: leaf(0),
            right: leaf(1),
        }),
        right: leaf(2),
    };
    agree_pairs("boolean_batch", TOL, |pa, pb| {
        let mut s = scene();
        let ids = [
            pa.add(&mut s, &lump()),
            pb.add(&mut s, &crossing()),
            pb.add(&mut s, &block([-0.2, -0.3, 0.2], [0.3, 0.2, 1.7])),
        ];
        let id = s
            .boolean_batch(&ids, &tree, &BooleanOptions::default())
            .unwrap();
        solid(&s, id, pa)
    });
}

#[test]
fn cuts() {
    for (origin, normal) in [
        ([0.1, 0.05, -0.2], [0.3, -0.5, 0.8]),
        ([60.0, 40.0, -2.5], [0.02, 0.03, 0.999]),
    ] {
        agree("cut_by_plane", TOL, |put| {
            let mut s = scene();
            let id = put.add(&mut s, &lump());
            let plane = Plane::new(put.world(origin), normal).unwrap();
            let r = s
                .cut_by_plane(id, &CutOptions { plane, cap: true })
                .unwrap();
            let mut out = solid(&s, r.added, put);
            out.extend(with_history(&mut s, id, put));
            out
        });
    }
}

/// Points on the lump around its +x end.
fn ring(count: u32, r: f64) -> Vec<V3> {
    (0..count)
        .map(|i| {
            let (sin, cos) = (f64::from(i) / f64::from(count) * std::f64::consts::TAU).sin_cos();
            [2.0, r * cos + 0.03, r * sin - 0.02]
        })
        .collect()
}

#[test]
fn trims() {
    for snap_tolerance in [0.0, 0.05] {
        agree("trim_by_curve", 5e-5, |put| {
            let mut s = scene();
            let id = put.add(&mut s, &lump());
            // Onto the surface first, so that every put has the same points.
            let points = ring(20, 0.45)
                .into_iter()
                .map(|p| {
                    let o = fast_mesh_scene::SceneRayOptions::default();
                    let hit = s.raycast(put.world(p), [-1.0, 0.0, 0.0], &o).unwrap();
                    hit.unwrap().point
                })
                .collect();
            let options = TrimCurveOptions {
                points,
                closed: true,
                delete_inside: true,
                snap_tolerance,
            };
            let r = s.trim_by_curve(id, &options).unwrap();
            let mut out = vec![r.removed_area];
            out.extend(with_history(&mut s, id, put));
            out
        });
    }
}

#[test]
fn joins() {
    agree_pairs("join", TOL, |pa, pb| {
        let mut s = scene();
        let a = pa.add(&mut s, &sheet(6));
        // The other way round, so that the rims can be bridged.
        let mut other = sheet(6).moved([0.1, 0.05, 0.9]);
        for t in &mut other.triangles {
            t.swap(1, 2);
        }
        let b = pb.add(&mut s, &other);
        let loop_of = |s: &Scene, id| boundary_loops(s.mesh(id).unwrap()).unwrap()[0].id;
        let options = JoinOptions {
            loop_a: loop_of(&s, a),
            loop_b: loop_of(&s, b),
            max_edges: None,
        };
        s.join(a, b, &options).unwrap();
        let joined = positions(&s, a, pa);
        // The bridge is triangulated in A's own space: as a whole, then.
        let before = joined.len();
        let mut out = joined;
        s.undo().unwrap();
        out.extend(positions(&s, a, pa));
        out.extend(positions(&s, b, pa));
        s.redo().unwrap();
        assert_eq!(positions(&s, a, pa).len(), before);
        out
    });
}

#[test]
fn drain_holes() {
    for through in [None, Some(0.6)] {
        agree("punch_drain_hole_at", TOL, |put| {
            let mut s = scene();
            let id = put.add(&mut s, &block([-1.0, -0.8, -0.7], [1.2, 0.9, 0.7]));
            let spec = DrainHoleSpec {
                surface_normal: [0.1, -0.05, 0.99],
                diameter: 0.4,
                depth: through,
                outer_chamfer: 0.05,
                segments: 16,
                ..DrainHoleSpec::default()
            };
            let options = DrainHoleOptions {
                generate_plug: true,
                plug_clearance: 0.02,
                ..DrainHoleOptions::default()
            };
            let at = put.world([0.1, 0.05, 0.7]);
            let (_, plugs) = s.punch_drain_hole_at(id, at, &spec, &options).unwrap();
            let mut out = Vec::new();
            // About its axis it may be turned: volumes and what is along it.
            for plug in &plugs {
                let whole = solid(&s, *plug, put);
                out.extend([whole[0], whole[1], whole[8], whole[9], whole[10]]);
            }
            let whole = with_history(&mut s, id, put);
            out.extend(&whole[..2]);
            out.extend(&whole[11..22]);
            out
        });
    }
}

/// A stamp without symmetries: an arrow, pointing along +x, its zero below
/// its middle.
fn arrow() -> Shape {
    let outline = [
        [-0.3, -0.08],
        [0.1, -0.08],
        [0.1, -0.2],
        [0.4, 0.05],
        [0.1, 0.2],
        [0.1, 0.08],
        [-0.3, 0.08],
    ];
    let n = outline.len() as u32;
    let mut points = Vec::new();
    for z in [-0.1, 0.15] {
        points.extend(outline.map(|[x, y]: [f64; 2]| [x, y, z]));
    }
    let mut triangles = Vec::new();
    // Fans that stay inside: from the tail's two ends and the head.
    for t in [[0, 1, 5], [0, 5, 6], [1, 2, 3], [1, 3, 5], [3, 4, 5]] {
        triangles.push([t[0] + n, t[1] + n, t[2] + n]);
        triangles.push([t[0], t[2], t[1]]);
    }
    for i in 0..n {
        let j = (i + 1) % n;
        triangles.push([i, j, j + n]);
        triangles.push([i, j + n, i + n]);
    }
    Shape { points, triangles }
}

#[test]
fn placed_meshes() {
    // Into a flat top, which is the same whatever the axes; every mode.
    let target = block([-1.0, -0.8, -0.7], [1.2, 0.9, 0.7]);
    for mode in [
        PlaceMode::KeepSeparate,
        PlaceMode::AppendShell,
        PlaceMode::EmbedUnion,
        PlaceMode::EngraveSubtract,
    ] {
        for conform_to_surface in [false, true] {
            agree_pairs("place_mesh_world", TOL, |pa, pb| {
                let mut s = scene();
                let id = pa.add(&mut s, &target);
                // Of the stamp's placement only the scale counts: its shape
                // scaled by it is the arrow.
                let put = Put { at: [0.0; 3], ..pb };
                let straight = Put {
                    turned: false,
                    origin: Origin::Zero,
                    ..put
                };
                let stamp = straight.add(&mut s, &arrow());
                let params = PlaceMeshParams {
                    up_axis: PlaceUpAxis::PosZ,
                    twist_angle_rad: 0.4,
                    uniform_scale: 1.3,
                    embed_depth: 0.03,
                    mode,
                    conform_to_surface,
                    ..PlaceMeshParams::default()
                };
                let at = pa.world([0.1, 0.05, 0.7]);
                let (r, new) = s
                    .place_mesh_world(id, stamp, at, [0.2, -0.1, 0.97], &params)
                    .unwrap();
                let mut out = solid(&s, new.unwrap_or(id), pa);
                // The transform it reports: of the stamp's points, times its
                // placement's scale.
                let t = r.transform;
                let m = |p: V3| {
                    let q = p.map(|c| t.scale * c);
                    let r = t.rigid.rotation;
                    let w: V3 = std::array::from_fn(|i| dot(r[i], q) + t.rigid.translation[i]);
                    pa.model(w)
                };
                out.extend(m([0.0; 3]).into_iter().chain(m([0.4, 0.05, 0.1])));
                s.undo().unwrap();
                out.extend(solid(&s, id, pa));
                out
            });
        }
    }
}

/// A stamp's own coordinates are its origin plus its positions, whatever the
/// one and the others are.
#[test]
fn stamps_do_not_go_by_their_origin() {
    let params = PlaceMeshParams {
        twist_angle_rad: 0.4,
        mode: PlaceMode::EmbedUnion,
        ..PlaceMeshParams::default()
    };
    let run = |origin: V3, put: Put| {
        let mut s = scene();
        let id = put.add(&mut s, &block([-1.0, -0.8, -0.7], [1.2, 0.9, 0.7]));
        let positions: Vec<f64> = arrow().points.into_iter().flatten().collect();
        let shape = arrow();
        let indices = shape.triangles.as_flattened();
        let mesh = Mesh::from_buffers_f64_with_origin(&positions, indices, origin).unwrap();
        let stamp = s.add_mesh(mesh, MeshOptions::default()).unwrap();
        let at = put.world([0.1, 0.05, 0.7]);
        s.place_mesh_world(id, stamp, at, [0.2, -0.1, 0.97], &params)
            .unwrap();
        solid(&s, id, put)
    };
    let want = run([0.0; 3], PLAIN);
    for put in some_puts_at(FAR) {
        for origin in [[0.0; 3], [0.1, -0.2, 0.3], [3.0, 2.0, -1.0]] {
            let d = off(&run(origin, put), &want);
            assert!(d < TOL, "{origin:?} {put:?}: {d}");
        }
    }
}

/// What a mesh goes through outside the geometry: a compaction, a removal and
/// its undo, a step cancelled.
#[test]
fn meshes_stay_where_they_are() {
    agree("compact, remove, cancel", 2e-6, |put| {
        let mut s = scene();
        let id = put.add(&mut s, &lump());
        let mut out = solid(&s, id, put);
        s.delete_selected_faces(id, &[0, 1, 2, 3]).unwrap();
        s.compact(id).unwrap();
        out.extend(solid(&s, id, put));
        s.remove_mesh_undoable(id).unwrap();
        s.undo().unwrap();
        out.extend(solid(&s, id, put));
        s.begin_step();
        let there = Placement {
            translation: [1.0, 2.0, 3.0],
            scale: [2.0; 3],
            ..s.placement(id).unwrap()
        };
        s.set_placement(id, there).unwrap();
        s.cancel_step().unwrap();
        out.extend(solid(&s, id, put));
        let copy = put.add(&mut s, &lump());
        out.extend(solid(&s, copy, put));
        out
    });
}

#[test]
fn extruded_faces() {
    for direction in [
        ExtrudeDirection::RegionNormal,
        ExtrudeDirection::VertexNormals,
        ExtrudeDirection::Fixed([0.8, 0.3, 0.4]),
    ] {
        agree("extrude_faces", TOL, |put| {
            let mut s = scene();
            let id = put.add(&mut s, &lump());
            let sel = s
                .select_faces_in_sphere_world(id, put.world([1.0, 0.3, 0.2]), 0.6, None)
                .unwrap();
            let options = ExtrudeFacesOptions {
                distance: 0.31,
                direction,
                check_self_intersections: true,
            };
            s.extrude_faces(id, &sel, &options).unwrap();
            let mut out = positions(&s, id, put);
            out.extend(with_history(&mut s, id, put));
            out
        });
    }
}

#[test]
fn extruded_bases() {
    // A sheet with a hole that is longer than its rim in the stretched axes
    // of a put, or not.
    let mut shape = sheet(12);
    let mut cell = 0..;
    shape.triangles.retain(|_| {
        let c: u32 = cell.next().unwrap() / 2;
        !((4..8).contains(&(c % 12)) && (3..9).contains(&(c / 12)))
    });
    let used: std::collections::BTreeSet<u32> = shape.triangles.iter().flatten().copied().collect();
    let index: std::collections::BTreeMap<u32, u32> = used
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i as u32))
        .collect();
    shape.points = used.iter().map(|&v| shape.points[v as usize]).collect();
    for t in &mut shape.triangles {
        *t = t.map(|v| index[&v]);
    }
    let n = unit([0.1, 0.2, -0.95]);
    for below in [true, false] {
        for loops in [BaseLoops::Largest, BaseLoops::All] {
            agree("extrude_base", 5e-5, |put| {
                let mut s = scene();
                let id = put.add(&mut s, &shape);
                let options = ExtrudeBaseOptions {
                    direction: n,
                    plane: if below {
                        BasePlane::Below(0.3)
                    } else {
                        BasePlane::Height(-dot(put.world([0.0, 0.0, -0.8]), n))
                    },
                    loops: loops.clone(),
                    refine: false,
                    check_self_intersections: true,
                };
                let r = s.extrude_base(id, &options).unwrap();
                let mut out = vec![r.height + dot(put.at, n)];
                out.extend(with_history(&mut s, id, put));
                out
            });
        }
    }
}

/// The largest loop is the world's: the strip along x, whatever the mesh's own
/// coordinates make of it.
#[test]
fn extruded_base_of_the_largest_loop() {
    let strip = |x: f64, y: f64, z: f64| Shape {
        points: vec![[0.0, 0.0, z], [x, 0.0, z], [x, y, z], [0.0, y, z]],
        triangles: vec![[0, 1, 2], [0, 2, 3]],
    };
    let mut shape = strip(3.0, 0.4, 0.0);
    let other = strip(0.3, 2.9, 0.0).moved([0.5, 1.0, 0.2]);
    shape.points.extend(other.points);
    shape
        .triangles
        .extend(other.triangles.iter().map(|t| t.map(|v| v + 4)));
    agree("extrude_base", 2e-5, |put| {
        let mut s = scene();
        let id = put.add(&mut s, &shape);
        let options = ExtrudeBaseOptions {
            direction: [0.0, 0.0, -1.0],
            plane: BasePlane::Below(0.5),
            loops: BaseLoops::Largest,
            refine: false,
            check_self_intersections: false,
        };
        s.extrude_base(id, &options).unwrap();
        with_history(&mut s, id, put)
    });
}

#[test]
fn hulls() {
    agree_pairs("convex_hull", TOL, |pa, pb| {
        let mut s = scene();
        let (a, b) = (pa.add(&mut s, &lump()), pb.add(&mut s, &crossing()));
        let mut out = Vec::new();
        for ids in [vec![a], vec![b], vec![a, b]] {
            let r = s.convex_hull(&ids).unwrap();
            out.extend(solid(&s, r.added, pa));
        }
        let points: Vec<V3> = lump().points.iter().map(|&p| pa.world(p)).collect();
        let r = s.convex_hull_points(&points).unwrap();
        out.extend(solid(&s, r.added, pa));
        out
    });
}

/// On voxels, which lie along the meshes' own axes when they have one rigid
/// placement and along the world's otherwise: only about the same.
#[test]
fn wraps() {
    let options = WrapOptions {
        gap: 0.2,
        voxel_size: Some(0.12),
        project: true,
        ..WrapOptions::default()
    };
    agree_few_pairs("wrap", 0.1, |pa, pb| {
        let mut s = scene();
        let (a, b) = (pa.add(&mut s, &lump()), pb.add(&mut s, &crossing()));
        let mut out = Vec::new();
        for ids in [vec![a], vec![a, b]] {
            let plan = s.wrap_plan(&ids, &options).unwrap();
            let r = s.wrap(&ids, &options).unwrap();
            assert_eq!(r.plan, plan);
            out.push(f64::from(plan.voxel_size));
            out.extend(&solid(&s, r.added, pa)[..8]);
        }
        out
    });
}

/// Its cells are aligned with the mesh's own axes, so it fills about as much.
#[test]
fn lattices() {
    let options = LatticeOptions {
        cell_size: 0.8,
        wall_thickness: 0.2,
        ..LatticeOptions::default()
    };
    agree_some("lattice_fill", 0.2, |put| {
        let mut s = scene();
        let id = put.add(&mut s, &block([-1.0, -0.8, -0.7], [1.2, 0.9, 0.7]));
        let plan = s.lattice_plan(id, &options).unwrap();
        let r = s.lattice_fill(id, &options).unwrap();
        let [volume, ..] = solid(&s, id, put)[..] else {
            panic!()
        };
        s.undo().unwrap();
        let mut out = vec![volume, r.solid_volume, plan.region_volume];
        out.extend(solid(&s, id, put));
        out
    });
}

/// The same mesh, its origin elsewhere only: the very same lattice.
#[test]
fn lattices_do_not_go_by_the_origin() {
    let options = LatticeOptions {
        cell_size: 0.8,
        wall_thickness: 0.2,
        ..LatticeOptions::default()
    };
    let shape = block([-1.0, -0.8, -0.7], [1.2, 0.9, 0.7]);
    for at in [NEAR, FAR] {
        let run = |origin| {
            let put = Put {
                scale: [1.0; 3],
                turned: false,
                origin,
                at,
            };
            let mut s = scene();
            let id = put.add(&mut s, &shape);
            s.lattice_fill(id, &options).unwrap();
            positions(&s, id, put)
        };
        let want = run(Origin::Centre);
        for origin in [Origin::Zero, Origin::Both, Origin::Away] {
            let d = off(&run(origin), &want);
            assert!(d < 1e-4, "{origin:?} {at:?}: {d}");
        }
    }
}
