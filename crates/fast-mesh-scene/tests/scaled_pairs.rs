//! Queries between two meshes (#288): whatever mix of scale, pose and mesh
//! origin puts them where they are, the answers are those for the same world
//! shapes at scale 1 around the world origin.

use fast_mesh_scene::{
    Contact, ContactFilter, DeviationOptions, DistanceField, MeshId, MeshOptions, Placement, Scene,
    SceneOptions, SignMode,
};
use fast_mesh_store::Mesh;

type V3 = [f64; 3];

const TOL: f64 = 1e-4;

/// The world boxes: A, one 8 units from it along x, and one it runs into.
const A: (V3, V3) = ([9.0, -1.0, -1.5], [11.0, 1.0, 1.5]);
const FAR: (V3, V3) = ([19.0, -1.2, -1.1], [21.0, 1.2, 1.1]);
const HIT: (V3, V3) = ([10.4, -0.7, -0.8], [12.6, 0.9, 0.7]);

const SCALES: [V3; 4] = [
    [1.0, 1.0, 1.0],
    [2.0, 2.0, 2.0],
    [0.25, 0.25, 0.25],
    [2.0, 1.5, 0.5],
];

/// A rotation about z, then about x.
fn turn(z: f64, x: f64) -> [[f64; 3]; 3] {
    let ((sz, cz), (sx, cx)) = (z.sin_cos(), x.sin_cos());
    [
        [cz, -sz, 0.0],
        [cx * sz, cx * cz, -sx],
        [sx * sz, sx * cz, cx],
    ]
}

fn corners((lo, hi): (V3, V3)) -> Vec<V3> {
    (0..8)
        .map(|i| std::array::from_fn(|k| if i >> k & 1 == 1 { hi[k] } else { lo[k] }))
        .collect()
}

#[rustfmt::skip]
const FACES: [u32; 36] = [
    0, 4, 6, 0, 6, 2, 1, 3, 7, 1, 7, 5, 0, 1, 5, 0, 5, 4,
    2, 6, 7, 2, 7, 3, 0, 2, 3, 0, 3, 1, 4, 5, 7, 4, 7, 6,
];

/// How a world box is put into the scene.
#[derive(Debug, Clone, Copy)]
struct Put {
    scale: V3,
    posed: bool,
    /// Whether the mesh's origin is its own centre, or zero.
    centred: bool,
}

const PLAIN: Put = Put {
    scale: [1.0; 3],
    posed: false,
    centred: false,
};

fn every_put() -> Vec<Put> {
    let mut out = Vec::new();
    for scale in SCALES {
        for posed in [false, true] {
            for centred in [false, true] {
                out.push(Put {
                    scale,
                    posed,
                    centred,
                });
            }
        }
    }
    out
}

/// Adds the world box `shape` the way `put` says.
fn add(scene: &mut Scene, shape: (V3, V3), put: Put) -> MeshId {
    let (rotation, translation) = if put.posed {
        (turn(0.4, -0.7), [3.0, -2.0, 1.0])
    } else {
        (Placement::IDENTITY.rotation, [0.0; 3])
    };
    let own = |w: V3| -> V3 {
        let d: V3 = std::array::from_fn(|k| w[k] - translation[k]);
        // The rotation's transpose, then the scale's inverse.
        std::array::from_fn(|k| (0..3).map(|r| rotation[r][k] * d[r]).sum::<f64>() / put.scale[k])
    };
    let positions: Vec<f64> = corners(shape).into_iter().flat_map(own).collect();
    let centre: V3 = std::array::from_fn(|k| 0.5 * (shape.0[k] + shape.1[k]));
    let origin = if put.centred { own(centre) } else { [0.0; 3] };
    let mesh = Mesh::from_buffers_f64_with_origin(&positions, &FACES, origin).unwrap();
    let placement = Placement {
        rotation,
        translation,
        scale: put.scale,
    };
    let options = MeshOptions {
        placement,
        ..MeshOptions::default()
    };
    scene.add_mesh(mesh, options).unwrap()
}

fn scene() -> Scene {
    Scene::new(SceneOptions::default())
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < TOL
}

/// Runs `check` for every way to put a box A and a second box.
fn every_pair(mut check: impl FnMut(Put, Put)) {
    for pa in every_put() {
        for pb in every_put() {
            check(pa, pb);
        }
    }
}

#[test]
fn distances_and_their_points() {
    every_pair(|pa, pb| {
        let mut s = scene();
        let (a, b) = (add(&mut s, A, pa), add(&mut s, FAR, pb));
        for (x, y, near, far) in [(a, b, 11.0, 19.0), (b, a, 19.0, 11.0)] {
            let d = s.mesh_distance(x, y, f64::INFINITY).unwrap().unwrap();
            assert!(close(d.distance, 8.0), "{pa:?} {pb:?}: {d:?}");
            assert!(close(d.point_a[0], near), "{pa:?} {pb:?}: {d:?}");
            assert!(close(d.point_b[0], far), "{pa:?} {pb:?}: {d:?}");
            assert!(s.mesh_distance(x, y, 7.9).unwrap().is_none());
        }
    });
}

#[test]
fn intersections_both_ways() {
    every_pair(|pa, pb| {
        let mut s = scene();
        let (a, hit) = (add(&mut s, A, pa), add(&mut s, HIT, pb));
        let far = add(&mut s, FAR, pb);
        assert!(s.intersects(a, hit).unwrap(), "{pa:?} {pb:?}");
        assert!(s.intersects(hit, a).unwrap(), "{pa:?} {pb:?}");
        assert!(!s.intersects(a, far).unwrap(), "{pa:?} {pb:?}");
        assert!(!s.intersects(far, a).unwrap(), "{pa:?} {pb:?}");
    });
}

#[test]
fn contacts_and_triangles_near() {
    let contacts = |pa: Put, pb: Put| -> (Vec<Contact>, [usize; 2]) {
        let mut s = scene();
        let a = add(&mut s, A, pa);
        let (hit, far) = (add(&mut s, HIT, pb), add(&mut s, FAR, pb));
        let mut out = Vec::new();
        s.find_contacts(8.1, &ContactFilter::default(), &mut out)
            .unwrap();
        let (mut ta, mut tb) = (Vec::new(), Vec::new());
        s.triangles_near(a, far, 8.1, &mut ta, &mut tb).unwrap();
        assert!(out.iter().any(|c| c.a == a && c.b == hit && c.intersecting));
        (out, [ta.len(), tb.len()])
    };
    let (want, near) = contacts(PLAIN, PLAIN);
    every_pair(|pa, pb| {
        let (got, n) = contacts(pa, pb);
        assert_eq!(n, near, "{pa:?} {pb:?}");
        assert_eq!(got.len(), want.len(), "{pa:?} {pb:?}");
        for (g, w) in got.iter().zip(&want) {
            assert_eq!(g.intersecting, w.intersecting, "{pa:?} {pb:?}");
            assert!(close(g.distance, w.distance), "{pa:?} {pb:?}: {g:?}");
        }
    });
}

#[test]
fn distance_fields() {
    let field = |pa: Put, pb: Put, shape: (V3, V3)| -> Vec<f32> {
        let mut s = scene();
        let (a, b) = (add(&mut s, A, pa), add(&mut s, shape, pb));
        let desc = DistanceField::new("d", a, b, f64::INFINITY, SignMode::Auto);
        let f = s.add_distance_field(desc).unwrap();
        s.update_fields().unwrap();
        s.field_values(f).unwrap().to_vec()
    };
    for shape in [FAR, HIT] {
        let want = field(PLAIN, PLAIN, shape);
        every_pair(|pa, pb| {
            for (g, w) in field(pa, pb, shape).iter().zip(&want) {
                assert!(
                    close(f64::from(*g), f64::from(*w)),
                    "{pa:?} {pb:?}: {g} {w}"
                );
            }
        });
    }
}

#[test]
fn deviation() {
    let deviation = |pa: Put, pb: Put, shape: (V3, V3), signed: bool| -> Vec<f64> {
        let mut s = scene();
        let (a, b) = (add(&mut s, A, pa), add(&mut s, shape, pb));
        let options = DeviationOptions {
            signed,
            max_distance: f32::INFINITY,
        };
        let mut values = Vec::new();
        let stats = s.mesh_deviation(a, b, &options, &mut values).unwrap();
        let mut out: Vec<f64> = values.iter().map(|&v| f64::from(v)).collect();
        out.extend([stats.min, stats.max, stats.mean, stats.rms].map(f64::from));
        out
    };
    for shape in [A, FAR, HIT] {
        for signed in [false, true] {
            let want = deviation(PLAIN, PLAIN, shape, signed);
            every_pair(|pa, pb| {
                let got = deviation(pa, pb, shape, signed);
                assert_eq!(got.len(), want.len());
                for (g, w) in got.iter().zip(&want) {
                    assert!(close(*g, *w), "{pa:?} {pb:?} signed {signed}: {g} {w}");
                }
            });
        }
    }
}
