//! Face selections.

use fast_mesh_ops::repair::{AdjacencyKind, ScreenSelectOptions, ScreenSelectionShape};
use fast_mesh_scene::{MeshId, ScreenCamera};

use crate::*;

fn ids(list: Vec<u32>) -> Vec<f64> {
    std::iter::once(-1.0)
        .chain(list.into_iter().map(f64::from))
        .collect()
}

#[test]
fn spheres() {
    agree("select_faces_in_sphere_world", 0.0, |put| {
        let mut s = scene();
        let id = put.add(&mut s, &lump());
        let mut out = Vec::new();
        for (c, r) in [
            ([1.0, 0.3, 0.2], 0.61),
            ([0.1, -0.9, -0.1], 0.83),
            ([0.0, 0.0, 1.3], 0.47),
        ] {
            for cone in [None, Some(([0.3, 0.2, 0.9], 0.9))] {
                let sel = s.select_faces_in_sphere_world(id, put.world(c), r, cone);
                out.extend(ids(sel.unwrap()));
            }
        }
        out
    });
}

#[test]
fn creases_shells_and_islands() {
    // A lump and two small bits of it apart.
    let mut shape = lump();
    for (k, at) in [[2.5, 0.0, 0.0], [0.0, 2.2, 0.4]].into_iter().enumerate() {
        let side = 0.2 + 0.1 * k as f64;
        let bit = block([0.0; 3], [side, 0.8 * side, 0.3]).moved(at);
        let n = shape.points.len() as u32;
        shape.points.extend(bit.points);
        shape
            .triangles
            .extend(bit.triangles.iter().map(|t| t.map(|v| v + n)));
    }
    agree("select_by_crease_angle", 0.0, |put| {
        let mut s = scene();
        let id = put.add(&mut s, &shape);
        let mut out = Vec::new();
        for degrees in [17.0f64, 23.0, 29.0] {
            let sel = s.select_by_crease_angle(id, 100, degrees.to_radians());
            out.extend(ids(sel.unwrap()));
        }
        out.extend(ids(s.select_connected_shell(id, 100).unwrap()));
        // Their areas are 0.272 and 0.474 in the world.
        for area in [None, Some(0.3), Some(0.5)] {
            let sel = s.select_small_face_islands(id, 20, area);
            out.extend(ids(sel.unwrap()));
        }
        let some = s.select_by_crease_angle(id, 100, 0.4).unwrap();
        let grown = s.grow_face_selection(id, &some, 1, AdjacencyKind::Edge);
        out.extend(ids(grown.unwrap()));
        out
    });
}

/// A camera at `eye` looking at `to`.
struct Camera {
    eye: V3,
    to: V3,
    ortho: bool,
}

impl Camera {
    fn view_proj(&self) -> [[f64; 4]; 4] {
        let f = unit(sub(self.to, self.eye));
        let r = unit(cross(f, [0.0, 0.0, 1.0]));
        let u = cross(r, f);
        let eye = self.eye;
        let view = [
            [r[0], u[0], -f[0], 0.0],
            [r[1], u[1], -f[1], 0.0],
            [r[2], u[2], -f[2], 0.0],
            [-dot(r, eye), -dot(u, eye), dot(f, eye), 1.0],
        ];
        let (near, far) = (0.1, 50.0);
        let mut proj = [[0.0; 4]; 4];
        if self.ortho {
            proj[0][0] = 1.0 / 1.6;
            proj[1][1] = 1.0 / 1.6;
            proj[2][2] = -2.0 / (far - near);
            proj[3][2] = -(far + near) / (far - near);
            proj[3][3] = 1.0;
        } else {
            proj[0][0] = 2.3;
            proj[1][1] = 2.3;
            proj[2][2] = -(far + near) / (far - near);
            proj[2][3] = -1.0;
            proj[3][2] = -2.0 * far * near / (far - near);
        }
        std::array::from_fn(|c| {
            std::array::from_fn(|r| (0..4).map(|k| proj[k][r] * view[c][k]).sum())
        })
    }
}

const CAMERAS: [Camera; 3] = [
    Camera {
        eye: [4.0, -3.0, 2.5],
        to: [0.1, 0.0, 0.0],
        ortho: false,
    },
    Camera {
        eye: [-3.0, -4.0, 1.5],
        to: [0.0, 0.1, -0.1],
        ortho: true,
    },
    // Close up.
    Camera {
        eye: [1.9, 0.2, 0.4],
        to: [0.0, 0.0, 0.0],
        ortho: false,
    },
];

fn shapes() -> [ScreenSelectionShape; 2] {
    [
        ScreenSelectionShape::Rect {
            min_ndc: [-0.31, -0.27],
            max_ndc: [0.43, 0.39],
        },
        ScreenSelectionShape::Lasso {
            polygon_ndc: vec![[-0.5, -0.41], [0.62, -0.17], [0.1, 0.57], [-0.2, 0.11]],
        },
    ]
}

fn on_screen(
    s: &mut Scene,
    id: MeshId,
    shape: &ScreenSelectionShape,
    camera: &Camera,
    put: Put,
    flags: [bool; 3],
) -> Vec<u32> {
    let world = Camera {
        eye: put.world(camera.eye),
        to: put.world(camera.to),
        ortho: camera.ortho,
    };
    let [visible_only, reject_backfaces, include_touching] = flags;
    let camera = ScreenCamera {
        view_proj: world.view_proj(),
        origin: world.eye,
        forward: sub(world.to, world.eye),
    };
    let options = ScreenSelectOptions {
        view_proj: camera.view_proj.map(|c| c.map(|x| x as f32)),
        camera_origin: camera.origin.map(|c| c as f32),
        camera_forward: camera.forward.map(|c| c as f32),
        is_orthographic: world.ortho,
        visible_only,
        reject_backfaces,
        include_touching,
    };
    let sel = s.select_faces_screen_from(id, shape, &camera, &options);
    if put.at != FAR {
        // Near, the options' own camera is as good.
        assert_eq!(s.select_faces_screen(id, shape, &options), sel);
    }
    sel.unwrap()
}

#[test]
fn screens() {
    let puts = [puts_at(NEAR), puts_at(FAR)].concat();
    agree_for("select_faces_screen", &puts, 0.0, |put| {
        let mut s = scene();
        let id = put.add(&mut s, &lump());
        let mut out = Vec::new();
        for camera in &CAMERAS {
            for shape in &shapes() {
                for flags in [
                    [false, false, true],
                    [true, true, true],
                    [true, false, false],
                    [false, true, false],
                ] {
                    out.extend(ids(on_screen(&mut s, id, shape, camera, put, flags)));
                }
            }
        }
        out
    });
}

#[test]
fn faces_taken_out_stay_where_they_are() {
    agree("extract and split faces", 2e-6, |put| {
        let mut s = scene();
        let id = put.add(&mut s, &lump());
        let sel = s
            .select_faces_in_sphere_world(id, put.world([1.0, 0.3, 0.2]), 0.8, None)
            .unwrap();
        let copy = s.extract_selected_faces(id, &sel).unwrap();
        let mut out = solid(&s, copy, put);
        let (part, _) = s.split_selected_faces(id, &sel).unwrap();
        out.extend(solid(&s, part, put));
        out.extend(solid(&s, id, put));
        s.undo().unwrap();
        out.extend(solid(&s, id, put));
        s.redo().unwrap();
        out.extend(solid(&s, part, put));
        s.delete_selected_faces(id, &[0, 1, 2]).unwrap();
        out.extend(solid(&s, id, put));
        out
    });
}
