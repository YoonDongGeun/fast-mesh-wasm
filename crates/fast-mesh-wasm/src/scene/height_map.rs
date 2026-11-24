//! `WasmScene` bindings for height maps (`Scene::height_map_frame`,
//! `Scene::height_map`) and reliefs (`Scene::add_heightfield`, and
//! [`fast_mesh_ops::heightfield`] on its own).
//!
//! A frame crosses as 13 values: `[width, height, origin x, y, z, right x, y,
//! z, up x, y, z, pixelSize, range]`.

use fast_mesh_ops::heightfield::{HeightfieldOptions, heightfield};
use fast_mesh_scene::{HeightMapFrame, HeightMapOptions, Placement};
use wasm_bindgen::prelude::*;

use super::{WasmScene, len, vec3};
use crate::error::Error;
use crate::primitives::pack;

#[wasm_bindgen]
impl WasmScene {
    /// Fits a height map's frame to the meshes `meshes` (flat `[index,
    /// generation, ...]`). `options` is `[direction x, y, z, up x, y, z,
    /// resolution, pixelSize, margin, low, high]`; a NaN `up x`, `pixelSize`
    /// or `low` takes the default (an up of its own, the pixel size of the
    /// resolution, the meshes' own range). Returns the frame.
    pub fn height_map_frame(&self, meshes: &[u32], options: &[f64]) -> Result<Vec<f64>, Error> {
        let ids = self.ids(meshes)?;
        let frame = self
            .scene
            .height_map_frame(&ids, &parse_options(options)?)?;
        Ok(encode_frame(&frame))
    }

    /// Samples the meshes `meshes` in `frame`: the heights of rows
    /// `first_row` up to `end_row`, row by row, NaN where no surface is seen.
    pub fn height_map(
        &mut self,
        meshes: &[u32],
        frame: &[f64],
        first_row: u32,
        end_row: u32,
    ) -> Result<Vec<f32>, Error> {
        let ids = self.ids(meshes)?;
        let frame = parse_frame(frame)?;
        Ok(self.scene.height_map(&ids, &frame, first_row..end_row)?)
    }

    /// Adds the relief of `heights` (a `width` by `height` grid, rows from
    /// the top left) as a new closed, outward-wound mesh, as one undo step
    /// (undo removes it). `options` is `[pixel size x, y, base]`; the
    /// placement is given as for `set_placement`; `normals` turns the mesh's
    /// smooth vertex normals on. Returns `[index, generation, vertices,
    /// triangles]`.
    #[allow(clippy::too_many_arguments)]
    pub fn add_heightfield(
        &mut self,
        heights: &[f32],
        width: u32,
        height: u32,
        options: &[f64],
        rotation: &[f64],
        translation: &[f64],
        scale: Option<Vec<f64>>,
        normals: bool,
    ) -> Result<Vec<u32>, Error> {
        let options = parse_field(options)?;
        let r: [f64; 9] = rotation
            .try_into()
            .map_err(|_| Error::invalid("rotation must have 9 values"))?;
        let scale = match scale {
            None => [1.0; 3],
            Some(s) => vec3(&s, "scale must have 3 values")?,
        };
        let placement = Placement {
            rotation: [[r[0], r[1], r[2]], [r[3], r[4], r[5]], [r[6], r[7], r[8]]],
            translation: vec3(translation, "translation must have 3 values")?,
            scale,
        };
        let report = self
            .scene
            .add_heightfield(heights, width, height, &options, placement)?;
        let mut out = self.added(report.added, normals)?;
        out.extend([len(report.vertices), len(report.triangles)]);
        Ok(out)
    }

    /// Generates the relief `add_heightfield` adds, in its own frame, and
    /// returns `[vertices, triangles, positions as f32 bits..., indices...]`
    /// as `generatePrimitive` does (a static, for the same reason).
    #[wasm_bindgen(js_name = generateHeightfield)]
    pub fn generate_heightfield(
        heights: &[f32],
        width: u32,
        height: u32,
        options: &[f64],
    ) -> Result<Vec<u32>, Error> {
        pack(&heightfield(
            heights,
            width,
            height,
            &parse_field(options)?,
        )?)
    }
}

/// A whole number in `u32` range.
fn whole(v: f64, what: &str) -> Result<u32, Error> {
    if v.fract() == 0.0 && (0.0..=f64::from(u32::MAX)).contains(&v) {
        Ok(v as u32)
    } else {
        Err(Error::invalid(what))
    }
}

fn parse_options(flat: &[f64]) -> Result<HeightMapOptions, Error> {
    let &[
        dx,
        dy,
        dz,
        ux,
        uy,
        uz,
        resolution,
        pixel_size,
        margin,
        low,
        high,
    ] = flat
    else {
        return Err(Error::invalid("height map options must hold 11 values"));
    };
    Ok(HeightMapOptions {
        direction: [dx, dy, dz],
        up: (!ux.is_nan()).then_some([ux, uy, uz]),
        resolution: whole(
            resolution,
            "a height map's resolution must be a whole number",
        )?,
        pixel_size: (!pixel_size.is_nan()).then_some(pixel_size),
        margin,
        range: (!low.is_nan()).then_some([low, high]),
    })
}

fn encode_frame(f: &HeightMapFrame) -> Vec<f64> {
    let mut out = vec![f64::from(f.width), f64::from(f.height)];
    out.extend(f.origin.iter().chain(&f.right).chain(&f.up));
    out.extend([f.pixel_size, f.range]);
    out
}

fn parse_frame(flat: &[f64]) -> Result<HeightMapFrame, Error> {
    let &[
        width,
        height,
        ox,
        oy,
        oz,
        rx,
        ry,
        rz,
        ux,
        uy,
        uz,
        size,
        range,
    ] = flat
    else {
        return Err(Error::invalid("a height map frame must hold 13 values"));
    };
    let what = "a height map's width and height must be whole numbers";
    Ok(HeightMapFrame {
        width: whole(width, what)?,
        height: whole(height, what)?,
        origin: [ox, oy, oz],
        right: [rx, ry, rz],
        up: [ux, uy, uz],
        pixel_size: size,
        range,
    })
}

fn parse_field(flat: &[f64]) -> Result<HeightfieldOptions, Error> {
    let &[x, y, base] = flat else {
        return Err(Error::invalid("height field options must hold 3 values"));
    };
    Ok(HeightfieldOptions {
        pixel_size: [x, y],
        base,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::codes;

    const CUBE: [f64; 24] = [
        0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 1., 1., 0., 1., 0., 1., 1., 1., 1.,
        1.,
    ];
    const CUBE_TRIS: [u32; 36] = [
        0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4, 2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6,
        1, 3, 5, 3, 7, 5,
    ];
    const NAN: f64 = f64::NAN;
    const IDENTITY: [f64; 9] = [1., 0., 0., 0., 1., 0., 0., 0., 1.];

    fn down() -> [f64; 11] {
        [0., 0., -1., NAN, NAN, NAN, 16., NAN, 0., NAN, NAN]
    }

    #[test]
    fn a_cube_maps_and_comes_back_as_a_relief() {
        let mut s = WasmScene::new();
        let a = s.add_mesh(&CUBE, &CUBE_TRIS, None, 0, false).unwrap();
        let frame = s.height_map_frame(&a, &down()).unwrap();
        let p = 1.0 / 16.0;
        let want = [
            16.,
            16.,
            0.5 * p,
            1.0 - 0.5 * p,
            0.,
            1.,
            0.,
            0.,
            0.,
            1.,
            0.,
            p,
            1.,
        ];
        assert_eq!(frame, want);
        let h = s.height_map(&a, &frame, 0, 16).unwrap();
        assert_eq!(h.len(), 256);
        assert!(h.iter().all(|h| (h - 1.0).abs() < 1e-6));
        assert_eq!(s.height_map(&a, &frame, 3, 7).unwrap(), h[48..112]);
        // Its own axes, pixel size and range.
        let mut o = down();
        o[3..6].copy_from_slice(&[1.0, 0.0, 0.0]);
        o[7..].copy_from_slice(&[0.5, 0.25, -1.0, 2.0]);
        let f = s.height_map_frame(&a, &o).unwrap();
        assert_eq!(f[..2], [3.0, 3.0]);
        assert_eq!(f[8..], [1.0, 0.0, 0.0, 0.5, 3.0]);

        s.take_changes();
        let options = [p, p, 0.5];
        let r = s
            .add_heightfield(
                &h,
                16,
                16,
                &options,
                &IDENTITY,
                &[0.5, 0.5, 0.0],
                None,
                true,
            )
            .unwrap();
        assert_eq!(r[2..], [256 + 60, 2 * 225 + 3 * 60 - 2]);
        assert_eq!(s.triangle_slots(r[0], r[1]).unwrap(), r[3]);
        let back = s.height_map(&r[..2], &frame, 0, 16).unwrap();
        for row in 1..15 {
            for col in 1..15 {
                assert!((back[row * 16 + col] - 1.0).abs() < 1e-6);
            }
        }
        assert!(s.undo().unwrap());
        assert!(s.mesh(r[0], r[1]).is_err());
        assert!(s.redo().unwrap());
        assert_eq!(s.mesh(r[0], r[1]).unwrap().triangle_count(), r[3] as usize);

        let raw = WasmScene::generate_heightfield(&h, 16, 16, &options).unwrap();
        assert_eq!(raw[..2], r[2..]);
        assert_eq!(raw.len(), 2 + 3 * (r[2] + r[3]) as usize);
    }

    #[test]
    fn bad_input_is_an_invalid_argument() {
        let mut s = WasmScene::new();
        let a = s.add_mesh(&CUBE, &CUBE_TRIS, None, 0, false).unwrap();
        let code = |e: Error| assert_eq!(e.code, codes::INVALID_ARGUMENT, "{}", e.text);
        code(s.height_map_frame(&a, &down()[..10]).unwrap_err());
        code(s.height_map_frame(&[], &down()).unwrap_err());
        for (i, v) in [(2, 0.0), (6, 1.5), (6, -1.0), (7, 0.0), (8, NAN)] {
            let mut o = down();
            o[i] = v;
            code(s.height_map_frame(&a, &o).unwrap_err());
        }
        assert_eq!(
            s.height_map_frame(&[9, 0], &down()).unwrap_err().code,
            codes::UNKNOWN_MESH
        );
        let frame = s.height_map_frame(&a, &down()).unwrap();
        code(s.height_map(&a, &frame[..12], 0, 1).unwrap_err());
        code(s.height_map(&a, &frame, 0, 17).unwrap_err());
        let mut bad = frame.clone();
        bad[0] = 2.5;
        code(s.height_map(&a, &bad, 0, 1).unwrap_err());

        let mut add = |heights: &[f32], options: &[f64], rotation: &[f64]| {
            s.add_heightfield(heights, 2, 2, options, rotation, &[0.; 3], None, false)
                .unwrap_err()
        };
        let ok = [1.0, 1.0, 1.0];
        code(add(&[0.; 3], &ok, &IDENTITY));
        code(add(&[0.; 4], &ok[..2], &IDENTITY));
        code(add(&[0.; 4], &[1.0, 1.0, 0.0], &IDENTITY));
        code(add(&[0.; 4], &ok, &IDENTITY[..8]));
        code(add(&[-2.0; 4], &ok, &IDENTITY));
        code(WasmScene::generate_heightfield(&[0.; 4], 2, 2, &[1.0]).unwrap_err());
        code(WasmScene::generate_heightfield(&[0.; 4], 1, 4, &ok).unwrap_err());
    }
}
