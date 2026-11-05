//! Writes a closed torus, for measuring the CLI on large files:
//!
//! ```sh
//! cargo run --release -p fast-mesh-cli --example gen_torus -- <triangles> <out> [holes] [offset]
//! ```
//!
//! `holes` removes that many well-separated triangles (default 0) and `offset`
//! moves the torus along all three axes (default 0).

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use fast_mesh_cli::io::{Format, MeshData, write_mesh};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [triangles, out, rest @ ..] = args.as_slice() else {
        return Err("usage: gen_torus <triangles> <out> [holes] [offset]".into());
    };
    let triangles: usize = triangles.parse()?;
    let holes: usize = rest.first().map_or(Ok(0), |s| s.parse())?;
    let offset: f64 = rest.get(1).map_or(Ok(0.0), |s| s.parse())?;

    let around = ((triangles as f64 / 2.0).sqrt().ceil() as usize).max(3);
    let tube = (triangles / (2 * around)).max(3);
    let (major, minor) = (3.0, 1.0);
    let mut positions = Vec::with_capacity(3 * around * tube);
    for i in 0..around {
        let u = std::f64::consts::TAU * i as f64 / around as f64;
        for j in 0..tube {
            let v = std::f64::consts::TAU * j as f64 / tube as f64;
            let r = major + minor * v.cos();
            positions.extend([
                r * u.cos() + offset,
                r * u.sin() + offset,
                minor * v.sin() + offset,
            ]);
        }
    }
    let total = 2 * around * tube;
    let every = total.checked_div(holes).unwrap_or(usize::MAX);
    let mut indices = Vec::with_capacity(3 * total);
    let id = |i: usize, j: usize| (i % around * tube + j % tube) as u32;
    let mut t = 0;
    for i in 0..around {
        for j in 0..tube {
            let (a, b, c, d) = (id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1));
            for tri in [[a, b, c], [a, c, d]] {
                if t % every != every / 2 {
                    indices.extend(tri);
                }
                t += 1;
            }
        }
    }
    let path = Path::new(out);
    let mut file = BufWriter::new(File::create(path)?);
    write_mesh(
        &MeshData { positions, indices },
        Format::from_path(path)?,
        false,
        &mut file,
    )?;
    Ok(())
}
