//! The file formats behind the `fast-mesh` command line.
//!
//! The binary itself (`src/main.rs`) parses arguments and calls the engine
//! crates; [`io`] is a library so that the fuzz targets can reach the parsers.

pub mod io;
