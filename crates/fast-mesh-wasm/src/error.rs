//! Errors thrown to JavaScript.
//!
//! Every fallible binding returns [`Error`], which crosses the boundary as a
//! `JsError` whose message is `"<CODE>: <text>"`. The TypeScript layer
//! parses the code into `FastMeshError { code, message }`.

use fast_mesh_bvh::BvhError;
use fast_mesh_fields::FieldsError;
use fast_mesh_ops::OpsError;
use fast_mesh_query::QueryError;
use fast_mesh_scene::SceneError;
use fast_mesh_store::StoreError;
use wasm_bindgen::{JsError, JsValue};

/// The error codes; `packages/fast-mesh-wasm/src/core.ts` lists the same.
/// `INSTANCE_POISONED` is set by the TypeScript layer alone, after a trap.
pub(crate) mod codes {
    pub const UNKNOWN_MESH: &str = "UNKNOWN_MESH";
    pub const UNKNOWN_FIELD: &str = "UNKNOWN_FIELD";
    pub const UNKNOWN_SECTION: &str = "UNKNOWN_SECTION";
    pub const INVALID_ARGUMENT: &str = "INVALID_ARGUMENT";
    pub const INVALID_INDEX: &str = "INVALID_INDEX";
    pub const STALE_BVH: &str = "STALE_BVH";
    pub const STEP_NOT_OPEN: &str = "STEP_NOT_OPEN";
    pub const CORRUPTED: &str = "CORRUPTED";
    pub const OUT_OF_MEMORY: &str = "OUT_OF_MEMORY";
    pub const STALE_LOOPS: &str = "STALE_LOOPS";
}

/// An error with its code, converted into a `JsError` at the boundary.
///
/// A plain struct rather than a `JsError`, so native tests can inspect it
/// (constructing a `JsError` needs a JavaScript host).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub code: &'static str,
    pub text: String,
}

impl Error {
    /// An `INVALID_ARGUMENT` error for input checked by the bindings.
    pub(crate) fn invalid(text: &str) -> Error {
        Error {
            code: codes::INVALID_ARGUMENT,
            text: text.to_owned(),
        }
    }

    /// The message JavaScript sees: `"<CODE>: <text>"`.
    pub(crate) fn message(&self) -> String {
        format!("{}: {}", self.code, self.text)
    }
}

impl From<Error> for JsValue {
    fn from(e: Error) -> JsValue {
        JsError::new(&e.message()).into()
    }
}

fn store_code(e: &StoreError) -> &'static str {
    match e {
        StoreError::OutOfMemory => codes::OUT_OF_MEMORY,
        StoreError::InvalidVertex(_) | StoreError::InvalidTriangle(_) => codes::INVALID_INDEX,
        _ => codes::INVALID_ARGUMENT,
    }
}

fn query_code(e: &QueryError) -> &'static str {
    match e {
        QueryError::OutOfMemory => codes::OUT_OF_MEMORY,
        QueryError::StaleBvh => codes::STALE_BVH,
        _ => codes::INVALID_ARGUMENT,
    }
}

/// The code of a scene error (spec §4.4): the scene's own variants map one
/// to one, any out-of-memory error to `OUT_OF_MEMORY`, store index errors to
/// `INVALID_INDEX`, and every other nested error to `INVALID_ARGUMENT`.
pub(crate) fn code(e: &SceneError) -> &'static str {
    match e {
        SceneError::UnknownMesh => codes::UNKNOWN_MESH,
        SceneError::UnknownField => codes::UNKNOWN_FIELD,
        SceneError::UnknownSection => codes::UNKNOWN_SECTION,
        SceneError::StaleBvh(_) => codes::STALE_BVH,
        SceneError::InvalidInput(_) => codes::INVALID_ARGUMENT,
        SceneError::StepNotOpen => codes::STEP_NOT_OPEN,
        SceneError::Corrupted => codes::CORRUPTED,
        SceneError::Store(e) => store_code(e),
        SceneError::Bvh(BvhError::OutOfMemory) => codes::OUT_OF_MEMORY,
        SceneError::Bvh(BvhError::Store(e)) => store_code(e),
        SceneError::Bvh(_) => codes::INVALID_ARGUMENT,
        SceneError::Query(e) => query_code(e),
        SceneError::Fields(FieldsError::OutOfMemory) => codes::OUT_OF_MEMORY,
        SceneError::Fields(FieldsError::Store(e)) => store_code(e),
        SceneError::Fields(FieldsError::Query(e)) => query_code(e),
        SceneError::Fields(_) => codes::INVALID_ARGUMENT,
        SceneError::Ops(OpsError::StaleLoops) => codes::STALE_LOOPS,
        SceneError::Ops(OpsError::Store(e)) => store_code(e),
        SceneError::Ops(OpsError::Query(e)) => query_code(e),
        SceneError::Ops(_) => codes::INVALID_ARGUMENT,
    }
}

impl From<SceneError> for Error {
    fn from(e: SceneError) -> Error {
        Error {
            code: code(&e),
            text: e.to_string(),
        }
    }
}

impl From<StoreError> for Error {
    fn from(e: StoreError) -> Error {
        SceneError::Store(e).into()
    }
}
