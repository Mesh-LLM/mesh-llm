//! Standalone Skippy command execution and output formatting.
//!
//! The model-command argument contract lives here; this crate executes parsed
//! commands against the shared Skippy API and renders human and JSON output.
//! It intentionally has no dependency on `skippy-serving` and adds no
//! serving options types of its own.

pub mod console;
pub mod models;
pub mod prompt;
pub mod runtime;
pub mod split;
