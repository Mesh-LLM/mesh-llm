//! Standalone Skippy command execution and output formatting.
//!
//! Argument parsing lives in `skippy-cli`; this crate executes the parsed
//! commands against the shared Skippy API and renders their JSON output.
//! It intentionally has no dependency on `skippy-server` and adds no
//! serving options types of its own.

pub mod console;
pub mod models;
pub mod runtime;
pub mod split;
