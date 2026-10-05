//! Standalone Skippy settings, shared by the CLI and serving.
//!
//! This crate sits below serving and the lifecycle API: it depends only on
//! protocol primitives, never on `skippy-api` or `skippy-serving`.

mod config;

pub use config::{example_config, load_json, validate_config};

pub mod speculative;

pub mod capacity;
pub mod local_serving;
