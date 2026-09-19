//! Standalone Skippy settings and path policy, shared by the CLI and serving.
//!
//! This crate sits below serving and the lifecycle API: it depends only on
//! protocol primitives and path policy, never on `skippy-api` or `skippy-server`.

mod config;

pub use config::{example_config, load_json, validate_config};

pub mod paths;
