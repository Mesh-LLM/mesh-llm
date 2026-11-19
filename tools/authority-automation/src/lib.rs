//! The commands the protected authority producers' consumers run: test-model
//! resolution, the runner authority audits, and the file digest helper, plus
//! this executable's bootstrap identity. Full xtask delegates the same
//! commands here, so each has one implementation.

pub mod bootstrap;
pub mod ci_operations;
pub mod file_projection;
pub mod model_registry;
