use std::path::Path;

use anyhow::{Context, Result};
use mesh_llm_config::memberships::MembershipsConfig;

/// Schema/path validation only: no credential reads or running host required.
pub fn run_memberships_validate(path: &Path, json: bool) -> Result<()> {
    match MembershipsConfig::load(path) {
        Ok(_) => super::handle_validation_result(path.to_owned(), Vec::new(), json),
        Err(error) => {
            super::print_validation_load_error(Some(path), &error, json)?;
            Err(error).context("memberships config validation failed")
        }
    }
}
