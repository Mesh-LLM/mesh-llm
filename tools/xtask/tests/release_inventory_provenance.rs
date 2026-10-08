//! Actual finite native Git repositories exercise the release inventory provenance owner.
#[path = "release_inventory_provenance/contracts.rs"]
mod contracts;
#[path = "../src/process/mod.rs"]
pub mod process;
#[path = "../src/release/inventory/provenance.rs"]
mod provenance;
#[path = "release_inventory_provenance/real_git.rs"]
mod real_git;
