//! Actual release inventory file owners and public xtask command qualification.
#![allow(dead_code)]
#[path = "release_inventory_publication/owner_projection.rs"]
mod inventory;
#[path = "../src/process/mod.rs"]
pub mod process;
#[path = "../src/release/inventory/provenance.rs"]
mod provenance;
#[path = "release_inventory_provenance/real_git.rs"]
mod real_git;

#[path = "release_inventory_publication/public_contracts.rs"]
mod public_contracts;
