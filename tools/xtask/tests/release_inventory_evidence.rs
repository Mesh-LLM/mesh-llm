//! Actual finite Git and inert-gh evidence contracts, without remote access.
#[path = "../src/release/inventory/evidence.rs"]
mod evidence;
#[cfg(unix)]
#[path = "release_inventory_evidence/gh_contracts.rs"]
mod gh_contracts;
#[path = "release_inventory_evidence/git_contracts.rs"]
mod git_contracts;
#[path = "../src/release/inventory/github.rs"]
mod github;
#[path = "../src/process/mod.rs"]
pub mod process;
#[path = "../src/release/inventory/provenance.rs"]
mod provenance;
#[path = "release_inventory_provenance/real_git.rs"]
mod real_git;
#[path = "../src/release/inventory/remotes.rs"]
mod remotes;
#[path = "../src/release/inventory/transport.rs"]
mod transport;
