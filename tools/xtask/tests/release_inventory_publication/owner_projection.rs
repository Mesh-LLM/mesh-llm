//! Exact production modules, projected only to give this integration target owning fixture access.
#[path = "../../src/release/inventory/dirty.rs"]
mod dirty;
#[path = "../../src/release/inventory/dirty_file.rs"]
mod dirty_file;
#[path = "../../src/release/inventory/github.rs"]
mod github;
#[path = "../../src/release/inventory/observation.rs"]
mod observation;
#[path = "../../src/release/inventory/publication.rs"]
mod publication;
#[path = "../../src/release/inventory/transport.rs"]
mod transport;
use crate::provenance;
#[cfg(unix)]
#[path = "cli_contracts.rs"]
mod cli_contracts;
#[path = "contracts.rs"]
mod contracts;

#[path = "identity_contracts.rs"]
mod identity_contracts;
