//! The local inference observation feed (`local_inference_observation.v1`).
//!
//! Each local inference frontend reports its lifecycle to an
//! [`adapter::ObservationAdapter`], which records executions in the process's
//! [`hub::ObservationHub`]. A plugin connection subscribes
//! ([`subscription`]), gets a snapshot, and a [`delivery`] task sends it the
//! events that follow. Skippy knows nothing of plugins: the adapter sits on
//! the Mesh side of its observer boundary.

pub(crate) mod adapter;
pub(crate) mod delivery;
pub(crate) mod hub;
pub(crate) mod subscription;

use super::proto;
use std::sync::{Arc, OnceLock};

/// This process's hub: one host generation per process.
pub(crate) fn hub() -> Arc<hub::ObservationHub> {
    static HUB: OnceLock<Arc<hub::ObservationHub>> = OnceLock::new();
    HUB.get_or_init(|| Arc::new(hub::ObservationHub::new()))
        .clone()
}

#[cfg(test)]
mod tests;
