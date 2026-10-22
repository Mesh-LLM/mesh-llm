//! Mesh transport primitives. Callers own peer admission, routing and service policy.
pub mod protocol;
mod relay;
pub mod transport;
pub mod transport_iroh;
pub mod tunnel;

pub use relay::{relay_bidirectional, stage_link_delay};
