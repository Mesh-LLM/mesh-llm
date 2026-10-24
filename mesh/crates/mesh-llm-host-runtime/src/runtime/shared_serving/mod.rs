//! One locally owned model, exposed through independent mesh memberships.

mod admission;
mod backend;
mod configuration;
mod control;
mod host;
mod local_http;
mod membership;

pub use admission::AdmissionStatus;
pub use configuration::{ServingMembershipConfig, SharedServingConfig};
pub use control::run_shared_serving;
pub use host::{SharedServingHost, SharedServingStatus};
pub use membership::{MembershipPhase, MembershipStatus};
