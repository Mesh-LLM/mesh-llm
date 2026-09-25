//! Pure payment data types shared by mesh core and the payments engine.
//!
//! No ledger, no SQLite, no wallet provider: mesh core (gossip, peer state,
//! the payment wire) depends on this crate only, so it never links the
//! settlement engine in `mesh-llm-payments`.
#![forbid(unsafe_code)]

pub mod lifetimes;
pub mod pricing;
pub mod terms;
pub mod wire;

pub use terms::RequestTerms;
