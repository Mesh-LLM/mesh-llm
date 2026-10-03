//! Versioned canonical logging contracts and lifecycle invariants.
//!
//! This module defines the semantic event types used by the logging system.
//! `OutputEvent` remains a presentation adapter and is never persisted raw.

pub mod artifacts;
pub mod envelope;
pub mod events;
pub mod identifiers;
pub mod lifecycle;
mod presentation;
pub mod proxy;
pub mod replay;
pub mod summaries;
pub mod timestamp;

#[cfg(test)]
mod tests;
