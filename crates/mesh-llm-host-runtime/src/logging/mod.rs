//! Operator logging privacy policy and redaction infrastructure.
//! Centralizes all credential/token/header/query/body/stack sanitization before any log event is serialized or persisted.
//!
//! Note: `#[allow(dead_code)]` and `#![allow(unused_imports)]` are intentional — this module defines the full surface
//! that will be wired into the runtime logging pipeline in subsequent tasks (Todo 6+).

#![allow(dead_code, unused_imports)]

mod bus;
pub mod foundation;
mod lifecycle;
mod limits;
mod persistence;
pub mod policy;
mod registry;
mod runtime_state;
mod sequences;
mod service;
#[cfg(test)]
mod service_tests;
mod writer;

pub use bus::{BusEntry, PushOutcome, ReplayBus};
pub use foundation::LoggingFoundation;
pub use lifecycle::{DuplicateTerminalError, LifecycleGuard, TerminalOutcome};
pub use limits::{DynamicLoggingLimits, LoggingDynamicLimits};
pub use persistence::LogStoreSink;
pub use registry::{RegistryConfig, RequestRegistry, RequestSummaryEntry};
pub use runtime_state::{
    ArtifactCaptureRequest, LoggingRuntimeApplyError, LoggingRuntimeHealth, LoggingRuntimeState,
};
pub use sequences::SequenceGenerators;
pub use service::{Clock, LoggingService, PersistSink, ServiceConfig, SystemClock};
pub use writer::{FailOpenWriter, RecursionGuard};
