use std::time::Duration;

use crate::{EventId, NativeSourceEnvelope, OperationScope, RuntimeFact};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeEventSchemaVersion(pub u16);

impl RuntimeEventSchemaVersion {
    pub const CURRENT: Self = Self(1);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProducerSource {
    Native,
    Rust,
    Reconciled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Trace,
    Debug,
    Info,
    Warning,
    Error,
    Fatal,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeEventEnvelope {
    pub schema_version: RuntimeEventSchemaVersion,
    pub event_id: EventId,
    pub operation: OperationScope,
    pub producer: ProducerSource,
    pub severity: Severity,
    pub wall_clock_unix_ns: u64,
    pub process_monotonic_time: Duration,
    pub native_source: Option<NativeSourceEnvelope>,
    pub fact: RuntimeFact,
}

impl RuntimeEventEnvelope {
    #[must_use]
    pub fn into_parts(self) -> (RuntimeFact, Option<NativeSourceEnvelope>) {
        (self.fact, self.native_source)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CarrierLocation {
    RuntimeEventEnvelope,
    NativeSourceEnvelope,
    FamilyFact,
}

impl CarrierLocation {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RuntimeEventEnvelope => "RuntimeEventEnvelope",
            Self::NativeSourceEnvelope => "NativeSourceEnvelope",
            Self::FamilyFact => "FamilyFact",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CarrierKind {
    SchemaVersion,
    CategoryKind,
    Producer,
    Severity,
    WallClockTime,
    ProcessMonotonicTime,
    NativeMonotonicTime,
    NativeSequence,
    ProcessInstanceId,
    EventSequence,
    RootOperationId,
    ChildOperationId,
    EmitterIdentity,
    ScopeIdentities,
    PreviousCurrentState,
    Progress,
    OutcomeReason,
    Duration,
    BoundedSummaries,
}

impl CarrierKind {
    pub const ALL: &'static [Self] = &[
        Self::SchemaVersion,
        Self::CategoryKind,
        Self::Producer,
        Self::Severity,
        Self::WallClockTime,
        Self::ProcessMonotonicTime,
        Self::NativeMonotonicTime,
        Self::NativeSequence,
        Self::ProcessInstanceId,
        Self::EventSequence,
        Self::RootOperationId,
        Self::ChildOperationId,
        Self::EmitterIdentity,
        Self::ScopeIdentities,
        Self::PreviousCurrentState,
        Self::Progress,
        Self::OutcomeReason,
        Self::Duration,
        Self::BoundedSummaries,
    ];

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SchemaVersion => "schema_version",
            Self::CategoryKind => "category_kind",
            Self::Producer => "producer",
            Self::Severity => "severity",
            Self::WallClockTime => "wall_clock_time",
            Self::ProcessMonotonicTime => "process_monotonic_time",
            Self::NativeMonotonicTime => "native_monotonic_time",
            Self::NativeSequence => "native_sequence",
            Self::ProcessInstanceId => "process_instance_id",
            Self::EventSequence => "event_sequence",
            Self::RootOperationId => "root_operation_id",
            Self::ChildOperationId => "child_operation_id",
            Self::EmitterIdentity => "emitter_identity",
            Self::ScopeIdentities => "scope_identities",
            Self::PreviousCurrentState => "previous_current_state",
            Self::Progress => "progress",
            Self::OutcomeReason => "outcome_reason",
            Self::Duration => "duration",
            Self::BoundedSummaries => "bounded_summaries",
        }
    }

    #[must_use]
    pub const fn location(self) -> CarrierLocation {
        match self {
            Self::SchemaVersion
            | Self::CategoryKind
            | Self::Producer
            | Self::Severity
            | Self::WallClockTime
            | Self::ProcessMonotonicTime
            | Self::ProcessInstanceId
            | Self::EventSequence
            | Self::RootOperationId
            | Self::ChildOperationId => CarrierLocation::RuntimeEventEnvelope,
            Self::NativeMonotonicTime | Self::NativeSequence | Self::EmitterIdentity => {
                CarrierLocation::NativeSourceEnvelope
            }
            Self::ScopeIdentities
            | Self::PreviousCurrentState
            | Self::Progress
            | Self::OutcomeReason
            | Self::Duration
            | Self::BoundedSummaries => CarrierLocation::FamilyFact,
        }
    }
}
