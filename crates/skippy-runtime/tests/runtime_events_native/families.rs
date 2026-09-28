//! Classification of native runtime-event records into the capability
//! families the probe confirms, and a per-family tally.

use skippy_runtime::{NativeEventRecord, RuntimeEventKind as Kind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    ModelLoadV2,
    Kv,
    Device,
    Diagnostic,
    Unload,
}

impl Family {
    pub const ALL: [Self; 5] = [
        Self::ModelLoadV2,
        Self::Kv,
        Self::Device,
        Self::Diagnostic,
        Self::Unload,
    ];

    pub fn of(kind: Kind) -> Option<Self> {
        match kind {
            Kind::ModelLoadPhaseChanged
            | Kind::ModelLoadMemoryAllocated
            | Kind::ModelLoadTensorsOffloaded
            | Kind::ModelLoadTokenizerReady
            | Kind::ModelLoadAuxComponentReady => Some(Self::ModelLoadV2),
            Kind::KvInitialized
            | Kind::KvPressureCrossed
            | Kind::KvPressureCleared
            | Kind::KvContextApproachingCapacity
            | Kind::KvContextCapacityExhausted => Some(Self::Kv),
            Kind::DeviceBackendInitialized
            | Kind::DeviceReady
            | Kind::DeviceDegraded
            | Kind::DeviceUnavailable
            | Kind::DeviceRecovered
            | Kind::DeviceLost
            | Kind::DeviceResourceAllocated
            | Kind::DeviceOutOfMemory
            | Kind::DeviceFallbackActivated => Some(Self::Device),
            Kind::DiagnosticWarningRaised
            | Kind::DiagnosticWarningCleared
            | Kind::DiagnosticRecoverableFailure
            | Kind::DiagnosticFatalFailure
            | Kind::DiagnosticInvariantViolation => Some(Self::Diagnostic),
            Kind::UnloadStarted
            | Kind::UnloadCompleted
            | Kind::UnloadFailed
            | Kind::UnloadForced
            | Kind::UnloadSessionDraining => Some(Self::Unload),
            Kind::ModelOpenStarted
            | Kind::ModelOpenProgress
            | Kind::BackendDeviceSelected
            | Kind::ModelOpenFinished
            | Kind::ModelOpenFailedHandled
            | Kind::Unknown(_) => None,
        }
    }

    pub fn feature_bit(self) -> u64 {
        match self {
            Self::ModelLoadV2 => skippy_ffi::FEATURE_MODEL_LOAD_EVENTS_V2,
            Self::Kv => skippy_ffi::FEATURE_KV_EVENTS,
            Self::Device => skippy_ffi::FEATURE_DEVICE_EVENTS,
            Self::Diagnostic => skippy_ffi::FEATURE_DIAGNOSTIC_EVENTS,
            Self::Unload => skippy_ffi::FEATURE_UNLOAD_EVENTS,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::ModelLoadV2 => "model_load_events_v2",
            Self::Kv => "kv_events",
            Self::Device => "device_events",
            Self::Diagnostic => "diagnostic_events",
            Self::Unload => "unload_events",
        }
    }

    /// Families whose absence during a real load + session fails the gate.
    /// KV and diagnostic absence is recorded as `NOT_OBSERVED` only.
    pub fn required_when_confirmed(self) -> bool {
        match self {
            Self::ModelLoadV2 | Self::Device | Self::Unload => true,
            Self::Kv | Self::Diagnostic => false,
        }
    }

    fn index(self) -> usize {
        match self {
            Self::ModelLoadV2 => 0,
            Self::Kv => 1,
            Self::Device => 2,
            Self::Diagnostic => 3,
            Self::Unload => 4,
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct FamilyTally {
    counts: [usize; 5],
}

impl FamilyTally {
    pub fn add(&mut self, records: &[NativeEventRecord]) {
        for record in records {
            if let Some(family) = Family::of(record.to_event().kind) {
                self.counts[family.index()] += 1;
            }
        }
    }

    pub fn merge(&mut self, other: &Self) {
        for (total, added) in self.counts.iter_mut().zip(other.counts) {
            *total += added;
        }
    }

    pub fn count(&self, family: Family) -> usize {
        self.counts[family.index()]
    }

    pub fn structured(&self) -> usize {
        self.counts.iter().sum()
    }
}

/// Take everything the process-global reporter ring has buffered.
pub fn drain_global() -> Vec<NativeEventRecord> {
    let mut records = Vec::new();
    skippy_runtime::drain_runtime_events(&mut records, usize::MAX);
    records
}
