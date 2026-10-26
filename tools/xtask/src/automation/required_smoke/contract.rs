use super::Rejection;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Model {
    Dense,
    Recurrent,
}

impl Model {
    pub(crate) const fn artifact_id(self) -> &'static str {
        match self {
            Self::Dense => "smollm2-q8-inference",
            Self::Recurrent => "family-granite-hybrid",
        }
    }

    pub(crate) const fn context_size(self) -> u32 {
        match self {
            Self::Dense => 256,
            Self::Recurrent => 128,
        }
    }

    pub(crate) const fn batch_sizes(self) -> Option<(u32, u32)> {
        match self {
            Self::Dense => None,
            Self::Recurrent => Some((128, 128)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stack {
    Default,
    Constrained,
}

impl Stack {
    pub(crate) const fn bytes(self) -> Option<u32> {
        match self {
            Self::Default => None,
            Self::Constrained => Some(2_097_152),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Variant {
    pub(crate) model: Model,
    pub(crate) stack: Stack,
}

impl Variant {
    pub(crate) const REQUIRED: [Self; 4] = [
        Self {
            model: Model::Dense,
            stack: Stack::Default,
        },
        Self {
            model: Model::Recurrent,
            stack: Stack::Default,
        },
        Self {
            model: Model::Dense,
            stack: Stack::Constrained,
        },
        Self {
            model: Model::Recurrent,
            stack: Stack::Constrained,
        },
    ];
}

pub(crate) enum Attestation {
    Disabled,
    Required { expected: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Check {
    InspectAttestation,
    Runtime,
    Models,
    RuntimeAttestation,
    Chat,
    Stream,
    Auto,
    HeadlessModels,
    HeadlessStatus,
    HeadlessAttestation,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Budget {
    readiness: Duration,
}

impl Budget {
    pub(crate) fn new(readiness: Duration) -> Result<Self, Rejection> {
        if readiness.is_zero() || readiness > Duration::from_secs(86_400) {
            return Err(Rejection::InvalidBudget);
        }
        Ok(Self { readiness })
    }

    pub(super) const fn for_check(self, check: Check) -> Duration {
        match check {
            Check::Runtime | Check::HeadlessModels | Check::HeadlessStatus => self.readiness,
            Check::InspectAttestation
            | Check::Models
            | Check::RuntimeAttestation
            | Check::Chat
            | Check::Stream
            | Check::Auto
            | Check::HeadlessAttestation => Duration::from_secs(60),
        }
    }
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            readiness: Duration::from_secs(180),
        }
    }
}
