use crate::api::status::ModelTargetCapacityAdviceState;

// ─── Intent types (desired-state declarations) ───────────────────────────────

/// Source that generated this intent.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum IntentSource {
    StartupConfig,
    ApiLoad,
    ApiUnload,
    #[expect(
        dead_code,
        reason = "reserved source for the local CLI intent producer"
    )]
    LocalCli,
    OwnerLoad,
    OwnerUnload,
    OwnerEnsure,
    OwnerDrain,
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "advisory mesh demand is modeled and covered by pure reconciliation tests"
        )
    )]
    MeshDemand,
}

impl IntentSource {
    pub(crate) fn precedence(self) -> u8 {
        match self {
            Self::ApiLoad | Self::ApiUnload | Self::LocalCli => 4,
            Self::OwnerLoad | Self::OwnerUnload | Self::OwnerEnsure | Self::OwnerDrain => 3,
            Self::StartupConfig => 2,
            Self::MeshDemand => 1,
        }
    }

    pub(crate) fn is_maintained(self) -> bool {
        matches!(self, Self::StartupConfig | Self::OwnerEnsure)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DesiredModelState {
    Present,
    Absent,
    Draining,
}

impl DesiredModelState {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Present => "present",
            Self::Absent => "absent",
            Self::Draining => "draining",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum IntentPersistence {
    Process,
    Session,
    Ephemeral,
}

/// Bounded desired-state record used by reconciliation and status surfaces.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DesiredRuntimeIntent {
    pub(crate) intent_id: String,
    pub(crate) canonical_model_ref: String,
    pub(crate) profile: String,
    pub(crate) instance_target: Option<String>,
    pub(crate) desired_state: DesiredModelState,
    pub(crate) source: IntentSource,
    pub(crate) persistence: IntentPersistence,
    pub(crate) created_at_secs: u64,
    pub(crate) updated_at_secs: u64,
    pub(crate) last_error: Option<String>,
}

impl DesiredRuntimeIntent {
    pub(crate) fn set_last_error(&mut self, error: impl Into<String>) {
        let mut error = error.into();
        if error.len() > 512 {
            let mut end = 512;
            while !error.is_char_boundary(end) {
                end -= 1;
            }
            error.truncate(end);
        }
        self.last_error = Some(error);
    }
}

/// A typed intent expressing a desired model lifecycle transition.
/// All sources (startup config/CLI, API requests, owner commands, mesh demand)
/// translate into one of these intents. The reconciler is the sole consumer.
#[derive(Debug)]
pub(crate) enum ModelIntent {
    /// Request to load a model (add to desired set).
    Load {
        intent_id: Option<String>,
        spec: String,
        config_model_id: Option<String>,
        profile: String,
        source: IntentSource,
        /// If Some, caller awaits synchronous result via this channel.
        completion:
            Option<tokio::sync::oneshot::Sender<anyhow::Result<crate::api::RuntimeLoadResponse>>>,
    },
    /// Request to unload a model (remove from desired set).
    Unload {
        intent_id: Option<String>,
        /// Canonical model identity when the execution target is a specific instance.
        canonical_model_ref: Option<String>,
        target: mesh_llm_node::serving::UnloadTarget,
        options: mesh_llm_node::serving::UnloadOptions,
        source: IntentSource,
        /// If Some, caller awaits synchronous result via this channel.
        completion:
            Option<tokio::sync::oneshot::Sender<anyhow::Result<crate::api::RuntimeUnloadResponse>>>,
    },
}

impl ModelIntent {
    #[expect(
        dead_code,
        reason = "source projection helper is retained for typed intent producers"
    )]
    pub(crate) fn source(&self) -> IntentSource {
        match self {
            ModelIntent::Load { source, .. } => *source,
            ModelIntent::Unload { source, .. } => *source,
        }
    }
}

// ─── Reconciliation candidate / action types ────────────────────────────────

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ModelTargetReconciliationCandidate {
    pub(crate) rank: usize,
    pub(crate) model_ref: String,
    pub(crate) profile: String,
    pub(crate) model_name: Option<String>,
    pub(crate) wanted: bool,
    pub(crate) wanted_reason: Option<&'static str>,
    pub(crate) request_count: u64,
    pub(crate) last_active_secs_ago: Option<u64>,
    pub(crate) serving_node_count: usize,
    pub(crate) capacity_state: ModelTargetReconciliationCapacityState,
    pub(crate) local_path: Option<std::path::PathBuf>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ModelTargetReconciliationCapacityState {
    AlreadyServing,
    SingleNodeFit,
    SplitCandidate,
    InsufficientCapacity,
    UnknownModelSize,
    UnknownCapacity,
    NoEligibleHosts,
}

impl From<ModelTargetCapacityAdviceState> for ModelTargetReconciliationCapacityState {
    fn from(value: ModelTargetCapacityAdviceState) -> Self {
        match value {
            ModelTargetCapacityAdviceState::AlreadyServing => Self::AlreadyServing,
            ModelTargetCapacityAdviceState::SingleNodeFit => Self::SingleNodeFit,
            ModelTargetCapacityAdviceState::SplitCandidate => Self::SplitCandidate,
            ModelTargetCapacityAdviceState::InsufficientCapacity => Self::InsufficientCapacity,
            ModelTargetCapacityAdviceState::UnknownModelSize => Self::UnknownModelSize,
            ModelTargetCapacityAdviceState::UnknownCapacity => Self::UnknownCapacity,
            ModelTargetCapacityAdviceState::NoEligibleHosts => Self::NoEligibleHosts,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ModelTargetReconciliationAction {
    pub(crate) model_ref: String,
    pub(crate) profile: String,
    pub(crate) model_name: Option<String>,
    pub(crate) load_spec: std::path::PathBuf,
    pub(crate) replace_model_ref: Option<String>,
}
