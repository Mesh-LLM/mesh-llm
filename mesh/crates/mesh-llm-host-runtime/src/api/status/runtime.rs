//! Host-owned lifecycle-to-management label conversions.
use mesh_llm_control_api::status::runtime::{ActivityPolicyStateLabel, IntentSourceLabel};

// ─── Intent Source Conversion ──────────────────────────────────────────────────

impl From<crate::runtime::IntentSource> for IntentSourceLabel {
    fn from(source: crate::runtime::IntentSource) -> Self {
        match source {
            crate::runtime::IntentSource::StartupConfig => IntentSourceLabel::StartupConfig,
            crate::runtime::IntentSource::LocalCli => IntentSourceLabel::Cli,
            crate::runtime::IntentSource::ApiLoad | crate::runtime::IntentSource::ApiUnload => {
                IntentSourceLabel::ApiRequest
            }
            crate::runtime::IntentSource::OwnerLoad
            | crate::runtime::IntentSource::OwnerUnload
            | crate::runtime::IntentSource::OwnerEnsure
            | crate::runtime::IntentSource::OwnerDrain => IntentSourceLabel::OwnerLifecycle,
            crate::runtime::IntentSource::MeshDemand => IntentSourceLabel::MeshDemand,
        }
    }
}

// ─── Activity Policy State Conversion ──────────────────────────────────────────

impl From<crate::runtime::activity_policy::ActivityPolicyState> for ActivityPolicyStateLabel {
    fn from(state: crate::runtime::activity_policy::ActivityPolicyState) -> Self {
        match state {
            crate::runtime::activity_policy::ActivityPolicyState::Accepting => {
                ActivityPolicyStateLabel::Accepting
            }
            crate::runtime::activity_policy::ActivityPolicyState::AcceptingDeprioritized => {
                ActivityPolicyStateLabel::AcceptingDeprioritized
            }
            crate::runtime::activity_policy::ActivityPolicyState::RemotePaused => {
                ActivityPolicyStateLabel::RemotePaused
            }
            crate::runtime::activity_policy::ActivityPolicyState::AllPaused => {
                ActivityPolicyStateLabel::AllPaused
            }
        }
    }
}
