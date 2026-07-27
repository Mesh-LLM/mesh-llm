use tokio::net::TcpStream;

use super::runtime::ensure_loopback_control_caller_for_peer_addr;
use crate::{
    api::{
        MeshApi,
        http::{respond_error, respond_json},
        status::{
            ActivityOverrideMode, ActivityPolicyStateLabel, ActivityPolicyStatus, DetectorCategory,
        },
    },
    runtime::activity_policy::{ActivityPolicyGuard, ActivityPolicyState, ManualActivityOverride},
};
use mesh_llm_system::activity::HostActivity;

impl ActivityOverrideMode {
    fn into_manual(self) -> ManualActivityOverride {
        match self {
            Self::Auto => ManualActivityOverride::Auto,
            Self::Active => ManualActivityOverride::Active,
            Self::Idle => ManualActivityOverride::Idle,
        }
    }
}

impl From<ManualActivityOverride> for ActivityOverrideMode {
    fn from(value: ManualActivityOverride) -> Self {
        match value {
            ManualActivityOverride::Auto => Self::Auto,
            ManualActivityOverride::Active => Self::Active,
            ManualActivityOverride::Idle => Self::Idle,
        }
    }
}

pub(super) async fn handle_get(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    if !ensure_loopback(stream).await? {
        return Ok(());
    }
    respond_status(stream, activity_guard(state).await).await
}

pub(super) async fn handle_put(
    stream: &mut TcpStream,
    state: &MeshApi,
    body: &str,
) -> anyhow::Result<()> {
    if !ensure_loopback(stream).await? {
        return Ok(());
    }
    let override_mode = match serde_json::from_str::<ActivityOverrideMode>(body) {
        Ok(mode) => mode.into_manual(),
        Err(error) => {
            return respond_error(stream, 400, &format!("invalid override mode: {error}")).await;
        }
    };
    let guard = activity_guard(state).await;
    guard.apply_manual_override(override_mode);
    respond_status(stream, guard).await
}

pub(super) async fn handle_delete(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    if !ensure_loopback(stream).await? {
        return Ok(());
    }
    let guard = activity_guard(state).await;
    guard.apply_manual_override(ManualActivityOverride::Auto);
    respond_status(stream, guard).await
}

async fn ensure_loopback(stream: &mut TcpStream) -> anyhow::Result<bool> {
    ensure_loopback_control_caller_for_peer_addr(stream, stream.peer_addr()).await
}

async fn activity_guard(state: &MeshApi) -> ActivityPolicyGuard {
    state.inner.lock().await.node.activity_policy_guard.clone()
}

async fn respond_status(stream: &mut TcpStream, guard: ActivityPolicyGuard) -> anyhow::Result<()> {
    let status = ActivityPolicyStatus {
        effective_state: match guard.effective_state() {
            ActivityPolicyState::Accepting => ActivityPolicyStateLabel::Accepting,
            ActivityPolicyState::AcceptingDeprioritized => {
                ActivityPolicyStateLabel::AcceptingDeprioritized
            }
            ActivityPolicyState::RemotePaused => ActivityPolicyStateLabel::RemotePaused,
            ActivityPolicyState::AllPaused => ActivityPolicyStateLabel::AllPaused,
        },
        override_mode: guard.manual_override().into(),
        detector_category: match guard.detector_state() {
            HostActivity::Active => DetectorCategory::Active,
            HostActivity::Idle => DetectorCategory::Idle,
            HostActivity::Unknown => DetectorCategory::Unavailable,
        },
    };
    respond_json(stream, 200, &status).await
}
