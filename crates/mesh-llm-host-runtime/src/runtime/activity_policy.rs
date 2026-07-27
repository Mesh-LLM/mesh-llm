//! Host-activity-driven inference admission policy.
//!
//! Enforces configurable pause behavior at ingress boundaries based on host
//! activity detection. Models stay loaded during pauses; only new inference
//! requests are rejected. Resume follows the idle debounce from config.

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use mesh_llm_config::{ActivityAdvertisement, ActivityResponse, RuntimeActivityConfig};
use mesh_llm_system::activity::{
    ActivityOverride, ActivityPolicy, HostActivity, HostActivityMonitor,
    NativeHostActivityDetector, NativePriorityController, PrioritySession, SystemClock,
};

// ── Effective admission state (maps to proto InferenceAdmissionState) ────────

/// Coarse inference admission state advertised to peers and enforced at ingress.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ActivityPolicyState {
    /// Normal operation — all inference accepted.
    #[default]
    Accepting,
    /// `reduce_priority` active — all inference accepted but process priority lowered.
    AcceptingDeprioritized,
    /// `pause_remote` active — remote work paused; local + plugin + management allowed.
    RemotePaused,
    /// `pause_all` active — all inference paused; management API still allowed.
    AllPaused,
}

impl ActivityPolicyState {
    pub fn to_proto(self) -> mesh_llm_protocol::proto::node::InferenceAdmissionState {
        match self {
            Self::Accepting => mesh_llm_protocol::proto::node::InferenceAdmissionState::Accepting,
            Self::AcceptingDeprioritized => {
                mesh_llm_protocol::proto::node::InferenceAdmissionState::AcceptingDeprioritized
            }
            Self::RemotePaused => {
                mesh_llm_protocol::proto::node::InferenceAdmissionState::RemotePaused
            }
            Self::AllPaused => mesh_llm_protocol::proto::node::InferenceAdmissionState::AllPaused,
        }
    }

    pub fn is_blocking(self) -> bool {
        matches!(self, Self::RemotePaused | Self::AllPaused)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ActivityAdvertisementDecision {
    pub(crate) admission_state: Option<mesh_llm_protocol::proto::node::InferenceAdmissionState>,
    pub(crate) withdraw_model_availability: bool,
}

/// Reduce configured activity policy to the privacy-safe gossip representation.
///
/// Availability withdrawal is intentionally explicit for every advertising
/// mode when a peer must stop routing work here. Older peers ignore the new
/// enum, so clearing both serving and hosted models is the compatibility path.
pub(crate) fn activity_advertisement_decision(
    enabled: bool,
    mode: ActivityAdvertisement,
    state: ActivityPolicyState,
    public_mesh: bool,
) -> ActivityAdvertisementDecision {
    if !enabled || mode == ActivityAdvertisement::None {
        return ActivityAdvertisementDecision::default();
    }

    let admission_state = match mode {
        ActivityAdvertisement::None | ActivityAdvertisement::AvailabilityOnly => None,
        ActivityAdvertisement::CoarseState => Some(state.to_proto()),
        ActivityAdvertisement::PrivateCoarseState if public_mesh => None,
        ActivityAdvertisement::PrivateCoarseState => Some(state.to_proto()),
    };

    ActivityAdvertisementDecision {
        admission_state,
        withdraw_model_availability: state.is_blocking(),
    }
}

// ── Ingress type classification ──────────────────────────────────────────────

/// Categorizes request sources for admission policy evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IngressType {
    /// Local HTTP OpenAI requests (port 9337 / API proxy).
    LocalOpenAi,
    /// Remote QUIC tunnelled HTTP from mesh peers.
    RemoteQuicHttp,
    /// Inbound stage transport for skippy split serving.
    StageTransport,
    /// Plugin model dispatch (external plugin inference endpoints).
    PluginDispatch,
    #[cfg(test)]
    /// Management API routes are represented only in policy matrix tests.
    ManagementApi,
}

impl IngressType {
    pub fn is_remote(self) -> bool {
        matches!(self, Self::RemoteQuicHttp | Self::StageTransport)
    }

    #[cfg(test)]
    pub fn is_inference(self) -> bool {
        !matches!(self, Self::ManagementApi)
    }
}

// ── Admission result ────────────────────────────────────────────────────────

/// Bounded retryable admission decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionResult {
    /// Request is allowed to proceed.
    Allowed,
    /// Request is paused with a reason and optional retry hint.
    Paused {
        /// Human-readable reason for the pause.
        reason: &'static str,
        /// Suggested retry-after duration in seconds (None means indefinite).
        retry_after_secs: Option<u64>,
    },
}

#[cfg(test)]
impl AdmissionResult {
    pub fn is_allowed(self) -> bool {
        matches!(self, Self::Allowed)
    }

    pub fn is_paused(self) -> bool {
        !self.is_allowed()
    }
}

/// Session-only manual override for activity state.
pub type ManualActivityOverride = ActivityOverride;

// ── Shared runtime state (Arc<Mutex<>> for cross-task sharing) ───────────────

/// Mutable runtime state updated by the control loop; read by ingress handlers.
pub struct ActivityPolicyRuntimeState {
    /// Latest raw detector sample (Active / Idle / Unknown).
    detector_state: HostActivity,
    /// Session-only manual override.
    manual_override: ManualActivityOverride,
    /// Whether reduce_priority failed and is degraded.
    priority_degraded: bool,
    /// Cached effective policy state (recomputed on update).
    effective_state: ActivityPolicyState,
}

impl ActivityPolicyRuntimeState {
    pub fn new(response: ActivityResponse) -> Self {
        Self {
            detector_state: HostActivity::Unknown,
            manual_override: ManualActivityOverride::Auto,
            priority_degraded: false,
            effective_state: compute_effective_state(
                HostActivity::Unknown,
                ManualActivityOverride::Auto,
                response,
                false,
            ),
        }
    }

    pub fn update_detector(&mut self, state: HostActivity, response: ActivityResponse) {
        self.detector_state = state;
        self.effective_state = compute_effective_state(
            state,
            self.manual_override,
            response,
            self.priority_degraded,
        );
    }

    pub fn apply_manual_override(
        &mut self,
        override_mode: ManualActivityOverride,
        response: ActivityResponse,
    ) {
        self.manual_override = override_mode;
        self.effective_state = compute_effective_state(
            self.detector_state,
            override_mode,
            response,
            self.priority_degraded,
        );
    }

    /// Mark priority reduction as degraded.
    pub fn mark_priority_degraded(&mut self, response: ActivityResponse) {
        self.priority_degraded = true;
        self.effective_state = compute_effective_state(
            self.detector_state,
            self.manual_override,
            response,
            self.priority_degraded,
        );
    }

    pub fn clear_priority_degraded(&mut self, response: ActivityResponse) {
        self.priority_degraded = false;
        self.effective_state = compute_effective_state(
            self.detector_state,
            self.manual_override,
            response,
            self.priority_degraded,
        );
    }

    pub fn effective_state(&self) -> ActivityPolicyState {
        self.effective_state
    }

    pub fn priority_degraded(&self) -> bool {
        self.priority_degraded
    }

    pub fn detector_state(&self) -> HostActivity {
        self.detector_state
    }

    pub fn manual_override(&self) -> ManualActivityOverride {
        self.manual_override
    }
}

// ── Pure computation (testable, no shared state) ─────────────────────────────

/// Compute the effective admission policy state from inputs.
///
/// This is the core decision function — pure and deterministic for testing.
pub fn compute_effective_state(
    detector: HostActivity,
    manual_override: ManualActivityOverride,
    response: ActivityResponse,
    priority_degraded: bool,
) -> ActivityPolicyState {
    // Resolve effective activity through manual override.
    let resolved = manual_override.resolve(detector);

    // Unknown detector → treat as active (don't pause on uncertainty).
    if resolved == HostActivity::Unknown {
        return ActivityPolicyState::Accepting;
    }

    match response {
        ActivityResponse::PauseRemote => {
            if resolved == HostActivity::Active {
                ActivityPolicyState::RemotePaused
            } else {
                ActivityPolicyState::Accepting
            }
        }
        ActivityResponse::PauseAll => {
            if resolved == HostActivity::Active {
                ActivityPolicyState::AllPaused
            } else {
                ActivityPolicyState::Accepting
            }
        }
        ActivityResponse::ReducePriority => {
            // Always accepting; priority_degraded flag tracks failures separately.
            if resolved == HostActivity::Active && !priority_degraded {
                ActivityPolicyState::AcceptingDeprioritized
            } else {
                ActivityPolicyState::Accepting
            }
        }
    }
}

/// Check admission for a given ingress type against the current effective state.
///
/// Rules:
/// - `pause_remote`: RemoteQuicHttp + StageTransport = Paused; others = Allowed
/// - `pause_all`: All inference types = Paused; management routes do not call this guard
/// - `reduce_priority` (AcceptingDeprioritized): All = Allowed
pub fn check_admission(state: ActivityPolicyState, ingress_type: IngressType) -> AdmissionResult {
    match state {
        ActivityPolicyState::Accepting => AdmissionResult::Allowed,

        // Reduce priority: all allowed, but tracked as degraded.
        ActivityPolicyState::AcceptingDeprioritized => AdmissionResult::Allowed,

        // Pause remote: block only remote ingress types.
        ActivityPolicyState::RemotePaused => {
            if ingress_type.is_remote() {
                AdmissionResult::Paused {
                    reason: "remote inference paused (host activity)",
                    retry_after_secs: None,
                }
            } else {
                AdmissionResult::Allowed
            }
        }

        // Pause all: block all inference; management API always allowed.
        ActivityPolicyState::AllPaused => {
            #[cfg(test)]
            if !ingress_type.is_inference() {
                return AdmissionResult::Allowed;
            }
            AdmissionResult::Paused {
                reason: "all inference paused (host activity)",
                retry_after_secs: None,
            }
        }
    }
}

// ── Shared guard (Arc<Mutex<>> wrapper for ingress handlers) ────────────────

/// Thread-safe shared admission policy state.
///
/// The runtime control loop writes to this; ingress handlers read snapshots.
#[derive(Clone)]
pub struct ActivityPolicyGuard {
    inner: Arc<Mutex<ActivityPolicyRuntimeState>>,
    config: RuntimeActivityConfig,
}

impl ActivityPolicyGuard {
    /// Create a new shared policy guard from config.
    pub fn new(config: &RuntimeActivityConfig) -> Self {
        Self {
            inner: Arc::new(Mutex::new(ActivityPolicyRuntimeState::new(config.response))),
            config: config.clone(),
        }
    }

    pub(crate) fn advertisement_decision(
        &self,
        public_mesh: bool,
    ) -> ActivityAdvertisementDecision {
        activity_advertisement_decision(
            self.config.enabled,
            self.config.advertisement,
            self.effective_state(),
            public_mesh,
        )
    }

    /// Update detector state (called by runtime control loop).
    pub fn update_detector_state(&self, state: HostActivity) {
        let mut inner = self.inner.lock().unwrap();
        inner.update_detector(state, self.config.response);
    }

    /// Apply manual override (called by API / CLI).
    pub fn apply_manual_override(&self, override_mode: ManualActivityOverride) {
        let mut inner = self.inner.lock().unwrap();
        inner.apply_manual_override(override_mode, self.config.response);
    }

    pub fn mark_priority_degraded(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.mark_priority_degraded(self.config.response);
    }

    pub fn clear_priority_degraded(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.clear_priority_degraded(self.config.response);
    }

    pub fn priority_degraded(&self) -> bool {
        let inner = self.inner.lock().unwrap();
        inner.priority_degraded()
    }

    pub fn effective_state(&self) -> ActivityPolicyState {
        let inner = self.inner.lock().unwrap();
        inner.effective_state()
    }

    /// Current manual override mode (auto = detector-driven).
    pub fn manual_override(&self) -> ManualActivityOverride {
        let inner = self.inner.lock().unwrap();
        inner.manual_override()
    }

    /// Current detector state (active/idle/unknown).
    pub fn detector_state(&self) -> HostActivity {
        let inner = self.inner.lock().unwrap();
        inner.detector_state()
    }

    /// Check admission for an ingress type. Returns Allowed or Paused.
    ///
    /// If policy is disabled, always returns Allowed.
    pub fn check_admission(&self, ingress_type: IngressType) -> AdmissionResult {
        if !self.config.enabled {
            return AdmissionResult::Allowed;
        }
        let inner = self.inner.lock().unwrap();
        check_admission(inner.effective_state(), ingress_type)
    }
}

/// Start coarse native activity sampling for a configured runtime.
///
/// The task shares only the debounced coarse state with the admission guard.
/// Detector internals and errors never leave the platform adapter.
pub fn spawn_native_activity_policy(
    guard: ActivityPolicyGuard,
    config: RuntimeActivityConfig,
) -> Option<tokio::task::JoinHandle<()>> {
    if !config.enabled {
        return None;
    }

    Some(tokio::spawn(async move {
        let policy = ActivityPolicy::new(
            Duration::from_secs(config.idle_after_secs),
            Duration::from_secs(config.resume_debounce_secs),
        );
        let mut monitor =
            HostActivityMonitor::new(NativeHostActivityDetector::default(), SystemClock, policy);
        let mut priority = PrioritySession::new(NativePriorityController::default());
        let mut interval =
            tokio::time::interval(Duration::from_secs(config.poll_interval_secs.max(1)));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            interval.tick().await;
            let blocking_guard = guard.clone();
            let blocking_result = tokio::task::spawn_blocking(move || {
                let sample = monitor.sample();
                blocking_guard.update_detector_state(sample.effective);
                let priority_status = if blocking_guard.effective_state()
                    == ActivityPolicyState::AcceptingDeprioritized
                {
                    priority.reduce()
                } else {
                    priority.restore()
                };
                if priority_status.is_degraded() {
                    blocking_guard.mark_priority_degraded();
                } else {
                    blocking_guard.clear_priority_degraded();
                }
                (monitor, priority)
            })
            .await;
            match blocking_result {
                Ok((next_monitor, next_priority)) => {
                    monitor = next_monitor;
                    priority = next_priority;
                }
                Err(error) => {
                    tracing::warn!(%error, "host activity sampling task failed");
                    break;
                }
            }
        }
    }))
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── compute_effective_state matrix ──────────────────────────────────────

    #[test]
    fn effective_state_unknown_detector_always_accepting() {
        for response in [
            ActivityResponse::PauseRemote,
            ActivityResponse::PauseAll,
            ActivityResponse::ReducePriority,
        ] {
            assert_eq!(
                compute_effective_state(
                    HostActivity::Unknown,
                    ManualActivityOverride::Auto,
                    response,
                    false
                ),
                ActivityPolicyState::Accepting,
                "Unknown detector should always be Accepting for {:?}",
                response
            );
        }
    }

    #[test]
    fn effective_state_active_pause_remote() {
        assert_eq!(
            compute_effective_state(
                HostActivity::Active,
                ManualActivityOverride::Auto,
                ActivityResponse::PauseRemote,
                false
            ),
            ActivityPolicyState::RemotePaused
        );
    }

    #[test]
    fn effective_state_active_pause_all() {
        assert_eq!(
            compute_effective_state(
                HostActivity::Active,
                ManualActivityOverride::Auto,
                ActivityResponse::PauseAll,
                false
            ),
            ActivityPolicyState::AllPaused
        );
    }

    #[test]
    fn effective_state_active_reduce_priority_not_degraded() {
        assert_eq!(
            compute_effective_state(
                HostActivity::Active,
                ManualActivityOverride::Auto,
                ActivityResponse::ReducePriority,
                false
            ),
            ActivityPolicyState::AcceptingDeprioritized
        );
    }

    #[test]
    fn effective_state_active_reduce_priority_degraded() {
        assert_eq!(
            compute_effective_state(
                HostActivity::Active,
                ManualActivityOverride::Auto,
                ActivityResponse::ReducePriority,
                true
            ),
            ActivityPolicyState::Accepting
        );
    }

    // ── manual override tests ───────────────────────────────────────────────

    #[test]
    fn effective_state_override_active_forces_response() {
        assert_eq!(
            compute_effective_state(
                HostActivity::Idle,
                ManualActivityOverride::Active,
                ActivityResponse::PauseAll,
                false
            ),
            ActivityPolicyState::AllPaused
        );
    }

    #[test]
    fn effective_state_override_idle_resumes_acceptance() {
        assert_eq!(
            compute_effective_state(
                HostActivity::Active,
                ManualActivityOverride::Idle,
                ActivityResponse::PauseRemote,
                false
            ),
            ActivityPolicyState::Accepting
        );
    }

    // ── check_admission matrix ──────────────────────────────────────────────

    #[test]
    fn admission_accepting_allows_everything() {
        for ingress in [
            IngressType::LocalOpenAi,
            IngressType::RemoteQuicHttp,
            IngressType::StageTransport,
            IngressType::PluginDispatch,
            IngressType::ManagementApi,
        ] {
            assert_eq!(
                check_admission(ActivityPolicyState::Accepting, ingress),
                AdmissionResult::Allowed,
                "Accepting should allow {:?}",
                ingress
            );
        }
    }

    #[test]
    fn admission_deprioritized_allows_everything() {
        for ingress in [
            IngressType::LocalOpenAi,
            IngressType::RemoteQuicHttp,
            IngressType::StageTransport,
            IngressType::PluginDispatch,
            IngressType::ManagementApi,
        ] {
            assert_eq!(
                check_admission(ActivityPolicyState::AcceptingDeprioritized, ingress),
                AdmissionResult::Allowed,
                "AcceptingDeprioritized should allow {:?}",
                ingress
            );
        }
    }

    #[test]
    fn admission_remote_paused_blocks_only_remote() {
        // Remote types blocked.
        assert!(
            check_admission(
                ActivityPolicyState::RemotePaused,
                IngressType::RemoteQuicHttp
            )
            .is_paused()
        );
        assert!(
            check_admission(
                ActivityPolicyState::RemotePaused,
                IngressType::StageTransport
            )
            .is_paused()
        );

        // Local types allowed.
        assert_eq!(
            check_admission(ActivityPolicyState::RemotePaused, IngressType::LocalOpenAi),
            AdmissionResult::Allowed
        );
        assert_eq!(
            check_admission(
                ActivityPolicyState::RemotePaused,
                IngressType::PluginDispatch
            ),
            AdmissionResult::Allowed
        );
        assert_eq!(
            check_admission(
                ActivityPolicyState::RemotePaused,
                IngressType::ManagementApi
            ),
            AdmissionResult::Allowed
        );
    }

    #[test]
    fn admission_all_paused_blocks_inference_only() {
        // All inference types blocked.
        assert!(
            check_admission(ActivityPolicyState::AllPaused, IngressType::LocalOpenAi).is_paused()
        );
        assert!(
            check_admission(ActivityPolicyState::AllPaused, IngressType::RemoteQuicHttp)
                .is_paused()
        );
        assert!(
            check_admission(ActivityPolicyState::AllPaused, IngressType::StageTransport)
                .is_paused()
        );
        assert!(
            check_admission(ActivityPolicyState::AllPaused, IngressType::PluginDispatch)
                .is_paused()
        );

        // Management API always allowed.
        assert_eq!(
            check_admission(ActivityPolicyState::AllPaused, IngressType::ManagementApi),
            AdmissionResult::Allowed
        );
    }

    // ── ingress type helpers ────────────────────────────────────────────────

    #[test]
    fn ingress_type_is_remote() {
        assert!(!IngressType::LocalOpenAi.is_remote());
        assert!(IngressType::RemoteQuicHttp.is_remote());
        assert!(IngressType::StageTransport.is_remote());
        assert!(!IngressType::PluginDispatch.is_remote());
        assert!(!IngressType::ManagementApi.is_remote());
    }

    #[test]
    fn ingress_type_is_inference() {
        assert!(IngressType::LocalOpenAi.is_inference());
        assert!(IngressType::RemoteQuicHttp.is_inference());
        assert!(IngressType::StageTransport.is_inference());
        assert!(IngressType::PluginDispatch.is_inference());
        assert!(!IngressType::ManagementApi.is_inference());
    }

    // ── proto mapping round-trip ────────────────────────────────────────────

    #[test]
    fn activity_policy_state_to_proto() {
        use mesh_llm_protocol::proto::node::InferenceAdmissionState as Proto;

        assert_eq!(ActivityPolicyState::Accepting.to_proto(), Proto::Accepting);
        assert_eq!(
            ActivityPolicyState::AcceptingDeprioritized.to_proto(),
            Proto::AcceptingDeprioritized
        );
        assert_eq!(
            ActivityPolicyState::RemotePaused.to_proto(),
            Proto::RemotePaused
        );
        assert_eq!(ActivityPolicyState::AllPaused.to_proto(), Proto::AllPaused);
    }

    // ── ActivityPolicyGuard integration tests ───────────────────────────────

    #[test]
    fn guard_disabled_always_allows() {
        let config = RuntimeActivityConfig {
            enabled: false,
            ..Default::default()
        };
        let guard = ActivityPolicyGuard::new(&config);
        // Even with idle state, disabled policy allows everything.
        guard.update_detector_state(HostActivity::Idle);
        assert_eq!(
            guard.check_admission(IngressType::RemoteQuicHttp),
            AdmissionResult::Allowed
        );
    }

    #[test]
    fn guard_pause_remote_blocks_remote_when_active() {
        let config = RuntimeActivityConfig {
            enabled: true,
            response: ActivityResponse::PauseRemote,
            ..Default::default()
        };
        let guard = ActivityPolicyGuard::new(&config);

        // Active host use → remote inference yields immediately.
        guard.update_detector_state(HostActivity::Active);
        assert!(
            guard
                .check_admission(IngressType::RemoteQuicHttp)
                .is_paused()
        );

        // Idle host use → inference resumes.
        guard.update_detector_state(HostActivity::Idle);
        assert_eq!(
            guard.check_admission(IngressType::RemoteQuicHttp),
            AdmissionResult::Allowed
        );
    }

    #[test]
    fn guard_manual_override_controls_host_activity_state() {
        let config = RuntimeActivityConfig {
            enabled: true,
            response: ActivityResponse::PauseAll,
            ..Default::default()
        };
        let guard = ActivityPolicyGuard::new(&config);

        guard.update_detector_state(HostActivity::Active);
        assert!(guard.check_admission(IngressType::LocalOpenAi).is_paused());

        // Forced idle resumes admission even while the detector reports active use.
        guard.apply_manual_override(ManualActivityOverride::Idle);
        assert_eq!(
            guard.check_admission(IngressType::LocalOpenAi),
            AdmissionResult::Allowed
        );
    }

    #[test]
    fn guard_priority_degraded_flag() {
        let config = RuntimeActivityConfig {
            enabled: true,
            response: ActivityResponse::ReducePriority,
            ..Default::default()
        };
        let guard = ActivityPolicyGuard::new(&config);

        assert!(!guard.priority_degraded());
        guard.mark_priority_degraded();
        assert!(guard.priority_degraded());
        guard.update_detector_state(HostActivity::Active);
        // Degraded → falls back to Accepting instead of Deprioritized.
        assert_eq!(guard.effective_state(), ActivityPolicyState::Accepting);
    }

    // ── full matrix: response x ingress_type ────────────────────────────────

    #[test]
    fn admission_matrix_pause_remote() {
        let state = ActivityPolicyState::RemotePaused;
        assert!(check_admission(state, IngressType::RemoteQuicHttp).is_paused());
        assert!(check_admission(state, IngressType::StageTransport).is_paused());
        assert_eq!(
            check_admission(state, IngressType::LocalOpenAi),
            AdmissionResult::Allowed
        );
        assert_eq!(
            check_admission(state, IngressType::PluginDispatch),
            AdmissionResult::Allowed
        );
        assert_eq!(
            check_admission(state, IngressType::ManagementApi),
            AdmissionResult::Allowed
        );
    }

    #[test]
    fn admission_matrix_pause_all() {
        let state = ActivityPolicyState::AllPaused;
        assert!(check_admission(state, IngressType::RemoteQuicHttp).is_paused());
        assert!(check_admission(state, IngressType::StageTransport).is_paused());
        assert!(check_admission(state, IngressType::LocalOpenAi).is_paused());
        assert!(check_admission(state, IngressType::PluginDispatch).is_paused());
        assert_eq!(
            check_admission(state, IngressType::ManagementApi),
            AdmissionResult::Allowed
        );
    }

    #[test]
    fn admission_matrix_reduce_priority() {
        let state = ActivityPolicyState::AcceptingDeprioritized;
        assert_eq!(
            check_admission(state, IngressType::RemoteQuicHttp),
            AdmissionResult::Allowed
        );
        assert_eq!(
            check_admission(state, IngressType::StageTransport),
            AdmissionResult::Allowed
        );
        assert_eq!(
            check_admission(state, IngressType::LocalOpenAi),
            AdmissionResult::Allowed
        );
        assert_eq!(
            check_admission(state, IngressType::PluginDispatch),
            AdmissionResult::Allowed
        );
        assert_eq!(
            check_admission(state, IngressType::ManagementApi),
            AdmissionResult::Allowed
        );
    }

    // ── paused result fields ────────────────────────────────────────────────

    #[test]
    fn paused_result_has_reason_and_retry() {
        let result = check_admission(
            ActivityPolicyState::RemotePaused,
            IngressType::RemoteQuicHttp,
        );
        match result {
            AdmissionResult::Paused {
                reason,
                retry_after_secs,
            } => {
                assert!(!reason.is_empty());
                assert!(reason.contains("remote"));
                assert_eq!(retry_after_secs, None);
            }
            _ => panic!("expected Paused"),
        }
    }

    #[test]
    fn activity_policy_ingress_matrix() {
        let ingresses = [
            IngressType::LocalOpenAi,
            IngressType::RemoteQuicHttp,
            IngressType::StageTransport,
            IngressType::PluginDispatch,
            IngressType::ManagementApi,
        ];
        let remote_paused = [false, true, true, false, false];
        let all_paused = [true, true, true, true, false];

        for (index, ingress) in ingresses.into_iter().enumerate() {
            assert_eq!(
                check_admission(ActivityPolicyState::RemotePaused, ingress).is_paused(),
                remote_paused[index],
                "pause_remote ingress={ingress:?}"
            );
            assert_eq!(
                check_admission(ActivityPolicyState::AllPaused, ingress).is_paused(),
                all_paused[index],
                "pause_all ingress={ingress:?}"
            );
            assert!(
                check_admission(ActivityPolicyState::AcceptingDeprioritized, ingress).is_allowed(),
                "reduce_priority ingress={ingress:?}"
            );
        }
    }

    #[test]
    fn activity_policy_legacy_peer_and_backend_failure() {
        let config = RuntimeActivityConfig {
            enabled: true,
            response: ActivityResponse::ReducePriority,
            advertisement: ActivityAdvertisement::CoarseState,
            ..Default::default()
        };
        let guard = ActivityPolicyGuard::new(&config);
        guard.update_detector_state(HostActivity::Active);
        guard.mark_priority_degraded();

        assert_eq!(guard.effective_state(), ActivityPolicyState::Accepting);
        assert!(guard.check_admission(IngressType::LocalOpenAi).is_allowed());
        assert!(
            guard
                .check_admission(IngressType::RemoteQuicHttp)
                .is_allowed()
        );
        let advertised = guard.advertisement_decision(false);
        assert_eq!(
            advertised.admission_state,
            Some(mesh_llm_protocol::proto::node::InferenceAdmissionState::Accepting)
        );
        assert!(!advertised.withdraw_model_availability);

        let legacy_pause = activity_advertisement_decision(
            true,
            ActivityAdvertisement::AvailabilityOnly,
            ActivityPolicyState::RemotePaused,
            false,
        );
        assert_eq!(legacy_pause.admission_state, None);
        assert!(legacy_pause.withdraw_model_availability);
    }

    // ── ActivityPolicyRuntimeState mutation tests ───────────────────────────

    #[test]
    fn runtime_state_updates_detector_and_recomputes() {
        let mut state = ActivityPolicyRuntimeState::new(ActivityResponse::PauseRemote);
        assert_eq!(state.effective_state(), ActivityPolicyState::Accepting); // Unknown → Accepting

        state.update_detector(HostActivity::Active, ActivityResponse::PauseRemote);
        assert_eq!(state.effective_state(), ActivityPolicyState::RemotePaused);

        state.update_detector(HostActivity::Idle, ActivityResponse::PauseRemote);
        assert_eq!(state.effective_state(), ActivityPolicyState::Accepting);
    }

    #[test]
    fn runtime_state_manual_override_persists_across_detector_updates() {
        let mut state = ActivityPolicyRuntimeState::new(ActivityResponse::PauseAll);
        state.apply_manual_override(ManualActivityOverride::Idle, ActivityResponse::PauseAll);

        // Even when detector says active, forced idle keeps admission accepting.
        state.update_detector(HostActivity::Active, ActivityResponse::PauseAll);
        assert_eq!(state.effective_state(), ActivityPolicyState::Accepting);
    }
}
