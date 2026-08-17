use super::local_package::{
    SPLIT_DEFAULT_MIN_PARTICIPANTS, SplitParticipant, SplitParticipantSnapshot,
    collect_split_participant_membership, collect_split_participants,
    ensure_split_participant_timeout_has_quorum,
};
use super::split_planning::{split_participant_exclusion_labels, split_participant_labels};
use crate::inference::skippy;
use crate::mesh;
use anyhow::Result;
use std::time::Duration;

const SPLIT_PARTICIPANT_POLL_INTERVAL: Duration = Duration::from_millis(500);
/// An automatic split cannot know how many nodes the operator intends to start.
/// Wait for additions/capacity changes to stop instead of claiming the first
/// two-node quorum that happens to become visible.
const SPLIT_MEMBERSHIP_SETTLE_DWELL: Duration = Duration::from_secs(8);
/// Even an immediately stable two-node quorum gets a bounded discovery window.
const SPLIT_FIRST_QUORUM_OBSERVATION: Duration = Duration::from_secs(8);

fn membership_settle_timeout(requested: Duration) -> Duration {
    requested.max(SPLIT_FIRST_QUORUM_OBSERVATION.max(SPLIT_MEMBERSHIP_SETTLE_DWELL))
}

type SplitMembershipSignature = Vec<(String, u64)>;

#[derive(Debug, Default)]
struct SplitMembershipSettleBarrier {
    signature: SplitMembershipSignature,
    first_quorum_observed: Option<tokio::time::Instant>,
    stable_since: Option<tokio::time::Instant>,
}

impl SplitMembershipSettleBarrier {
    fn observe(&mut self, participants: &[SplitParticipant], now: tokio::time::Instant) -> bool {
        let signature = split_membership_signature(participants);
        if signature != self.signature {
            self.signature = signature;
            self.stable_since = Some(now);
        }
        if participants.len() >= SPLIT_DEFAULT_MIN_PARTICIPANTS
            && self.first_quorum_observed.is_none()
        {
            self.first_quorum_observed = Some(now);
        }
        self.is_ready(participants.len(), now)
    }

    fn is_ready(&self, participant_count: usize, now: tokio::time::Instant) -> bool {
        if participant_count < SPLIT_DEFAULT_MIN_PARTICIPANTS {
            return false;
        }
        let Some(first_quorum_observed) = self.first_quorum_observed else {
            return false;
        };
        let Some(stable_since) = self.stable_since else {
            return false;
        };
        now.saturating_duration_since(first_quorum_observed) >= SPLIT_FIRST_QUORUM_OBSERVATION
            && now.saturating_duration_since(stable_since) >= SPLIT_MEMBERSHIP_SETTLE_DWELL
    }

    fn stable_for(&self, now: tokio::time::Instant) -> Duration {
        self.stable_since
            .map(|stable_since| now.saturating_duration_since(stable_since))
            .unwrap_or_default()
    }
}

struct SplitMembershipWait<'a> {
    node: &'a mesh::Node,
    model_name: &'a str,
    model_ref: &'a str,
    deadline: tokio::time::Instant,
    barrier: SplitMembershipSettleBarrier,
    last_logged_signature: SplitMembershipSignature,
}

impl<'a> SplitMembershipWait<'a> {
    fn new(
        node: &'a mesh::Node,
        model_name: &'a str,
        model_ref: &'a str,
        timeout: Duration,
    ) -> Self {
        Self {
            node,
            model_name,
            model_ref,
            deadline: tokio::time::Instant::now() + membership_settle_timeout(timeout),
            barrier: SplitMembershipSettleBarrier::default(),
            last_logged_signature: Vec::new(),
        }
    }

    async fn run(mut self) -> Result<SplitParticipantSnapshot> {
        loop {
            let snapshot =
                collect_split_participant_membership(self.node, self.model_name, self.model_ref)
                    .await;
            let signature = split_membership_signature(&snapshot.participants);
            self.log_membership_change(&snapshot, &signature);
            let now = tokio::time::Instant::now();
            if self.barrier.observe(&snapshot.participants, now) {
                self.log_accepted(&snapshot, now);
                return Ok(snapshot);
            }
            if now >= self.deadline {
                return self.finish_at_timeout().await;
            }
            tokio::time::sleep(SPLIT_PARTICIPANT_POLL_INTERVAL).await;
        }
    }

    fn log_membership_change(
        &mut self,
        snapshot: &SplitParticipantSnapshot,
        signature: &SplitMembershipSignature,
    ) {
        if signature == &self.last_logged_signature {
            return;
        }
        tracing::info!(
            model_ref = self.model_ref,
            members = ?split_participant_labels(&snapshot.participants),
            excluded = ?split_participant_exclusion_labels(&snapshot.excluded),
            "split topology stable membership changed"
        );
        self.last_logged_signature = signature.clone();
    }

    fn log_accepted(&self, snapshot: &SplitParticipantSnapshot, now: tokio::time::Instant) {
        tracing::info!(
            model_ref = self.model_ref,
            stable_for_ms = self.barrier.stable_for(now).as_millis(),
            participants = ?split_participant_labels(&snapshot.participants),
            "split topology membership accepted for canonical coordinator election"
        );
    }

    /// Elect only from a freshly revalidated snapshot.
    ///
    /// A best-ever set can name peers that have since vanished, which puts
    /// dead nodes into the elected topology. Re-collect at the deadline so the
    /// final membership reflects peers that are still present.
    async fn finish_at_timeout(self) -> Result<SplitParticipantSnapshot> {
        let snapshot =
            collect_split_participant_membership(self.node, self.model_name, self.model_ref).await;
        ensure_split_participant_timeout_has_quorum(
            self.model_ref,
            &snapshot.participants,
            &snapshot.excluded,
        )?;
        tracing::warn!(
            model_ref = self.model_ref,
            participants = ?split_participant_labels(&snapshot.participants),
            excluded = ?split_participant_exclusion_labels(&snapshot.excluded),
            "split topology membership settle timed out; using revalidated final snapshot"
        );
        Ok(snapshot)
    }
}

struct SplitEligibilityWait<'a> {
    node: &'a mesh::Node,
    model_name: &'a str,
    model_ref: &'a str,
    package: &'a skippy::SkippyPackageIdentity,
    local_vram_override: Option<u64>,
    expected_node_ids: Vec<String>,
    deadline: tokio::time::Instant,
}

impl<'a> SplitEligibilityWait<'a> {
    async fn run(self) -> Result<SplitParticipantSnapshot> {
        loop {
            let snapshot = self.collect_snapshot().await;
            if self.snapshot_is_complete(&snapshot) {
                self.log_complete(&snapshot);
                return Ok(snapshot);
            }
            if tokio::time::Instant::now() >= self.deadline {
                return self.finish_at_timeout().await;
            }
            self.log_pending(&snapshot);
            tokio::time::sleep(SPLIT_PARTICIPANT_POLL_INTERVAL).await;
        }
    }

    async fn collect_snapshot(&self) -> SplitParticipantSnapshot {
        collect_split_participants(
            self.node,
            self.model_name,
            self.model_ref,
            self.package,
            self.local_vram_override,
        )
        .await
    }

    fn snapshot_is_complete(&self, snapshot: &SplitParticipantSnapshot) -> bool {
        split_membership_node_ids(&snapshot.participants) == self.expected_node_ids
    }

    fn log_complete(&self, snapshot: &SplitParticipantSnapshot) {
        tracing::info!(
            model_ref = self.model_ref,
            participants = ?split_participant_labels(&snapshot.participants),
            "canonical coordinator accepted full split package inventory"
        );
    }

    /// Elect only from a freshly revalidated eligible snapshot; see
    /// `SplitMembershipWait::finish_at_timeout`.
    async fn finish_at_timeout(self) -> Result<SplitParticipantSnapshot> {
        let snapshot = self.collect_snapshot().await;
        ensure_split_participant_timeout_has_quorum(
            self.model_ref,
            &snapshot.participants,
            &snapshot.excluded,
        )?;
        tracing::warn!(
            model_ref = self.model_ref,
            participants = ?split_participant_labels(&snapshot.participants),
            excluded = ?split_participant_exclusion_labels(&snapshot.excluded),
            "split package inventory wait timed out; using revalidated final snapshot"
        );
        Ok(snapshot)
    }

    fn log_pending(&self, snapshot: &SplitParticipantSnapshot) {
        tracing::debug!(
            model_ref = self.model_ref,
            expected_members = ?self.expected_node_ids,
            eligible = ?split_participant_labels(&snapshot.participants),
            excluded = ?split_participant_exclusion_labels(&snapshot.excluded),
            "canonical coordinator waiting for full split package inventory"
        );
    }
}

pub(super) async fn wait_for_split_membership(
    node: &mesh::Node,
    model_name: &str,
    model_ref: &str,
    timeout: Duration,
) -> Result<SplitParticipantSnapshot> {
    SplitMembershipWait::new(node, model_name, model_ref, timeout)
        .run()
        .await
}

pub(super) async fn wait_for_split_participants(
    node: &mesh::Node,
    model_name: &str,
    model_ref: &str,
    package: &skippy::SkippyPackageIdentity,
    local_vram_override: Option<u64>,
    expected_membership: &[SplitParticipant],
    timeout: Duration,
) -> Result<SplitParticipantSnapshot> {
    SplitEligibilityWait {
        node,
        model_name,
        model_ref,
        package,
        local_vram_override,
        expected_node_ids: split_membership_node_ids(expected_membership),
        deadline: tokio::time::Instant::now() + timeout,
    }
    .run()
    .await
}

fn split_membership_signature(participants: &[SplitParticipant]) -> SplitMembershipSignature {
    participants
        .iter()
        .map(|participant| (participant.node_id.to_string(), participant.vram_bytes))
        .collect()
}

fn split_membership_node_ids(participants: &[SplitParticipant]) -> Vec<String> {
    participants
        .iter()
        .map(|participant| participant.node_id.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::local_package::{
        SplitParticipantPackageSignal, split_participant_signature,
    };

    fn make_id(seed: u8) -> iroh::EndpointId {
        let secret = iroh::SecretKey::from_bytes(&[seed; 32]);
        secret.public()
    }

    fn participant(seed: u8) -> SplitParticipant {
        SplitParticipant::new(make_id(seed), u64::from(seed) * 1_000_000_000, None)
    }

    #[test]
    fn membership_settle_waits_for_two_to_six_arrivals() {
        let start = tokio::time::Instant::now();
        let mut barrier = SplitMembershipSettleBarrier::default();

        assert!(!barrier.observe(&[participant(1), participant(2)], start));
        for (seconds, count) in [(2, 3), (4, 4), (6, 5), (7, 6)] {
            let participants = (1..=count).map(participant).collect::<Vec<_>>();
            assert!(!barrier.observe(&participants, start + Duration::from_secs(seconds)));
        }
        let participants = (1..=6).map(participant).collect::<Vec<_>>();
        assert!(!barrier.observe(&participants, start + Duration::from_secs(14)));
        assert!(barrier.observe(&participants, start + Duration::from_secs(15)));
    }

    #[test]
    fn two_node_cohort_settles_after_bounded_discovery_window() {
        let start = tokio::time::Instant::now();
        let participants = vec![participant(1), participant(2)];
        let mut barrier = SplitMembershipSettleBarrier::default();

        assert!(!barrier.observe(&participants, start));
        assert!(!barrier.observe(&participants, start + Duration::from_secs(7)));
        assert!(barrier.observe(&participants, start + Duration::from_secs(8)));
    }

    #[test]
    fn short_caller_timeout_still_allows_the_settle_barrier() {
        assert_eq!(
            membership_settle_timeout(Duration::from_secs(1)),
            Duration::from_secs(8)
        );
        assert_eq!(
            membership_settle_timeout(Duration::from_secs(30)),
            Duration::from_secs(30)
        );
    }

    #[test]
    fn volatile_package_and_rtt_signals_do_not_reset_membership_dwell() {
        let start = tokio::time::Instant::now();
        let mut participants = vec![participant(1), participant(2)];
        let mut barrier = SplitMembershipSettleBarrier::default();
        assert!(!barrier.observe(&participants, start));

        participants[0] = participants[0].with_package_signals(
            SplitParticipantPackageSignal {
                cached_slice_bytes: 10_000,
                missing_artifact_bytes: 90_000,
                availability_score: 3,
            },
            Some(80),
            true,
        );
        participants[1] = participants[1].with_package_signals(
            SplitParticipantPackageSignal {
                cached_slice_bytes: 100_000,
                missing_artifact_bytes: 0,
                availability_score: 30,
            },
            Some(4),
            true,
        );

        assert!(barrier.observe(&participants, start + Duration::from_secs(8)));
        assert_eq!(
            split_membership_signature(&participants),
            vec![
                (make_id(1).to_string(), 1_000_000_000),
                (make_id(2).to_string(), 2_000_000_000),
            ]
        );
        assert_ne!(split_participant_signature(&participants), Vec::new());
    }

    #[test]
    fn capacity_change_restarts_membership_dwell() {
        let start = tokio::time::Instant::now();
        let participants = vec![participant(1), participant(2)];
        let mut barrier = SplitMembershipSettleBarrier::default();
        assert!(!barrier.observe(&participants, start));

        let changed = vec![
            participant(1),
            SplitParticipant::new(make_id(2), 9_000, None),
        ];
        assert!(!barrier.observe(&changed, start + Duration::from_secs(7)));
        assert!(!barrier.observe(&changed, start + Duration::from_secs(14)));
        assert!(barrier.observe(&changed, start + Duration::from_secs(15)));
    }
}
