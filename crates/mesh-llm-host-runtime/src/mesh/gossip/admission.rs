//! Authenticate the direct gossip sender before payload or liveness mutations.

use super::{AnnouncedPeerContext, EndpointAddr, EndpointId, Node, PeerAnnouncement, Result};

/// Returns the single announcement the authenticated remote made for itself.
///
/// The frame must carry exactly one entry for the direct sender. `apply_announced_peers`
/// validates the direct announcement once and then applies every entry as already
/// validated, so a duplicate self-entry could be admitted without re-running the direct
/// mesh requirements. Rejecting duplicates up front keeps the validated entry and the
/// applied entry the same object.
fn direct_announcement(
    announcements: &[(EndpointAddr, PeerAnnouncement)],
    remote: EndpointId,
) -> Result<&PeerAnnouncement> {
    let mut direct = announcements
        .iter()
        .filter_map(|(addr, ann)| (addr.id == remote).then_some(ann));
    let announcement = direct.next().ok_or_else(|| {
        anyhow::anyhow!(
            "gossip payload from {} omitted its direct announcement",
            remote.fmt_short()
        )
    })?;
    if direct.next().is_some() {
        anyhow::bail!(
            "gossip payload from {} contains multiple direct announcements",
            remote.fmt_short()
        );
    }
    Ok(announcement)
}

impl Node {
    pub(super) async fn validate_direct_announcement_before_payload_apply(
        &self,
        their_announcements: &[(EndpointAddr, PeerAnnouncement)],
        context: AnnouncedPeerContext,
    ) -> Result<()> {
        let direct_announcement = direct_announcement(their_announcements, context.remote)?;
        if let Err(reason) = self
            .validate_direct_peer_requirements(
                context.remote,
                direct_announcement,
                context.negotiated_protocol_generation,
            )
            .await
        {
            self.record_mesh_requirement_rejection(
                crate::mesh::requirements::MeshRequirementRejectionSource::Gossip,
                Some(context.remote),
                reason.clone(),
            )
            .await;
            self.state
                .lock()
                .await
                .requirement_rejected_peers
                .insert(context.remote);
            anyhow::bail!(
                "peer {} rejected by mesh requirements: {}",
                context.remote.fmt_short(),
                reason.code()
            );
        }
        self.validate_direct_owner_before_payload_apply(their_announcements, context)
            .await
    }

    pub(super) async fn validate_direct_owner_before_payload_apply(
        &self,
        their_announcements: &[(EndpointAddr, PeerAnnouncement)],
        context: AnnouncedPeerContext,
    ) -> Result<()> {
        let direct_announcement = direct_announcement(their_announcements, context.remote)?;
        let owner_summary = self
            .direct_peer_owner_summary(context.remote, direct_announcement)
            .await;
        if self
            .reject_direct_peer_for_policy(context.remote, &owner_summary)
            .await
        {
            self.capture_peer_rejected(
                context.remote,
                &direct_announcement.addr,
                direct_announcement,
                &owner_summary,
                "direct",
                None,
            );
            anyhow::bail!(
                "peer {} rejected by owner policy: {:?}",
                context.remote.fmt_short(),
                owner_summary.status
            );
        }
        Ok(())
    }
}
