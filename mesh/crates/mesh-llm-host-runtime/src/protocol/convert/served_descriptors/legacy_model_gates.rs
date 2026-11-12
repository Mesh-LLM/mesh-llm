//! Translate retired wire gates into generic channel requirements, not payment state.

use crate::mesh::{PeerAnnouncement, public_model_id_from_identity};
use crate::proto::node::LegacyModelGate;
use mesh_llm_types::mesh::ChannelRequirement;
use std::collections::HashSet;

/// Distinct from any current plugin channel: the retired built-in protocol is
/// not implemented by a plugin declaring its own modern payment channel.
const LEGACY_LIGHTNING_CHANNEL: &str = "mesh-llm.legacy-lightning";

#[cfg(test)]
mod tests;

pub(super) fn restore(gates: &[LegacyModelGate], ann: &mut PeerAnnouncement) -> Option<()> {
    // Keep the old trust-boundary limits. Never truncate a gate list: a dropped
    // tail could turn its paid models into apparently ungated ones.
    if gates.len() > 128
        || gates
            .iter()
            .any(|gate| gate.model.is_empty() || gate.model.len() > 1024)
    {
        return None;
    }
    let models: HashSet<_> = gates.iter().map(|gate| gate.model.as_str()).collect();
    if models.len() != gates.len() {
        return None;
    }
    for descriptor in &mut ann.served_model_descriptors {
        let public_id = public_model_id_from_identity(&descriptor.identity);
        if (models.contains(descriptor.identity.model_name.as_str())
            || public_id.as_deref().is_some_and(|id| models.contains(id)))
            && !descriptor
                .requires
                .iter()
                .any(|requirement| requirement.channel == LEGACY_LIGHTNING_CHANNEL)
        {
            descriptor.requires.push(ChannelRequirement {
                channel: LEGACY_LIGHTNING_CHANNEL.into(),
                terms: Vec::new(),
            });
        }
    }
    Some(())
}
