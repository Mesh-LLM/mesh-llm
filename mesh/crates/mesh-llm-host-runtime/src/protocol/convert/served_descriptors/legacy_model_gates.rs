//! Translate retired wire gates into generic requirements; relay their opaque bytes.

use crate::mesh::{PeerAnnouncement, public_model_id_from_identity};
use crate::proto::node::LegacyModelGate;
use mesh_llm_types::mesh::ChannelRequirement;
use prost::Message;
use std::collections::HashSet;

/// Distinct from any current plugin channel: the retired built-in protocol is
/// not implemented by a plugin declaring its own modern payment channel.
const LEGACY_LIGHTNING_CHANNEL: &str = "mesh-llm.legacy-lightning";

#[cfg(test)]
mod tests;

pub(super) fn restore(gates: &[Vec<u8>], ann: &mut PeerAnnouncement) -> Option<()> {
    // Never truncate a gate list: a dropped tail could appear ungated. A valid
    // old message fits in 4 KiB, including its at-most-1024-byte model name.
    if gates.len() > 128 || gates.iter().any(|gate| gate.len() > 4096) {
        return None;
    }
    let mut models = HashSet::new();
    for encoded in gates {
        let gate = LegacyModelGate::decode(encoded.as_slice()).ok()?;
        if gate.model.is_empty() || gate.model.len() > 1024 || !models.insert(gate.model.clone()) {
            return None;
        }
        for descriptor in &mut ann.served_model_descriptors {
            let public_id = public_model_id_from_identity(&descriptor.identity);
            if descriptor.identity.model_name != gate.model
                && public_id.as_deref() != Some(gate.model.as_str())
            {
                continue;
            }
            let requirement = ChannelRequirement {
                channel: LEGACY_LIGHTNING_CHANNEL.into(),
                // Unknown fields must survive: name-only re-encoding would
                // erase the gate for old readers or make them reject the peer.
                terms: encoded.clone(),
            };
            if !descriptor.requires.contains(&requirement) {
                descriptor.requires.push(requirement);
            }
        }
    }
    Some(())
}

pub(in super::super) fn relay(ann: &PeerAnnouncement) -> Vec<Vec<u8>> {
    let mut gates = ann
        .served_model_descriptors
        .iter()
        .flat_map(|descriptor| &descriptor.requires)
        .filter(|requirement| {
            requirement.channel == LEGACY_LIGHTNING_CHANNEL && !requirement.terms.is_empty()
        })
        .map(|requirement| requirement.terms.clone())
        .collect::<Vec<_>>();
    gates.sort();
    gates.dedup();
    gates
}
