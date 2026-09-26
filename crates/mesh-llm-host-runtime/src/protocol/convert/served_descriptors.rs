//! Restore authoritative model descriptors without reviving rejected legacy routes.

use super::{
    proto_capability_level_to_local, proto_identity_to_local, proto_model_metadata_to_local,
};
use crate::mesh::{PeerAnnouncement, ServedModelDescriptor};
use crate::proto::node;

/// Only peers that omit descriptors may use identity/name-based legacy inference.
pub(super) fn restore_served_descriptors(
    source: &node::PeerAnnouncement,
    ann: &mut PeerAnnouncement,
) {
    if source.served_model_descriptors.is_empty() {
        ann.served_model_descriptors = source
            .served_model_identities
            .iter()
            .filter(|identity| !identity.model_name.is_empty())
            .map(legacy_descriptor_from_identity)
            .collect();
        crate::mesh::backfill_legacy_descriptors(ann);
    } else {
        ann.served_model_descriptors = source
            .served_model_descriptors
            .iter()
            .filter_map(proto_descriptor_to_local)
            .collect();
    }
}

/// Convert a legacy identity without inventing capability or workload metadata.
fn legacy_descriptor_from_identity(
    identity: &crate::proto::node::ServedModelIdentity,
) -> crate::mesh::ServedModelDescriptor {
    crate::mesh::ServedModelDescriptor {
        identity: proto_identity_to_local(identity),
        capabilities_known: false,
        capabilities: crate::models::ModelCapabilities::default(),
        topology: None,
        metadata: None,
    }
}

/// Retain complete metadata for valid descriptors and discard missing/empty identities.
fn proto_descriptor_to_local(
    descriptor: &node::ServedModelDescriptor,
) -> Option<ServedModelDescriptor> {
    let identity = descriptor
        .identity
        .as_ref()
        .filter(|id| !id.model_name.is_empty())?;
    let capabilities = descriptor
        .capabilities
        .as_ref()
        .map(|caps| crate::models::ModelCapabilities {
            multimodal: caps.multimodal,
            vision: proto_capability_level_to_local(caps.vision),
            audio: proto_capability_level_to_local(caps.audio),
            reasoning: proto_capability_level_to_local(caps.reasoning),
            tool_use: proto_capability_level_to_local(caps.tool_use),
            moe: caps.moe,
        })
        .unwrap_or_default();
    Some(ServedModelDescriptor {
        identity: proto_identity_to_local(identity),
        capabilities_known: descriptor
            .capabilities_known
            .unwrap_or(capabilities != crate::models::ModelCapabilities::default()),
        capabilities,
        topology: descriptor
            .topology
            .as_ref()
            .map(|topology| crate::models::ModelTopology {
                moe: topology
                    .moe
                    .as_ref()
                    .map(|moe| crate::models::ModelMoeInfo {
                        expert_count: moe.expert_count,
                        used_expert_count: moe.used_expert_count,
                        min_experts_per_node: moe.min_experts_per_node,
                        source: moe.source.clone(),
                        ranking_source: moe.ranking_source.clone(),
                        ranking_origin: moe.ranking_origin.clone(),
                        ranking: moe.ranking.clone(),
                        ranking_prompt_count: moe.ranking_prompt_count,
                        ranking_tokens: moe.ranking_tokens,
                        ranking_layer_scope: moe.ranking_layer_scope.clone(),
                    }),
            }),
        metadata: descriptor
            .metadata
            .as_ref()
            .map(proto_model_metadata_to_local),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::convert::proto_ann_to_local;

    /// Include both legacy fallback sources so malformed descriptors cannot hide the regression.
    fn announcement_with_legacy_routes() -> node::PeerAnnouncement {
        node::PeerAnnouncement {
            endpoint_id: vec![1; 32],
            serving_models: vec!["legacy-model".to_string()],
            model_source: Some("legacy-model".to_string()),
            served_model_identities: vec![node::ServedModelIdentity {
                model_name: "legacy-model".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    /// A present-but-invalid descriptor list must never recover a legacy model route.
    #[test]
    fn invalid_descriptors_do_not_backfill_from_legacy_names_or_identities() {
        for descriptor in [
            node::ServedModelDescriptor::default(),
            node::ServedModelDescriptor {
                identity: Some(node::ServedModelIdentity::default()),
                ..Default::default()
            },
        ] {
            let source = node::PeerAnnouncement {
                served_model_descriptors: vec![descriptor],
                ..announcement_with_legacy_routes()
            };
            let (_, ann) = proto_ann_to_local(&source).unwrap();
            assert!(ann.served_model_descriptors.is_empty());
        }
    }

    /// Older identity-aware peers still restore the identities they actually advertise.
    #[test]
    fn absent_descriptors_preserve_legacy_identity_fallback() {
        let (_, ann) = proto_ann_to_local(&announcement_with_legacy_routes()).unwrap();
        assert_eq!(ann.served_model_descriptors.len(), 1);
        let descriptor = &ann.served_model_descriptors[0];
        assert_eq!(descriptor.identity.model_name, "legacy-model");
        assert!(!descriptor.capabilities_known);
        assert!(descriptor.metadata.is_none());
    }

    /// Name-only peers retain the original primary-model inference contract.
    #[test]
    fn absent_descriptors_and_identities_preserve_legacy_name_fallback() {
        let source = node::PeerAnnouncement {
            served_model_identities: Vec::new(),
            ..announcement_with_legacy_routes()
        };
        let (_, ann) = proto_ann_to_local(&source).unwrap();
        assert_eq!(ann.served_model_descriptors.len(), 1);
        let descriptor = &ann.served_model_descriptors[0];
        assert_eq!(descriptor.identity.model_name, "legacy-model");
        assert!(descriptor.identity.is_primary);
        assert!(descriptor.metadata.is_none());
    }

    /// Empty legacy identities must not suppress a valid name-only fallback.
    #[test]
    fn empty_legacy_identities_preserve_name_fallback() {
        let source = node::PeerAnnouncement {
            served_model_identities: vec![node::ServedModelIdentity::default()],
            ..announcement_with_legacy_routes()
        };
        let (_, ann) = proto_ann_to_local(&source).unwrap();
        assert_eq!(ann.served_model_descriptors.len(), 1);
        assert_eq!(
            ann.served_model_descriptors[0].identity.model_name,
            "legacy-model"
        );
        assert!(ann.served_model_descriptors[0].identity.is_primary);
    }

    /// A valid advertised identity stays authoritative beside malformed legacy entries.
    #[test]
    fn empty_legacy_identities_do_not_replace_valid_identities() {
        let mut source = announcement_with_legacy_routes();
        source
            .served_model_identities
            .insert(0, node::ServedModelIdentity::default());
        source.served_model_identities[1].model_name = "identity-model".to_string();
        let (_, ann) = proto_ann_to_local(&source).unwrap();
        assert_eq!(ann.served_model_descriptors.len(), 1);
        assert_eq!(
            ann.served_model_descriptors[0].identity.model_name,
            "identity-model"
        );
    }

    /// Partial invalidity must not drop valid workload metadata or restore extra legacy names.
    #[test]
    fn valid_descriptors_remain_authoritative_among_invalid_entries() {
        let source = node::PeerAnnouncement {
            served_model_descriptors: vec![
                node::ServedModelDescriptor::default(),
                node::ServedModelDescriptor {
                    identity: Some(node::ServedModelIdentity {
                        model_name: "embedding-model".to_string(),
                        ..Default::default()
                    }),
                    capabilities_known: Some(true),
                    metadata: Some(node::ServedModelMetadata {
                        workload_class: Some(node::ModelWorkloadClass::Embedding as i32),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            ],
            ..announcement_with_legacy_routes()
        };
        let (_, ann) = proto_ann_to_local(&source).unwrap();
        assert_eq!(ann.served_model_descriptors.len(), 1);
        let descriptor = &ann.served_model_descriptors[0];
        assert_eq!(descriptor.identity.model_name, "embedding-model");
        assert!(descriptor.capabilities_known);
        assert_eq!(
            descriptor.metadata.as_ref().unwrap().workload_class,
            Some(crate::mesh::ModelWorkloadClass::Embedding)
        );
    }
}
