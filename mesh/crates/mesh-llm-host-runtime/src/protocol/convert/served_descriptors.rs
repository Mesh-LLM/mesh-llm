//! Restore authoritative model descriptors without reviving rejected legacy routes.

use super::{
    proto_capability_level_to_local, proto_identity_to_local, proto_model_metadata_to_local,
};
use crate::mesh::{PeerAnnouncement, ServedModelDescriptor};
use crate::proto::node;

mod legacy_model_gates;
pub(super) use legacy_model_gates::relay as relay_legacy_model_gates;

/// Only peers that omit descriptors may use identity/name-based legacy inference.
pub(super) fn restore_served_descriptors(
    source: &node::PeerAnnouncement,
    ann: &mut PeerAnnouncement,
) -> Option<()> {
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
    legacy_model_gates::restore(&source.legacy_lightning_offers, ann)
}

/// Convert a legacy identity without inventing capability or workload metadata.
fn legacy_descriptor_from_identity(
    identity: &crate::proto::node::ServedModelIdentity,
) -> crate::mesh::ServedModelDescriptor {
    crate::mesh::ServedModelDescriptor {
        identity: proto_identity_to_local(identity),
        requires: Vec::new(),
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
            system_one: proto_capability_level_to_local(caps.system_one),
        })
        .unwrap_or_default();
    Some(ServedModelDescriptor {
        requires: descriptor
            .requires
            .iter()
            .map(|r| mesh_llm_types::mesh::ChannelRequirement {
                channel: r.channel.clone(),
                terms: r.terms.clone(),
            })
            .collect(),
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

    /// The legacy wire schema is deliberately independent of the current reader.
    #[derive(Clone, PartialEq, prost::Message)]
    struct LegacyAnnouncement {
        #[prost(string, repeated, tag = "10")]
        serving_models: Vec<String>,
        #[prost(string, repeated, tag = "23")]
        hosted_models: Vec<String>,
        #[prost(message, repeated, tag = "52")]
        lightning_offers: Vec<LegacyOffer>,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    struct LegacyOffer {
        #[prost(string, tag = "1")]
        model: String,
        #[prost(uint64, tag = "2")]
        input_msat_per_million: u64,
        #[prost(uint64, tag = "3")]
        output_msat_per_million: u64,
        #[prost(uint64, tag = "4")]
        minimum_invoice_msat: u64,
        #[prost(uint32, tag = "5")]
        cache_discount_percent: u32,
        #[prost(uint32, tag = "6")]
        payment_protocol: u32,
        #[prost(bytes = "vec", tag = "7")]
        unknown_extension: Vec<u8>,
    }

    #[test]
    fn legacy_paid_offer_survives_wire_decode_as_model_requirement() {
        use prost::Message;

        let source = node::PeerAnnouncement {
            endpoint_id: vec![1; 32],
            version: Some("0.78.0".into()),
            serving_models: vec!["gated".into(), "free".into()],
            hosted_models: vec!["gated".into(), "free".into()],
            served_model_descriptors: ["gated", "free"]
                .into_iter()
                .map(|model| node::ServedModelDescriptor {
                    identity: Some(node::ServedModelIdentity {
                        model_name: model.into(),
                        ..Default::default()
                    }),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        };
        let mut bytes = source.encode_to_vec();
        bytes.extend(
            LegacyAnnouncement {
                lightning_offers: vec![LegacyOffer {
                    model: "gated".into(),
                    input_msat_per_million: 50_000_000,
                    output_msat_per_million: 50_000_000,
                    minimum_invoice_msat: 1,
                    cache_discount_percent: 20,
                    payment_protocol: 3,
                    unknown_extension: vec![42],
                }],
                ..Default::default()
            }
            .encode_to_vec(),
        );
        let decoded = node::PeerAnnouncement::decode(bytes.as_slice()).unwrap();
        let (_, local) = proto_ann_to_local(&decoded).unwrap();
        let gated = &local.served_model_descriptors[0];
        assert_eq!(gated.identity.model_name, "gated");
        assert_eq!(gated.requires.len(), 1, "a legacy paid offer is not free");
        assert_eq!(
            gated.requires[0].terms, decoded.legacy_lightning_offers[0],
            "the old gate stays opaque"
        );
        assert!(local.served_model_descriptors[1].requires.is_empty());

        let relayed = crate::protocol::local_ann_to_proto_ann(&local);
        let old_source = LegacyAnnouncement::decode(bytes.as_slice()).unwrap();
        let old_recipient = LegacyAnnouncement::decode(relayed.encode_to_vec().as_slice()).unwrap();
        assert_eq!(old_recipient.serving_models, vec!["gated", "free"]);
        assert_eq!(old_recipient.hosted_models, vec!["gated", "free"]);
        assert_eq!(
            relayed.legacy_lightning_offers,
            decoded.legacy_lightning_offers
        );
        assert_eq!(
            old_recipient.lightning_offers, old_source.lightning_offers,
            "an old recipient must receive the original gate, not an ungated model or fabricated prices"
        );
        let (_, restored) = proto_ann_to_local(&relayed).unwrap();
        assert_eq!(
            restored.served_model_descriptors,
            local.served_model_descriptors
        );
    }

    #[test]
    fn system_one_capability_decodes_from_additive_proto_field() {
        let descriptor = node::ServedModelDescriptor {
            identity: Some(node::ServedModelIdentity {
                model_name: "decision-model".to_string(),
                ..Default::default()
            }),
            capabilities_known: Some(true),
            capabilities: Some(node::ModelCapabilities {
                system_one: node::CapabilityLevel::Supported as i32,
                ..Default::default()
            }),
            ..Default::default()
        };

        let local = proto_descriptor_to_local(&descriptor).expect("valid descriptor");
        assert!(local.capabilities.supports_system_one_runtime());
    }

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
