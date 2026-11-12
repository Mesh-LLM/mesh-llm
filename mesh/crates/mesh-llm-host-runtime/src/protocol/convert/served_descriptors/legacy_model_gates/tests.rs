use super::*;
use crate::proto::node;
use crate::protocol::proto_ann_to_local;

fn source() -> node::PeerAnnouncement {
    node::PeerAnnouncement {
        endpoint_id: vec![1; 32],
        serving_models: vec!["gated".into(), "free".into()],
        hosted_models: vec!["gated".into(), "free".into()],
        legacy_lightning_offers: vec![LegacyModelGate {
            model: "gated".into(),
        }],
        ..Default::default()
    }
}

#[test]
fn gates_cover_both_name_only_and_identity_only_legacy_announcements() {
    for identities in [
        Vec::new(),
        ["gated", "free"]
            .map(|name| node::ServedModelIdentity {
                model_name: name.into(),
                ..Default::default()
            })
            .to_vec(),
    ] {
        let mut source = source();
        source.served_model_identities = identities;
        let (_, local) = proto_ann_to_local(&source).unwrap();
        let gated = local
            .served_model_descriptors
            .iter()
            .find(|descriptor| descriptor.identity.model_name == "gated")
            .unwrap();
        let free = local
            .served_model_descriptors
            .iter()
            .find(|descriptor| descriptor.identity.model_name == "free")
            .unwrap();
        assert_eq!(gated.requires[0].channel, LEGACY_LIGHTNING_CHANNEL);
        assert!(gated.requires[0].terms.is_empty());
        assert!(free.requires.is_empty());
    }
}

#[test]
fn gates_match_public_ids_and_preserve_modern_requirements() {
    let mut source = source();
    source.served_model_descriptors = vec![node::ServedModelDescriptor {
        identity: Some(node::ServedModelIdentity {
            model_name: "gated".into(),
            source_kind: node::ModelSourceKind::Catalog as i32,
            canonical_ref: Some("acme/public-model".into()),
            ..Default::default()
        }),
        requires: vec![node::ChannelRequirement {
            channel: "plugin.v1".into(),
            terms: vec![42],
        }],
        ..Default::default()
    }];
    source.legacy_lightning_offers[0].model = "acme/public-model".into();
    let (_, mut local) = proto_ann_to_local(&source).unwrap();
    assert_eq!(local.served_model_descriptors[0].requires.len(), 2);
    assert_eq!(
        local.served_model_descriptors[0].requires[0].terms,
        vec![42]
    );
    assert_eq!(
        local.served_model_descriptors[0].requires[1].channel,
        LEGACY_LIGHTNING_CHANNEL
    );
    restore(&source.legacy_lightning_offers, &mut local).unwrap();
    assert_eq!(
        local.served_model_descriptors[0].requires.len(),
        2,
        "translation is idempotent"
    );
}

#[test]
fn malformed_gate_lists_reject_the_announcement_instead_of_appearing_free() {
    let gate = |model: String| LegacyModelGate { model };
    for gates in [
        vec![gate(String::new())],
        vec![gate("x".repeat(1025))],
        vec![gate("gated".into()), gate("gated".into())],
        (0..129)
            .map(|index| gate(format!("gated-{index}")))
            .collect(),
    ] {
        let mut source = source();
        source.legacy_lightning_offers = gates;
        assert!(proto_ann_to_local(&source).is_none());
    }
    let mut source = source();
    source.legacy_lightning_offers = (0..128)
        .map(|index| gate(format!("gated-{index}")))
        .collect();
    assert!(proto_ann_to_local(&source).is_some());
}
