use super::{local_model_metadata_to_proto, proto_model_metadata_to_local};
use crate::mesh::ServedModelMetadata;
use prost::Message;

#[test]
fn thinking_efforts_preserve_presence_and_order_through_protobuf() {
    for efforts in [None, Some(vec![]), Some(vec!["high".into(), "low".into()])] {
        let metadata = ServedModelMetadata {
            thinking_efforts: efforts.clone(),
            ..Default::default()
        };
        let bytes = local_model_metadata_to_proto(&metadata).encode_to_vec();
        let decoded = crate::proto::node::ServedModelMetadata::decode(bytes.as_slice()).unwrap();
        assert_eq!(
            proto_model_metadata_to_local(&decoded).thinking_efforts,
            efforts
        );
    }
}

#[test]
fn thinking_efforts_keep_legacy_json_unknown_and_observed_empty_distinct() {
    let legacy: ServedModelMetadata = serde_json::from_str("{}").unwrap();
    assert_eq!(legacy.thinking_efforts, None);
    assert!(legacy.is_empty());
    let observed = ServedModelMetadata {
        thinking_efforts: Some(vec![]),
        ..Default::default()
    };
    assert!(!observed.is_empty());
    let json = serde_json::to_value(&observed).unwrap();
    assert_eq!(json, serde_json::json!({"thinking_efforts": []}));
    assert_eq!(
        serde_json::from_value::<ServedModelMetadata>(json).unwrap(),
        observed
    );
}

#[test]
fn legacy_protobuf_readers_ignore_the_new_thinking_metadata() {
    // An older reader knows the existing architecture tag but no effort field.
    #[derive(Clone, PartialEq, Message)]
    struct LegacyMetadata {
        #[prost(string, optional, tag = "1")]
        architecture: Option<String>,
    }
    let metadata = ServedModelMetadata {
        architecture: Some("test".into()),
        thinking_efforts: Some(vec!["max".into()]),
        ..Default::default()
    };
    let bytes = local_model_metadata_to_proto(&metadata).encode_to_vec();
    let legacy = LegacyMetadata::decode(bytes.as_slice()).unwrap();
    assert_eq!(legacy.architecture.as_deref(), Some("test"));
    let decoded =
        crate::proto::node::ServedModelMetadata::decode(legacy.encode_to_vec().as_slice()).unwrap();
    assert_eq!(
        proto_model_metadata_to_local(&decoded).thinking_efforts,
        None
    );
}
