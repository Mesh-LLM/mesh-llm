use super::*;

#[test]
fn unknown_workloads_never_inherit_legacy_admission() {
    let metadata: mesh::ServedModelMetadata = serde_json::from_value(serde_json::json!({
        "workload_class": "future_non_chat_contract"
    }))
    .unwrap();
    assert_eq!(metadata.workload_class, Some(ModelWorkloadClass::Unknown));
    let descriptors = [mesh::ServedModelDescriptor {
        metadata: Some(metadata),
        ..local_gguf_descriptor("future")
    }];
    for path in [
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/responses",
        "/v1/embeddings",
        "/v1/rerank",
        "/v1/audio/speech",
        "/v1/audio/transcriptions",
        "/v1/audio/translations",
    ] {
        assert!(
            !model_satisfies_request_workload(
                "future",
                request_workload_class(path).unwrap(),
                path,
                &descriptors
            ),
            "{path}"
        );
    }
    let relayed = serde_json::to_value(&descriptors[0].metadata).unwrap();
    assert_eq!(relayed["workload_class"], "unknown");
    let absent: mesh::ServedModelMetadata = serde_json::from_value(serde_json::json!({})).unwrap();
    assert_eq!(absent.workload_class, None);
}

fn local_gguf_descriptor(model_name: &str) -> ServedModelDescriptor {
    ServedModelDescriptor {
        identity: mesh::ServedModelIdentity {
            model_name: model_name.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn descriptor_with_workload(
    model_name: &str,
    workload_class: mesh::ModelWorkloadClass,
) -> mesh::ServedModelDescriptor {
    mesh::ServedModelDescriptor {
        metadata: Some(mesh::ServedModelMetadata {
            workload_class: Some(workload_class),
            ..Default::default()
        }),
        ..local_gguf_descriptor(model_name)
    }
}

#[test]
fn legacy_descriptors_are_compatible_only_with_generation_routes() {
    let descriptors = vec![local_gguf_descriptor("legacy")];

    assert!(model_satisfies_workload_class(
        "legacy",
        mesh::ModelWorkloadClass::CausalGeneration,
        &descriptors
    ));
    assert!(!model_satisfies_workload_class(
        "legacy",
        mesh::ModelWorkloadClass::Embedding,
        &descriptors
    ));
    assert!(!model_satisfies_workload_class(
        "legacy",
        mesh::ModelWorkloadClass::SpeechSynthesis,
        &descriptors
    ));
}

#[test]
fn workload_routes_require_an_exact_advertised_class() {
    let descriptors = vec![
        descriptor_with_workload("embed", mesh::ModelWorkloadClass::Embedding),
        descriptor_with_workload("rank", mesh::ModelWorkloadClass::Rerank),
    ];

    assert!(model_satisfies_workload_class(
        "embed",
        mesh::ModelWorkloadClass::Embedding,
        &descriptors
    ));
    assert!(!model_satisfies_workload_class(
        "embed",
        mesh::ModelWorkloadClass::Rerank,
        &descriptors
    ));
    assert!(model_satisfies_workload_class(
        "rank",
        mesh::ModelWorkloadClass::Rerank,
        &descriptors
    ));
}

#[test]
fn workload_metadata_is_independent_of_descriptor_order() {
    let legacy = local_gguf_descriptor("shared-model");
    let current = descriptor_with_workload("shared-model", mesh::ModelWorkloadClass::Embedding);
    for descriptors in [vec![legacy.clone(), current.clone()], vec![current, legacy]] {
        assert!(model_satisfies_workload_class(
            "shared-model",
            mesh::ModelWorkloadClass::Embedding,
            &descriptors,
        ));
    }
}

#[test]
fn encoder_decoder_models_can_serve_generation_routes() {
    let descriptors = vec![descriptor_with_workload(
        "t5",
        mesh::ModelWorkloadClass::EncoderDecoder,
    )];

    assert!(model_satisfies_workload_class(
        "t5",
        mesh::ModelWorkloadClass::CausalGeneration,
        &descriptors
    ));
    assert!(!model_satisfies_workload_class(
        "t5",
        mesh::ModelWorkloadClass::Embedding,
        &descriptors
    ));
}
