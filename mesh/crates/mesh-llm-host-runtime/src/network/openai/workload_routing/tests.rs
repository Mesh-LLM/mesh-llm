use super::*;

/// The inference surface the OpenAI frontend router recognizes: every route
/// the router can dispatch to a model target, plus the metadata routes it
/// serves without model execution. The coverage test below requires every
/// entry to be workload-classified, System One, or explicitly exempt — a new
/// route fails CI until it is classified.
const ROUTER_INFERENCES_PATHS: [&str; 19] = [
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/responses",
    "/v1/messages",
    "/v1/embeddings",
    "/v1/rerank",
    "/v1/audio/speech",
    "/v1/audio/transcriptions",
    "/v1/audio/translations",
    "/v1/realtime",
    "/v1/models",
    "/models",
    "/v1/tokenize",
    "/v1/messages/count_tokens",
    "/systemone",
    "/v1/decisions",
    "/mesh/v1/documents/ocr",
    "/mesh/v1/images/similarity",
    "/mesh/v1/images/depth",
];

#[test]
fn document_ocr_uses_only_the_mesh_namespace() {
    for path in ["/mesh/v1/documents/ocr", "/mesh/v1/documents/ocr?trace=1"] {
        assert!(is_document_ocr_path(path));
        assert_eq!(
            request_workload_class(path),
            Some(ModelWorkloadClass::DocumentOcr)
        );
    }
    assert!(!is_document_ocr_path("/v1/documents/ocr"));
    assert_eq!(request_workload_class("/v1/documents/ocr"), None);
}

#[test]
/// Every inference path the frontend router recognizes must carry a workload
/// class (or be System One). An unclassified route gets no workload admission
/// at all (fail-open), so a new route fails this gate until it is classified.
fn router_inference_paths_are_all_classified() {
    for path in ROUTER_INFERENCES_PATHS {
        // Metadata routes are served without model execution: no workload class by design.
        let exempt = matches!(
            path,
            "/v1/models" | "/models" | "/v1/tokenize" | "/v1/messages/count_tokens"
        );
        let classified = request_workload_class(path).is_some() || is_system_one_path(path);
        assert!(
            classified || exempt,
            "inference path {path} is unclassified and unexempt: it would be admitted without workload filtering"
        );
    }
}

#[test]
/// Distinguish unrecognized explicit metadata from the old absent-field contract.
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

/// Construct a local artifact identity without implying workload capability.
fn local_gguf_descriptor(model_name: &str) -> ServedModelDescriptor {
    ServedModelDescriptor {
        identity: mesh::ServedModelIdentity {
            model_name: model_name.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Attach an explicit class to the same model identity for admission comparisons.
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
/// Preserve old generation peers while rejecting unadvertised non-chat endpoints.
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
/// A known but different workload never satisfies an endpoint's requirement.
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
/// Reordering equivalent metadata must not change route eligibility.
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
/// Encoder-decoder models retain supported direct generation endpoints.
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

#[test]
/// A decision model serves only System One reads; generation and other
/// non-chat endpoints never reach it.
fn decision_models_are_excluded_from_every_classed_endpoint() {
    let descriptors = vec![descriptor_with_workload(
        "laya",
        mesh::ModelWorkloadClass::Decision,
    )];
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
                "laya",
                request_workload_class(path).unwrap(),
                path,
                &descriptors
            ),
            "{path}"
        );
    }
    assert_eq!(request_workload_class("/systemone"), None);
    assert_eq!(request_workload_class("/v1/decisions"), None);
    assert_eq!(
        required_request_workload("/systemone?trace=1"),
        Some(mesh::ModelWorkloadClass::Decision)
    );
    assert_eq!(
        required_request_workload("/v1/decisions?trace=1"),
        Some(mesh::ModelWorkloadClass::Decision)
    );
    assert_eq!(
        unsupported_workload_message("/systemone", mesh::ModelWorkloadClass::Decision),
        "no served model advertises System One support"
    );
}

#[test]
/// Admit System One by its runtime capability, not by workload or architecture.
fn system_one_requires_its_own_advertised_capability() {
    for workload in [
        mesh::ModelWorkloadClass::Decision,
        mesh::ModelWorkloadClass::CausalGeneration,
    ] {
        let mut capable = descriptor_with_workload("system-one", workload);
        capable.capabilities_known = true;
        capable.capabilities.system_one = crate::models::CapabilityLevel::Supported;

        assert!(model_satisfies_request(
            "system-one",
            "/systemone?trace=1",
            std::slice::from_ref(&capable)
        ));
        assert!(model_satisfies_request(
            "system-one",
            "/v1/decisions?trace=1",
            std::slice::from_ref(&capable)
        ));
    }

    let unsupported =
        descriptor_with_workload("plain-decision-model", mesh::ModelWorkloadClass::Decision);
    assert!(!model_satisfies_request(
        "plain-decision-model",
        "/systemone",
        std::slice::from_ref(&unsupported)
    ));
    assert!(!model_satisfies_request(
        "plain-decision-model",
        "/v1/decisions",
        std::slice::from_ref(&unsupported)
    ));
}

#[test]
/// A speech-recognition model with verified audio takes both audio-to-text
/// endpoints and nothing that generates chat text.
fn speech_recognition_models_serve_only_audio_to_text() {
    let mut whisper =
        descriptor_with_workload("whisper", mesh::ModelWorkloadClass::SpeechRecognition);
    whisper.capabilities_known = true;
    whisper.capabilities = crate::models::ModelCapabilities {
        audio: crate::models::CapabilityLevel::Supported,
        multimodal: true,
        ..Default::default()
    };
    let descriptors = [whisper];
    for path in ["/v1/audio/transcriptions", "/v1/audio/translations"] {
        assert!(
            model_satisfies_request("whisper", path, &descriptors),
            "{path}"
        );
    }
    for path in [
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/responses",
        "/v1/embeddings",
        "/v1/audio/speech",
    ] {
        assert!(
            !model_satisfies_request("whisper", path, &descriptors),
            "{path}"
        );
    }
}

#[test]
/// The class alone is not enough: audio upload still needs verified audio.
fn speech_recognition_without_verified_audio_is_not_routable() {
    let descriptors = [descriptor_with_workload(
        "whisper",
        mesh::ModelWorkloadClass::SpeechRecognition,
    )];
    assert!(!model_satisfies_request(
        "whisper",
        "/v1/audio/transcriptions",
        &descriptors
    ));
}

#[test]
/// The mixed-version promise: a peer whose metadata decodes as an unknown
/// class (an older node reading a newer class, or the reverse) is never sent
/// audio, even when it advertises verified audio.
fn unknown_class_peers_never_receive_audio() {
    let mut future = descriptor_with_workload("whisper", mesh::ModelWorkloadClass::Unknown);
    future.capabilities_known = true;
    future.capabilities = crate::models::ModelCapabilities {
        audio: crate::models::CapabilityLevel::Supported,
        multimodal: true,
        ..Default::default()
    };
    let descriptors = [future];
    for path in [
        "/v1/audio/transcriptions",
        "/v1/audio/translations",
        "/v1/chat/completions",
    ] {
        assert!(
            !model_satisfies_request("whisper", path, &descriptors),
            "{path}"
        );
    }
}
