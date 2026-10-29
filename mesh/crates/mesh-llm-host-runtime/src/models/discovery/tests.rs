use super::*;

#[test]
fn advertised_operations_use_the_same_runtime_evidence_as_endpoint_admission() {
    use crate::models::CapabilityLevel;
    let mut decision = descriptor("decision", ModelWorkloadClass::Decision);
    assert!(!descriptor_operations(Some(&decision)).contains(&"system_one".to_string()));
    decision.capabilities.system_one = CapabilityLevel::Supported;
    assert!(!descriptor_operations(Some(&decision)).contains(&"system_one".to_string()));
    decision.capabilities_known = true;
    assert!(descriptor_operations(Some(&decision)).contains(&"system_one".to_string()));

    let mut speech = descriptor("speech", ModelWorkloadClass::SpeechRecognition);
    assert_eq!(descriptor_operations(Some(&speech)), ["live_transcription"]);
    speech.capabilities_known = true;
    speech.capabilities.audio = CapabilityLevel::Supported;
    assert_eq!(
        descriptor_operations(Some(&speech)),
        ["live_transcription", "transcription", "translation"]
    );

    let mut causal = descriptor("multimodal-llm", ModelWorkloadClass::CausalGeneration);
    causal.capabilities_known = true;
    causal.capabilities.audio = CapabilityLevel::Supported;
    causal.capabilities.system_one = CapabilityLevel::Supported;
    assert_eq!(
        descriptor_operations(Some(&causal)),
        [
            "chat",
            "completions",
            "responses",
            "system_one",
            "transcription",
            "translation"
        ]
    );
}

fn descriptor(name: &str, class: ModelWorkloadClass) -> ServedModelDescriptor {
    ServedModelDescriptor {
        identity: mesh::ServedModelIdentity {
            model_name: name.into(),
            ..Default::default()
        },
        metadata: Some(mesh::ServedModelMetadata {
            workload_class: Some(class),
            architecture: Some("test-architecture".into()),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn model(name: &str, class: ModelWorkloadClass) -> MeshModel {
    let mut catalog = BTreeMap::new();
    insert_observations(
        &mut catalog,
        name,
        &[descriptor(name, class)],
        &[],
        &ProviderContext {
            provider_id: "provider-a",
            local: false,
            available: true,
            endpoint: None,
        },
    );
    finish_model(catalog.into_values().next().unwrap())
}

#[test]
fn discovery_keeps_all_workloads_and_unknown_metadata() {
    let mut models = vec![
        model("llm", ModelWorkloadClass::CausalGeneration),
        model("ocr", ModelWorkloadClass::DocumentOcr),
        model("speech", ModelWorkloadClass::SpeechRecognition),
        model("decision", ModelWorkloadClass::Decision),
        model("embedding", ModelWorkloadClass::Embedding),
        model("unknown", ModelWorkloadClass::Unknown),
    ];
    let mut catalog = BTreeMap::new();
    insert_observations(
        &mut catalog,
        "legacy",
        &[],
        &[],
        &ProviderContext {
            provider_id: "old-peer",
            local: false,
            available: true,
            endpoint: None,
        },
    );
    models.extend(catalog.into_values().map(finish_model));
    let page = query_models(models.clone(), MeshModelQuery::default()).unwrap();
    assert_eq!(page.data.len(), 7);
    assert_eq!(
        page.data
            .iter()
            .filter(|model| model.is_available_llm())
            .count(),
        1
    );
    let ocr = query_models(
        models,
        MeshModelQuery {
            workload_class: Some(ModelWorkloadClass::DocumentOcr),
            available: Some(true),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(ocr.data[0].id, "ocr");
    assert_eq!(ocr.data[0].operations, ["document_ocr"]);
}

#[test]
fn shared_ocr_and_chat_identity_preserves_provider_contracts() {
    let mut catalog = BTreeMap::new();
    for (provider, class, available, context_length) in [
        ("ocr-provider", ModelWorkloadClass::DocumentOcr, true, 8192),
        (
            "chat-provider",
            ModelWorkloadClass::CausalGeneration,
            false,
            32768,
        ),
    ] {
        insert_observations(
            &mut catalog,
            "shared",
            &[descriptor("shared", class)],
            &[ModelRuntimeDescriptor {
                model_name: "shared".into(),
                identity_hash: None,
                context_length: Some(context_length),
                ready: available,
            }],
            &ProviderContext {
                provider_id: provider,
                local: false,
                available,
                endpoint: None,
            },
        );
    }
    let shared = finish_model(catalog.into_values().next().unwrap());
    assert_eq!(shared.providers.len(), 2);
    assert_eq!(shared.workload_class, None);
    assert!(shared.metadata.is_none());
    assert!(shared.is_llm());
    assert!(
        !shared.is_available_llm(),
        "OCR availability must not authorize an unavailable LLM"
    );
    assert!(
        shared
            .providers
            .iter()
            .all(|provider| provider.thinking.is_none())
    );
    assert!(
        shared
            .providers
            .iter()
            .find(|provider| provider.workload_class == Some(ModelWorkloadClass::DocumentOcr))
            .unwrap()
            .runtime
            .is_none()
    );
    let chat = shared
        .providers
        .iter()
        .find(|provider| provider.workload_class == Some(ModelWorkloadClass::CausalGeneration))
        .unwrap();
    assert_eq!(chat.runtime.as_ref().unwrap().context_length, Some(32768));
    for query in [
        MeshModelQuery {
            workload_class: Some(ModelWorkloadClass::CausalGeneration),
            available: Some(true),
            ..Default::default()
        },
        MeshModelQuery {
            kind: Some(MeshModelKind::Llm),
            available: Some(true),
            ..Default::default()
        },
    ] {
        assert!(
            query_models(vec![shared.clone()], query)
                .unwrap()
                .data
                .is_empty()
        );
    }
    assert_eq!(
        query_models(
            vec![shared],
            MeshModelQuery {
                workload_class: Some(ModelWorkloadClass::DocumentOcr),
                available: Some(true),
                ..Default::default()
            }
        )
        .unwrap()
        .data
        .len(),
        1
    );
}

#[test]
fn profile_ids_and_unknown_capabilities_are_preserved() {
    let mut catalog = BTreeMap::new();
    let descriptor = descriptor("org/model:Q4", ModelWorkloadClass::CausalGeneration);
    for name in ["org/model:Q4#small", "org/model:Q4#large"] {
        insert_observations(
            &mut catalog,
            name,
            std::slice::from_ref(&descriptor),
            &[],
            &ProviderContext {
                provider_id: "peer",
                local: false,
                available: true,
                endpoint: None,
            },
        );
    }
    let page = query_models(
        catalog.into_values().map(finish_model).collect(),
        MeshModelQuery::default(),
    )
    .unwrap();
    assert_eq!(page.data.len(), 2);
    assert!(page.data.iter().all(|model| model.is_available_llm()));
    assert!(
        page.data
            .iter()
            .all(|model| model.providers[0].capabilities.is_none())
    );
    assert_eq!(
        lookup(page.data, "org/model:Q4#small").unwrap().id,
        "org/model:Q4#small"
    );
}

#[test]
fn pagination_rejects_stale_or_invalid_cursors_and_combines_filters() {
    let models = vec![
        model("a", ModelWorkloadClass::CausalGeneration),
        model("b", ModelWorkloadClass::CausalGeneration),
        model("c", ModelWorkloadClass::DocumentOcr),
    ];
    let first = query_models(
        models.clone(),
        MeshModelQuery {
            limit: Some(1),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(first.data[0].id, "a");
    let next = MeshModelQuery {
        limit: Some(1),
        cursor: first.next_cursor,
        ..Default::default()
    };
    let second = query_models(models.clone(), next.clone()).unwrap();
    assert_eq!(second.data[0].id, "b");
    assert!(query_models(models[..2].to_vec(), next).is_err());
    assert!(
        query_models(
            models.clone(),
            MeshModelQuery {
                cursor: Some("garbage".into()),
                ..Default::default()
            }
        )
        .is_err()
    );
    for limit in [0, 501] {
        assert!(
            query_models(
                models.clone(),
                MeshModelQuery {
                    limit: Some(limit),
                    ..Default::default()
                }
            )
            .is_err()
        );
    }
    let empty = query_models(
        models,
        MeshModelQuery {
            workload_class: Some(ModelWorkloadClass::DocumentOcr),
            kind: Some(MeshModelKind::Llm),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(empty.data.is_empty());
}
