use super::*;
use mesh_llm_types::mesh::ModelWorkloadClass;
use mesh_llm_types::models::discovery::{MeshModelAvailability, MeshModelKind, MeshModelProvider};

fn model(id: &str, workload: Option<ModelWorkloadClass>) -> MeshModel {
    let causal = workload == Some(ModelWorkloadClass::CausalGeneration);
    MeshModel {
        id: id.into(),
        display_name: format!("Name: {id}"),
        kind: if causal {
            MeshModelKind::Llm
        } else {
            MeshModelKind::Model
        },
        workload_class: workload,
        availability: MeshModelAvailability::Available,
        operations: if causal { vec!["chat".into()] } else { vec![] },
        metadata: None,
        virtual_model: None,
        providers: vec![MeshModelProvider {
            provider_id: "node-a".into(),
            local: false,
            endpoint_id: None,
            plugin: None,
            workload_class: workload,
            available: true,
            operations: if causal { vec!["chat".into()] } else { vec![] },
            capabilities: None,
            metadata: None,
            runtime: None,
            thinking: Some(json!({"toggle": true})),
            requires: Some(json!([{"channel": "example.offer.v1"}])),
        }],
    }
}

fn llm(id: &str) -> MeshModel {
    model(id, Some(ModelWorkloadClass::CausalGeneration))
}

#[test]
fn openai_list_is_llm_only_and_contains_only_standard_fields() {
    let mut models = vec![llm("llm"), model("legacy", None)];
    for workload in [
        ModelWorkloadClass::Embedding,
        ModelWorkloadClass::Rerank,
        ModelWorkloadClass::EncoderDecoder,
        ModelWorkloadClass::SpeechSynthesis,
        ModelWorkloadClass::SpeechRecognition,
        ModelWorkloadClass::Decision,
        ModelWorkloadClass::DocumentOcr,
        ModelWorkloadClass::Unknown,
    ] {
        models.push(model(&format!("{workload:?}"), Some(workload)));
    }
    let (status, body) = models_response(&models, "/v1/models", false);
    assert_eq!(status, 200);
    assert_eq!(
        body,
        json!({"object":"list", "data":[{
            "id":"llm", "object":"model", "created":0,"owned_by":"mesh-llm"
        }]})
    );
}

#[test]
fn availability_is_checked_on_the_llm_provider_not_an_ocr_provider() {
    let mut shared = llm("shared");
    shared.providers[0].available = false;
    let mut ocr = shared.providers[0].clone();
    ocr.available = true;
    ocr.workload_class = Some(ModelWorkloadClass::DocumentOcr);
    shared.providers.push(ocr);
    assert_eq!(
        models_response(&[shared], "/models", false).1["data"],
        json!([])
    );
}

#[test]
fn available_llm_provider_keeps_a_shared_ocr_identity_discoverable() {
    let mut shared = llm("shared");
    shared.workload_class = None;
    let mut ocr = shared.providers[0].clone();
    ocr.workload_class = Some(ModelWorkloadClass::DocumentOcr);
    shared.providers.push(ocr);
    assert_eq!(
        models_response(&[shared], "/models", false).1["data"][0]["id"],
        "shared"
    );
}

#[test]
fn anthropic_has_its_own_envelope_and_bidirectional_pagination() {
    let models = vec![llm("c"), llm("a"), llm("b")];
    let (_, first) = models_response(&models, "/v1/models?limit=1", true);
    assert_eq!(
        first["data"],
        json!([{
            "id":"a", "type":"model", "display_name":"Name: a",
            "created_at":"1970-01-01T00:00:00Z", "lifecycle":"active"
        }])
    );
    assert_eq!(first["first_id"], "a");
    assert_eq!(first["last_id"], "a");
    assert_eq!(first["has_more"], true);
    let (_, next) = models_response(&models, "/v1/models?limit=1&after_id=a", true);
    assert_eq!(next["data"][0]["id"], "b");
    let (_, previous) = models_response(&models, "/v1/models?limit=1&before_id=c", true);
    assert_eq!(previous["data"][0]["id"], "b");
    assert_eq!(previous["has_more"], true);
    let (_, empty) = models_response(&models, "/v1/models?after_id=c", true);
    assert_eq!(empty["data"], json!([]));
    assert_eq!(empty["first_id"], Value::Null);
    assert_eq!(empty["last_id"], Value::Null);
    assert_eq!(empty["has_more"], false);
}

#[test]
fn detail_uses_the_same_llm_filter_and_decodes_ids_exactly_once() {
    let models = vec![
        llm("org/model@revision#profile+one"),
        llm("literal%2Fvalue"),
        model("ocr", Some(ModelWorkloadClass::DocumentOcr)),
    ];
    let (status, body) = models_response(
        &models,
        "/v1/models/org%2Fmodel%40revision%23profile+one",
        false,
    );
    assert_eq!(status, 200);
    assert_eq!(body["id"], "org/model@revision#profile+one");
    assert_eq!(
        models_response(&models, "/models/literal%252Fvalue", true).1["id"],
        "literal%2Fvalue"
    );
    let (status, body) = models_response(&models, "/v1/models/ocr", true);
    assert_eq!(status, 404);
    assert_eq!(body["error"]["type"], "not_found_error");
    for path in [
        "/models/",
        "/models/%",
        "/models/%GG",
        "/models/%FF",
        "/models/%00",
    ] {
        assert_eq!(models_response(&models, path, false).0, 400, "{path}");
    }
}

#[test]
fn pagination_rejects_invalid_limits_cursors_and_duplicate_parameters() {
    let models = vec![llm("a")];
    for query in [
        "limit=0",
        "limit=1001",
        "limit=nope",
        "limit=1&limit=2",
        "after_id=missing",
        "before_id=missing",
        "before_id=a&after_id=a",
        "extra=true",
    ] {
        assert_eq!(
            models_response(&models, &format!("/models?{query}"), true).0,
            400,
            "{query}"
        );
    }
}

#[test]
fn anthropic_cursors_reject_invalid_utf8_without_substituting_another_model_id() {
    let models = vec![llm("\u{fffd}")];
    for encoded in ["%FF", "%C3%28", "%", "%XY"] {
        assert_eq!(
            models_response(&models, &format!("/models?after_id={encoded}"), true).0,
            400,
            "{encoded}",
        );
    }
    assert_eq!(
        models_response(&models, "/models?after_id=%EF%BF%BD", true).0,
        200,
    );
}

#[test]
fn public_ids_keep_quantization_profiles_and_virtual_routes_distinct() {
    let mut virtual_model = llm("mesh");
    virtual_model.kind = MeshModelKind::Virtual;
    virtual_model.virtual_model = Some(json!({"plugin":"mesh-moa"}));
    let models = vec![
        llm("org/model:Q4#small"),
        llm("org/model:Q8#large"),
        virtual_model,
    ];
    let (_, body) = models_response(&models, "/models", false);
    assert_eq!(body["data"].as_array().unwrap().len(), 3);
    assert_eq!(body["data"][0]["id"], "mesh");
    assert!(body["data"][0].get("virtual_model").is_none());
}
