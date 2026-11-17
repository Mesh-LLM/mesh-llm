use super::*;
use serde_json::json;

#[test]
fn legacy_host_inference_defaults_to_concrete_only() {
    let request: HostInferenceRequest = serde_json::from_value(json!({
        "model_id": "worker",
        "request": {"messages": []}
    }))
    .unwrap();
    assert!(!request.single_model_only);
    assert!(request.target_node_id.is_none());
    assert!(request.timeout_ms.is_none());
}

#[test]
fn single_model_directive_round_trips_without_losing_placement() {
    let value = json!({
        "model_id": "mesh",
        "single_model_only": true,
        "target_node_id": "physical-worker",
        "request": {"messages": []},
        "timeout_ms": 1500
    });
    let request: HostInferenceRequest = serde_json::from_value(value.clone()).unwrap();
    assert!(request.single_model_only);
    assert_eq!(serde_json::to_value(request).unwrap(), value);
}
