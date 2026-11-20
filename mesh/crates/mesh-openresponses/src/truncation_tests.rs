use super::*;
use serde_json::json;
use skippy_guardrails::MESH_COMPACT_FIELD;

#[test]
fn accepted_responses_truncation_modes_force_compaction_off() {
    for mut body in [
        json!({"model": "m", "input": "default"}),
        json!({"model": "m", "input": "null", "truncation": null}),
        json!({"model": "m", "input": "disabled", "truncation": "disabled"}),
        json!({"model": "m", "input": "override", "mesh_compact": true}),
    ] {
        normalize_openai_compat_request("/v1/responses?transport=http", &mut body).unwrap();
        assert_eq!(body[MESH_COMPACT_FIELD], false, "normalized body: {body}");
    }
}

#[test]
fn automatic_responses_truncation_remains_rejected() {
    let mut body = json!({"model": "m", "input": "auto", "truncation": "auto"});
    let error = normalize_openai_compat_request("/v1/responses", &mut body).unwrap_err();

    assert_eq!(error.body().error.param.as_deref(), Some("truncation"));
    assert_eq!(error.status(), axum::http::StatusCode::BAD_REQUEST);
}

#[test]
fn chat_and_explicit_responses_compaction_keep_their_override() {
    for path in ["/v1/chat/completions", "/v1/responses/compact"] {
        let mut body = json!({"model": "m", "messages": [], "mesh_compact": true});
        normalize_openai_compat_request(path, &mut body).unwrap();
        assert_eq!(body[MESH_COMPACT_FIELD], true, "path: {path}");
    }
}
