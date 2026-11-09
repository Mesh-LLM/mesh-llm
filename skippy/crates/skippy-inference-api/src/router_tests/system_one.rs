use super::*;

#[tokio::test]
async fn system_one_is_mounted_under_openai_v1() {
    let response = post_json(
        "/v1/systemone",
        json!({
            "state": "release candidate",
            "model": "openjev-latest",
            "questions": {
                "safe": {
                    "type": "noul",
                    "instructions": "Is this safe?"
                }
            }
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_body_json(response).await;
    assert_eq!(body["model"], "openjev-latest");
    assert_eq!(body["answers"]["safe"]["type"], "noul");
    assert_eq!(body["answers"]["safe"]["noul"], 0.875);
    assert_eq!(body["usage"]["input_tokens"], 12);
    assert_eq!(body["usage"]["output_tokens"], 0);
}

#[tokio::test]
async fn system_one_legacy_mount_matches_canonical_mount() {
    // Both mounts share one handler, so the answers must be identical.
    let payload = json!({
        "state": "release candidate",
        "model": "openjev-latest",
        "questions": {
            "safe": {
                "type": "noul",
                "instructions": "Is this safe?"
            }
        }
    });
    let canonical = post_json("/v1/systemone", payload.clone()).await;
    let legacy = post_json("/systemone", payload).await;
    assert_eq!(canonical.status(), StatusCode::OK);
    assert_eq!(legacy.status(), StatusCode::OK);
    assert_eq!(
        response_body_json(canonical).await,
        response_body_json(legacy).await
    );
}

#[tokio::test]
async fn system_one_route_preserves_jev_response_shape() {
    let response = post_json(
        "/systemone",
        json!({
            "state": "release candidate",
            "model": "openjev-latest",
            "questions": {
                "safe": {
                    "type": "noul",
                    "instructions": "Is this safe?"
                }
            }
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_body_json(response).await;
    assert_eq!(body["model"], "openjev-latest");
    assert_eq!(body["answers"]["safe"]["type"], "noul");
    assert_eq!(body["answers"]["safe"]["noul"], 0.875);
    assert_eq!(body["usage"]["input_tokens"], 12);
    assert_eq!(body["usage"]["output_tokens"], 0);
}
