use super::*;

#[tokio::test]
async fn system_one_is_not_mounted_under_openai_v1() {
    let response = post_json("/v1/systemone", json!({})).await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
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
