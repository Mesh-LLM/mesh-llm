use super::*;

#[tokio::test]
async fn decisions_preview_maps_all_observed_question_and_answer_types() {
    let response = post_json("/v1/decisions", json!({
        "model": "laya-test", "input": "I was charged twice",
        "questions": [
            {"type": "predicate", "name": "urgent", "instructions": "Does this need action today?"},
            {"type": "choice", "name": "department", "instructions": "Which team?", "choices": [
                {"value": "billing", "description": "Payments"},
                {"value": "technical", "description": "Bugs"}
            ]},
            {"type": "score", "name": "frustration", "instructions": "How frustrated?", "levels": [
                {"label": "0", "description": "Calm"},
                {"label": "1", "description": "Frustrated"},
                {"label": "2", "description": "Angry"}
            ]}
        ]
    })).await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_body_json(response).await;
    assert_eq!(body["model"], "laya-test");
    assert_eq!(
        body["answers"][0],
        json!({"type":"predicate", "name":"urgent", "probability":0.875})
    );
    assert_eq!(body["answers"][1]["choice"], "billing");
    assert_eq!(
        body["answers"][1]["probabilities"][0],
        json!({"value":"billing", "probability":0.75})
    );
    assert_eq!(body["answers"][2]["score"], 1.25);
    assert_eq!(
        body["answers"][2]["probabilities"][2],
        json!({"value":2, "label":"2", "probability":0.5})
    );
    assert_eq!(
        body["usage"],
        json!({"input_tokens":12, "output_tokens":0, "total_tokens":12})
    );
}

#[tokio::test]
async fn decisions_preview_rejects_duplicate_question_names() {
    let response = post_json(
        "/v1/decisions",
        json!({
            "model":"laya-test", "input":"x", "questions":[
                {"type":"predicate", "name":"same"},
                {"type":"predicate", "name":"same"}
            ]
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}
