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
        json!({"input_tokens":12, "input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0}, "output_tokens":0, "output_tokens_details":{"reasoning_tokens":0}, "total_tokens":12})
    );
}

#[tokio::test]
async fn decisions_accepts_unnamed_and_duplicate_named_questions() {
    let response = post_json(
        "/v1/decisions",
        json!({
            "model":"laya-test", "input":"x", "questions":[
                {"type":"predicate", "instructions":"Is it relevant?"},
                {"type":"predicate", "name":"same", "instructions":"Is it urgent?"},
                {"type":"predicate", "name":"same", "instructions":"Is it safe?"}
            ]
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_body_json(response).await;
    assert_eq!(body["answers"][0]["name"], Value::Null);
    assert_eq!(body["answers"][1]["name"], "same");
    assert_eq!(body["answers"][2]["name"], "same");
}

#[tokio::test]
async fn decisions_accepts_text_messages_and_boolean_choices() {
    let response = post_json(
        "/v1/decisions",
        json!({
            "model":"laya-test",
            "input":[{"role":"user","content":[{"type":"input_text","text":"Route this"}]}],
            "questions":[{"type":"choice","instructions":"Is it urgent?","choices":[
                {"value":true,"description":"Urgent"},
                {"value":false,"description":"Routine"}
            ]}]
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_body_json(response).await;
    assert!(body["answers"][0]["choice"].is_boolean());
    assert_eq!(body["answers"][0]["probabilities"][0]["value"], true);
    assert_eq!(body["answers"][0]["probabilities"][1]["value"], false);
}

#[tokio::test]
async fn decisions_rejects_images_explicitly() {
    let response = post_json("/v1/decisions", json!({
        "model":"laya-test",
        "input":[{"role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,AA=="}]}],
        "questions":[{"type":"predicate","instructions":"Is it damaged?"}]
    })).await;
    assert!(!response.status().is_success());
    let body = response_body_json(response).await;
    assert!(body.to_string().contains("image input is not supported"));
}

#[tokio::test]
async fn decisions_preserves_string_and_boolean_choice_identity() {
    let response = post_json(
        "/v1/decisions",
        json!({
            "model":"laya-test", "input":"x", "questions":[{
                "type":"choice", "instructions":"Select one", "choices":[
                    {"value":"true", "description":"The word true"},
                    {"value":true, "description":"Boolean true"},
                    {"value":false, "description":"Boolean false"}
                ]
            }]
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_body_json(response).await;
    assert_eq!(body["answers"][0]["choice"], "true");
    assert_eq!(body["answers"][0]["probabilities"][0]["value"], "true");
    assert_eq!(body["answers"][0]["probabilities"][1]["value"], true);
    assert_eq!(body["answers"][0]["probabilities"][2]["value"], false);
}

#[tokio::test]
async fn decisions_rejects_former_and_undocumented_request_shapes() {
    let requests = [
        json!({"model":"laya-test","input":"x","questions":[{"type":"predicate","name":"urgent"}]}),
        json!({"model":"laya-test","input":"x","questions":[{"type":"predicate","instructions":"Urgent?","criteria":"legacy"}]}),
        json!({"model":"laya-test","input":"x","questions":[{"type":"predicate","instructions":"Urgent?","name":null}]}),
        json!({"model":"laya-test","input":"x","questions":[{"type":"predicate","instructions":"Urgent?"}],"state":"legacy"}),
        json!({"model":"laya-test","input":[{"role":"user","content":"x","metadata":"legacy"}],"questions":[{"type":"predicate","instructions":"Urgent?"}]}),
        json!({"model":"laya-test","input":[{"role":"user","type":null,"content":"x"}],"questions":[{"type":"predicate","instructions":"Urgent?"}]}),
    ];
    for request in requests {
        let response = post_json("/v1/decisions", request).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}
