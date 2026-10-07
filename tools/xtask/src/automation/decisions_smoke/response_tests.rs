use super::*;
fn valid() -> Value {
    serde_json::json!({"model":"actual","answers":[
        {"type":"predicate","name":"urgent","probability":0.6},
        {"type":"choice","name":"team","choice":"billing","confidence":0.8,
         "probabilities":[{"value":"billing","probability":0.8},{"value":"support","probability":0.2}]},
        {"type":"score","name":"frustration","score":1.25,"confidence":0.7,
         "probabilities":[{"value":0,"label":"0","probability":0.3},{"value":1,"label":"1","probability":0.7}]}],
        "usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}})
}
#[test]
fn valid_scores_are_finite_without_invented_range_or_sum_constraints() {
    let mut body = valid();
    body["answers"][2]["score"] = serde_json::json!(-3.5);
    body["usage"]["total_tokens"] = serde_json::json!(0);
    body["answers"][1]["probabilities"][0]["probability"] = serde_json::json!(0.1);
    assert!(validate(&body, "actual").is_ok());
}
#[test]
fn response_refuses_typed_value_identity_and_order_corruption() {
    for (pointer, value) in [
        ("/model", serde_json::json!("other")),
        ("/answers/0/probability", serde_json::json!(true)),
        ("/answers/0/probability", serde_json::json!(-0.1)),
        ("/answers/0/probability", serde_json::json!(1.1)),
        ("/answers/1/choice", serde_json::json!("other")),
        ("/answers/1/confidence", serde_json::json!(false)),
        (
            "/answers/1/probabilities/0/value",
            serde_json::json!("support"),
        ),
        ("/answers/2/score", serde_json::json!(true)),
        ("/answers/2/probabilities/0/value", serde_json::json!(false)),
        (
            "/answers/2/probabilities/1/label",
            serde_json::json!("other"),
        ),
        ("/usage/input_tokens", serde_json::json!(true)),
        ("/usage/output_tokens", serde_json::json!(-1)),
        ("/usage/total_tokens", serde_json::json!(1.5)),
    ] {
        let mut body = valid();
        *body.pointer_mut(pointer).unwrap() = value;
        assert!(
            validate(&body, "actual").is_err(),
            "accepted {pointer}: {body}"
        );
    }
    let mut body = valid();
    body["answers"].as_array_mut().unwrap().swap(0, 1);
    assert!(validate(&body, "actual").is_err());
}
#[test]
fn discovery_excludes_aliases_and_requires_advertised_exact_capability() {
    let models = serde_json::json!({"data":[
        {"id":"mesh","capabilities":["system_one"]},
        {"id":"auto","capabilities":["system_one"]},
        {"id":"ordinary","capabilities":[]},
        {"id":"first","capabilities":["system_one"]},
        {"id":"second","capabilities":["system_one"]}]});
    assert_eq!(select_model(&models, None).unwrap(), "first");
    assert_eq!(select_model(&models, Some("second")).unwrap(), "second");
    for requested in ["mesh", "auto", "ordinary", "absent"] {
        assert!(select_model(&models, Some(requested)).is_err());
    }
    assert!(select_model(&serde_json::json!({"data":[]}), None).is_err());
    assert!(select_model(&serde_json::json!({"data":null}), None).is_err());
}
