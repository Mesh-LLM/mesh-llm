use serde_json::{Value, json};
use skippy_inference_api::{chat::ChatCompletionRequest, generation::GenerationRequest};

#[test]
fn wire_and_generation_validation_preserve_field_errors() {
    let cases = [
        ("model", json!(" "), "model is required"),
        ("messages", json!([]), "messages is required"),
        (
            "max_tokens",
            json!(0),
            "max_tokens must be greater than zero",
        ),
        (
            "max_completion_tokens",
            json!(0),
            "max_tokens must be greater than zero",
        ),
        ("n", json!(0), "n must be greater than zero"),
        (
            "n",
            json!(2),
            "n > 1 is parsed but multiple choices are not yet implemented",
        ),
        (
            "top_logprobs",
            json!(2),
            "top_logprobs requires logprobs=true",
        ),
        ("tools", json!({}), "tools must be an array"),
        ("tools", json!([1]), "tools[0] must be an object"),
        ("tools", json!([{}]), "tools[0].type must be `function`"),
        (
            "tools",
            json!([{"type":"function"}]),
            "tools[0].function must be an object",
        ),
        (
            "tools",
            json!([{"type":"function","function":{"name":" "}}]),
            "tools[0].function.name must be a non-empty string",
        ),
        (
            "tools",
            json!([{"type":"function","function":{"name":"f","parameters":[]}}]),
            "tools[0].function.parameters must be an object",
        ),
        (
            "response_format",
            json!({"type":7}),
            "response_format must be an object with a type field",
        ),
    ];
    for (field, value, message) in cases {
        let mut input = json!({"model":"model","messages":[{"role":"user","content":"hello"}]});
        input[field] = value;
        let wire: ChatCompletionRequest = serde_json::from_value(input).unwrap();
        let wire_error = wire.validate().unwrap_err();
        let generation_error = GenerationRequest::from(wire).validate().unwrap_err();
        assert_eq!(wire_error.message(), message, "{field}");
        assert_eq!(wire_error.status(), generation_error.status(), "{field}");
        assert_eq!(
            serde_json::to_value(wire_error.body()).unwrap(),
            serde_json::to_value(generation_error.body()).unwrap(),
            "{field}"
        );
    }
}

#[test]
fn wire_and_generation_validation_preserve_characterized_requests() {
    let cases: Value =
        serde_json::from_str(include_str!("fixtures/generation-wire-main.json")).unwrap();
    let mut validated = 0;
    for case in cases.as_array().unwrap() {
        for backend in case["backend"].as_array().unwrap() {
            let Some(request) = backend.get("request") else {
                continue;
            };
            let wire: ChatCompletionRequest = serde_json::from_value(request.clone()).unwrap();
            let wire_result = wire
                .validate()
                .map_err(|error| serde_json::to_value(error.body()).unwrap());
            let generation_result = GenerationRequest::from(wire)
                .validate()
                .map_err(|error| serde_json::to_value(error.body()).unwrap());
            assert_eq!(wire_result, generation_result);
            validated += 1;
        }
    }
    assert!(
        validated >= 20,
        "exercise the preserved wire fixture corpus"
    );
}
