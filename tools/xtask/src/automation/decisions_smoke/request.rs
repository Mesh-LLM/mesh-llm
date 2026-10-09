pub(super) fn questions(model: &str) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "input": "I was charged twice. Please refund me today.",
        "questions": [
            {"type":"predicate","name":"urgent","instructions":"Does this need action today?"},
            {"type":"choice","name":"team","instructions":"Which team?","choices":[
                {"value":"billing","description":"Payments and refunds"},
                {"value":"support","description":"Technical help"}]},
            {"type":"score","name":"frustration","instructions":"How frustrated?","levels":[
                {"label":"0","description":"Calm"},{"label":"1","description":"Frustrated"}]}
        ]
    })
}

pub(super) fn boolean_choice(model: &str) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "input": [{"role":"user","content":[{"type":"input_text","text":"Please review this request."}]}],
        "questions": [{"type":"choice","instructions":"Is this a request?","choices":[
            {"value":true,"description":"A request"},
            {"value":false,"description":"Not a request"}]}]
    })
}
