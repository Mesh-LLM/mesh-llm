use super::tool_calls::{Call, TOOL_NAME, assistant};
use serde_json::{Value, json};

fn initial_messages(attempt: u32) -> Value {
    json!([
        {"role":"system","content":"You are a strict OpenAI tool-call compatibility probe."},
        {"role":"user","content":format!("Attempt {attempt}: call the {TOOL_NAME} tool with key=codeword. Do not answer directly before the tool call.")}
    ])
}

pub(super) fn tool(model: &str, attempt: u32, stream: bool) -> Value {
    json!({
        "model":model,"messages":initial_messages(attempt),
        "tools":[{"type":"function","function":{
            "name":TOOL_NAME,"description":"Return one deterministic fact from the agent reliability fixture.",
            "parameters":{"type":"object","properties":{"key":{"type":"string","enum":["checksum","codeword"]}},
                "required":["key"],"additionalProperties":false}
        }}],
        "tool_choice":{"type":"function","function":{"name":TOOL_NAME}},
        "parallel_tool_calls":false,"stream":stream,"max_tokens":96,"temperature":0,
        "chat_template_kwargs":{"enable_thinking":false}
    })
}

pub(super) fn continuation(
    model: &str,
    attempt: u32,
    call: &Call,
    original: Option<&Value>,
    stream: bool,
) -> Value {
    let mut messages = initial_messages(attempt);
    let rows = messages.as_array_mut().expect("owned message array");
    rows.push(assistant(call, original));
    rows.push(
        json!({"role":"tool","tool_call_id":call.id,"name":TOOL_NAME,
        "content":json!({"key":call.key,"value":call.key.fact()}).to_string()}),
    );
    json!({"model":model,"messages":messages,"stream":stream,"max_tokens":64,"temperature":0,
        "chat_template_kwargs":{"enable_thinking":false}})
}

pub(super) fn surface(model: &str, attempt: u32, stream: bool) -> Value {
    let sentinel = if stream { "STREAM_OK" } else { "STABILITY_OK" };
    let mut payload = json!({
        "model":model,"messages":[
            {"role":"system","content":"You are a deterministic mesh-llm stability probe."},
            {"role":"user","content":format!("Attempt {attempt}: reply with exactly {sentinel} and no extra text.")}
        ],
        "stream":stream,"max_tokens":32,"temperature":0,
        "chat_template_kwargs":{"enable_thinking":false}
    });
    if stream {
        payload["stream_options"] = json!({"include_usage":true});
    }
    payload
}
