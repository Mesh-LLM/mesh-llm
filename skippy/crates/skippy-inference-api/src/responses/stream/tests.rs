use std::sync::Arc;

use serde_json::{Value, json};

use super::{
    ResponsesStream, ResponsesStreamEvent, obfuscation, retained_chunk_bytes,
    stream_events_for_response,
};
use crate::responses::{
    ResponseContinuation,
    store::{CallerScope, ResponseSessionCache, ResponseStore},
    turn::ResponsesTurn,
};

fn stream_with(continuation: ResponseContinuation) -> ResponsesStream {
    stream_for(
        json!({
        "model": "m",
        "input": "hi",
        "tools": [
            {"type":"function","name":"weather"},
            {"type":"function","name":"time"},
            {"type":"function","name":"f"}
        ]
        }),
        continuation,
    )
}

fn stream_for(mut body: Value, continuation: ResponseContinuation) -> ResponsesStream {
    ResponsesStream::new(Arc::new(
        ResponsesTurn::prepare(&mut body, continuation).unwrap(),
    ))
}

fn stream() -> ResponsesStream {
    stream_with(ResponseContinuation::disabled())
}

fn text(delta: &str) -> Value {
    json!({"choices": [{"delta": {"content": delta}}]})
}

fn finish(reason: &str) -> Value {
    json!({"choices": [{"delta": {}, "finish_reason": reason}], "usage": {"prompt_tokens": 4, "completion_tokens": 2}})
}

fn run(chunks: &[Value]) -> Vec<ResponsesStreamEvent> {
    let mut stream = stream();
    let mut events = stream.start();
    for chunk in chunks {
        events.extend(stream.push_chunk(chunk));
    }
    events.extend(stream.finish());
    events
}

fn names(events: &[ResponsesStreamEvent]) -> Vec<&'static str> {
    events.iter().map(|event| event.event).collect()
}

/// Invariants every stream must hold.
fn assert_well_formed(events: &[ResponsesStreamEvent]) {
    for (index, event) in events.iter().enumerate() {
        assert_eq!(event.data["type"], event.event, "event name matches type");
        assert_eq!(event.data["sequence_number"], index as u64);
    }
    let id = &events[0].data["response"]["id"];
    for event in events
        .iter()
        .filter(|event| event.data.get("response").is_some())
    {
        assert_eq!(&event.data["response"]["id"], id, "one response id");
    }
    let terminal = events
        .iter()
        .filter(|event| {
            matches!(
                event.event,
                "response.completed" | "response.incomplete" | "response.failed"
            )
        })
        .count();
    assert_eq!(terminal, 1, "exactly one terminal event");
    assert!(events.last().unwrap().event.starts_with("response."));
}

fn assert_profiled(event: &ResponsesStreamEvent) {
    let padding = event.data["obfuscation"]
        .as_str()
        .expect("eligible delta has obfuscation");
    assert!((1..=256).contains(&padding.len()));
    assert!(
        padding
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    );
    assert_eq!(serde_json::to_vec(&event.data).unwrap().len() % 256, 0);
}

#[test]
fn text_stream_follows_the_item_lifecycle() {
    let events = run(&[text("Hel"), text("lo"), finish("stop")]);
    assert_well_formed(&events);
    assert_eq!(
        names(&events),
        [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    let completed = &events.last().unwrap().data["response"];
    assert_eq!(completed["status"], "completed");
    assert_eq!(completed["output"][0]["content"][0]["text"], "Hello");
    assert_eq!(completed["usage"]["total_tokens"], 6);
    assert_eq!(events[6].data["text"], "Hello");
    assert!(events[0].data["response"]["completed_at"].is_null());
    for event in events
        .iter()
        .filter(|event| event.event == "response.output_text.delta")
    {
        assert_profiled(event);
    }
    assert!(
        events
            .iter()
            .filter(|event| {
                matches!(
                    event.event,
                    "response.output_text.done" | "response.completed"
                )
            })
            .all(|event| event.data.get("obfuscation").is_none())
    );
}

#[test]
fn explicit_false_disables_padding_without_changing_output() {
    let mut stream = stream_for(
        json!({
            "model":"m",
            "input":"hi",
            "stream":true,
            "stream_options":{"include_obfuscation":false},
        }),
        ResponseContinuation::disabled(),
    );
    let mut events = stream.start();
    events.extend(stream.push_chunk(&text("plain")));
    events.extend(stream.push_chunk(&finish("stop")));
    events.extend(stream.finish());

    let delta = events
        .iter()
        .find(|event| event.event == "response.output_text.delta")
        .unwrap();
    assert!(delta.data.get("obfuscation").is_none());
    let completed = &events.last().unwrap().data["response"];
    assert_eq!(completed["output_text"], "plain");
    assert!(!completed.to_string().contains("obfuscation"));
}

#[test]
fn omitted_null_empty_and_true_stream_options_enable_padding() {
    for stream_options in [
        None,
        Some(Value::Null),
        Some(json!({})),
        Some(json!({
            "include_obfuscation":true,
        })),
    ] {
        let mut body = json!({"model":"m","input":"hi","stream":true});
        if let Some(stream_options) = stream_options {
            body["stream_options"] = stream_options;
        }
        let delta = stream_for(body, ResponseContinuation::disabled())
            .push_chunk(&text("padded"))
            .into_iter()
            .find(|event| event.event == "response.output_text.delta")
            .unwrap();
        assert_profiled(&delta);
    }
}

#[test]
fn original_turn_policy_survives_backend_mutation_and_does_not_inherit() {
    let store = Arc::new(ResponseStore::default());
    let continuation = ResponseContinuation::new(Some(store.clone()), Some(CallerScope::LocalHost));
    let mut parent_body = json!({
        "model":"m",
        "input":"parent",
        "stream":true,
        "stream_options":{"include_obfuscation":false},
    });
    let parent = ResponsesTurn::prepare(&mut parent_body, continuation.clone()).unwrap();
    let parent_id = parent.id().to_string();
    parent.finish_chat_completion(&json!({
        "choices":[{"message":{"content":"saved"},"finish_reason":"stop"}],
    }));

    let mut child_body = json!({"previous_response_id":parent_id,"input":"child","stream":true});
    let child = Arc::new(ResponsesTurn::prepare(&mut child_body, continuation).unwrap());
    child
        .normalize_backend_request("/v1/responses", &mut child_body)
        .unwrap();
    child_body["stream_options"]["include_obfuscation"] = Value::Bool(false);
    let mut child_stream = ResponsesStream::new(child);
    let delta = child_stream
        .push_chunk(&text("padded"))
        .into_iter()
        .find(|event| event.event == "response.output_text.delta")
        .unwrap();
    assert_profiled(&delta);

    let mut disabled_body = json!({
        "model":"m",
        "input":"disabled",
        "stream":true,
        "stream_options":{"include_obfuscation":false},
    });
    let disabled = Arc::new(
        ResponsesTurn::prepare(&mut disabled_body, ResponseContinuation::disabled()).unwrap(),
    );
    disabled
        .normalize_backend_request("/v1/responses", &mut disabled_body)
        .unwrap();
    disabled_body["stream_options"]["include_obfuscation"] = Value::Bool(true);
    let delta = ResponsesStream::new(disabled)
        .push_chunk(&text("plain"))
        .into_iter()
        .find(|event| event.event == "response.output_text.delta")
        .unwrap();
    assert!(delta.data.get("obfuscation").is_none());
}

#[test]
fn reasoning_streams_as_its_own_item_before_the_message() {
    let events = run(&[
        json!({"choices": [{"delta": {"reasoning_content": "think"}}]}),
        text("answer"),
        finish("stop"),
    ]);
    assert_well_formed(&events);
    let names = names(&events);
    let delta = names
        .iter()
        .position(|name| *name == "response.reasoning.delta")
        .unwrap();
    let done = names
        .iter()
        .position(|name| *name == "response.reasoning.done")
        .unwrap();
    let text = names
        .iter()
        .position(|name| *name == "response.output_text.delta")
        .unwrap();
    assert!(delta < done && done < text);
    let output = &events.last().unwrap().data["response"]["output"];
    assert_eq!(output[0]["type"], "reasoning");
    assert_eq!(output[0]["content"][0]["text"], "think");
    assert_eq!(output[1]["type"], "message");
    assert_eq!(events[delta].data["output_index"], 0);
    assert_profiled(&events[delta]);
}

#[test]
fn assistant_phase_is_carried_by_streamed_output_items() {
    let events = run(&[
        json!({"choices": [{"delta": {"phase": "commentary", "content": "checking"}}]}),
        json!({"choices": [{"delta": {"phase": "final_answer", "content": "done"}}]}),
        finish("stop"),
    ]);
    assert_well_formed(&events);
    let output = &events.last().unwrap().data["response"]["output"];
    assert_eq!(output[0]["phase"], "commentary");
    assert_eq!(output[1]["phase"], "final_answer");
}

#[test]
fn tool_calls_buffer_arguments_until_named_and_keep_call_ids() {
    let events = run(&[
        text("let me check"),
        json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_a", "function": {"arguments": "{\"ci"}}]}}]}),
        json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "weather", "arguments": "ty\":"}}]}}]}),
        json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\"Oslo\"}"}}]}}]}),
        json!({"choices": [{"delta": {"tool_calls": [{"index": 1, "id": "call_b", "function": {"name": "time", "arguments": "{}"}}]}}]}),
        finish("tool_calls"),
    ]);
    assert_well_formed(&events);
    let response = &events.last().unwrap().data["response"];
    assert_eq!(response["status"], "completed");
    let output = response["output"].as_array().unwrap();
    assert_eq!(output.len(), 3);
    assert_eq!(output[0]["status"], "completed");
    assert_eq!(output[1]["call_id"], "call_a");
    assert_eq!(output[1]["name"], "weather");
    assert_eq!(output[1]["arguments"], "{\"city\":\"Oslo\"}");
    assert_eq!(output[2]["call_id"], "call_b");
    let arguments_done: Vec<_> = events
        .iter()
        .filter(|event| event.event == "response.function_call_arguments.done")
        .map(|event| event.data["arguments"].clone())
        .collect();
    assert_eq!(arguments_done, [json!("{\"city\":\"Oslo\"}"), json!("{}")]);
    // The message closed before the first call was announced.
    let names = names(&events);
    let message_done = names
        .iter()
        .position(|name| *name == "response.output_text.done")
        .unwrap();
    let call_added = names
        .iter()
        .enumerate()
        .filter(|(_, name)| **name == "response.output_item.added")
        .nth(1)
        .unwrap()
        .0;
    assert!(message_done < call_added);
    for event in events
        .iter()
        .filter(|event| event.event == "response.function_call_arguments.delta")
    {
        assert_profiled(event);
    }
    assert!(!response.to_string().contains("obfuscation"));
}

#[test]
fn length_cutoff_is_incomplete_and_marks_the_last_item() {
    let events = run(&[text("partial"), finish("length")]);
    assert_well_formed(&events);
    let last = events.last().unwrap();
    assert_eq!(last.event, "response.incomplete");
    assert_eq!(last.data["response"]["status"], "incomplete");
    assert_eq!(
        last.data["response"]["incomplete_details"]["reason"],
        "max_output_tokens"
    );
    assert_eq!(last.data["response"]["output"][0]["status"], "incomplete");

    let events = run(&[
        json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c", "function": {"name": "f", "arguments": "{\"a"}}]}}]}),
        finish("length"),
    ]);
    let response = &events.last().unwrap().data["response"];
    assert_eq!(response["status"], "incomplete");
    assert_eq!(response["output"][0]["status"], "incomplete");
}

#[test]
fn failure_emits_error_then_failed_and_saves_nothing() {
    let store = Arc::new(ResponseStore::default());
    let continuation = ResponseContinuation::new(Some(store.clone()), Some(CallerScope::LocalHost));
    let mut stream = stream_with(continuation.clone());
    let mut events = stream.start();
    events.extend(stream.push_chunk(&text("half")));
    events.extend(stream.fail("server_error", "upstream closed"));
    assert!(
        stream.finish().is_empty(),
        "nothing after the terminal event"
    );
    assert_well_formed(&events);
    let names = names(&events);
    assert_eq!(&names[names.len() - 2..], ["error", "response.failed"]);
    let failed = &events.last().unwrap().data["response"];
    assert_eq!(failed["status"], "failed");
    assert_eq!(failed["error"]["code"], "server_error");
    assert_eq!(failed["output"], json!([]));
    let id = failed["id"].as_str().unwrap();
    assert!(store.get(&CallerScope::LocalHost, id).is_err());

    let mut stream = stream_with(continuation);
    stream.push_chunk(&text("ok"));
    stream.push_chunk(&finish("stop"));
    let completed = stream.finish();
    let id = completed.last().unwrap().data["response"]["id"]
        .as_str()
        .unwrap();
    assert!(store.get(&CallerScope::LocalHost, id).is_ok());
}

#[test]
fn eof_without_a_finish_reason_is_a_failed_terminal() {
    let mut stream = stream();
    let mut events = stream.start();
    events.extend(stream.push_chunk(&text("partial")));
    events.extend(stream.finish());
    assert_well_formed(&events);
    assert_eq!(events.last().unwrap().event, "response.failed");
    assert_eq!(
        events.last().unwrap().data["response"]["error"]["code"],
        "upstream_stream_incomplete"
    );
}

#[test]
fn error_finish_reason_is_a_failed_terminal() {
    let events = run(&[text("partial"), finish("error")]);
    assert_eq!(events.last().unwrap().event, "response.failed");
    assert_eq!(
        events.last().unwrap().data["response"]["error"]["code"],
        "upstream_response_failed"
    );
}

#[test]
fn reconstructed_output_is_bounded_and_failure_omits_partial_items() {
    let mut stream = stream();
    stream.max_output_bytes = 4;
    let events = stream.push_chunk(&text("too large"));
    let failed = &events.last().unwrap().data["response"];
    assert_eq!(events.last().unwrap().event, "response.failed");
    assert_eq!(failed["error"]["code"], "response_output_too_large");
    assert_eq!(failed["output"], json!([]));
}

#[test]
fn phase_only_chunks_consume_the_active_output_budget() {
    let chunk = json!({"choices": [{"delta": {"phase": "commentary"}}]});
    let choice = &chunk["choices"][0];
    let mut stream = stream();
    stream.max_output_bytes = retained_chunk_bytes(choice);
    assert!(!stream.push_chunk(&chunk).is_empty());
    let events = stream.push_chunk(&chunk);
    assert_eq!(events.last().unwrap().event, "response.failed");
    assert_eq!(
        events.last().unwrap().data["response"]["error"]["code"],
        "response_output_too_large"
    );
}

#[test]
fn an_empty_stream_still_reports_one_message() {
    let events = run(&[finish("stop")]);
    assert_well_formed(&events);
    let output = &events.last().unwrap().data["response"]["output"];
    assert_eq!(output.as_array().unwrap().len(), 1);
    assert_eq!(output[0]["content"][0]["text"], "");
}

#[test]
fn replayed_responses_follow_the_same_lifecycle() {
    let mut body = json!({
        "model":"m", "input":"hi",
        "tools":[{"type":"function","name":"f"}]
    });
    let turn = ResponsesTurn::prepare(&mut body, ResponseContinuation::disabled()).unwrap();
    let response = turn.finish_chat_completion(&json!({
        "choices": [{"message": {
            "reasoning_content": "r",
            "content": "hello",
            "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}],
        }, "finish_reason": "tool_calls"}],
    }));
    let events = stream_events_for_response(&response);
    assert_well_formed(&events);
    let names = names(&events);
    assert_eq!(names.first(), Some(&"response.created"));
    assert_eq!(names.last(), Some(&"response.completed"));
    assert!(names.contains(&"response.reasoning.delta"));
    assert!(names.contains(&"response.function_call_arguments.done"));
    assert_eq!(events[0].data["response"]["status"], "in_progress");
}

#[test]
fn malformed_buffered_calls_fail_before_executable_replay() {
    let response = json!({
        "id": "resp_1",
        "status": "completed",
        "output": [{
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "",
            "arguments": {}
        }]
    });
    let events = stream_events_for_response(&response);
    assert_well_formed(&events);
    assert_eq!(
        names(&events),
        ["response.created", "error", "response.failed"]
    );
    assert_eq!(
        events.last().unwrap().data["response"]["error"]["code"],
        "upstream_response_invalid"
    );
}

#[test]
fn fragmented_names_are_authorized_only_after_exact_terminal_assembly() {
    let store = Arc::new(ResponseStore::default());
    let continuation = ResponseContinuation::new(Some(store.clone()), Some(CallerScope::LocalHost));
    let mut stream = stream_for(
        json!({
            "model":"m", "input":"hi",
            "tools":[
                {"type":"function","name":"safe"},
                {"type":"function","name":"safe_admin"}
            ],
            "tool_choice":{"type":"allowed_tools","tools":[
                {"type":"function","name":"safe"}
            ]}
        }),
        continuation,
    );
    let id = stream.turn().id().to_string();
    let mut events = stream.start();
    events.extend(
        stream.push_chunk(&json!({"choices":[{"delta":{"tool_calls":[{
            "index":0,"id":"call_1","function":{"name":"safe","arguments":"{}"}
        }]}}]})),
    );
    events.extend(
        stream.push_chunk(&json!({"choices":[{"delta":{"tool_calls":[{
        "index":0,"function":{"name":"_admin"}
    }]},"finish_reason":"tool_calls"}]})),
    );
    events.extend(stream.finish());

    assert_well_formed(&events);
    assert_eq!(events.last().unwrap().event, "response.failed");
    assert_eq!(
        events.last().unwrap().data["response"]["error"]["code"],
        "tool_choice_violation"
    );
    assert!(!names(&events).contains(&"response.output_item.added"));
    assert!(store.get(&CallerScope::LocalHost, &id).is_err());
}

#[test]
fn allowed_fragmented_name_replays_one_complete_executable_call() {
    let mut stream = stream_for(
        json!({
            "model":"m", "input":"hi",
            "tools":[{"type":"function","name":"weather"}],
            "tool_choice":{"type":"allowed_tools","tools":[
                {"type":"function","name":"weather"}
            ],"mode":"required"}
        }),
        ResponseContinuation::disabled(),
    );
    let mut events = stream.start();
    for chunk in [
        json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"wea","arguments":"{"}}]}}]}),
        json!({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"ther","arguments":"}"}}]},"finish_reason":"tool_calls"}]}),
    ] {
        assert!(stream.push_chunk(&chunk).is_empty());
    }
    events.extend(stream.finish());
    assert_well_formed(&events);
    let response = &events.last().unwrap().data["response"];
    assert_eq!(response["output"][0]["name"], "weather");
    assert_eq!(response["output"][0]["arguments"], "{}");
}

#[test]
fn none_and_required_modes_follow_completed_vs_incomplete_semantics() {
    let mut none = stream_for(
        json!({
            "model":"m", "input":"hi", "tool_choice":"none",
            "tools":[{"type":"function","name":"safe"}]
        }),
        ResponseContinuation::disabled(),
    );
    let events = none.push_chunk(&json!({"choices":[{"delta":{"tool_calls":[{
        "index":0,"id":"c","function":{"arguments":"{}"}
    }]},"finish_reason":"length"}]}));
    assert!(
        events
            .iter()
            .all(|event| event.event != "response.output_item.added")
    );
    let failed = none.finish();
    assert_eq!(failed.last().unwrap().event, "response.failed");
    assert_eq!(
        failed.last().unwrap().data["response"]["error"]["code"],
        "tool_choice_violation"
    );

    for (reason, terminal) in [
        ("stop", "response.failed"),
        ("length", "response.incomplete"),
    ] {
        let mut required = stream_for(
            json!({
                "model":"m", "input":"hi", "tool_choice":"required",
                "tools":[{"type":"function","name":"safe"}]
            }),
            ResponseContinuation::disabled(),
        );
        required.push_chunk(&finish(reason));
        assert_eq!(required.finish().last().unwrap().event, terminal);
    }
}

#[test]
fn malformed_calls_and_quarantine_bounds_fail_without_sequence_gaps() {
    let malformed = json!({"choices":[{"delta":{"tool_calls":[{
        "id":"c","function":{"name":"weather","arguments":"{}"}
    }]}}]});
    let mut malformed_stream = stream();
    let events = malformed_stream.push_chunk(&malformed);
    assert_well_formed(&events);
    assert_eq!(events.last().unwrap().event, "response.failed");

    let mut bounded = stream();
    // The two replay events fit; the first closing event must still be
    // admitted before any executable suffix can escape.
    bounded.max_quarantined_events = 2;
    let mut events = bounded.push_chunk(&json!({"choices":[{"delta":{"tool_calls":[{
        "index":0,"id":"c","function":{"name":"weather","arguments":"{}"}
    }]}}]}));
    events.extend(bounded.push_chunk(&finish("tool_calls")));
    events.extend(bounded.finish());
    assert_well_formed(&events);
    assert_eq!(events.last().unwrap().event, "response.failed");
    assert_eq!(
        events.last().unwrap().data["response"]["error"]["code"],
        "response_output_too_large"
    );
    assert!(!names(&events).contains(&"response.output_item.added"));
}

#[test]
fn quarantine_terminal_event_is_inside_the_byte_budget_and_precedes_commit() {
    let chunk = json!({"choices":[{
        "delta":{"tool_calls":[{
            "index":0,"id":"c","function":{"name":"weather","arguments":"{}"}
        }]},
        "finish_reason":"tool_calls"
    }]});

    let mut baseline = stream();
    baseline.push_chunk(&chunk);
    let suffix = baseline.finish();
    let before_terminal = suffix[..suffix.len() - 1]
        .iter()
        .map(|event| serde_json::to_vec(&event.data).unwrap().len())
        .sum();

    let store = Arc::new(ResponseStore::default());
    let session = Arc::new(ResponseSessionCache::new(1024 * 1024));
    let continuation = ResponseContinuation::for_session(
        Some(store.clone()),
        Some(CallerScope::LocalHost),
        session.clone(),
        true,
    );
    let mut bounded = stream_with(continuation);
    let id = bounded.turn().id().to_string();
    let mut events = bounded.push_chunk(&chunk);
    bounded.max_quarantined_event_bytes = before_terminal;
    events.extend(bounded.finish());

    assert_well_formed(&events);
    assert_eq!(events.last().unwrap().event, "response.failed");
    assert!(!names(&events).contains(&"response.output_item.added"));
    assert!(store.get(&CallerScope::LocalHost, &id).is_err());
    assert!(session.get(&id).is_none());
}

#[test]
fn quarantine_obfuscation_failure_emits_only_a_contiguous_failed_lifecycle() {
    fn failing_random_source() -> Result<[u8; 256], obfuscation::ObfuscationError> {
        Err(obfuscation::ObfuscationError)
    }

    let store = Arc::new(ResponseStore::default());
    let session = Arc::new(ResponseSessionCache::new(1024 * 1024));
    let continuation = ResponseContinuation::for_session(
        Some(store.clone()),
        Some(CallerScope::LocalHost),
        session.clone(),
        true,
    );
    let mut stream = stream_with(continuation);
    let id = stream.turn().id().to_string();
    stream.set_quarantine_random_source(failing_random_source);

    let mut events = stream.start();
    events.extend(stream.push_chunk(&json!({"choices":[{
        "delta":{"tool_calls":[{
            "index":0,
            "id":"call_rng_failure",
            "function":{"name":"weather","arguments":"{}"}
        }]},
        "finish_reason":"tool_calls"
    }]})));
    events.extend(stream.finish());

    assert_well_formed(&events);
    assert_eq!(
        names(&events),
        [
            "response.created",
            "response.in_progress",
            "error",
            "response.failed"
        ]
    );
    assert_eq!(events[2].data["error"]["code"], "stream_obfuscation_failed");
    assert_eq!(
        events[3].data["response"]["error"]["code"],
        "stream_obfuscation_failed"
    );
    assert_eq!(events[3].data["response"]["output"], json!([]));
    assert!(store.get(&CallerScope::LocalHost, &id).is_err());
    assert!(session.get(&id).is_none());
}

#[test]
fn rejected_stream_does_not_replace_websocket_session_state() {
    let session = Arc::new(ResponseSessionCache::new(1024 * 1024));
    let continuation = ResponseContinuation::for_session(None, None, session.clone(), false);
    let mut stream = stream_for(
        json!({
            "model":"m", "input":"hi",
            "tools":[{"type":"function","name":"safe"}],
            "tool_choice":{"type":"allowed_tools","tools":[
                {"type":"function","name":"safe"}
            ]}
        }),
        continuation,
    );
    let id = stream.turn().id().to_string();
    stream.push_chunk(&json!({"choices":[{"delta":{"tool_calls":[{
        "index":0,"id":"c","function":{"name":"other","arguments":"{}"}
    }]},"finish_reason":"tool_calls"}]}));
    assert_eq!(stream.finish().last().unwrap().event, "response.failed");
    assert!(session.get(&id).is_none());
}

#[test]
fn max_tool_calls_failure_quarantines_repeated_calls_and_commits_nothing() {
    let store = Arc::new(ResponseStore::default());
    let session = Arc::new(ResponseSessionCache::new(1024 * 1024));
    let continuation = ResponseContinuation::for_session(
        Some(store.clone()),
        Some(CallerScope::LocalHost),
        session.clone(),
        true,
    );
    let mut stream = stream_for(
        json!({
            "model":"m", "input":"hi", "max_tool_calls": 1,
            "tools":[{"type":"function","name":"same"}]
        }),
        continuation,
    );
    let id = stream.turn().id().to_string();
    let mut events = stream.start();
    events.extend(stream.push_chunk(&json!({"choices":[{
        "delta":{"tool_calls":[
            {"index":0,"id":"c0","function":{"name":"same","arguments":"{}"}},
            {"index":1,"id":"c1","function":{"name":"same","arguments":"{}"}}
        ]},
        "finish_reason":"tool_calls"
    }]})));
    events.extend(stream.finish());

    assert_well_formed(&events);
    assert_eq!(events.last().unwrap().event, "response.failed");
    assert_eq!(
        events.last().unwrap().data["response"]["error"]["code"],
        "tool_call_limit_exceeded"
    );
    assert_eq!(events.last().unwrap().data["response"]["output"], json!([]));
    assert!(events.iter().all(|event| {
        !matches!(
            event.event,
            "response.output_item.added"
                | "response.function_call_arguments.delta"
                | "response.function_call_arguments.done"
        )
    }));
    assert!(store.get(&CallerScope::LocalHost, &id).is_err());
    assert!(session.get(&id).is_none());
}
