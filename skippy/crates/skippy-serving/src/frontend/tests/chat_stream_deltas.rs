//! Streaming chat-completion contract at the event/chunk boundary.
//!
//! `tool_call_stream` covers the per-index delta arithmetic. These tests drive
//! `ChatStreamDeltas` — the type the live streaming path actually uses — and
//! convert its events through `generation_event_to_chat_chunk`, so they pin the
//! shape a client receives on the wire rather than an internal representation.

use super::*;

/// Snapshots recorded from the native parser; see the fixture header.
const RECORDED: &str = include_str!("../../../tests/fixtures/qwen35_partial_tool_calls.txt");

fn recorded_single_call() -> (Vec<Value>, Value) {
    let fixture = recorded_fixture(RECORDED, "single_call");
    (
        fixture.snapshots.into_iter().map(Value::Array).collect(),
        Value::Array(fixture.final_call),
    )
}

fn parsed(tool_calls: Option<Value>, content: Option<&str>) -> ParsedChatMessage {
    ParsedChatMessage {
        content: content.map(ToString::to_string),
        reasoning_content: None,
        tool_calls,
    }
}

/// The `delta.tool_calls` arrays a client would receive, in order, after each
/// event has gone through the real chunk conversion.
fn streamed_tool_call_chunks(events: Vec<GenerationStreamEvent>) -> Vec<Value> {
    events
        .into_iter()
        .filter_map(|event| {
            let chunk = generation_event_to_chat_chunk(Ok(event), "test").expect("chunk");
            let value = serde_json::to_value(&chunk).expect("serialize chunk");
            value["choices"][0]["delta"]
                .get("tool_calls")
                .filter(|tool_calls| !tool_calls.is_null())
                .cloned()
        })
        .collect()
}

#[test]
fn a_tool_call_reaches_the_wire_as_several_chunks_before_generation_ends() {
    let (snapshots, _) = recorded_single_call();
    let mut deltas = ChatStreamDeltas::new(false);

    let chunks = snapshots
        .iter()
        .flat_map(|snapshot| {
            streamed_tool_call_chunks(
                deltas.events_for_parse(&parsed(Some(snapshot.clone()), None), true),
            )
        })
        .collect::<Vec<_>>();

    assert!(
        chunks.len() > 1,
        "the whole point of #1369: a tool call must not arrive as one terminal chunk, got {} chunk(s)",
        chunks.len()
    );
    assert_eq!(chunks[0][0]["index"], 0);
    assert_eq!(chunks[0][0]["type"], "function");
    assert_eq!(chunks[0][0]["function"]["name"], "read_file");
    assert!(chunks[0][0]["id"].as_str().is_some_and(|id| !id.is_empty()));
}

#[test]
fn concatenated_wire_fragments_equal_the_non_streaming_arguments() {
    let (snapshots, final_call) = recorded_single_call();
    let mut deltas = ChatStreamDeltas::new(false);

    let mut chunks = snapshots
        .iter()
        .flat_map(|snapshot| {
            streamed_tool_call_chunks(
                deltas.events_for_parse(&parsed(Some(snapshot.clone()), None), true),
            )
        })
        .collect::<Vec<_>>();
    chunks.extend(streamed_tool_call_chunks(
        deltas.events_for_parse(&parsed(Some(final_call.clone()), None), false),
    ));

    let arguments = chunks
        .iter()
        .filter_map(|tool_calls| tool_calls[0]["function"]["arguments"].as_str())
        .collect::<String>();
    assert_eq!(
        arguments,
        final_call[0]["function"]["arguments"].as_str().unwrap(),
        "a client concatenating fragments must reconstruct the non-streaming arguments"
    );
    assert_eq!(
        deltas.finish_reason(FinishReason::Stop),
        FinishReason::ToolCalls
    );
}

#[test]
fn only_the_first_chunk_for_an_index_carries_the_id_and_name() {
    let (snapshots, _) = recorded_single_call();
    let mut deltas = ChatStreamDeltas::new(false);

    let chunks = snapshots
        .iter()
        .flat_map(|snapshot| {
            streamed_tool_call_chunks(
                deltas.events_for_parse(&parsed(Some(snapshot.clone()), None), true),
            )
        })
        .collect::<Vec<_>>();

    for tool_calls in &chunks[1..] {
        assert!(
            tool_calls[0].get("id").is_none(),
            "a repeated id breaks tool-result correlation: {tool_calls}"
        );
        assert!(tool_calls[0]["function"].get("name").is_none());
    }
}

#[test]
fn a_generation_truncated_mid_tool_call_does_not_report_tool_calls() {
    let (snapshots, _) = recorded_single_call();
    let mut deltas = ChatStreamDeltas::new(false);

    // Every partial parse, and then no terminal parse at all — the shape of a
    // generation that ran out of tokens while emitting the call.
    for snapshot in &snapshots {
        deltas.events_for_parse(&parsed(Some(snapshot.clone()), None), true);
    }

    assert_eq!(
        deltas.finish_reason(FinishReason::Length),
        FinishReason::Length,
        "streaming a partial call must not upgrade the finish reason to tool_calls"
    );
}

#[test]
fn a_request_without_tool_calls_streams_content_exactly_as_before() {
    let mut deltas = ChatStreamDeltas::new(false);

    let first = deltas.events_for_parse(&parsed(None, Some("Hello")), true);
    let second = deltas.events_for_parse(&parsed(None, Some("Hello there")), true);
    let terminal = deltas.events_for_parse(&parsed(None, Some("Hello there")), false);

    assert!(matches!(
        first.as_slice(),
        [GenerationStreamEvent::Delta(delta)] if delta == "Hello"
    ));
    assert!(matches!(
        second.as_slice(),
        [GenerationStreamEvent::Delta(delta)] if delta == " there"
    ));
    assert!(
        terminal.is_empty(),
        "an unchanged terminal parse adds nothing"
    );
    assert_eq!(deltas.finish_reason(FinishReason::Stop), FinishReason::Stop);
}

#[test]
fn reasoning_prefix_frames_reconstruct_on_auto_wire_and_remain_absent_when_hidden() {
    for format in ["auto", "hidden"] {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "test",
            "messages": [{"role": "user", "content": "choose"}],
            "reasoning_format": format
        }))
        .unwrap();
        let options =
            chat_template_options(&request, &EmbeddedOpenAiRequestDefaults::default()).unwrap();
        let mut deltas = ChatStreamDeltas::new(template_exposes_reasoning(&options));
        let mut wire = Vec::new();
        for (reasoning, content) in [
            ("Weigh", None),
            ("Weigh both ", None),
            ("Weigh both options.", None),
            ("Weigh both options.", Some("Ans")),
            ("Weigh both options.", Some("Answer.")),
        ] {
            let parsed = ParsedChatMessage {
                content: content.map(str::to_owned),
                reasoning_content: Some(reasoning.to_owned()),
                tool_calls: None,
            };
            for event in deltas.events_for_parse(&parsed, true) {
                let chunk = generation_event_to_chat_chunk(Ok(event), "test").unwrap();
                wire.push(serde_json::to_value(chunk).unwrap());
            }
        }
        let terminal = ParsedChatMessage {
            content: Some("Answer.".to_owned()),
            reasoning_content: Some("Weigh both options.".to_owned()),
            tool_calls: None,
        };
        assert!(deltas.events_for_parse(&terminal, false).is_empty());
        let reasoning = wire
            .iter()
            .filter_map(|chunk| chunk["choices"][0]["delta"]["reasoning_content"].as_str())
            .collect::<Vec<_>>();
        let content = wire
            .iter()
            .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
            .collect::<String>();
        assert_eq!(content, "Answer.", "format={format}");
        if format == "auto" {
            assert_eq!(reasoning, ["Weigh", " both ", "options."]);
            assert_eq!(reasoning.concat(), "Weigh both options.");
        } else {
            assert!(reasoning.is_empty(), "hidden reasoning leaked: {wire:?}");
            assert_eq!(wire.len(), 2);
        }
        assert_eq!(deltas.finish_reason(FinishReason::Stop), FinishReason::Stop);
    }
}
