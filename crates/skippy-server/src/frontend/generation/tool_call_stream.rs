//! Incremental tool-call deltas for streaming chat completions.
//!
//! The native chat parser re-parses the whole generated prefix on every decoded
//! token and returns the complete tool-call array each time, so a call that is
//! still being generated reappears with a longer `arguments` string on every
//! call. This module converts that sequence of full snapshots into the OpenAI
//! streaming shape: one delta per index carrying `id` and `function.name` first,
//! then `function.arguments` fragments that the client concatenates.

use crate::frontend::generation::incremental_text::suffix_delta;
use serde_json::Value;
use serde_json::json;

/// Streaming state for a single tool call, keyed by its wire `index`.
///
/// The `id` is not stored here at all: `deltas` emits it once, on the parse
/// where an index is not yet present in `streamed`, reading it directly off
/// that call. Once the index is pushed here, later parses take the
/// argument-delta branch below and never touch `id` again — which is what
/// keeps a single stable id on the wire even though `ensure_tool_call_ids`
/// mints a fresh UUID on every parse for any call the model did not id
/// itself. `emitted_name` and `emitted_arguments` are what a client has
/// actually received for this index, so `streamed_matches` can confirm the
/// terminal parse reconstructs both.
struct StreamedToolCall {
    emitted_name: String,
    emitted_arguments: String,
}

/// Tracks which tool-call bytes have already gone out on the wire.
#[derive(Default)]
pub(in crate::frontend) struct ToolCallStreamState {
    streamed: Vec<StreamedToolCall>,
    completed: bool,
}

impl ToolCallStreamState {
    /// Record a parse of the tool-call array and return the deltas that have not
    /// been sent yet, or `None` when nothing new is available.
    ///
    /// `is_partial` mirrors the parser argument: a non-partial parse is the
    /// authoritative terminal one. It is the only parse that can mark the calls
    /// complete, and it does so only if what went out on the wire actually
    /// reconstructs it.
    pub(in crate::frontend) fn record(
        &mut self,
        tool_calls: &[Value],
        is_partial: bool,
    ) -> Option<Value> {
        let deltas = self.deltas(tool_calls);
        if !is_partial {
            self.completed = self.streamed_matches(tool_calls);
        }
        deltas
    }

    /// Whether the terminal parse produced tool calls that the client can
    /// actually reconstruct from the deltas it received.
    ///
    /// A partial parse can be revised downward on malformed output, and once a
    /// revision breaks the append-only contract this state can no longer catch
    /// up (re-sending would double-append for any client that concatenates). In
    /// that case the calls are NOT complete: reporting
    /// `finish_reason: "tool_calls"` would invite the client to execute a
    /// tool call whose arguments it never fully received.
    pub(in crate::frontend) fn completed(&self) -> bool {
        self.completed
    }

    /// Does every streamed index reproduce the terminal parse byte for byte?
    fn streamed_matches(&self, tool_calls: &[Value]) -> bool {
        if tool_calls.len() != self.streamed.len() {
            return false;
        }
        tool_calls
            .iter()
            .zip(&self.streamed)
            .all(|(call, streamed)| {
                tool_call_name(call) == Some(streamed.emitted_name.as_str())
                    && tool_call_arguments(call).unwrap_or_default() == streamed.emitted_arguments
            })
    }

    fn deltas(&mut self, tool_calls: &[Value]) -> Option<Value> {
        let mut deltas = Vec::new();
        for (index, call) in tool_calls.iter().enumerate() {
            let arguments = tool_call_arguments(call);
            if let Some(streamed) = self.streamed.get_mut(index) {
                // `suffix_delta` yields `None` when the new value is not an
                // extension of what was already sent. The parser can revise a
                // partial guess downward on malformed output; re-sending would
                // corrupt clients that concatenate, so hold what we have and let
                // the request's own error surface.
                if let Some(delta) = suffix_delta(arguments, &mut streamed.emitted_arguments) {
                    deltas.push(json!({
                        "index": index,
                        "function": {"arguments": delta},
                    }));
                }
                continue;
            }
            // A new index only becomes streamable once its name is known: the
            // first delta for an index must carry the id and the function name.
            // The parser can surface a call before the name is parsed, so stop
            // here rather than skipping ahead and renumbering later indexes.
            let Some(name) = tool_call_name(call) else {
                break;
            };
            let mut emitted_arguments = String::new();
            let argument_delta = suffix_delta(arguments, &mut emitted_arguments);
            let mut function = serde_json::Map::new();
            function.insert("name".to_string(), Value::String(name.to_string()));
            if let Some(delta) = argument_delta {
                function.insert("arguments".to_string(), Value::String(delta));
            }
            deltas.push(json!({
                "index": index,
                "id": tool_call_id(call).unwrap_or_else(minted_tool_call_id),
                "type": "function",
                "function": Value::Object(function),
            }));
            self.streamed.push(StreamedToolCall {
                emitted_name: name.to_string(),
                emitted_arguments,
            });
        }
        (!deltas.is_empty()).then_some(Value::Array(deltas))
    }
}

fn tool_call_name(call: &Value) -> Option<&str> {
    call.get("function")
        .and_then(|function| function.get("name"))
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
}

fn tool_call_arguments(call: &Value) -> Option<&str> {
    call.get("function")
        .and_then(|function| function.get("arguments"))
        .and_then(Value::as_str)
}

fn tool_call_id(call: &Value) -> Option<String> {
    call.get("id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(ToString::to_string)
}

fn minted_tool_call_id() -> String {
    format!("call_{}", uuid::Uuid::new_v4().simple())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::generation::incremental_text::{RecordedFixture, recorded_fixture};

    /// One partial parse snapshot: `(name, arguments)` for a single call.
    fn call(name: &str, arguments: &str) -> Value {
        json!({"type": "function", "function": {"name": name, "arguments": arguments}})
    }

    /// Collected `(index, id, name, arguments)` from every delta produced by
    /// feeding the given snapshots in order.
    fn drive(snapshots: &[Vec<Value>]) -> Vec<Value> {
        let mut state = ToolCallStreamState::default();
        snapshots
            .iter()
            .filter_map(|snapshot| state.record(snapshot, true))
            .flat_map(|delta| delta.as_array().cloned().unwrap_or_default())
            .collect()
    }

    /// Argument fragments a client would concatenate for one index.
    fn joined_arguments(deltas: &[Value], index: u64) -> String {
        deltas
            .iter()
            .filter(|delta| delta["index"] == index)
            .filter_map(|delta| delta["function"]["arguments"].as_str())
            .collect()
    }

    /// Snapshots recorded from the real native parser. See the fixture header
    /// for the capture method; the file holds only the prefixes at which the
    /// parsed `tool_calls` array changed, plus the terminal parse.
    const RECORDED: &str = include_str!("../../../tests/fixtures/qwen35_partial_tool_calls.txt");

    /// Fetch the recorded snapshots for `fixture` via the reader shared with
    /// `chat_stream_deltas` (`incremental_text::recorded_fixture`).
    fn recorded(fixture: &str) -> RecordedFixture {
        recorded_fixture(RECORDED, fixture)
    }

    fn final_arguments(calls: &[Value], index: usize) -> String {
        calls[index]
            .pointer("/function/arguments")
            .and_then(Value::as_str)
            .expect("final arguments")
            .to_string()
    }

    #[test]
    fn streams_more_than_one_delta_for_a_single_tool_call() {
        let deltas = drive(&recorded("single_call").snapshots);

        assert!(
            deltas.len() > 1,
            "a tool call generated over several tokens must produce several deltas, got {deltas:#?}"
        );
    }

    #[test]
    fn first_delta_for_an_index_carries_id_type_and_name() {
        let deltas = drive(&recorded("single_call").snapshots);
        let first = &deltas[0];

        assert_eq!(first["index"], 0);
        assert_eq!(first["type"], "function");
        assert_eq!(first["function"]["name"], "read_file");
        assert!(
            first["id"].as_str().is_some_and(|id| !id.is_empty()),
            "first delta must carry a tool-call id: {first:#?}"
        );
    }

    #[test]
    fn later_deltas_carry_only_argument_fragments() {
        let deltas = drive(&recorded("single_call").snapshots);

        for delta in &deltas[1..] {
            assert!(delta.get("id").is_none(), "id must not repeat: {delta:#?}");
            assert!(
                delta["function"].get("name").is_none(),
                "name must not repeat: {delta:#?}"
            );
            assert_eq!(delta["index"], 0);
        }
    }

    #[test]
    fn concatenated_argument_fragments_equal_the_final_arguments() {
        let fixture = recorded("single_call");

        let deltas = drive(&fixture.snapshots);

        assert_eq!(
            joined_arguments(&deltas, 0),
            final_arguments(&fixture.final_call, 0),
            "a client concatenating the fragments must reconstruct the non-streaming arguments"
        );
    }

    #[test]
    fn recorded_parallel_calls_reconstruct_both_argument_strings() {
        let fixture = recorded("parallel_calls");

        let deltas = drive(&fixture.snapshots);

        let headers = deltas
            .iter()
            .filter(|delta| delta.get("id").is_some())
            .collect::<Vec<_>>();
        assert_eq!(headers.len(), 2, "one header per call: {deltas:#?}");
        assert_eq!(headers[0]["index"], 0);
        assert_eq!(headers[1]["index"], 1);
        assert_ne!(headers[0]["id"], headers[1]["id"]);
        assert_eq!(
            joined_arguments(&deltas, 0),
            final_arguments(&fixture.final_call, 0)
        );
        assert_eq!(
            joined_arguments(&deltas, 1),
            final_arguments(&fixture.final_call, 1)
        );
    }

    #[test]
    fn repeated_identical_snapshots_emit_nothing_new() {
        let mut state = ToolCallStreamState::default();
        let snapshot = vec![call("read_file", "{\"path\":\"a\"}")];

        assert!(state.record(&snapshot, true).is_some());
        assert!(
            state.record(&snapshot, true).is_none(),
            "an unchanged parse must not re-emit"
        );
    }

    #[test]
    fn parallel_calls_keep_their_own_index_and_id() {
        let deltas = drive(&[
            vec![call("read_file", "{\"path\":\"a")],
            vec![
                call("read_file", "{\"path\":\"a\"}"),
                call("list_dir", "{\"path\":\"b"),
            ],
            vec![
                call("read_file", "{\"path\":\"a\"}"),
                call("list_dir", "{\"path\":\"b\"}"),
            ],
        ]);

        let headers = deltas
            .iter()
            .filter(|delta| delta.get("id").is_some())
            .collect::<Vec<_>>();
        assert_eq!(headers.len(), 2, "one header per call: {deltas:#?}");
        assert_eq!(headers[0]["index"], 0);
        assert_eq!(headers[0]["function"]["name"], "read_file");
        assert_eq!(headers[1]["index"], 1);
        assert_eq!(headers[1]["function"]["name"], "list_dir");
        assert_ne!(
            headers[0]["id"], headers[1]["id"],
            "distinct calls need distinct ids"
        );
        assert_eq!(joined_arguments(&deltas, 0), "{\"path\":\"a\"}");
        assert_eq!(joined_arguments(&deltas, 1), "{\"path\":\"b\"}");
    }

    #[test]
    fn second_call_delta_is_not_labelled_with_the_first_index() {
        let mut state = ToolCallStreamState::default();
        state
            .record(&[call("read_file", "{\"path\":\"a\"}")], true)
            .expect("first call header");

        let deltas = state
            .record(
                &[
                    call("read_file", "{\"path\":\"a\"}"),
                    call("list_dir", "{\"path\":\"b\"}"),
                ],
                true,
            )
            .expect("second call header");

        let deltas = deltas.as_array().unwrap();
        assert_eq!(deltas.len(), 1, "only the new call changed: {deltas:#?}");
        assert_eq!(
            deltas[0]["index"], 1,
            "a delta for the second call must not claim index 0"
        );
    }

    #[test]
    fn an_index_whose_name_is_not_yet_parsed_is_withheld() {
        let mut state = ToolCallStreamState::default();

        assert!(
            state.record(&[call("", "{")], true).is_none(),
            "no delta may go out before the function name is known"
        );

        let deltas = state
            .record(&[call("read_file", "{")], true)
            .expect("header");
        assert_eq!(deltas[0]["function"]["name"], "read_file");
    }

    #[test]
    fn a_later_call_waits_for_an_earlier_unnamed_one() {
        let mut state = ToolCallStreamState::default();

        assert!(
            state
                .record(&[call("", "{"), call("list_dir", "{}")], true)
                .is_none(),
            "emitting the second call first would renumber indexes"
        );
    }

    #[test]
    fn arguments_that_regress_are_not_re_emitted() {
        // Observed on malformed output (a Pythonic `True` for a schema-declared
        // boolean): the parser drops back to a shorter argument string. The
        // request fails at the terminal parse either way; the streaming path must
        // not double-append in the meantime.
        let mut state = ToolCallStreamState::default();
        state
            .record(&[call("list_dir", "{\"recursive\":")], true)
            .expect("header");

        assert!(
            state.record(&[call("list_dir", "{")], true).is_none(),
            "a shorter argument string must not produce a delta"
        );
    }

    #[test]
    fn a_diverged_call_is_not_reported_as_complete() {
        // Once a revision breaks the append-only contract the client can never
        // receive the rest of the arguments, so the terminal parse must not claim
        // the calls are complete — otherwise the client executes a tool call it
        // only partly received.
        let mut state = ToolCallStreamState::default();
        state
            .record(&[call("list_dir", "{\"recursive\":True")], true)
            .expect("header");

        // The terminal parse normalised the Pythonic scalar, so what the client
        // already received is no longer a prefix of the truth.
        assert!(
            state
                .record(&[call("list_dir", "{\"recursive\":true}")], false)
                .is_none(),
            "a diverged call cannot be repaired by sending more bytes"
        );
        assert!(
            !state.completed(),
            "finish_reason must not claim tool_calls for a call the client cannot reconstruct"
        );
    }

    #[test]
    fn a_call_whose_name_diverged_from_the_header_is_not_reported_as_complete() {
        // The header already went out with `read_file`; if the terminal parse
        // revised the name, the client still believes it received `read_file`
        // and a matching `finish_reason: "tool_calls"` would tell it to execute
        // a call under the wrong name.
        let mut state = ToolCallStreamState::default();
        state
            .record(&[call("read_file", "{\"path\":\"a\"}")], true)
            .expect("header");

        assert!(
            state
                .record(&[call("list_dir", "{\"path\":\"a\"}")], false)
                .is_none(),
            "arguments alone matching must not be enough to repair a renamed call"
        );
        assert!(
            !state.completed(),
            "finish_reason must not claim tool_calls when the emitted name diverged"
        );
    }

    #[test]
    fn a_call_that_appears_only_at_the_terminal_parse_is_still_delivered() {
        let mut state = ToolCallStreamState::default();

        let deltas = state
            .record(&[call("read_file", "{\"path\":\"a\"}")], false)
            .expect("terminal delta");

        assert_eq!(deltas[0]["function"]["name"], "read_file");
        assert_eq!(deltas[0]["function"]["arguments"], "{\"path\":\"a\"}");
        assert!(state.completed());
    }

    #[test]
    fn a_call_whose_count_shrank_is_not_reported_as_complete() {
        let mut state = ToolCallStreamState::default();
        state
            .record(
                &[
                    call("read_file", "{\"path\":\"a\"}"),
                    call("list_dir", "{\"path\":\"b\"}"),
                ],
                true,
            )
            .expect("two headers");

        state.record(&[call("read_file", "{\"path\":\"a\"}")], false);

        assert!(
            !state.completed(),
            "the client received a second call the terminal parse does not contain"
        );
    }

    #[test]
    fn a_call_still_growing_at_the_terminal_parse_is_completed_by_the_final_suffix() {
        let mut state = ToolCallStreamState::default();
        state
            .record(&[call("read_file", "{\"path\":\"")], true)
            .expect("header");

        let deltas = state
            .record(&[call("read_file", "{\"path\":\"a\"}")], false)
            .expect("terminal suffix");

        assert_eq!(deltas[0]["function"]["arguments"], "a\"}");
        assert!(state.completed());
    }

    #[test]
    fn an_id_that_changes_between_parses_is_emitted_only_once() {
        // `ensure_tool_call_ids` re-mints an id on every parse, so consecutive
        // partial parses of the same call disagree on `id`. Only the first may
        // reach the wire, or the client cannot correlate the tool result.
        let mut state = ToolCallStreamState::default();
        let first = state
            .record(
                &[json!({
                    "id": "call_first_parse",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{"},
                })],
                true,
            )
            .expect("header");
        let second = state
            .record(
                &[json!({
                    "id": "call_second_parse",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                })],
                true,
            )
            .expect("argument fragment");

        assert_eq!(first[0]["id"], "call_first_parse");
        assert!(
            second[0].get("id").is_none(),
            "a re-minted id must not reach the wire: {second:#?}"
        );
    }

    #[test]
    fn a_parser_supplied_id_is_preserved() {
        let mut state = ToolCallStreamState::default();
        let deltas = state
            .record(
                &[json!({
                    "id": "call_from_model",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                })],
                true,
            )
            .expect("header");

        assert_eq!(deltas[0]["id"], "call_from_model");
    }

    #[test]
    fn streaming_deltas_alone_do_not_mark_the_calls_complete() {
        let fixture = recorded("single_call");
        let mut state = ToolCallStreamState::default();

        for snapshot in &fixture.snapshots {
            state.record(snapshot, true);
        }

        assert!(
            !state.completed(),
            "a generation truncated mid-call must not report finish_reason: tool_calls"
        );
    }

    #[test]
    fn the_terminal_parse_marks_the_calls_complete() {
        let fixture = recorded("single_call");
        let mut state = ToolCallStreamState::default();

        for snapshot in &fixture.snapshots {
            state.record(snapshot, true);
        }
        state.record(&fixture.final_call, false);

        assert!(state.completed());
    }

    #[test]
    fn the_terminal_parse_emits_nothing_when_streaming_already_caught_up() {
        let fixture = recorded("single_call");
        let mut state = ToolCallStreamState::default();
        for snapshot in &fixture.snapshots {
            state.record(snapshot, true);
        }

        assert!(
            state.record(&fixture.final_call, false).is_none(),
            "re-sending the whole call at finish would double-append arguments"
        );
    }
}
