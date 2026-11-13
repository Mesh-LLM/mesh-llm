//! The Open Responses streaming state machine, shared by the direct Skippy
//! handler and the Mesh relay so both emit the same events.
//!
//! It turns Chat Completions stream chunks into semantic events with one
//! monotonic `sequence_number`: `response.created`, `response.in_progress`,
//! then for each output item `output_item.added` → `content_part.added` →
//! deltas → `<kind>.done` → `content_part.done` → `output_item.done`, and one
//! terminal `response.completed`, `response.incomplete` or `response.failed`.
//! Every event's `type` is also its SSE `event` name.

use std::{collections::BTreeMap, sync::Arc};

use serde_json::{Value, json};

use super::resource::{
    ResponseError, ResponseStatus, ResponsesUsage, function_call_item, message_item, new_item_id,
    reasoning_item, responses_logprobs,
};
use super::turn::ResponsesTurn;

/// Maximum reconstructed text, reasoning, tool-call and logprob state held by
/// one active response before it fails closed.
const MAX_ACTIVE_OUTPUT_BYTES: usize = 8 * 1024 * 1024;

/// One server-sent event: the SSE `event` name and its JSON data, whose
/// `type` is the same name.
#[derive(Debug, Clone, PartialEq)]
pub struct ResponsesStreamEvent {
    pub event: &'static str,
    pub data: Value,
}

#[derive(Debug)]
enum OutputSlot {
    Reasoning {
        id: String,
        text: String,
    },
    Message {
        id: String,
        text: String,
        logprobs: Vec<Value>,
        phase: Option<String>,
    },
    Call(CallSlot),
}

#[derive(Debug, Default)]
struct CallSlot {
    id: String,
    call_id: String,
    name: String,
    arguments: String,
}

/// A tool call the model has started but not yet named. It takes an output
/// index only once it is announced.
#[derive(Debug, Default)]
struct PendingCall {
    call_id: Option<String>,
    name: Option<String>,
    arguments: String,
    output_index: Option<usize>,
}

/// Translates one Chat Completions stream into Responses events.
#[derive(Debug)]
pub struct ResponsesStream {
    turn: Arc<ResponsesTurn>,
    sequence_number: u64,
    output: Vec<OutputSlot>,
    /// The output index of the reasoning or message item still streaming.
    open: Option<usize>,
    calls: BTreeMap<u64, PendingCall>,
    usage: Option<ResponsesUsage>,
    finish_reason: Option<String>,
    active_output_bytes: usize,
    max_output_bytes: usize,
    started: bool,
    terminated: bool,
    failed: bool,
}

impl ResponsesStream {
    pub fn new(turn: Arc<ResponsesTurn>) -> Self {
        Self {
            turn,
            sequence_number: 0,
            output: Vec::new(),
            open: None,
            calls: BTreeMap::new(),
            usage: None,
            finish_reason: None,
            active_output_bytes: 0,
            max_output_bytes: MAX_ACTIVE_OUTPUT_BYTES,
            started: false,
            terminated: false,
            failed: false,
        }
    }

    pub fn turn(&self) -> &ResponsesTurn {
        &self.turn
    }

    /// Whether a terminal event has been emitted.
    pub fn is_terminated(&self) -> bool {
        self.terminated
    }

    /// Whether the terminal event represents a failed response.
    pub fn is_failed(&self) -> bool {
        self.failed
    }

    /// `response.created` and `response.in_progress`. Idempotent.
    pub fn start(&mut self) -> Vec<ResponsesStreamEvent> {
        let mut events = Vec::new();
        self.start_into(&mut events);
        events
    }

    fn start_into(&mut self, events: &mut Vec<ResponsesStreamEvent>) {
        if self.started {
            return;
        }
        self.started = true;
        let response = self
            .turn
            .response(ResponseStatus::InProgress, &[], None, None);
        self.emit(
            events,
            "response.created",
            json!({"response": response.clone()}),
        );
        self.emit(
            events,
            "response.in_progress",
            json!({"response": response}),
        );
    }

    /// Events for one parsed Chat Completions stream chunk.
    pub fn push_chunk(&mut self, chunk: &Value) -> Vec<ResponsesStreamEvent> {
        let mut events = Vec::new();
        if self.terminated {
            return events;
        }
        self.start_into(&mut events);
        if let Some(usage) = chunk.get("usage").and_then(ResponsesUsage::from_chat_value) {
            self.usage = Some(usage);
        }
        let Some(choice) = chunk
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
        else {
            return events;
        };
        let delta = choice.get("delta");
        let retained_bytes = retained_chunk_bytes(choice);
        if self.active_output_bytes.saturating_add(retained_bytes) > self.max_output_bytes {
            return self.fail_into(
                events,
                "response_output_too_large",
                "upstream response output exceeded the active-state limit",
            );
        }
        self.active_output_bytes += retained_bytes;
        let phase = delta
            .and_then(|delta| delta.get("phase"))
            .and_then(Value::as_str)
            .filter(|phase| matches!(*phase, "commentary" | "final_answer"));
        let text_of = |key: &str| {
            delta
                .and_then(|delta| delta.get(key))
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
        };
        if let Some(reasoning) = text_of("reasoning_content") {
            self.push_reasoning(&mut events, reasoning);
        }
        if let Some(text) = text_of("content") {
            let logprobs = choice
                .get("logprobs")
                .filter(|value| !value.is_null())
                .cloned()
                .map(responses_logprobs)
                .and_then(|value| value.as_array().cloned())
                .unwrap_or_default();
            self.push_text(&mut events, text, logprobs, phase);
        } else if let Some(phase) = phase {
            self.open_message(&mut events, Some(phase));
        }
        if let Some(calls) = delta
            .and_then(|delta| delta.get("tool_calls"))
            .and_then(Value::as_array)
        {
            for call in calls {
                self.push_call(&mut events, call);
            }
        }
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            self.finish_reason = Some(reason.to_string());
        }
        events
    }

    /// Close every item and emit the terminal event for a stream that ended
    /// normally. A completed response is saved for continuation here, from
    /// the same shared path for every route.
    pub fn finish(&mut self) -> Vec<ResponsesStreamEvent> {
        let mut events = Vec::new();
        if self.terminated {
            return events;
        }
        self.start_into(&mut events);
        if self.finish_reason.is_none() {
            return self.fail_into(
                events,
                "upstream_stream_incomplete",
                "upstream stream ended before a finish reason",
            );
        }
        if self.finish_reason.as_deref() == Some("error") {
            return self.fail_into(
                events,
                "upstream_response_failed",
                "upstream stream ended with an error finish reason",
            );
        }
        let status = ResponseStatus::from_finish_reason(self.finish_reason.as_deref());
        self.close_open(&mut events, status);
        // A call the model never named cannot be announced complete.
        let unnamed: Vec<u64> = self
            .calls
            .iter()
            .filter(|(_, call)| call.output_index.is_none())
            .map(|(index, _)| *index)
            .collect();
        for index in unnamed {
            if let Some(call) = self.calls.get_mut(&index) {
                call.name.get_or_insert_with(|| "tool".to_string());
            }
            self.announce_call(&mut events, index);
        }
        let call_indexes: Vec<usize> = self
            .calls
            .values()
            .filter_map(|call| call.output_index)
            .collect();
        for output_index in call_indexes {
            self.close_call(&mut events, output_index, status);
        }
        if self.output.is_empty() {
            // A response always says something, even if it is empty.
            self.open_message(&mut events, None);
            self.close_open(&mut events, status);
        }
        let output = self.output_items(status);
        let response = self.turn.finish(status, &output, self.usage);
        let event = match status {
            ResponseStatus::Incomplete(_) => "response.incomplete",
            _ => "response.completed",
        };
        self.emit(&mut events, event, json!({"response": response}));
        self.terminated = true;
        events
    }

    /// `error` then `response.failed`, for a stream that broke. Nothing is
    /// saved.
    pub fn fail(&mut self, code: &str, message: &str) -> Vec<ResponsesStreamEvent> {
        let mut events = Vec::new();
        if self.terminated {
            return events;
        }
        self.start_into(&mut events);
        self.fail_into(events, code, message)
    }

    fn fail_into(
        &mut self,
        mut events: Vec<ResponsesStreamEvent>,
        code: &str,
        message: &str,
    ) -> Vec<ResponsesStreamEvent> {
        self.emit(
            &mut events,
            "error",
            json!({"error": {"type": "server_error", "code": code, "message": message, "param": null}}),
        );
        let response = self
            .turn
            .fail(&[], self.usage, ResponseError { code, message });
        self.emit(
            &mut events,
            "response.failed",
            json!({"response": response}),
        );
        self.failed = true;
        self.terminated = true;
        events
    }

    fn emit(&mut self, events: &mut Vec<ResponsesStreamEvent>, event: &'static str, body: Value) {
        let mut data = body;
        data["type"] = Value::String(event.to_string());
        data["sequence_number"] = Value::from(self.sequence_number);
        self.sequence_number += 1;
        events.push(ResponsesStreamEvent { event, data });
    }

    fn push_reasoning(&mut self, events: &mut Vec<ResponsesStreamEvent>, delta: &str) {
        let index = match self.open {
            Some(index) if matches!(self.output[index], OutputSlot::Reasoning { .. }) => index,
            _ => {
                self.close_open(events, ResponseStatus::Completed);
                let id = new_item_id("rs");
                let index = self.output.len();
                self.output.push(OutputSlot::Reasoning {
                    id: id.clone(),
                    text: String::new(),
                });
                self.open = Some(index);
                let item = json!({"id": id, "type": "reasoning", "summary": [], "content": []});
                self.emit(
                    events,
                    "response.output_item.added",
                    json!({"output_index": index, "item": item}),
                );
                self.emit(
                    events,
                    "response.content_part.added",
                    json!({"item_id": id, "output_index": index, "content_index": 0, "part": {"type": "reasoning_text", "text": ""}}),
                );
                index
            }
        };
        let OutputSlot::Reasoning { id, text } = &mut self.output[index] else {
            return;
        };
        text.push_str(delta);
        let id = id.clone();
        self.emit(
            events,
            "response.reasoning.delta",
            json!({"item_id": id, "output_index": index, "content_index": 0, "delta": delta}),
        );
    }

    fn open_message(
        &mut self,
        events: &mut Vec<ResponsesStreamEvent>,
        phase: Option<&str>,
    ) -> usize {
        self.close_open(events, ResponseStatus::Completed);
        let id = new_item_id("msg");
        let index = self.output.len();
        self.output.push(OutputSlot::Message {
            id: id.clone(),
            text: String::new(),
            logprobs: Vec::new(),
            phase: phase.map(ToString::to_string),
        });
        self.open = Some(index);
        let mut item = json!({"id": id, "type": "message", "status": "in_progress", "role": "assistant", "content": []});
        if let Some(phase) = phase {
            item["phase"] = Value::String(phase.to_string());
        }
        self.emit(
            events,
            "response.output_item.added",
            json!({"output_index": index, "item": item}),
        );
        self.emit(
            events,
            "response.content_part.added",
            json!({"item_id": id, "output_index": index, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}}),
        );
        index
    }

    fn push_text(
        &mut self,
        events: &mut Vec<ResponsesStreamEvent>,
        delta: &str,
        delta_logprobs: Vec<Value>,
        phase: Option<&str>,
    ) {
        let index = match self.open {
            Some(index)
                if matches!(
                    &self.output[index],
                    OutputSlot::Message { phase: current, .. }
                        if phase.is_none() || current.as_deref() == phase
                ) =>
            {
                index
            }
            _ => self.open_message(events, phase),
        };
        let OutputSlot::Message {
            id, text, logprobs, ..
        } = &mut self.output[index]
        else {
            return;
        };
        text.push_str(delta);
        logprobs.extend(delta_logprobs.iter().cloned());
        let id = id.clone();
        self.emit(
            events,
            "response.output_text.delta",
            json!({"item_id": id, "output_index": index, "content_index": 0, "delta": delta, "logprobs": delta_logprobs}),
        );
    }

    fn push_call(&mut self, events: &mut Vec<ResponsesStreamEvent>, call: &Value) {
        let index = call.get("index").and_then(Value::as_u64).unwrap_or(0);
        let function = call.get("function");
        let pending = self.calls.entry(index).or_default();
        if let Some(id) = call
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
        {
            pending.call_id.get_or_insert_with(|| id.to_string());
        }
        if let Some(name) = function
            .and_then(|function| function.get("name"))
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
        {
            pending.name.get_or_insert_with(|| name.to_string());
        }
        let fragment = function
            .and_then(|function| function.get("arguments"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        match pending.output_index {
            Some(output_index) => self.push_arguments(events, output_index, &fragment),
            None => {
                pending.arguments.push_str(&fragment);
                if pending.name.is_some() {
                    self.announce_call(events, index);
                }
            }
        }
    }

    /// Give a named call its output item and flush the arguments buffered
    /// before its name arrived.
    fn announce_call(&mut self, events: &mut Vec<ResponsesStreamEvent>, index: u64) {
        self.close_open(events, ResponseStatus::Completed);
        let output_index = self.output.len();
        let Some(pending) = self.calls.get_mut(&index) else {
            return;
        };
        pending.output_index = Some(output_index);
        let slot = CallSlot {
            id: new_item_id("fc"),
            call_id: pending
                .call_id
                .clone()
                .unwrap_or_else(|| new_item_id("call")),
            name: pending.name.clone().unwrap_or_default(),
            arguments: String::new(),
        };
        let buffered = std::mem::take(&mut pending.arguments);
        let item = function_call_item(&slot.id, "in_progress", &slot.call_id, &slot.name, "");
        self.output.push(OutputSlot::Call(slot));
        self.emit(
            events,
            "response.output_item.added",
            json!({"output_index": output_index, "item": item}),
        );
        self.push_arguments(events, output_index, &buffered);
    }

    fn push_arguments(
        &mut self,
        events: &mut Vec<ResponsesStreamEvent>,
        index: usize,
        fragment: &str,
    ) {
        if fragment.is_empty() {
            return;
        }
        let OutputSlot::Call(slot) = &mut self.output[index] else {
            return;
        };
        slot.arguments.push_str(fragment);
        let id = slot.id.clone();
        self.emit(
            events,
            "response.function_call_arguments.delta",
            json!({"item_id": id, "output_index": index, "delta": fragment}),
        );
    }

    /// Finish the streaming reasoning or message item.
    fn close_open(&mut self, events: &mut Vec<ResponsesStreamEvent>, status: ResponseStatus) {
        let Some(index) = self.open.take() else {
            return;
        };
        let item = self.item(index, status);
        match &self.output[index] {
            OutputSlot::Reasoning { id, text } => {
                let (id, text) = (id.clone(), text.clone());
                self.emit(
                    events,
                    "response.reasoning.done",
                    json!({"item_id": id, "output_index": index, "content_index": 0, "text": text}),
                );
                self.emit(
                    events,
                    "response.content_part.done",
                    json!({"item_id": id, "output_index": index, "content_index": 0, "part": {"type": "reasoning_text", "text": text}}),
                );
            }
            OutputSlot::Message {
                id, text, logprobs, ..
            } => {
                let (id, text, logprobs) = (id.clone(), text.clone(), logprobs.clone());
                self.emit(
                    events,
                    "response.output_text.done",
                    json!({"item_id": id, "output_index": index, "content_index": 0, "text": text, "logprobs": logprobs}),
                );
                self.emit(
                    events,
                    "response.content_part.done",
                    json!({"item_id": id, "output_index": index, "content_index": 0, "part": {"type": "output_text", "text": text, "annotations": []}}),
                );
            }
            OutputSlot::Call(_) => {}
        }
        self.emit(
            events,
            "response.output_item.done",
            json!({"output_index": index, "item": item}),
        );
    }

    fn close_call(
        &mut self,
        events: &mut Vec<ResponsesStreamEvent>,
        index: usize,
        status: ResponseStatus,
    ) {
        let OutputSlot::Call(slot) = &self.output[index] else {
            return;
        };
        let (id, arguments) = (slot.id.clone(), slot.arguments.clone());
        let item = self.item(index, status);
        self.emit(
            events,
            "response.function_call_arguments.done",
            json!({"item_id": id, "output_index": index, "arguments": arguments}),
        );
        self.emit(
            events,
            "response.output_item.done",
            json!({"output_index": index, "item": item}),
        );
    }

    /// The finished form of one output item. Only the items that were still
    /// being generated when the response ended inherit its incomplete status.
    fn item(&self, index: usize, status: ResponseStatus) -> Value {
        let last = index + 1 == self.output.len();
        let item_status = if last {
            status.item_status()
        } else {
            "completed"
        };
        match &self.output[index] {
            OutputSlot::Reasoning { id, text } => reasoning_item(id, text),
            OutputSlot::Message {
                id,
                text,
                logprobs,
                phase,
            } => message_item(
                id,
                item_status,
                text,
                (!logprobs.is_empty()).then(|| Value::Array(logprobs.clone())),
                phase.as_deref(),
            ),
            OutputSlot::Call(slot) => function_call_item(
                &slot.id,
                item_status,
                &slot.call_id,
                &slot.name,
                &slot.arguments,
            ),
        }
    }

    fn output_items(&self, status: ResponseStatus) -> Vec<Value> {
        (0..self.output.len())
            .map(|index| self.item(index, status))
            .collect()
    }
}

fn retained_chunk_bytes(choice: &Value) -> usize {
    // Charge the whole choice, not just strings copied into output slots. A
    // phase-only stream can allocate a new slot without retaining text, and
    // every such allocation must still consume the active-state budget.
    serde_json::to_vec(choice).map_or(usize::MAX, |bytes| bytes.len())
}

/// Replay a complete response as the events that would have streamed it,
/// for answers that are only available whole (such as a virtual model's).
pub fn stream_events_for_response(response: &Value) -> Vec<ResponsesStreamEvent> {
    let mut sequence_number = 0u64;
    let mut events = Vec::new();
    let mut emit = |event: &'static str, body: Value| {
        let mut data = body;
        data["type"] = Value::String(event.to_string());
        data["sequence_number"] = Value::from(sequence_number);
        sequence_number += 1;
        events.push(ResponsesStreamEvent { event, data });
    };
    let mut in_progress = response.clone();
    in_progress["status"] = Value::String("in_progress".into());
    in_progress["output"] = Value::Array(Vec::new());
    in_progress["completed_at"] = Value::Null;
    in_progress["usage"] = Value::Null;
    in_progress["output_text"] = Value::String(String::new());
    emit("response.created", json!({"response": in_progress.clone()}));
    emit("response.in_progress", json!({"response": in_progress}));
    let output = response
        .get("output")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    for (index, item) in output.iter().enumerate() {
        replay_item(&mut emit, index, item);
    }
    let terminal = match response.get("status").and_then(Value::as_str) {
        Some("incomplete") => "response.incomplete",
        Some("failed") => "response.failed",
        _ => "response.completed",
    };
    emit(terminal, json!({"response": response}));
    events
}

fn replay_item(emit: &mut impl FnMut(&'static str, Value), index: usize, item: &Value) {
    let id = item.get("id").cloned().unwrap_or(Value::Null);
    let kind = item.get("type").and_then(Value::as_str);
    let mut added = item.clone();
    match kind {
        Some("message") | Some("reasoning") => {
            added["content"] = Value::Array(Vec::new());
            if kind == Some("message") {
                added["status"] = Value::String("in_progress".into());
            }
        }
        Some("function_call") => {
            added["status"] = Value::String("in_progress".into());
            added["arguments"] = Value::String(String::new());
        }
        _ => {}
    }
    emit(
        "response.output_item.added",
        json!({"output_index": index, "item": added}),
    );
    let parts = item
        .get("content")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    for (content_index, part) in parts.iter().enumerate() {
        let text = part.get("text").and_then(Value::as_str).unwrap_or_default();
        let (delta, done) = match part.get("type").and_then(Value::as_str) {
            Some("reasoning_text") => ("response.reasoning.delta", "response.reasoning.done"),
            Some("output_text") => ("response.output_text.delta", "response.output_text.done"),
            _ => continue,
        };
        let mut empty = part.clone();
        empty["text"] = Value::String(String::new());
        let at = json!({"item_id": id, "output_index": index, "content_index": content_index});
        let with = |extra: Value| {
            let mut body = at.clone();
            for (key, value) in extra.as_object().into_iter().flatten() {
                body[key] = value.clone();
            }
            body
        };
        emit("response.content_part.added", with(json!({"part": empty})));
        if !text.is_empty() {
            emit(delta, with(json!({"delta": text})));
        }
        emit(done, with(json!({"text": text})));
        emit("response.content_part.done", with(json!({"part": part})));
    }
    if kind == Some("function_call") {
        let arguments = item.get("arguments").cloned().unwrap_or(Value::from(""));
        let at = json!({"item_id": id, "output_index": index});
        if arguments
            .as_str()
            .is_some_and(|arguments| !arguments.is_empty())
        {
            let mut body = at.clone();
            body["delta"] = arguments.clone();
            emit("response.function_call_arguments.delta", body);
        }
        let mut body = at;
        body["arguments"] = arguments;
        emit("response.function_call_arguments.done", body);
    }
    emit(
        "response.output_item.done",
        json!({"output_index": index, "item": item}),
    );
}

#[cfg(test)]
mod tests;
