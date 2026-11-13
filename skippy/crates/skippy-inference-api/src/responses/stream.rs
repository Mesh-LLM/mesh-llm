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
use super::tools::{MAX_TOOL_CALLS, MAX_TOOL_NAME_BYTES, ToolOutputViolation};
use super::turn::ResponsesTurn;
use replay::{EventBuffer, ReplayLimit, ReplayLimits};

mod replay;

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
    name: String,
    arguments: String,
    arguments_seen: bool,
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
    quarantined_choices: Vec<Value>,
    quarantined_bytes: usize,
    max_quarantined_event_bytes: usize,
    max_quarantined_events: usize,
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
            quarantined_choices: Vec::new(),
            quarantined_bytes: 0,
            max_quarantined_event_bytes: replay::MAX_REPLAY_EVENT_BYTES,
            max_quarantined_events: replay::MAX_REPLAY_EVENTS,
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
        let mut events = EventBuffer::unbounded(self.sequence_number);
        self.start_into(&mut events)
            .expect("unbounded event buffer accepts response start");
        self.accept(events)
    }

    fn start_into(&mut self, events: &mut EventBuffer) -> Result<(), ReplayLimit> {
        if self.started {
            return Ok(());
        }
        self.started = true;
        let response = self
            .turn
            .response(ResponseStatus::InProgress, &[], None, None);
        events.emit("response.created", json!({"response": response.clone()}))?;
        events.emit("response.in_progress", json!({"response": response}))
    }

    /// Events for one parsed Chat Completions stream chunk.
    pub fn push_chunk(&mut self, chunk: &Value) -> Vec<ResponsesStreamEvent> {
        let mut events = EventBuffer::unbounded(self.sequence_number);
        if self.terminated {
            return events.into_events();
        }
        self.start_into(&mut events)
            .expect("unbounded event buffer accepts response start");
        if let Some(usage) = chunk.get("usage").and_then(ResponsesUsage::from_chat_value) {
            self.usage = Some(usage);
        }
        let Some(choice) = chunk
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
        else {
            return self.accept(events);
        };
        let retained_bytes = retained_chunk_bytes(choice);
        if self.active_output_bytes.saturating_add(retained_bytes) > self.max_output_bytes {
            return self.fail_into(
                events,
                "response_output_too_large",
                "upstream response output exceeded the active-state limit",
            );
        }
        self.active_output_bytes += retained_bytes;
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            self.finish_reason = Some(reason.to_string());
        }
        if !self.quarantined_choices.is_empty() || has_tool_call_fragment(choice) {
            if let Err(violation) = self.quarantine_choice(choice, retained_bytes) {
                return self.fail_into(events, violation.code, violation.message);
            }
            return self.accept(events);
        }
        self.process_choice(&mut events, choice, false)
            .expect("unbounded event buffer accepts choice events");
        self.accept(events)
    }

    fn process_choice(
        &mut self,
        events: &mut EventBuffer,
        choice: &Value,
        replaying_quarantine: bool,
    ) -> Result<(), ReplayLimit> {
        let delta = choice.get("delta");
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
            self.push_reasoning(events, reasoning)?;
        }
        if let Some(text) = text_of("content") {
            let logprobs = choice
                .get("logprobs")
                .filter(|value| !value.is_null())
                .cloned()
                .map(responses_logprobs)
                .and_then(|value| value.as_array().cloned())
                .unwrap_or_default();
            self.push_text(events, text, logprobs, phase)?;
        } else if let Some(phase) = phase {
            self.open_message(events, Some(phase))?;
        }
        if let Some(calls) = delta
            .and_then(|delta| delta.get("tool_calls"))
            .and_then(Value::as_array)
        {
            for call in calls {
                if replaying_quarantine {
                    self.push_call(events, call)?;
                }
            }
        }
        Ok(())
    }

    /// Close every item and emit the terminal event for a stream that ended
    /// normally. A completed response is saved for continuation here, from
    /// the same shared path for every route.
    pub fn finish(&mut self) -> Vec<ResponsesStreamEvent> {
        let mut events = EventBuffer::unbounded(self.sequence_number);
        if self.terminated {
            return events.into_events();
        }
        self.start_into(&mut events)
            .expect("unbounded event buffer accepts response start");
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
        if let Err(violation) = self.validate_quarantined_calls(status) {
            return self.fail_into(events, violation.code, violation.message);
        }
        let mut accepted = self.accept(events);
        let limits = ReplayLimits {
            max_bytes: self.max_quarantined_event_bytes,
            max_events: self.max_quarantined_events,
        };
        let mut suffix = if self.quarantined_choices.is_empty() {
            EventBuffer::unbounded(self.sequence_number)
        } else {
            EventBuffer::bounded(self.sequence_number, limits)
        };
        if self.replay_quarantine(&mut suffix).is_err() {
            accepted.extend(self.fail("response_output_too_large", quarantine_limit().message));
            return accepted;
        }
        if self.close_open(&mut suffix, status).is_err() {
            accepted.extend(self.fail("response_output_too_large", quarantine_limit().message));
            return accepted;
        }
        let call_indexes: Vec<usize> = self
            .calls
            .values()
            .filter_map(|call| call.output_index)
            .collect();
        for output_index in call_indexes {
            if self.close_call(&mut suffix, output_index, status).is_err() {
                accepted.extend(self.fail("response_output_too_large", quarantine_limit().message));
                return accepted;
            }
        }
        if self.output.is_empty() {
            // A response always says something, even if it is empty.
            if self.open_message(&mut suffix, None).is_err()
                || self.close_open(&mut suffix, status).is_err()
            {
                accepted.extend(self.fail("response_output_too_large", quarantine_limit().message));
                return accepted;
            }
        }
        let output = self.output_items(status);
        let staged = match self.turn.stage_finish(status, output, self.usage) {
            Ok(staged) => staged,
            Err(violation) => {
                accepted.extend(self.fail(violation.code, violation.message));
                return accepted;
            }
        };
        let event = match status {
            ResponseStatus::Incomplete(_) => "response.incomplete",
            _ => "response.completed",
        };
        if suffix
            .emit(event, json!({"response": staged.response()}))
            .is_err()
        {
            accepted.extend(self.fail("response_output_too_large", quarantine_limit().message));
            return accepted;
        }
        let stored = staged.commit_storage();
        patch_response_store(suffix.events_mut(), stored);
        self.terminated = true;
        accepted.extend(self.accept(suffix));
        accepted
    }

    fn quarantine_choice(
        &mut self,
        choice: &Value,
        retained_bytes: usize,
    ) -> Result<(), ToolOutputViolation> {
        if self.quarantined_choices.len() >= self.max_quarantined_events
            || self.quarantined_bytes.saturating_add(retained_bytes)
                > self.max_quarantined_event_bytes
        {
            return Err(quarantine_limit());
        }
        if let Some(calls) = choice
            .get("delta")
            .and_then(|delta| delta.get("tool_calls"))
        {
            let calls = calls.as_array().ok_or_else(malformed_stream_call)?;
            let mut chunk_indexes = std::collections::BTreeSet::new();
            for call in calls {
                let index = call
                    .get("index")
                    .and_then(Value::as_u64)
                    .ok_or_else(malformed_stream_call)?;
                if !chunk_indexes.insert(index) {
                    return Err(malformed_stream_call());
                }
                self.accumulate_call(index, call)?;
            }
        }
        self.quarantined_bytes += retained_bytes;
        self.quarantined_choices.push(choice.clone());
        Ok(())
    }

    fn accumulate_call(&mut self, index: u64, call: &Value) -> Result<(), ToolOutputViolation> {
        if !self.calls.contains_key(&index) && self.calls.len() >= MAX_TOOL_CALLS {
            return Err(ToolOutputViolation::new(
                "tool_call_limit_exceeded",
                "upstream response exceeded the tool-call limit",
            ));
        }
        let call = call.as_object().ok_or_else(malformed_stream_call)?;
        if call
            .get("type")
            .is_some_and(|kind| kind.as_str() != Some("function"))
            || call.contains_key("name")
            || call.contains_key("arguments")
        {
            return Err(malformed_stream_call());
        }
        let pending = self.calls.entry(index).or_default();
        if let Some(id) = call.get("id") {
            let id = id
                .as_str()
                .filter(|id| !id.is_empty())
                .ok_or_else(malformed_stream_call)?;
            if pending.call_id.is_some() {
                return Err(malformed_stream_call());
            }
            pending.call_id.get_or_insert_with(|| id.to_string());
        }
        let Some(function) = call.get("function") else {
            return Ok(());
        };
        let function = function.as_object().ok_or_else(malformed_stream_call)?;
        if let Some(name) = function.get("name") {
            let name = name.as_str().ok_or_else(malformed_stream_call)?;
            pending.name.push_str(name);
            if pending.name.len() > MAX_TOOL_NAME_BYTES {
                return Err(malformed_stream_call());
            }
        }
        if let Some(arguments) = function.get("arguments") {
            pending.arguments_seen = true;
            pending
                .arguments
                .push_str(arguments.as_str().ok_or_else(malformed_stream_call)?);
        }
        Ok(())
    }

    fn validate_quarantined_calls(
        &self,
        status: ResponseStatus,
    ) -> Result<(), ToolOutputViolation> {
        self.turn.validate_stream_tool_calls(
            status,
            !self.calls.is_empty(),
            self.calls.values().map(|call| call.name.as_str()),
        )?;
        let mut call_ids = std::collections::BTreeSet::new();
        for call in self.calls.values() {
            if call.call_id.is_none() || call.name.is_empty() || !call.arguments_seen {
                return Err(malformed_stream_call());
            }
            if !call_ids.insert(call.call_id.as_deref().expect("checked call id")) {
                return Err(malformed_stream_call());
            }
        }
        Ok(())
    }

    fn replay_quarantine(&mut self, events: &mut EventBuffer) -> Result<(), ReplayLimit> {
        let choices = std::mem::take(&mut self.quarantined_choices);
        for choice in &choices {
            self.process_choice(events, choice, true)?;
        }
        Ok(())
    }

    /// `error` then `response.failed`, for a stream that broke. Nothing is
    /// saved.
    pub fn fail(&mut self, code: &str, message: &str) -> Vec<ResponsesStreamEvent> {
        let mut events = EventBuffer::unbounded(self.sequence_number);
        if self.terminated {
            return events.into_events();
        }
        self.start_into(&mut events)
            .expect("unbounded event buffer accepts response start");
        self.fail_into(events, code, message)
    }

    fn fail_into(
        &mut self,
        mut events: EventBuffer,
        code: &str,
        message: &str,
    ) -> Vec<ResponsesStreamEvent> {
        events
            .emit(
            "error",
            json!({"error": {"type": "server_error", "code": code, "message": message, "param": null}}),
        )
            .expect("unbounded event buffer accepts error");
        let response = self
            .turn
            .fail(&[], self.usage, ResponseError { code, message });
        events
            .emit("response.failed", json!({"response": response}))
            .expect("unbounded event buffer accepts response.failed");
        self.failed = true;
        self.terminated = true;
        self.accept(events)
    }

    fn accept(&mut self, events: EventBuffer) -> Vec<ResponsesStreamEvent> {
        self.sequence_number = events.next_sequence();
        events.into_events()
    }

    fn push_reasoning(&mut self, events: &mut EventBuffer, delta: &str) -> Result<(), ReplayLimit> {
        let index = match self.open {
            Some(index) if matches!(self.output[index], OutputSlot::Reasoning { .. }) => index,
            _ => {
                self.close_open(events, ResponseStatus::Completed)?;
                let id = new_item_id("rs");
                let index = self.output.len();
                self.output.push(OutputSlot::Reasoning {
                    id: id.clone(),
                    text: String::new(),
                });
                self.open = Some(index);
                let item = json!({"id": id, "type": "reasoning", "summary": [], "content": []});
                events.emit(
                    "response.output_item.added",
                    json!({"output_index": index, "item": item}),
                )?;
                events.emit(
                    "response.content_part.added",
                    json!({"item_id": id, "output_index": index, "content_index": 0, "part": {"type": "reasoning_text", "text": ""}}),
                )?;
                index
            }
        };
        let OutputSlot::Reasoning { id, text } = &mut self.output[index] else {
            return Ok(());
        };
        text.push_str(delta);
        let id = id.clone();
        events.emit(
            "response.reasoning.delta",
            json!({"item_id": id, "output_index": index, "content_index": 0, "delta": delta}),
        )
    }

    fn open_message(
        &mut self,
        events: &mut EventBuffer,
        phase: Option<&str>,
    ) -> Result<usize, ReplayLimit> {
        self.close_open(events, ResponseStatus::Completed)?;
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
        events.emit(
            "response.output_item.added",
            json!({"output_index": index, "item": item}),
        )?;
        events.emit(
            "response.content_part.added",
            json!({"item_id": id, "output_index": index, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}}),
        )?;
        Ok(index)
    }

    fn push_text(
        &mut self,
        events: &mut EventBuffer,
        delta: &str,
        delta_logprobs: Vec<Value>,
        phase: Option<&str>,
    ) -> Result<(), ReplayLimit> {
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
            _ => self.open_message(events, phase)?,
        };
        let OutputSlot::Message {
            id, text, logprobs, ..
        } = &mut self.output[index]
        else {
            return Ok(());
        };
        text.push_str(delta);
        logprobs.extend(delta_logprobs.iter().cloned());
        let id = id.clone();
        events.emit(
            "response.output_text.delta",
            json!({"item_id": id, "output_index": index, "content_index": 0, "delta": delta, "logprobs": delta_logprobs}),
        )
    }

    fn push_call(&mut self, events: &mut EventBuffer, call: &Value) -> Result<(), ReplayLimit> {
        let Some(index) = call.get("index").and_then(Value::as_u64) else {
            return Ok(());
        };
        let fragment = call
            .get("function")
            .and_then(|function| function.get("arguments"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let output_index = self.calls.get(&index).and_then(|call| call.output_index);
        match output_index {
            Some(output_index) => self.push_arguments(events, output_index, &fragment)?,
            None => {
                self.announce_call(events, index)?;
                if let Some(output_index) =
                    self.calls.get(&index).and_then(|call| call.output_index)
                {
                    self.push_arguments(events, output_index, &fragment)?;
                }
            }
        }
        Ok(())
    }

    /// Announce a fully assembled and authorized call at its first fragment.
    fn announce_call(&mut self, events: &mut EventBuffer, index: u64) -> Result<(), ReplayLimit> {
        self.close_open(events, ResponseStatus::Completed)?;
        let output_index = self.output.len();
        let Some(pending) = self.calls.get_mut(&index) else {
            return Ok(());
        };
        pending.output_index = Some(output_index);
        let slot = CallSlot {
            id: new_item_id("fc"),
            call_id: pending.call_id.clone().expect("validated call id"),
            name: pending.name.clone(),
            arguments: String::new(),
        };
        let item = function_call_item(&slot.id, "in_progress", &slot.call_id, &slot.name, "");
        self.output.push(OutputSlot::Call(slot));
        events.emit(
            "response.output_item.added",
            json!({"output_index": output_index, "item": item}),
        )
    }

    fn push_arguments(
        &mut self,
        events: &mut EventBuffer,
        index: usize,
        fragment: &str,
    ) -> Result<(), ReplayLimit> {
        if fragment.is_empty() {
            return Ok(());
        }
        let OutputSlot::Call(slot) = &mut self.output[index] else {
            return Ok(());
        };
        slot.arguments.push_str(fragment);
        let id = slot.id.clone();
        events.emit(
            "response.function_call_arguments.delta",
            json!({"item_id": id, "output_index": index, "delta": fragment}),
        )
    }

    /// Finish the streaming reasoning or message item.
    fn close_open(
        &mut self,
        events: &mut EventBuffer,
        status: ResponseStatus,
    ) -> Result<(), ReplayLimit> {
        let Some(index) = self.open.take() else {
            return Ok(());
        };
        let item = self.item(index, status);
        match &self.output[index] {
            OutputSlot::Reasoning { id, text } => {
                let (id, text) = (id.clone(), text.clone());
                events.emit(
                    "response.reasoning.done",
                    json!({"item_id": id, "output_index": index, "content_index": 0, "text": text}),
                )?;
                events.emit(
                    "response.content_part.done",
                    json!({"item_id": id, "output_index": index, "content_index": 0, "part": {"type": "reasoning_text", "text": text}}),
                )?;
            }
            OutputSlot::Message {
                id, text, logprobs, ..
            } => {
                let (id, text, logprobs) = (id.clone(), text.clone(), logprobs.clone());
                events.emit(
                    "response.output_text.done",
                    json!({"item_id": id, "output_index": index, "content_index": 0, "text": text, "logprobs": logprobs}),
                )?;
                events.emit(
                    "response.content_part.done",
                    json!({"item_id": id, "output_index": index, "content_index": 0, "part": {"type": "output_text", "text": text, "annotations": []}}),
                )?;
            }
            OutputSlot::Call(_) => {}
        }
        events.emit(
            "response.output_item.done",
            json!({"output_index": index, "item": item}),
        )
    }

    fn close_call(
        &mut self,
        events: &mut EventBuffer,
        index: usize,
        status: ResponseStatus,
    ) -> Result<(), ReplayLimit> {
        let OutputSlot::Call(slot) = &self.output[index] else {
            return Ok(());
        };
        let (id, arguments) = (slot.id.clone(), slot.arguments.clone());
        let item = self.item(index, status);
        events.emit(
            "response.function_call_arguments.done",
            json!({"item_id": id, "output_index": index, "arguments": arguments}),
        )?;
        events.emit(
            "response.output_item.done",
            json!({"output_index": index, "item": item}),
        )
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

fn has_tool_call_fragment(choice: &Value) -> bool {
    choice
        .get("delta")
        .and_then(|delta| delta.get("tool_calls"))
        .is_some_and(|calls| {
            !calls.is_null() && calls.as_array().is_none_or(|calls| !calls.is_empty())
        })
}

fn malformed_stream_call() -> ToolOutputViolation {
    ToolOutputViolation::new(
        "upstream_tool_call_invalid",
        "upstream stream contained a malformed function call",
    )
}

fn quarantine_limit() -> ToolOutputViolation {
    ToolOutputViolation::new(
        "response_output_too_large",
        "upstream response exceeded the tool-call quarantine limit",
    )
}

/// Replay a complete response as the events that would have streamed it,
/// for answers that are only available whole (such as a virtual model's).
pub fn stream_events_for_response(response: &Value) -> Vec<ResponsesStreamEvent> {
    replay::stream_events_for_response(response)
}

/// Translate and replay a complete Chat Completions body for `turn`. The turn
/// commits only after the full replay has passed incremental admission.
pub fn stream_events_for_chat_completion(
    turn: &ResponsesTurn,
    chat: &Value,
) -> Vec<ResponsesStreamEvent> {
    replay::stream_events_for_chat_completion(turn, chat)
}

fn patch_response_store(events: &mut [ResponsesStreamEvent], stored: bool) {
    for event in events {
        if let Some(response) = event.data.get_mut("response") {
            response["store"] = Value::Bool(stored);
        }
    }
}

#[cfg(test)]
mod tests;
