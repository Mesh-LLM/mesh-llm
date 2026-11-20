//! OpenAI request phases as spans. A phase that will be exported can be
//! entered so the work inside it, including native calls, nests beneath it;
//! one that will not be exported is never entered, so nothing points at it.

use std::collections::BTreeMap;

use serde_json::{Value, json};

use crate::frontend::generation::{PhaseTimer, StageOpenAiBackend};
use crate::telemetry::now_unix_nanos;
use crate::telemetry::trace_context::{Entered, TaskEntered};

impl StageOpenAiBackend {
    /// Nest synchronous work under a phase emitted by [`Self::emit_openai_phase`].
    pub(in crate::frontend) fn enter_openai_phase(&self, timer: &PhaseTimer) -> Option<Entered> {
        self.telemetry
            .is_debug_enabled()
            .then(|| timer.frame.enter())
    }

    /// Nest async work under a phase emitted by [`Self::emit_openai_phase`].
    pub(in crate::frontend) fn enter_openai_phase_task(
        &self,
        timer: &PhaseTimer,
    ) -> Option<TaskEntered> {
        self.telemetry
            .is_debug_enabled()
            .then(|| timer.frame.enter_task())
    }

    /// Nest synchronous work under a phase emitted by [`Self::emit_openai_summary`].
    pub(in crate::frontend) fn enter_openai_summary(&self, timer: &PhaseTimer) -> Option<Entered> {
        self.telemetry.is_enabled().then(|| timer.frame.enter())
    }

    /// Nest async work under a phase emitted by [`Self::emit_openai_summary`].
    pub(in crate::frontend) fn enter_openai_summary_task(
        &self,
        timer: &PhaseTimer,
    ) -> Option<TaskEntered> {
        self.telemetry
            .is_enabled()
            .then(|| timer.frame.enter_task())
    }

    pub(in crate::frontend) fn emit_openai_phase(
        &self,
        name: &str,
        timer: PhaseTimer,
        mut attrs: BTreeMap<String, Value>,
    ) -> f64 {
        let elapsed_ms = timer.elapsed_ms();
        attrs.insert("llama_stage.elapsed_ms".to_string(), json!(elapsed_ms));
        let end = now_unix_nanos() as u64;
        self.telemetry.emit_debug_frame_span(
            &timer.frame,
            name,
            attrs,
            timer.start_unix_nanos,
            end,
        );
        elapsed_ms
    }

    pub(in crate::frontend) fn emit_openai_summary(
        &self,
        name: &str,
        timer: PhaseTimer,
        mut attrs: BTreeMap<String, Value>,
    ) -> f64 {
        let elapsed_ms = timer.elapsed_ms();
        attrs.insert("llama_stage.elapsed_ms".to_string(), json!(elapsed_ms));
        let end = now_unix_nanos() as u64;
        self.telemetry.emit_frame_span(
            &timer.frame,
            name,
            attrs,
            timer.start_unix_nanos,
            end,
            Vec::new(),
        );
        elapsed_ms
    }
}
