//! A span owned by the section of code it measures. Child spans and events
//! attach to it while it is entered; it is exported when it ends or drops.

use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

use serde_json::Value;

use super::{
    Telemetry, TelemetryLevel, now_unix_nanos,
    trace_context::{self, Entered, Frame, SpanContext},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SpanLevel {
    Summary,
    Debug,
}

impl SpanLevel {
    fn recorded_at(self, level: TelemetryLevel) -> bool {
        match self {
            Self::Summary => level != TelemetryLevel::Off,
            Self::Debug => level == TelemetryLevel::Debug,
        }
    }
}

#[must_use = "a span is exported when it ends; dropping it immediately records nothing useful"]
pub(crate) struct ActiveSpan {
    /// `None` when not recorded: the frame then passes the parent through, so
    /// children never point at a span that is not exported.
    telemetry: Option<Telemetry>,
    frame: Frame,
    name: &'static str,
    start_unix_nanos: u64,
    started: Instant,
    attributes: BTreeMap<String, Value>,
    links: Vec<SpanContext>,
}

impl ActiveSpan {
    /// Start a span under whatever is active here.
    pub(crate) fn start(
        telemetry: Option<&Telemetry>,
        name: &'static str,
        level: SpanLevel,
    ) -> Self {
        Self::start_under(
            &trace_context::current_frame_or_root(),
            telemetry,
            name,
            level,
        )
    }

    /// Start a span directly under the process root, for a component that
    /// outlives the section that creates it, such as a worker thread.
    pub(crate) fn start_under_process(
        telemetry: Option<&Telemetry>,
        name: &'static str,
        level: SpanLevel,
    ) -> Self {
        Self::start_under(trace_context::process().root(), telemetry, name, level)
    }

    fn start_under(
        parent: &Frame,
        telemetry: Option<&Telemetry>,
        name: &'static str,
        level: SpanLevel,
    ) -> Self {
        let telemetry = telemetry
            .filter(|telemetry| level.recorded_at(telemetry.level))
            .cloned();
        let frame = if telemetry.is_some() {
            parent.new_child()
        } else {
            parent.clone()
        };
        Self {
            telemetry,
            frame,
            name,
            start_unix_nanos: u64::try_from(now_unix_nanos()).unwrap_or_default(),
            started: Instant::now(),
            attributes: BTreeMap::new(),
            links: Vec::new(),
        }
    }

    /// Make this the active span on this thread (synchronous code).
    pub(crate) fn enter(&self) -> Entered {
        self.frame.enter()
    }

    pub(crate) fn frame(&self) -> &Frame {
        &self.frame
    }

    pub(crate) fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    pub(crate) fn extend_attributes(
        &mut self,
        attributes: impl IntoIterator<Item = (String, Value)>,
    ) {
        self.attributes.extend(attributes);
    }

    /// Link to a span this one served, such as a request in a batch.
    pub(crate) fn link(&mut self, context: SpanContext) {
        if context != self.frame.context() && !self.links.contains(&context) {
            self.links.push(context);
        }
    }

    pub(crate) fn end(mut self) {
        self.finish();
    }

    /// Drop without exporting, for a section that turned out to do no work.
    pub(crate) fn discard(mut self) {
        self.telemetry = None;
    }

    fn finish(&mut self) {
        let Some(telemetry) = self.telemetry.take() else {
            return;
        };
        let elapsed = u64::try_from(self.started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        telemetry.emit_frame_span(
            &self.frame,
            self.name,
            std::mem::take(&mut self.attributes),
            self.start_unix_nanos,
            self.start_unix_nanos.saturating_add(elapsed),
            std::mem::take(&mut self.links),
        );
    }
}

impl Drop for ActiveSpan {
    fn drop(&mut self) {
        self.finish();
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::mpsc as std_mpsc, thread};

    use super::*;
    use crate::telemetry::{CapturedEvent, capture_stage_config as config};

    /// Exported spans by name, as (context, parent span id).
    fn spans(
        rx: &std_mpsc::Receiver<CapturedEvent>,
    ) -> HashMap<String, (SpanContext, Option<[u8; 8]>)> {
        rx.try_iter()
            .filter_map(|event| event.span.map(|span| (event.event, span)))
            .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn request_work_nests_under_its_request_across_threads() {
        let (telemetry, rx) = Telemetry::captured(config(), TelemetryLevel::Debug);
        let root = trace_context::process().root().context();
        let request = ActiveSpan::start(
            Some(&telemetry),
            "stage.openai_http_request",
            SpanLevel::Summary,
        );
        let handler_telemetry = telemetry.clone();
        trace_context::in_task(request.frame().clone(), async move {
            let telemetry = handler_telemetry;
            let phase = ActiveSpan::start(
                Some(&telemetry),
                "stage.openai_generation",
                SpanLevel::Debug,
            );
            let (blocking, origin) = {
                let _phase = phase.enter();
                let native = telemetry.clone();
                let blocking = tokio::task::spawn_blocking(trace_context::propagate(move || {
                    native.emit_debug_span("native.skippy_prefill", BTreeMap::new(), 1, 2);
                }));
                (blocking, trace_context::current_frame())
            };
            blocking.await.expect("blocking work");
            let scheduler = telemetry.clone();
            thread::spawn(move || {
                let _origin = origin.as_ref().map(Frame::enter);
                scheduler.emit_debug_span("native.skippy_decode", BTreeMap::new(), 3, 4);
            })
            .join()
            .expect("scheduler thread");
            phase.end();
        })
        .await;
        request.end();

        let spans = spans(&rx);
        let (request, request_parent) = spans["stage.openai_http_request"];
        let (phase, phase_parent) = spans["stage.openai_generation"];
        assert_eq!(request_parent, Some(root.span_id));
        assert_eq!(phase_parent, Some(request.span_id));
        assert_eq!(spans["native.skippy_prefill"].1, Some(phase.span_id));
        assert_eq!(spans["native.skippy_decode"].1, Some(phase.span_id));
        assert!(
            spans
                .values()
                .all(|(context, _)| context.trace_id == root.trace_id)
        );
    }

    #[test]
    fn an_unrecorded_span_passes_its_parent_through() {
        let (telemetry, rx) = Telemetry::captured(config(), TelemetryLevel::Summary);
        let outer = ActiveSpan::start(Some(&telemetry), "stage.outer", SpanLevel::Summary);
        let entered = outer.enter();
        let skipped = ActiveSpan::start(Some(&telemetry), "stage.debug_only", SpanLevel::Debug);
        let inner = {
            let _skipped = skipped.enter();
            ActiveSpan::start(Some(&telemetry), "stage.inner", SpanLevel::Summary)
        };
        inner.end();
        skipped.end();
        drop(entered);
        outer.end();

        let spans = spans(&rx);
        assert!(!spans.contains_key("stage.debug_only"));
        assert_eq!(spans["stage.inner"].1, Some(spans["stage.outer"].0.span_id));
    }
}
