//! Encoding of Skippy telemetry records as OTLP trace messages.

use std::{collections::BTreeMap, sync::atomic::Ordering};

use opentelemetry_proto::tonic::{
    common::v1::{AnyValue, ArrayValue, InstrumentationScope, KeyValue, KeyValueList, any_value},
    resource::v1::Resource,
    trace::v1::{ResourceSpans, ScopeSpans, Span, span},
};
use serde_json::{Value, json};
use skippy_metrics::metric;
use skippy_protocol::StageConfig;

use super::{
    TelemetryCounters, resource_attrs::resource_attrs, sink::SpanRecord, span_components::component,
};

/// One stage's spans, one resource per system component (see
/// [`super::span_components`]), so viewers that colour by service tell the
/// parts apart; the spans keep one trace.
pub(super) fn resource_spans_by_component(
    config: &StageConfig,
    spans: Vec<Span>,
) -> Vec<ResourceSpans> {
    let mut groups: BTreeMap<&'static str, Vec<Span>> = BTreeMap::new();
    for span in spans {
        groups.entry(component(&span.name)).or_default().push(span);
    }
    groups
        .into_iter()
        .map(|(service, spans)| resource_spans(config, service, spans))
        .collect()
}

/// Spans of one stage and component under their resource and the serving
/// instrumentation scope.
fn resource_spans(config: &StageConfig, service: &str, spans: Vec<Span>) -> ResourceSpans {
    let attributes = resource_attrs(config, service);
    ResourceSpans {
        resource: Some(Resource {
            attributes: key_values(attributes),
            dropped_attributes_count: 0,
            entity_refs: vec![],
        }),
        scope_spans: vec![ScopeSpans {
            scope: Some(InstrumentationScope {
                name: "skippy-serving".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                attributes: vec![],
                dropped_attributes_count: 0,
            }),
            spans,
            schema_url: String::new(),
        }],
        schema_url: String::new(),
    }
}

pub(super) fn event_to_span(event: &SpanRecord, stats: &TelemetryCounters) -> Span {
    let mut attributes = event.attributes.clone();
    attributes.insert(
        metric::OTEL_DROPPED_EVENTS.to_string(),
        json!(stats.dropped.load(Ordering::Relaxed)),
    );
    attributes.insert(
        metric::OTEL_EXPORT_ERRORS.to_string(),
        json!(stats.export_errors.load(Ordering::Relaxed)),
    );
    Span {
        trace_id: event.context.trace_id.to_vec(),
        span_id: event.context.span_id.to_vec(),
        trace_state: String::new(),
        parent_span_id: event
            .parent
            .map(|parent| parent.to_vec())
            .unwrap_or_default(),
        flags: 1,
        name: event.name.clone(),
        kind: span::SpanKind::Internal as i32,
        start_time_unix_nano: event.start_time_unix_nanos,
        end_time_unix_nano: event.end_time_unix_nanos,
        attributes: key_values(attributes),
        dropped_attributes_count: 0,
        events: event
            .events
            .iter()
            .map(|recorded| span::Event {
                time_unix_nano: recorded.time_unix_nanos,
                name: recorded.name.clone(),
                attributes: key_values(recorded.attributes.clone()),
                dropped_attributes_count: 0,
            })
            .collect(),
        dropped_events_count: 0,
        links: event
            .links
            .iter()
            .map(|linked| span::Link {
                trace_id: linked.trace_id.to_vec(),
                span_id: linked.span_id.to_vec(),
                trace_state: String::new(),
                attributes: vec![],
                dropped_attributes_count: 0,
                flags: 1,
            })
            .collect(),
        dropped_links_count: 0,
        status: None,
    }
}

pub(super) fn key_values(attributes: BTreeMap<String, Value>) -> Vec<KeyValue> {
    attributes
        .into_iter()
        .map(|(key, value)| KeyValue {
            key,
            value: Some(json_to_any_value(value)),
            ..Default::default()
        })
        .collect()
}

fn json_to_any_value(value: Value) -> AnyValue {
    let value = match value {
        Value::String(value) => any_value::Value::StringValue(value),
        Value::Bool(value) => any_value::Value::BoolValue(value),
        Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                any_value::Value::IntValue(value)
            } else {
                any_value::Value::DoubleValue(value.as_f64().unwrap_or_default())
            }
        }
        Value::Array(values) => any_value::Value::ArrayValue(ArrayValue {
            values: values.into_iter().map(json_to_any_value).collect(),
        }),
        Value::Object(values) => any_value::Value::KvlistValue(KeyValueList {
            values: key_values(values.into_iter().collect()),
        }),
        Value::Null => any_value::Value::StringValue(String::new()),
    };
    AnyValue { value: Some(value) }
}
