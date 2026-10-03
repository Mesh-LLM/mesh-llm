//! Cached runtime observations and their management metrics/slots projection.
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RuntimeLlamaEndpointStatus {
    Ready,
    #[default]
    Unavailable,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RuntimeLlamaMetricSample {
    pub name: String,
    pub labels: BTreeMap<String, String>,
    pub value: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RuntimeLlamaMetricsSnapshot {
    pub status: RuntimeLlamaEndpointStatus,
    pub last_attempt_unix_ms: Option<u64>,
    pub last_success_unix_ms: Option<u64>,
    pub error: Option<String>,
    pub raw_text: Option<String>,
    pub samples: Vec<RuntimeLlamaMetricSample>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RuntimeLlamaSlotSnapshot {
    pub id: Option<u64>,
    pub id_task: Option<u64>,
    pub n_ctx: Option<u64>,
    pub speculative: Option<bool>,
    pub is_processing: Option<bool>,
    pub next_token: Option<Value>,
    pub params: Option<Value>,
    pub extra: Value,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RuntimeLlamaSlotsSnapshot {
    pub status: RuntimeLlamaEndpointStatus,
    pub model: Option<String>,
    pub instance_id: Option<String>,
    pub last_attempt_unix_ms: Option<u64>,
    pub last_success_unix_ms: Option<u64>,
    pub error: Option<String>,
    pub slots: Vec<RuntimeLlamaSlotSnapshot>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RuntimeLlamaMetricItem {
    pub name: String,
    pub labels: BTreeMap<String, String>,
    pub value: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RuntimeLlamaSlotItem {
    pub index: usize,
    pub id: Option<u64>,
    pub id_task: Option<u64>,
    pub n_ctx: Option<u64>,
    pub is_processing: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RuntimeLlamaRuntimeItems {
    pub metrics: Vec<RuntimeLlamaMetricItem>,
    pub slots: Vec<RuntimeLlamaSlotItem>,
    pub slots_total: usize,
    pub slots_busy: usize,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RuntimeLlamaRuntimeSnapshot {
    pub metrics: RuntimeLlamaMetricsSnapshot,
    pub slots: RuntimeLlamaSlotsSnapshot,
    pub items: RuntimeLlamaRuntimeItems,
}

#[derive(Clone, Debug, Serialize)]
pub struct RuntimeLlamaPayload {
    pub metrics: RuntimeLlamaMetricsPayload,
    pub slots: RuntimeLlamaSlotsPayload,
    pub items: RuntimeLlamaItemsPayload,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub instances: Vec<RuntimeLlamaInstancePayload>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RuntimeLlamaInstancePayload {
    pub instance_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub metrics: RuntimeLlamaMetricsPayload,
    pub slots: RuntimeLlamaSlotsPayload,
    pub items: RuntimeLlamaItemsPayload,
}

#[derive(Clone, Debug, Serialize)]
pub struct RuntimeLlamaMetricsPayload {
    pub status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_attempt_unix_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_success_unix_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_text: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub samples: Vec<RuntimeLlamaMetricSamplePayload>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RuntimeLlamaMetricSamplePayload {
    pub name: String,
    #[serde(skip_serializing_if = "BTreeMap::is_empty", default)]
    pub labels: BTreeMap<String, String>,
    pub value: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct RuntimeLlamaSlotsPayload {
    pub status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instance_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_attempt_unix_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_success_unix_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub slots: Vec<RuntimeLlamaSlotPayload>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RuntimeLlamaSlotPayload {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id_task: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_ctx: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speculative: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_processing: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_token: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "serde_json::Value::is_null")]
    pub extra: serde_json::Value,
}

#[derive(Clone, Debug, Serialize)]
pub struct RuntimeLlamaItemsPayload {
    pub metrics: Vec<RuntimeLlamaMetricItemPayload>,
    pub slots: Vec<RuntimeLlamaSlotItemPayload>,
    pub slots_total: usize,
    pub slots_busy: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct RuntimeLlamaMetricItemPayload {
    pub name: String,
    #[serde(skip_serializing_if = "BTreeMap::is_empty", default)]
    pub labels: BTreeMap<String, String>,
    pub value: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct RuntimeLlamaSlotItemPayload {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id_task: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_ctx: Option<u64>,
    pub is_processing: bool,
}

pub fn build_runtime_llama_payload(
    snapshot: RuntimeLlamaRuntimeSnapshot,
    snapshots_by_instance: BTreeMap<String, RuntimeLlamaRuntimeSnapshot>,
) -> RuntimeLlamaPayload {
    let instances = snapshots_by_instance
        .into_iter()
        .map(|(instance_id, snapshot)| {
            let model = snapshot.slots.model.clone();
            let (metrics, slots, items) = build_runtime_llama_snapshot_payload(snapshot);
            RuntimeLlamaInstancePayload {
                instance_id,
                model,
                metrics,
                slots,
                items,
            }
        })
        .collect();
    let (metrics, slots, items) = build_runtime_llama_snapshot_payload(snapshot);
    RuntimeLlamaPayload {
        metrics,
        slots,
        items,
        instances,
    }
}

fn build_runtime_llama_snapshot_payload(
    snapshot: RuntimeLlamaRuntimeSnapshot,
) -> (
    RuntimeLlamaMetricsPayload,
    RuntimeLlamaSlotsPayload,
    RuntimeLlamaItemsPayload,
) {
    (
        RuntimeLlamaMetricsPayload {
            status: runtime_llama_endpoint_status(snapshot.metrics.status),
            last_attempt_unix_ms: snapshot.metrics.last_attempt_unix_ms,
            last_success_unix_ms: snapshot.metrics.last_success_unix_ms,
            error: snapshot.metrics.error,
            raw_text: snapshot.metrics.raw_text,
            samples: snapshot
                .metrics
                .samples
                .into_iter()
                .map(|sample| RuntimeLlamaMetricSamplePayload {
                    name: sample.name,
                    labels: sample.labels,
                    value: sample.value,
                })
                .collect(),
        },
        RuntimeLlamaSlotsPayload {
            status: runtime_llama_endpoint_status(snapshot.slots.status),
            model: snapshot.slots.model,
            instance_id: snapshot.slots.instance_id,
            last_attempt_unix_ms: snapshot.slots.last_attempt_unix_ms,
            last_success_unix_ms: snapshot.slots.last_success_unix_ms,
            error: snapshot.slots.error,
            slots: snapshot
                .slots
                .slots
                .into_iter()
                .map(|slot| RuntimeLlamaSlotPayload {
                    id: slot.id,
                    id_task: slot.id_task,
                    n_ctx: slot.n_ctx,
                    speculative: slot.speculative,
                    is_processing: slot.is_processing,
                    next_token: slot.next_token,
                    params: slot.params,
                    extra: slot.extra,
                })
                .collect(),
        },
        RuntimeLlamaItemsPayload {
            metrics: snapshot
                .items
                .metrics
                .into_iter()
                .map(|item| RuntimeLlamaMetricItemPayload {
                    name: item.name,
                    labels: item.labels,
                    value: item.value,
                })
                .collect(),
            slots: snapshot
                .items
                .slots
                .into_iter()
                .map(|item| RuntimeLlamaSlotItemPayload {
                    index: item.index,
                    id: item.id,
                    id_task: item.id_task,
                    n_ctx: item.n_ctx,
                    is_processing: item.is_processing,
                })
                .collect(),
            slots_total: snapshot.items.slots_total,
            slots_busy: snapshot.items.slots_busy,
        },
    )
}

fn runtime_llama_endpoint_status(status: RuntimeLlamaEndpointStatus) -> &'static str {
    match status {
        RuntimeLlamaEndpointStatus::Ready => "ready",
        RuntimeLlamaEndpointStatus::Unavailable => "unavailable",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn empty_snapshot_preserves_unavailable_status_and_optional_field_omission() {
        let payload = build_runtime_llama_payload(Default::default(), BTreeMap::new());
        assert_eq!(
            serde_json::to_value(payload).unwrap(),
            json!({
                "metrics": { "status": "unavailable" },
                "slots": { "status": "unavailable" },
                "items": { "metrics": [], "slots": [], "slots_total": 0, "slots_busy": 0 }
            })
        );
    }

    #[test]
    fn snapshot_projection_preserves_values_and_orders_instances_by_identity() {
        let snapshot = RuntimeLlamaRuntimeSnapshot {
            metrics: RuntimeLlamaMetricsSnapshot {
                status: RuntimeLlamaEndpointStatus::Ready,
                last_attempt_unix_ms: Some(11),
                last_success_unix_ms: Some(10),
                raw_text: Some("tokens_total 7".into()),
                error: Some("previous attempt".into()),
                samples: vec![RuntimeLlamaMetricSample {
                    name: "tokens_total".into(),
                    labels: BTreeMap::from([("model".into(), "example".into())]),
                    value: 7.0,
                }],
            },
            slots: RuntimeLlamaSlotsSnapshot {
                status: RuntimeLlamaEndpointStatus::Ready,
                model: Some("example".into()),
                instance_id: Some("instance-b".into()),
                last_attempt_unix_ms: Some(12),
                last_success_unix_ms: Some(12),
                error: None,
                slots: vec![RuntimeLlamaSlotSnapshot {
                    id: Some(3),
                    id_task: Some(4),
                    n_ctx: Some(1024),
                    speculative: Some(true),
                    is_processing: Some(true),
                    next_token: Some(json!({"token": 7})),
                    params: Some(json!({"temperature": 0})),
                    extra: json!({"additive": "retained"}),
                }],
            },
            items: RuntimeLlamaRuntimeItems {
                metrics: vec![RuntimeLlamaMetricItem {
                    name: "tokens_total".into(),
                    labels: BTreeMap::new(),
                    value: 7.0,
                }],
                slots: vec![RuntimeLlamaSlotItem {
                    index: 0,
                    id: Some(3),
                    id_task: Some(4),
                    n_ctx: Some(1024),
                    is_processing: true,
                }],
                slots_total: 1,
                slots_busy: 1,
            },
        };
        let payload = build_runtime_llama_payload(
            snapshot.clone(),
            BTreeMap::from([
                ("instance-b".into(), snapshot.clone()),
                ("instance-a".into(), snapshot),
            ]),
        );
        let json = serde_json::to_value(payload).unwrap();
        assert_eq!(json["metrics"]["samples"][0]["labels"]["model"], "example");
        assert_eq!(json["metrics"]["last_attempt_unix_ms"], 11);
        assert_eq!(json["metrics"]["last_success_unix_ms"], 10);
        assert_eq!(json["metrics"]["error"], "previous attempt");
        assert_eq!(
            json["slots"]["slots"][0]["extra"],
            json!({"additive": "retained"})
        );
        assert_eq!(
            json["slots"]["slots"][0]["params"],
            json!({"temperature": 0})
        );
        assert_eq!(json["items"]["slots_busy"], 1);
        assert_eq!(json["instances"][0]["instance_id"], "instance-a");
        assert_eq!(json["instances"][1]["instance_id"], "instance-b");
        assert_eq!(json["instances"][0]["metrics"], json["metrics"]);
        assert_eq!(json["instances"][0]["slots"], json["slots"]);
        assert_eq!(json["instances"][0]["items"], json["items"]);
    }
}
