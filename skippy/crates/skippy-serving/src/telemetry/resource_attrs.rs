//! The resource every signal carries: the stage's lifecycle attributes, the
//! serving configuration that changes what the GPU does (KV precision, flash
//! attention, context, lanes, batch sizes), and whatever the operator set in
//! `OTEL_RESOURCE_ATTRIBUTES`, so runs with different settings can be told
//! apart and compared from their telemetry alone.

use std::{collections::BTreeMap, env, sync::OnceLock};

use serde_json::{Value, json};
use skippy_protocol::StageConfig;

use super::{lifecycle_attrs, span_components::SERVICE_NAMESPACE};

/// The standard OpenTelemetry variable for extra resource attributes.
const RESOURCE_ATTRIBUTES_ENV: &str = "OTEL_RESOURCE_ATTRIBUTES";

/// Resource attributes for `service`. Operator attributes come first, so the
/// stage's own identity and configuration win over a conflicting key.
pub(super) fn resource_attrs(config: &StageConfig, service: &str) -> BTreeMap<String, Value> {
    let mut attributes = operator_attributes()
        .iter()
        .map(|(key, value)| (key.clone(), json!(value)))
        .collect::<BTreeMap<_, _>>();
    attributes.extend(lifecycle_attrs(config));
    attributes.extend(serving_config_attrs(config));
    attributes.insert("service.name".to_string(), json!(service));
    attributes.insert("service.namespace".to_string(), json!(SERVICE_NAMESPACE));
    attributes
}

/// The settings that decide which kernels run and how much KV they read.
fn serving_config_attrs(config: &StageConfig) -> BTreeMap<String, Value> {
    let mut attributes = BTreeMap::from([
        (
            "skippy.kv.cache_type_k".to_string(),
            json!(config.cache_type_k),
        ),
        (
            "skippy.kv.cache_type_v".to_string(),
            json!(config.cache_type_v),
        ),
        (
            "skippy.attention.flash".to_string(),
            serde_json::to_value(config.flash_attn_type).unwrap_or(Value::Null),
        ),
        ("skippy.kv.ctx_size".to_string(), json!(config.ctx_size)),
        (
            "skippy.scheduler.lanes".to_string(),
            json!(config.lane_count),
        ),
    ]);
    let optional = [
        ("skippy.batch.size", config.n_batch.map(Value::from)),
        ("skippy.batch.ubatch_size", config.n_ubatch.map(Value::from)),
        ("skippy.kv.unified", config.kv_unified.map(Value::from)),
        ("skippy.kv.offload", config.kv_offload.map(Value::from)),
        ("skippy.kv.swa_full", config.swa_full.map(Value::from)),
    ];
    attributes.extend(
        optional
            .into_iter()
            .filter_map(|(key, value)| Some((key.to_string(), value?))),
    );
    attributes
}

fn operator_attributes() -> &'static [(String, String)] {
    static PARSED: OnceLock<Vec<(String, String)>> = OnceLock::new();
    PARSED.get_or_init(|| {
        env::var(RESOURCE_ATTRIBUTES_ENV)
            .map(|value| parse_resource_attributes(&value))
            .unwrap_or_default()
    })
}

/// `key=value` pairs separated by commas, values percent-encoded, as the
/// OpenTelemetry SDK environment specification defines; malformed entries
/// are skipped.
fn parse_resource_attributes(value: &str) -> Vec<(String, String)> {
    value
        .split(',')
        .filter_map(|entry| {
            let (key, value) = entry.split_once('=')?;
            let key = key.trim();
            (!key.is_empty()).then(|| (key.to_string(), percent_decode(value.trim())))
        })
        .collect()
}

fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut rest = bytes;
    while let Some((&first, tail)) = rest.split_first() {
        let escaped = (first == b'%')
            .then(|| tail.get(..2))
            .flatten()
            .and_then(|hex| u8::from_str_radix(std::str::from_utf8(hex).ok()?, 16).ok());
        match escaped {
            Some(byte) => {
                decoded.push(byte);
                rest = &tail[2..];
            }
            None => {
                decoded.push(first);
                rest = tail;
            }
        }
    }
    String::from_utf8_lossy(&decoded).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stage_config() -> StageConfig {
        serde_json::from_value(json!({
            "run_id": "run",
            "topology_id": "topology",
            "model_id": "org/model:Q4_K_M",
            "stage_id": "stage-0",
            "stage_index": 0,
            "layer_start": 0,
            "layer_end": 4,
            "load_mode": "runtime-slice",
            "execution_contract": "",
            "bind_addr": "127.0.0.1:0",
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "flash_attn_type": "enabled",
            "ctx_size": 16384,
            "lane_count": 4,
            "n_ubatch": 512,
        }))
        .expect("stage config")
    }

    #[test]
    fn the_resource_names_the_kv_and_attention_configuration() {
        let attributes = resource_attrs(&stage_config(), "skippy");
        assert_eq!(attributes["skippy.kv.cache_type_k"], json!("q8_0"));
        assert_eq!(attributes["skippy.attention.flash"], json!("enabled"));
        assert_eq!(attributes["skippy.kv.ctx_size"], json!(16384));
        assert_eq!(attributes["skippy.scheduler.lanes"], json!(4));
        assert_eq!(attributes["skippy.batch.ubatch_size"], json!(512));
        assert!(!attributes.contains_key("skippy.batch.size"));
        assert_eq!(attributes["service.name"], json!("skippy"));
    }

    #[test]
    fn operator_attributes_are_percent_decoded_pairs() {
        assert_eq!(
            parse_resource_attributes("skippy.run.label=q8%20fa, team=perf,=x,broken, empty="),
            vec![
                ("skippy.run.label".to_string(), "q8 fa".to_string()),
                ("team".to_string(), "perf".to_string()),
                ("empty".to_string(), String::new()),
            ]
        );
        assert_eq!(percent_decode("50%25%2"), "50%%2");
    }
}
