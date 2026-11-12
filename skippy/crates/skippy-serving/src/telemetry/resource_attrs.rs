//! The resource every signal carries: the keys that make runs comparable
//! across machines, backends and commits, the stage's lifecycle attributes,
//! the serving configuration that changes what the GPU does (KV precision,
//! flash attention, context, lanes, batch sizes), and whatever the operator
//! set in `OTEL_RESOURCE_ATTRIBUTES`, so runs with different settings can be
//! told apart and compared from their telemetry alone.

use std::{collections::BTreeMap, env, sync::OnceLock};

use serde_json::{Value, json};
use skippy_protocol::{StageConfig, StageKvRetention};
use skippy_runtime::{BackendDeviceType, backend_devices_untraced};

use super::{lifecycle_attrs, span_components::SERVICE_NAMESPACE};

/// The standard OpenTelemetry variable for extra resource attributes.
const RESOURCE_ATTRIBUTES_ENV: &str = "OTEL_RESOURCE_ATTRIBUTES";

/// The run this process is: the stage's run id.
pub const RUN_ID: &str = "run.id";
/// The backend the runtime computes on (`MTL`, `CUDA`, `CPU`), once it has
/// computed anything; spans exported before that carry no backend.
pub const BACKEND_NAME: &str = "backend.name";
/// The commit skippy-serving was built from. `test.name`, the workload that
/// compares across runs, is the operator's alone and passes through.
pub const GIT_COMMIT_SHA: &str = "git.commit.sha";
/// The machine the run happened on; the host name unless the operator sets it.
pub const MACHINE_ID: &str = "machine.id";
/// `warmup` or `measured`; `measured` unless the operator sets it.
pub const RUN_PHASE: &str = "run.phase";

/// Resource attributes for `service`. Operator attributes override the
/// defaults for the machine and phase; the stage's own identity, build and
/// configuration win over a conflicting key.
pub(super) fn resource_attrs(config: &StageConfig, service: &str) -> BTreeMap<String, Value> {
    resource_attrs_with(operator_attributes(), config, service)
}

fn resource_attrs_with(
    operator: &[(String, String)],
    config: &StageConfig,
    service: &str,
) -> BTreeMap<String, Value> {
    let mut attributes = run_defaults();
    attributes.extend(
        operator
            .iter()
            .map(|(key, value)| (key.clone(), json!(value))),
    );
    attributes.extend(run_identity(config));
    attributes.extend(lifecycle_attrs(config));
    attributes.extend(serving_config_attrs(config));
    attributes.insert("service.name".to_string(), json!(service));
    attributes.insert("service.namespace".to_string(), json!(SERVICE_NAMESPACE));
    attributes
}

/// What the operator may override: the machine and the run's phase.
fn run_defaults() -> BTreeMap<String, Value> {
    let mut attributes = BTreeMap::from([(RUN_PHASE.to_string(), json!("measured"))]);
    attributes.extend(host_name().map(|name| (MACHINE_ID.to_string(), json!(name))));
    attributes
}

/// What the process knows about itself: its run, its build and, once known,
/// its backend.
fn run_identity(config: &StageConfig) -> BTreeMap<String, Value> {
    let mut attributes = BTreeMap::from([
        (RUN_ID.to_string(), json!(config.run_id)),
        (GIT_COMMIT_SHA.to_string(), json!(env!("SKIPPY_GIT_COMMIT"))),
    ]);
    attributes.extend(backend_name().map(|name| (BACKEND_NAME.to_string(), json!(name))));
    attributes
}

/// The backend of the first GPU the runtime reports, else of its first
/// device, named as GPU command spans name it: the device name without its
/// index (`MTL0` is `MTL`). Devices are queried once the runtime has made a
/// traced native call, so the query is never the call that starts it.
fn backend_name() -> Option<&'static str> {
    static NAME: OnceLock<String> = OnceLock::new();
    if let Some(name) = NAME.get() {
        return Some(name);
    }
    if !skippy_runtime::native_tracing::native_tracing_active() {
        return None;
    }
    let devices = backend_devices_untraced().ok()?;
    let device = devices
        .iter()
        .find(|device| {
            matches!(
                device.device_type,
                BackendDeviceType::Gpu | BackendDeviceType::IntegratedGpu
            )
        })
        .or_else(|| devices.first())?;
    let name = device
        .name
        .trim_end_matches(|c: char| c.is_ascii_digit())
        .to_string();
    Some(NAME.get_or_init(|| name))
}

/// The host name from the kernel, as `machine.id` when the operator sets none.
#[cfg(unix)]
fn host_name() -> Option<String> {
    let mut bytes = [0_u8; 256];
    // SAFETY: bytes is writable for the exact length passed to libc.
    if unsafe { libc::gethostname(bytes.as_mut_ptr().cast(), bytes.len()) } != 0 {
        return None;
    }
    let end = bytes.iter().position(|byte| *byte == 0)?;
    let name = std::str::from_utf8(&bytes[..end]).ok()?.trim();
    (!name.is_empty()).then(|| name.to_string())
}

#[cfg(not(unix))]
fn host_name() -> Option<String> {
    env::var("COMPUTERNAME")
        .or_else(|_| env::var("HOSTNAME"))
        .ok()
        .filter(|name| !name.trim().is_empty())
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
        (
            "skippy.kv.pool_tokens",
            config
                .kv_pool_tokens
                .map(|tokens| Value::from(tokens.get())),
        ),
        (
            "skippy.kv.lazy_clear",
            config.kv_lazy_clear.map(Value::from),
        ),
        (
            "skippy.kv.release_pages",
            config.kv_release_pages.map(Value::from),
        ),
        (
            "skippy.kv.row_attention",
            config.kv_row_attention.map(Value::from),
        ),
        (
            "skippy.kv.retention_tokens",
            config.kv_cache.as_ref().map(|cache| match cache.retention {
                StageKvRetention::Tokens(tokens) => Value::from(tokens),
                StageKvRetention::Unbounded => Value::from("unbounded"),
            }),
        ),
        (
            "skippy.scheduler.prefill_coalesce_max_ms",
            config.prefill_coalesce_max_ms.map(Value::from),
        ),
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
    fn the_resource_identifies_the_run_build_and_machine() {
        let attributes = resource_attrs_with(&[], &stage_config(), "skippy");
        assert_eq!(attributes[RUN_ID], json!("run"));
        assert_eq!(attributes[RUN_PHASE], json!("measured"));
        assert!(!attributes[GIT_COMMIT_SHA].as_str().unwrap().is_empty());
        assert!(!attributes.contains_key("test.name"));
        assert!(!attributes.contains_key(BACKEND_NAME));
        assert_eq!(attributes.contains_key(MACHINE_ID), host_name().is_some());
    }

    #[test]
    fn the_operator_sets_the_test_phase_and_machine_but_not_the_run() {
        let operator = [
            ("test.name".to_string(), "benchy-pp512-tg128-c4".to_string()),
            (RUN_PHASE.to_string(), "warmup".to_string()),
            (MACHINE_ID.to_string(), "spark-137".to_string()),
            (RUN_ID.to_string(), "not-this-run".to_string()),
        ];
        let attributes = resource_attrs_with(&operator, &stage_config(), "skippy");
        assert_eq!(attributes["test.name"], json!("benchy-pp512-tg128-c4"));
        assert_eq!(attributes[RUN_PHASE], json!("warmup"));
        assert_eq!(attributes[MACHINE_ID], json!("spark-137"));
        assert_eq!(attributes[RUN_ID], json!("run"));
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
