//! CPU stage-cache smoke profile; model identity is shared with ordinary OpenAI smoke.
use super::ModelStage;
use crate::{command::DynResult, repository::check_args::Grammar};
use serde::Serialize;
use std::{fs, io::Write, net::SocketAddr};
const GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool automation openai-smoke-config cache --output PATH --model-id ID --model-path PATH --layer-end N --ctx-size N --bind-addr ADDR --payload KIND --flash-attn KIND --n-batch N --n-ubatch N [--upstream-endpoint driver]",
    values: &[
        "--output",
        "--model-id",
        "--model-path",
        "--layer-end",
        "--ctx-size",
        "--bind-addr",
        "--payload",
        "--flash-attn",
        "--n-batch",
        "--n-ubatch",
        "--upstream-endpoint",
    ],
    flags: &["--help"],
};
#[derive(Serialize)]
struct Upstream<'a> {
    stage_id: &'static str,
    stage_index: u32,
    endpoint: &'a str,
}
#[derive(Serialize)]
struct Kv<'a> {
    mode: &'static str,
    payload: &'a str,
    max_entries: u32,
    max_bytes: u32,
    min_tokens: u32,
    shared_prefix_stride_tokens: u32,
    shared_prefix_record_limit: u32,
}
#[derive(Serialize)]
struct Config<'a> {
    run_id: &'static str,
    topology_id: &'static str,
    #[serde(flatten)]
    stage: ModelStage<'a>,
    lane_count: u32,
    n_batch: u32,
    n_ubatch: u32,
    cache_type_k: &'static str,
    cache_type_v: &'static str,
    flash_attn_type: &'a str,
    bind_addr: &'a str,
    upstream: Option<Upstream<'a>>,
    downstream: Option<()>,
    kv_cache: Kv<'a>,
}
pub(super) fn run(args: &[String]) -> DynResult<()> {
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return crate::repository::check_report::CheckReport::success(format!(
            "{}\n",
            GRAMMAR.usage
        ))
        .emit();
    }
    if !parsed.positionals.is_empty() {
        return Err("unexpected cache smoke positional arguments".into());
    }
    let required = |key| -> DynResult<&str> {
        parsed
            .last(key)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| format!("missing {key}").into())
    };
    let layer_end = required("--layer-end")?.parse()?;
    let ctx_size = required("--ctx-size")?.parse()?;
    let n_batch: u32 = required("--n-batch")?.parse()?;
    let n_ubatch: u32 = required("--n-ubatch")?.parse()?;
    if n_batch == 0 || n_ubatch == 0 {
        return Err("invalid cache smoke batch sizes".into());
    }
    let bind = required("--bind-addr")?;
    let address: SocketAddr = bind.parse()?;
    if !address.ip().is_loopback() || address.port() == 0 {
        return Err("cache smoke requires a nonzero loopback bind address".into());
    }
    let payload = required("--payload")?;
    if !matches!(
        payload,
        "resident-kv" | "kv-recurrent" | "full-state" | "auto"
    ) {
        return Err("unsupported cache payload".into());
    }
    let flash = required("--flash-attn")?;
    if !matches!(flash, "auto" | "enabled" | "disabled") {
        return Err("unsupported flash attention setting".into());
    }
    let upstream = match parsed
        .last("--upstream-endpoint")
        .filter(|value| !value.is_empty())
    {
        None => None,
        Some("driver") => Some(Upstream {
            stage_id: "stage-0",
            stage_index: 0,
            endpoint: "driver",
        }),
        Some(_) => return Err("cache smoke upstream must be driver or absent".into()),
    };
    let config = Config {
        run_id: "skippy-ci-smoke",
        topology_id: "skippy-ci-smoke-single-stage",
        stage: ModelStage::new(
            required("--model-id")?,
            required("--model-path")?,
            layer_end,
            ctx_size,
        )?,
        lane_count: 4,
        n_batch,
        n_ubatch,
        cache_type_k: "f16",
        cache_type_v: "f16",
        flash_attn_type: flash,
        bind_addr: bind,
        upstream,
        downstream: None,
        kv_cache: Kv {
            mode: "lookup-record",
            payload,
            max_entries: 32,
            max_bytes: 0,
            min_tokens: 64,
            shared_prefix_stride_tokens: 128,
            shared_prefix_record_limit: 2,
        },
    };
    let mut bytes = serde_json::to_vec_pretty(&config)?;
    bytes.push(b'\n');
    fs::File::create(required("--output")?)?.write_all(&bytes)?;
    Ok(())
}
