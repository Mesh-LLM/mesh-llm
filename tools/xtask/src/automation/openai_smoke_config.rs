//! CPU smoke stage identity, shared by ordinary OpenAI and cache profiles.
mod cache;
use crate::{
    command::DynResult,
    repository::{check_args::Grammar, check_report::CheckReport},
};
use serde::Serialize;
use std::{fs, io::Write, path::Path};
const GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool automation openai-smoke-config --output PATH --model-id ID --model-path PATH --layer-end N --ctx-size N",
    values: &[
        "--output",
        "--model-id",
        "--model-path",
        "--layer-end",
        "--ctx-size",
    ],
    flags: &["--help"],
};
#[derive(Serialize)]
struct ModelStage<'a> {
    model_id: &'a str,
    model_path: &'a str,
    source_model_sha256: String,
    stage_id: &'static str,
    stage_index: u32,
    layer_start: u32,
    layer_end: u32,
    ctx_size: u32,
    n_gpu_layers: u32,
    load_mode: &'static str,
    execution_contract: &'static str,
}
impl<'a> ModelStage<'a> {
    fn new(
        model_id: &'a str,
        model_path: &'a str,
        layer_end: u32,
        ctx_size: u32,
    ) -> DynResult<Self> {
        if model_id.is_empty() || layer_end == 0 || ctx_size == 0 {
            return Err(
                "model identity, layer end and context size must be nonempty/positive".into(),
            );
        }
        let source_model_sha256 = crate::product::digest::file_sha256(Path::new(model_path))
            .map_err(|failure| failure.error)?;
        Ok(Self {
            model_id,
            model_path,
            source_model_sha256,
            stage_id: "stage-0",
            stage_index: 0,
            layer_start: 0,
            layer_end,
            ctx_size,
            n_gpu_layers: 0,
            load_mode: "runtime-slice",
            execution_contract: "",
        })
    }
}
#[derive(Serialize)]
struct Config<'a> {
    run_id: &'static str,
    topology_id: &'static str,
    #[serde(flatten)]
    stage: ModelStage<'a>,
    bind_addr: &'static str,
    upstream: Option<()>,
    downstream: Option<()>,
    kv_server: Option<()>,
}
pub(crate) fn run(args: &[String]) -> DynResult<()> {
    if let Some((profile, rest)) = args.split_first()
        && profile == "cache"
    {
        return cache::run(rest);
    }
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage)).emit();
    }
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error("unexpected positional arguments").emit();
    }
    let required = |key| -> DynResult<&str> {
        parsed
            .last(key)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| format!("missing {key}").into())
    };
    let config = Config {
        run_id: "openai-smoke",
        topology_id: "openai-smoke-single-stage",
        stage: ModelStage::new(
            required("--model-id")?,
            required("--model-path")?,
            required("--layer-end")?.parse()?,
            required("--ctx-size")?.parse()?,
        )?,
        bind_addr: "127.0.0.1:19000",
        upstream: None,
        downstream: None,
        kv_server: None,
    };
    let output = required("--output")?;
    let mut bytes = serde_json::to_vec_pretty(&config)?;
    bytes.push(b'\n');
    fs::File::create(output)?.write_all(&bytes)?;
    Ok(())
}
