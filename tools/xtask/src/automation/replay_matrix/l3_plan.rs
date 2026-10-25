use super::l3_contract::Config;
use crate::{
    command::DynResult,
    repository::{
        check_args::{Grammar, ParsedArgs},
        check_report::CheckReport,
    },
};
use std::path::Path;
const GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool automation replay-matrix l3-plan --ref LABEL=GIT_REF --model MODEL --trajectory-manifest PATH [--repo PATH] [--backend BACKEND] [--require-source-dataset SOURCE] [--concurrency N]",
    values: &[
        "--repo",
        "--ref",
        "--model",
        "--backend",
        "--trajectory-manifest",
        "--lifecycle-cohort",
        "--require-source-dataset",
        "--concurrency",
        "--prompt-token-range",
        "--cold-samples",
        "--restart-samples",
        "--identical-repeats",
        "--max-output-tokens",
        "--disk-budget",
        "--low-space-disk-budget",
        "--minimum-free",
        "--low-space-minimum-free",
        "--max-l3-ttft-ratio",
        "--max-payload-write-amplification",
        "--max-decode-p99-regression-pct",
    ],
    flags: &["--help"],
};
pub(super) fn config(parsed: &ParsedArgs) -> DynResult<Config> {
    let text = |name, default| parsed.last(name).unwrap_or(default).to_owned();
    let count = |name, default| -> DynResult<usize> { Ok(text(name, default).parse()?) };
    let decimal = |name, default| -> DynResult<f64> { Ok(text(name, default).parse()?) };
    let range = text("--prompt-token-range", "18000:24000");
    let (minimum, maximum) = range
        .split_once(':')
        .ok_or("prompt range must be MIN:MAX")?;
    let concurrency = parsed.all("--concurrency");
    Ok(Config {
        model: parsed.last("--model").ok_or("missing --model")?.into(),
        backend: text("--backend", "metal"),
        lifecycle_cohort: text("--lifecycle-cohort", "l3"),
        required_sources: parsed
            .all("--require-source-dataset")
            .into_iter()
            .map(str::to_owned)
            .collect(),
        concurrency: if concurrency.is_empty() {
            vec![64, 128, 256]
        } else {
            concurrency
                .into_iter()
                .map(str::parse)
                .collect::<Result<_, _>>()?
        },
        prompt_min: minimum.parse()?,
        prompt_max: maximum.parse()?,
        cold_samples: count("--cold-samples", "3")?,
        restart_samples: count("--restart-samples", "3")?,
        identical_repeats: count("--identical-repeats", "100")?,
        max_output_tokens: text("--max-output-tokens", "2048").parse()?,
        disk_budget: text("--disk-budget", "auto"),
        minimum_free: text("--minimum-free", "1GiB"),
        low_space_disk_budget: text("--low-space-disk-budget", "32GiB"),
        low_space_minimum_free: text("--low-space-minimum-free", "1TiB"),
        max_l3_ttft_ratio: decimal("--max-l3-ttft-ratio", "0.5")?,
        max_payload_write_amplification: decimal("--max-payload-write-amplification", "1.2")?,
        max_decode_p99_regression_pct: decimal("--max-decode-p99-regression-pct", "5")?,
    })
}
pub(in crate::automation) fn run(root: Option<&Path>, args: &[String]) -> DynResult<()> {
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
    crate::command::print_json(&document(root, &parsed)?)
}
pub(super) fn document(root: Option<&Path>, parsed: &ParsedArgs) -> DynResult<serde_json::Value> {
    let config = config(parsed)?;
    let mut plan = super::l3_execution::plan(&config)?;
    let specification = parsed.last("--ref").ok_or("missing --ref")?;
    let (label, reference) = specification
        .split_once('=')
        .ok_or("ref must be LABEL=GIT_REF")?;
    if label.is_empty() || reference.is_empty() || reference.starts_with('-') {
        return Err("invalid ref specification".into());
    }
    let repo =
        crate::repository::RepositoryRoot::resolve(parsed.last("--repo").map(Path::new).or(root))?;
    plan["repo"] = repo.as_path().to_string_lossy().into_owned().into();
    plan["ref"] = serde_json::json!({"label":label,"ref":reference});
    plan["input"] = serde_json::json!({"trajectory_manifest":parsed.last("--trajectory-manifest").ok_or("missing --trajectory-manifest")?,"lifecycle_cohort":config.lifecycle_cohort,"required_source_datasets":config.required_sources,"high_load_cohorts":config.concurrency.iter().map(ToString::to_string).collect::<Vec<_>>(),"prompt_token_range":[config.prompt_min,config.prompt_max],"low_space_disk_budget":config.low_space_disk_budget,"low_space_minimum_free":config.low_space_minimum_free});
    plan["server_commands"] = serde_json::json!({"disk_off":["<release-binary>","serve","--model",config.model,"--log-format","json"],"disk_on":["<release-binary>","serve","--model",config.model,"--log-format","json","--kv-cache-disk",config.disk_budget,"--kv-cache-disk-dir","<persistent-cache-root>","--kv-cache-min-free",config.minimum_free]});
    plan["gates"] = serde_json::json!({"all_requests_succeed":true,"greedy_seeded_output_sha256_matches_disk_off":true,"post_restart_l3_ttft_p50_ratio_max":config.max_l3_ttft_ratio,"identical_prefix_repeats":config.identical_repeats,"physical_fill_delta":1,"physical_write_delta":1,"payload_write_amplification_max":config.max_payload_write_amplification,"forced_low_space_state":"read_only_low_space","high_load_decode_inter_token_p99_regression_max_pct":config.max_decode_p99_regression_pct});
    Ok(plan)
}
