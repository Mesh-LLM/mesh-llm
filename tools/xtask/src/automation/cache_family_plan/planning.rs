use super::contract::{Case, Corpus, Input, UseCase};
use crate::command::DynResult;
use serde_json::{Value, json};
use std::{collections::BTreeSet, path::Path};
pub(super) fn catalog() -> DynResult<Vec<Case>> {
    Ok(serde_json::from_slice(include_bytes!("catalog.json"))?)
}
fn hash(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(bytes))
}
fn use_cases(input: &Input) -> DynResult<Vec<Option<UseCase>>> {
    let Some(pin) = &input.corpus else {
        return Ok(vec![None]);
    };
    let bytes = crate::automation::receipt_files::bounded(&pin.path, 4 * 1024 * 1024)?;
    if hash(&bytes) != pin.sha256 {
        return Err("use-case corpus byte pin mismatch".into());
    }
    let corpus: Corpus = serde_json::from_slice(&bytes)?;
    let mut keys = BTreeSet::new();
    if corpus.version != 1 || corpus.use_cases.is_empty() || corpus.use_cases.len() > 128 {
        return Err("invalid cache corpus shape".into());
    }
    for item in &corpus.use_cases {
        if !super::contract::key(&item.key)
            || !keys.insert(&item.key)
            || item.label.is_empty()
            || item.label.len() > 4096
            || item.prompt.is_empty()
            || item.prompt.len() > 65536
            || item.prompt.contains('\0')
            || item.prefix_tokens == 0
        {
            return Err("invalid cache corpus member".into());
        }
    }
    if input.use_cases != ["all"] && input.use_cases.iter().any(|k| !keys.contains(k)) {
        return Err("unknown cache use case".into());
    }
    Ok(corpus
        .use_cases
        .into_iter()
        .filter(|u| input.use_cases == ["all"] || input.use_cases.contains(&u.key))
        .map(Some)
        .collect())
}
fn model(root: &Path, case: &Case, pin: Option<&String>) -> DynResult<Value> {
    let requested = root.join(&case.snapshot_relative);
    let canonical = match requested.canonicalize() {
        Ok(p) => p,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(
                json!({"status":"missing-model","requested":requested,"declared_revision":case.revision,"declared_byte_sha256":pin}),
            );
        }
        Err(e) => return Err(e.into()),
    };
    if !canonical.starts_with(root) {
        return Err("catalog model resolves outside admitted cache root".into());
    }
    let metadata = std::fs::symlink_metadata(&canonical)?;
    let expected_directory = case.stage_load_mode == "layer-package";
    if expected_directory && !metadata.is_dir() || !expected_directory && !metadata.is_file() {
        return Err("catalog model has unsupported file kind".into());
    }
    Ok(
        json!({"status":"present-unqualified","requested":requested,"canonical":canonical,"runtime_entrypoint":if case.key=="minimax_m27"{requested.parent().ok_or("primary parent")?.canonicalize()?.join(requested.file_name().ok_or("primary filename")?)}else{canonical.clone()},"declared_revision":case.revision,"declared_byte_sha256":pin,"kind":if expected_directory{"layer-package-tree"}else if case.key=="minimax_m27"{"split-gguf-first-shard"}else{"single-gguf"},"custody":"path_metadata_only_not_byte_or_model_admission"}),
    )
}
fn effective(
    case: &Case,
    use_case: Option<&UseCase>,
    prefix: Option<u32>,
    input: &Input,
) -> DynResult<Case> {
    let mut effective = case.clone();
    let override_prefix = prefix.or(use_case.map(|u| u.prefix_tokens));
    if let Some(prefix) = override_prefix {
        effective.prefix_tokens = prefix;
        effective.ctx_size = effective.ctx_size.max(
            prefix
                .checked_add(128)
                .ok_or("cache prefix context overflow")?,
        );
    }
    effective.n_gpu_layers = input.n_gpu_layers.unwrap_or(case.n_gpu_layers);
    effective.cache_hit_repeats = input.cache_hit_repeats.unwrap_or(case.cache_hit_repeats);
    Ok(effective)
}
fn tasks(case: &Case, observation: &Value, input: &Input) -> DynResult<Value> {
    let missing = observation["status"] == "missing-model";
    let package = case.stage_load_mode == "layer-package";
    let baseline_reason = if missing {
        Some("missing-model")
    } else if package {
        Some("layer-package-not-full-GGUF")
    } else if input.skip_llama_server {
        Some("skipped-by-request")
    } else {
        None
    };
    let lanes = input
        .runtime_lane_count
        .unwrap_or(*input.concurrency.iter().max().ok_or("concurrency")?);
    let serving_ctx = input.serving_ctx_size.unwrap_or(
        case.ctx_size
            .checked_mul(lanes)
            .ok_or("serving context overflow")?,
    );
    let rungs: Vec<_> = input
        .concurrency
        .iter()
        .map(|c| json!({"concurrency":c,"requests":input.concurrent_requests.max(*c)}))
        .collect();
    Ok(
        json!({"correctness":{"planned":!missing,"payload":case.payload,"stage_load_mode":case.stage_load_mode,"state_range":[case.state_layer_start,case.state_layer_end],"state_stage_index":case.state_stage_index,"adapter":if package{"layer-package-admission-required"}else if case.key=="minimax_m27"{"complete-shard-admission-required"}else{"cache-family-correctness-stage"},"admission":"native-byte-and-GGUF-admission-before-execution"},"serial_baseline":{"planned":baseline_reason.is_none(),"skip_reason":baseline_reason,"warmup_excluded":true,"runs":input.llama_repeats,"output_tokens":1,"prompt_source":"accepted_correctness_benchmark_prompt_text"},"concurrent_baseline":{"planned":baseline_reason.is_none(),"skip_reason":baseline_reason,"parallel":input.llama_parallel,"output_tokens":128,"rungs":rungs},"paired_serving":{"planned":!missing&&!package&&input.old_server.is_some(),"old":input.old_server,"new":input.new_server,"runtime_lanes":lanes,"shared_ctx_size":serving_ctx,"output_tokens":input.concurrent_output_tokens,"rungs":rungs,"greedy_fastpath":true,"ttft_slo_ms":input.ttft_slo_ms,"tpot_slo_ms":input.tpot_slo_ms,"parity_requirement":"complete_matching_request_ID_roster_and_content","prompt_source":"accepted_correctness_benchmark_prompt_text"}}),
    )
}
pub(super) fn plan(input: &Input) -> DynResult<Value> {
    input.validate()?;
    let root = input.cache_root.canonicalize()?;
    if !std::fs::symlink_metadata(&root)?.is_dir() {
        return Err("cache root must be a directory".into());
    }
    let catalog = catalog()?;
    let keys: BTreeSet<_> = catalog.iter().map(|c| &c.key).collect();
    if input.cases.iter().any(|k| !keys.contains(k))
        || input.model_sha256.keys().any(|k| !keys.contains(k))
    {
        return Err("unknown current cache-family case".into());
    }
    for pin in [&input.old_server, &input.new_server].into_iter().flatten() {
        let path = pin.path.canonicalize()?;
        if !std::fs::symlink_metadata(path)?.is_file() {
            return Err("paired serving binary must resolve to regular file".into());
        }
    }
    let selected: Vec<_> = catalog
        .iter()
        .filter(|c| input.cases.is_empty() || input.cases.contains(&c.key))
        .collect();
    let use_cases = use_cases(input)?;
    let prefixes = if let Some(prefix) = input.prefix_tokens {
        vec![Some(prefix)]
    } else if input.prefix_sweep.is_empty() {
        vec![None]
    } else {
        input.prefix_sweep.iter().copied().map(Some).collect()
    };
    let total = selected
        .len()
        .checked_mul(use_cases.len())
        .and_then(|n| n.checked_mul(prefixes.len()))
        .filter(|n| *n <= 8192)
        .ok_or("cache matrix exceeds8192 cells")?;
    let corpus_bytes = use_cases
        .iter()
        .try_fold(0_usize, |total, item| -> DynResult<usize> {
            total
                .checked_add(serde_json::to_vec(item)?.len())
                .ok_or_else(|| "cache corpus projection overflow".into())
        })?;
    let projection = corpus_bytes
        .checked_mul(selected.len())
        .and_then(|n| n.checked_mul(prefixes.len()))
        .and_then(|n| {
            total
                .checked_mul(4096)
                .and_then(|overhead| n.checked_add(overhead))
        })
        .ok_or("cache projected matrix overflow")?;
    if projection > 64 * 1024 * 1024 {
        return Err("cache projected matrix exceeds64MiB".into());
    }
    let mut cells = Vec::with_capacity(total);
    for prefix in prefixes {
        for use_case in &use_cases {
            for case in &selected {
                let effective = effective(case, use_case.as_ref(), prefix, input)?;
                let model = model(&root, case, input.model_sha256.get(&case.key))?;
                let tasks = tasks(&effective, &model, input)?;
                let relative = use_case.as_ref().map_or_else(
                    || format!("{}-p{}", case.key, effective.prefix_tokens),
                    |u| format!("{}/{}-p{}", u.key, case.key, effective.prefix_tokens),
                );
                cells.push(json!({"key":case.key,"case":effective,"use_case":use_case,"model_observation":model,"output_relative":relative,"tasks":tasks}));
            }
        }
    }
    // An override can collapse distinct source sweep entries onto the same effective
    // prefix. Refuse output collisions instead of silently reusing a trial directory.
    let mut outputs = BTreeSet::new();
    if cells
        .iter()
        .any(|c| !outputs.insert(c["output_relative"].as_str().unwrap_or_default()))
    {
        return Err("cache plan output collision".into());
    }
    Ok(
        json!({"schema_version":1,"scope":"declared_cache_family_producer_plan_not_execution","catalog_sha256":hash(include_bytes!("catalog.json")),"input":input,"admitted_cache_root":root,"cells":cells,"cell_count":total,"baseline_comparison_scope":"serial1token_concurrent128tokens_skippy_declared_output_not_interchangeable","execution":null,"promotion":null,"missing_model_policy":"retain_row_without_launch","report_owner":"automation cache-family-report after final producer row projection"}),
    )
}
