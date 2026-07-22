// Benchmark candidate generation and discovery.

use super::{TuneBenchmarkCandidate, TuneBenchmarkRunRequest, TuneBenchmarkSpeculativeCandidate};

use crate::gpus::tune::{
    TuneBoolOrAutoValue, TuneField, TuneFieldStatus, TuneFlashAttentionValue, TuneKvCacheType,
    TunePlan, TuneRecommendedValue,
};

/// Generate benchmark candidates for a target.
pub(crate) fn benchmark_candidates(
    request: &TuneBenchmarkRunRequest<'_>,
    prepared: &crate::gpus::tune_apply::PreparedTunePlan,
) -> Vec<TuneBenchmarkCandidate> {
    let default_ctx = default_model_fit_u32(request, prepared, TuneField::CtxSize).unwrap_or(8192);
    let contexts = if request.ctx_sizes.is_empty() {
        default_context_sizes(default_ctx)
    } else {
        unique_positive(request.ctx_sizes)
    };
    let batches = if request.batch_sizes.is_empty() {
        vec![default_model_fit_u32(request, prepared, TuneField::Batch).unwrap_or(512)]
    } else {
        unique_positive(request.batch_sizes)
    };
    let ubatches = if request.ubatch_sizes.is_empty() {
        vec![default_model_fit_u32(request, prepared, TuneField::Ubatch).unwrap_or(128)]
    } else {
        unique_positive(request.ubatch_sizes)
    };
    let cache_type_k = recommended_cache_type(&prepared.plan, TuneField::CacheTypeK)
        .unwrap_or(TuneKvCacheType::Q8_0);
    let cache_type_v =
        recommended_cache_type(&prepared.plan, TuneField::CacheTypeV).unwrap_or(cache_type_k);
    let mmap_values = benchmark_mmap_values(request.mmap_values, &prepared.plan);
    let mlock_values = benchmark_mlock_values(request.mlock_values, &prepared.plan);
    let speculative_values = benchmark_speculative_values(request, prepared);
    let flash_attention_values = benchmark_flash_attention_values(request.flash_attention_values);

    let mut candidates = Vec::new();
    for &fa in &flash_attention_values {
        for ctx_size in &contexts {
            for &batch in &batches {
                for &ubatch in &ubatches {
                    if ubatch > batch {
                        continue;
                    }
                    for &mmap in &mmap_values {
                        for &mlock in &mlock_values {
                            for speculative in &speculative_values {
                                candidates.push(TuneBenchmarkCandidate {
                                    ctx_size: *ctx_size,
                                    batch,
                                    ubatch,
                                    cache_type_k,
                                    cache_type_v,
                                    mmap,
                                    mlock,
                                    speculative: speculative.clone(),
                                    flash_attention: fa,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    candidates
}

pub(crate) fn default_model_fit_u32(
    request: &TuneBenchmarkRunRequest<'_>,
    prepared: &crate::gpus::tune_apply::PreparedTunePlan,
    field: TuneField,
) -> Option<u32> {
    recommended_u32(&prepared.plan, field).or_else(|| {
        preserved_model_fit_u32(
            benchmark_model_entry(request.config, prepared),
            request.config.defaults.as_ref(),
            field,
        )
    })
}

pub(crate) fn benchmark_model_entry<'a>(
    config: &'a mesh_llm_config::MeshConfig,
    prepared: &crate::gpus::tune_apply::PreparedTunePlan,
) -> Option<&'a mesh_llm_config::ModelConfigEntry> {
    config
        .models
        .get(prepared.target.config_matches.first()?.row_index)
}

pub(crate) fn preserved_model_fit_u32(
    model_entry: Option<&mesh_llm_config::ModelConfigEntry>,
    defaults: Option<&mesh_llm_config::ModelConfigDefaults>,
    field: TuneField,
) -> Option<u32> {
    model_entry
        .and_then(|entry| entry.model_fit.as_ref())
        .and_then(|fit| match field {
            TuneField::CtxSize => fit.ctx_size,
            TuneField::Batch => fit.batch,
            TuneField::Ubatch => fit.ubatch,
            _ => None,
        })
        .or_else(|| {
            defaults
                .and_then(|defaults| defaults.model_fit.as_ref())
                .and_then(|fit| match field {
                    TuneField::CtxSize => fit.ctx_size,
                    TuneField::Batch => fit.batch,
                    TuneField::Ubatch => fit.ubatch,
                    _ => None,
                })
        })
}

pub(crate) fn default_context_sizes(planned: u32) -> Vec<u32> {
    let mut values = [4096, 8192, 16_384, 32_768, 65_536, planned]
        .into_iter()
        .filter(|value| *value > 0 && *value <= planned.max(4096))
        .collect::<Vec<_>>();
    values.sort_unstable();
    values.dedup();
    values
}

pub(crate) fn unique_positive(values: &[u32]) -> Vec<u32> {
    let mut values = values
        .iter()
        .copied()
        .filter(|value| *value > 0)
        .collect::<Vec<_>>();
    values.sort_unstable();
    values.dedup();
    values
}

pub(crate) fn benchmark_mmap_values(
    requested: &[mesh_llm_cli::benchmark::BenchmarkBoolOrAuto],
    _plan: &TunePlan,
) -> Vec<TuneBoolOrAutoValue> {
    if requested.is_empty() {
        return vec![
            TuneBoolOrAutoValue::Auto,
            TuneBoolOrAutoValue::Enabled,
            TuneBoolOrAutoValue::Disabled,
        ];
    }
    let mut values = requested
        .iter()
        .copied()
        .map(|value| match value {
            mesh_llm_cli::benchmark::BenchmarkBoolOrAuto::Auto => TuneBoolOrAutoValue::Auto,
            mesh_llm_cli::benchmark::BenchmarkBoolOrAuto::Enabled => TuneBoolOrAutoValue::Enabled,
            mesh_llm_cli::benchmark::BenchmarkBoolOrAuto::Disabled => TuneBoolOrAutoValue::Disabled,
        })
        .collect::<Vec<_>>();
    values.sort_by_key(|value| match value {
        TuneBoolOrAutoValue::Auto => 0,
        TuneBoolOrAutoValue::Enabled => 1,
        TuneBoolOrAutoValue::Disabled => 2,
    });
    values.dedup();
    values
}

pub(crate) fn benchmark_mlock_values(
    requested: &[mesh_llm_cli::benchmark::BenchmarkBool],
    plan: &TunePlan,
) -> Vec<bool> {
    if requested.is_empty() {
        return if recommended_bool(plan, TuneField::Mlock).unwrap_or(false) {
            vec![false, true]
        } else {
            vec![false]
        };
    }
    let mut values = requested
        .iter()
        .copied()
        .map(|value| match value {
            mesh_llm_cli::benchmark::BenchmarkBool::Enabled => true,
            mesh_llm_cli::benchmark::BenchmarkBool::Disabled => false,
        })
        .collect::<Vec<_>>();
    values.sort_unstable();
    values.dedup();
    values
}

pub(crate) fn benchmark_flash_attention_values(
    requested: &[mesh_llm_cli::benchmark::BenchmarkFlashAttention],
) -> Vec<Option<TuneFlashAttentionValue>> {
    if requested.is_empty() {
        return vec![None];
    }
    let mut values = requested
        .iter()
        .copied()
        .map(|value| match value {
            mesh_llm_cli::benchmark::BenchmarkFlashAttention::On => {
                Some(TuneFlashAttentionValue::Enabled)
            }
            mesh_llm_cli::benchmark::BenchmarkFlashAttention::Off => {
                Some(TuneFlashAttentionValue::Disabled)
            }
        })
        .collect::<Vec<_>>();
    values.sort_unstable_by_key(|v| v.is_none());
    values.dedup();
    values
}

pub(crate) fn benchmark_speculative_values(
    request: &TuneBenchmarkRunRequest<'_>,
    prepared: &crate::gpus::tune_apply::PreparedTunePlan,
) -> Vec<TuneBenchmarkSpeculativeCandidate> {
    if request.no_speculative_tune {
        return vec![TuneBenchmarkSpeculativeCandidate::Disabled];
    }
    let requested = requested_speculative_types(request.speculative_types);
    let mut candidates = Vec::new();
    for requested_type in requested {
        match requested_type {
            mesh_llm_cli::benchmark::BenchmarkSpeculativeType::Auto => {
                push_auto_speculative_candidates(&mut candidates, request, prepared);
            }
            mesh_llm_cli::benchmark::BenchmarkSpeculativeType::Disabled => {
                candidates.push(TuneBenchmarkSpeculativeCandidate::Disabled);
            }
            mesh_llm_cli::benchmark::BenchmarkSpeculativeType::Mtp => {
                push_mtp_speculative_candidates(&mut candidates, request, prepared);
            }
            mesh_llm_cli::benchmark::BenchmarkSpeculativeType::Draft => {
                push_draft_speculative_candidates(&mut candidates, request, prepared);
            }
            mesh_llm_cli::benchmark::BenchmarkSpeculativeType::MtpNgram => {
                push_ngram_speculative_candidates(&mut candidates, request);
            }
        }
    }
    dedup_speculative_candidates(candidates)
}

pub(crate) fn requested_speculative_types(
    requested: &[mesh_llm_cli::benchmark::BenchmarkSpeculativeType],
) -> Vec<mesh_llm_cli::benchmark::BenchmarkSpeculativeType> {
    if requested.is_empty() {
        return vec![mesh_llm_cli::benchmark::BenchmarkSpeculativeType::Auto];
    }
    let mut values = requested.to_vec();
    values.sort_by_key(|value| speculative_type_priority(*value));
    values.dedup();
    values
}

pub(crate) fn speculative_type_priority(
    value: mesh_llm_cli::benchmark::BenchmarkSpeculativeType,
) -> u8 {
    match value {
        mesh_llm_cli::benchmark::BenchmarkSpeculativeType::Auto => 0,
        mesh_llm_cli::benchmark::BenchmarkSpeculativeType::Mtp => 1,
        mesh_llm_cli::benchmark::BenchmarkSpeculativeType::Draft => 2,
        mesh_llm_cli::benchmark::BenchmarkSpeculativeType::MtpNgram => 3,
        mesh_llm_cli::benchmark::BenchmarkSpeculativeType::Disabled => 4,
    }
}

pub(crate) fn push_auto_speculative_candidates(
    candidates: &mut Vec<TuneBenchmarkSpeculativeCandidate>,
    request: &TuneBenchmarkRunRequest<'_>,
    prepared: &crate::gpus::tune_apply::PreparedTunePlan,
) {
    if looks_like_mtp_target(prepared) {
        push_mtp_speculative_candidates(candidates, request, prepared);
        push_ngram_speculative_candidates(candidates, request);
    }
    push_draft_speculative_candidates(candidates, request, prepared);
    candidates.push(TuneBenchmarkSpeculativeCandidate::Disabled);
}

pub(crate) fn push_mtp_speculative_candidates(
    candidates: &mut Vec<TuneBenchmarkSpeculativeCandidate>,
    request: &TuneBenchmarkRunRequest<'_>,
    prepared: &crate::gpus::tune_apply::PreparedTunePlan,
) {
    let draft_models = discover_draft_model_candidates(request, prepared);
    let draft_models = if draft_models.is_empty() {
        vec![None]
    } else {
        draft_models.into_iter().map(Some).collect()
    };
    let max_tokens = positive_or_default(request.spec_draft_max_tokens, &[2, 3, 4]);
    let min_tokens = values_or_default_allow_zero(request.spec_draft_min_tokens, &[0]);
    let acceptance_thresholds =
        optional_probability_values(request.spec_draft_acceptance_threshold);
    let split_probabilities = optional_probability_values(request.spec_draft_split_probability);
    for draft_model in draft_models {
        for draft_max_tokens in &max_tokens {
            for draft_min_tokens in &min_tokens {
                if *draft_min_tokens > *draft_max_tokens {
                    continue;
                }
                push_mtp_threshold_cross_product(
                    candidates,
                    draft_model.clone(),
                    *draft_max_tokens,
                    *draft_min_tokens,
                    &acceptance_thresholds,
                    &split_probabilities,
                );
            }
        }
    }
}

pub(crate) fn push_mtp_threshold_cross_product(
    candidates: &mut Vec<TuneBenchmarkSpeculativeCandidate>,
    draft_model: Option<String>,
    draft_max_tokens: u32,
    draft_min_tokens: u32,
    acceptance_thresholds: &[f64],
    split_probabilities: &[f64],
) {
    push_threshold_cross_product(
        candidates,
        acceptance_thresholds,
        split_probabilities,
        |draft_acceptance_threshold, draft_split_probability| {
            TuneBenchmarkSpeculativeCandidate::Mtp {
                draft_model: draft_model.clone(),
                draft_max_tokens,
                draft_min_tokens,
                draft_acceptance_threshold,
                draft_split_probability,
            }
        },
    );
}

pub(crate) fn push_draft_speculative_candidates(
    candidates: &mut Vec<TuneBenchmarkSpeculativeCandidate>,
    request: &TuneBenchmarkRunRequest<'_>,
    prepared: &crate::gpus::tune_apply::PreparedTunePlan,
) {
    let draft_models = discover_draft_model_candidates(request, prepared);
    if draft_models.is_empty() {
        return;
    }
    let max_tokens = positive_or_default(request.spec_draft_max_tokens, &[4, 8, 16]);
    let min_tokens = optional_positive_values(request.spec_draft_min_tokens);
    let acceptance_thresholds =
        optional_probability_values(request.spec_draft_acceptance_threshold);
    let split_probabilities = optional_probability_values(request.spec_draft_split_probability);
    for draft_model in draft_models {
        for draft_max_tokens in &max_tokens {
            if min_tokens.is_empty() {
                push_draft_threshold_cross_product(
                    candidates,
                    draft_model.clone(),
                    *draft_max_tokens,
                    None,
                    &acceptance_thresholds,
                    &split_probabilities,
                );
                continue;
            }
            for draft_min_tokens in &min_tokens {
                if draft_min_tokens <= draft_max_tokens {
                    push_draft_threshold_cross_product(
                        candidates,
                        draft_model.clone(),
                        *draft_max_tokens,
                        Some(*draft_min_tokens),
                        &acceptance_thresholds,
                        &split_probabilities,
                    );
                }
            }
        }
    }
}

pub(crate) fn push_draft_threshold_cross_product(
    candidates: &mut Vec<TuneBenchmarkSpeculativeCandidate>,
    draft_model: String,
    draft_max_tokens: u32,
    draft_min_tokens: Option<u32>,
    acceptance_thresholds: &[f64],
    split_probabilities: &[f64],
) {
    push_threshold_cross_product(
        candidates,
        acceptance_thresholds,
        split_probabilities,
        |draft_acceptance_threshold, draft_split_probability| {
            TuneBenchmarkSpeculativeCandidate::Draft {
                draft_model: draft_model.clone(),
                draft_max_tokens,
                draft_min_tokens,
                draft_acceptance_threshold,
                draft_split_probability,
            }
        },
    );
}

fn push_threshold_cross_product<F>(
    candidates: &mut Vec<TuneBenchmarkSpeculativeCandidate>,
    acceptance_thresholds: &[f64],
    split_probabilities: &[f64],
    mut build_candidate: F,
) where
    F: FnMut(Option<f64>, Option<f64>) -> TuneBenchmarkSpeculativeCandidate,
{
    if acceptance_thresholds.is_empty() && split_probabilities.is_empty() {
        candidates.push(build_candidate(None, None));
        return;
    }
    if acceptance_thresholds.is_empty() {
        for split_probability in split_probabilities {
            candidates.push(build_candidate(None, Some(*split_probability)));
        }
        return;
    }
    if split_probabilities.is_empty() {
        for acceptance_threshold in acceptance_thresholds {
            candidates.push(build_candidate(Some(*acceptance_threshold), None));
        }
        return;
    }
    for acceptance_threshold in acceptance_thresholds {
        for split_probability in split_probabilities {
            candidates.push(build_candidate(
                Some(*acceptance_threshold),
                Some(*split_probability),
            ));
        }
    }
}

pub(crate) fn push_ngram_speculative_candidates(
    candidates: &mut Vec<TuneBenchmarkSpeculativeCandidate>,
    request: &TuneBenchmarkRunRequest<'_>,
) {
    let ngram_min_values = positive_or_default(request.spec_ngram_min, &[2]);
    let ngram_max_values = positive_or_default(request.spec_ngram_max, &[4]);
    for ngram_min in &ngram_min_values {
        for ngram_max in &ngram_max_values {
            if ngram_min <= ngram_max {
                candidates.push(TuneBenchmarkSpeculativeCandidate::MtpNgram {
                    ngram_min: *ngram_min,
                    ngram_max: *ngram_max,
                });
            }
        }
    }
}

pub(crate) fn positive_or_default(requested: &[u32], defaults: &[u32]) -> Vec<u32> {
    if requested.is_empty() {
        return defaults.to_vec();
    }
    unique_positive(requested)
}

pub(crate) fn optional_positive_values(requested: &[u32]) -> Vec<u32> {
    if requested.is_empty() {
        return Vec::new();
    }
    unique_positive(requested)
}

pub(crate) fn optional_probability_values(requested: &[f64]) -> Vec<f64> {
    if requested.is_empty() {
        return Vec::new();
    }
    let mut values = requested
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value >= 0.0 && *value <= 1.0)
        .collect::<Vec<_>>();
    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values.dedup_by(|a, b| a == b);
    values
}

pub(crate) fn values_or_default_allow_zero(requested: &[u32], defaults: &[u32]) -> Vec<u32> {
    if requested.is_empty() {
        return defaults.to_vec();
    }
    let mut values = requested.to_vec();
    values.sort_unstable();
    values.dedup();
    values
}

pub(crate) fn discover_draft_model_candidates(
    request: &TuneBenchmarkRunRequest<'_>,
    prepared: &crate::gpus::tune_apply::PreparedTunePlan,
) -> Vec<String> {
    let mut candidates = request
        .spec_draft_models
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>();
    if let Some(model_entry) = benchmark_model_entry(request.config, prepared)
        && let Some(path) = model_entry
            .speculative
            .as_ref()
            .and_then(|speculative| speculative.draft_model.as_ref())
    {
        candidates.push(path.clone());
    }
    if let Some(path) = request
        .config
        .defaults
        .as_ref()
        .and_then(|defaults| defaults.speculative.as_ref())
        .and_then(|speculative| speculative.draft_model.as_ref())
    {
        candidates.push(path.clone());
    }
    candidates.extend(discover_sibling_draft_models(
        &prepared.target.resolved_path,
    ));
    candidates.sort();
    candidates.dedup();
    candidates
}

pub(crate) fn discover_sibling_draft_models(model_path: &std::path::Path) -> Vec<String> {
    let Some(parent) = model_path.parent() else {
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(parent) else {
        return Vec::new();
    };
    let model_file_name = model_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path != model_path)
        .filter(|path| {
            path.extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
        })
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| looks_like_draft_model_name(name, model_file_name))
        })
        .map(|path| path.display().to_string())
        .collect()
}

pub(crate) fn looks_like_draft_model_name(name: &str, target_name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    let target_name = target_name.to_ascii_lowercase();
    (name.contains("draft") || name.contains("eagle"))
        && !target_name.is_empty()
        && shares_model_family_token(&name, &target_name)
}

pub(crate) fn shares_model_family_token(left: &str, right: &str) -> bool {
    left.split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|token| token.len() >= 4)
        .any(|token| right.contains(token))
}

pub(crate) fn looks_like_mtp_target(prepared: &crate::gpus::tune_apply::PreparedTunePlan) -> bool {
    [
        &prepared.target.requested_input,
        &prepared.target.canonical_model_ref,
    ]
    .into_iter()
    .any(|value| mesh_llm_system::util::contains_mtp_marker_str(value))
        || prepared
            .target
            .resolved_path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(mesh_llm_system::util::contains_mtp_marker_str)
}

pub(crate) fn dedup_speculative_candidates(
    mut candidates: Vec<TuneBenchmarkSpeculativeCandidate>,
) -> Vec<TuneBenchmarkSpeculativeCandidate> {
    candidates.sort_by_key(speculative_candidate_sort_key);
    candidates.dedup();
    candidates
}

pub(crate) fn speculative_candidate_sort_key(
    candidate: &TuneBenchmarkSpeculativeCandidate,
) -> String {
    fn fmt_prob(value: Option<f64>) -> String {
        value
            .map(|v| format!("{v:.6}"))
            .unwrap_or_else(|| "-".to_string())
    }
    match candidate {
        TuneBenchmarkSpeculativeCandidate::Mtp {
            draft_model,
            draft_max_tokens,
            draft_min_tokens,
            draft_acceptance_threshold,
            draft_split_probability,
        } => format!(
            "0:mtp:{}:{draft_max_tokens}:{draft_min_tokens}:{}:{}",
            draft_model.as_deref().unwrap_or(""),
            fmt_prob(*draft_acceptance_threshold),
            fmt_prob(*draft_split_probability),
        ),
        TuneBenchmarkSpeculativeCandidate::Draft {
            draft_model,
            draft_max_tokens,
            draft_min_tokens,
            draft_acceptance_threshold,
            draft_split_probability,
        } => format!(
            "1:draft:{draft_model}:{draft_max_tokens}:{}:{}:{}",
            draft_min_tokens.unwrap_or(0),
            fmt_prob(*draft_acceptance_threshold),
            fmt_prob(*draft_split_probability),
        ),
        TuneBenchmarkSpeculativeCandidate::MtpNgram {
            ngram_min,
            ngram_max,
        } => format!("2:mtp-ngram:{ngram_min}:{ngram_max}"),
        TuneBenchmarkSpeculativeCandidate::Disabled => "9:disabled".to_string(),
    }
}

pub(crate) fn recommended_u32(plan: &TunePlan, field: TuneField) -> Option<u32> {
    tune_field_recommendation(plan, field).and_then(|recommendation| match recommendation {
        TuneRecommendedValue::ContextSize(value)
        | TuneRecommendedValue::Batch(value)
        | TuneRecommendedValue::Ubatch(value) => Some(*value),
        _ => None,
    })
}

pub(crate) fn recommended_bool(plan: &TunePlan, field: TuneField) -> Option<bool> {
    tune_field_recommendation(plan, field).and_then(|recommendation| match recommendation {
        TuneRecommendedValue::Bool(value) => Some(*value),
        _ => None,
    })
}

pub(crate) fn recommended_cache_type(plan: &TunePlan, field: TuneField) -> Option<TuneKvCacheType> {
    tune_field_recommendation(plan, field).and_then(|recommendation| match recommendation {
        TuneRecommendedValue::KvCacheType(value) => Some(*value),
        _ => None,
    })
}

fn tune_field_recommendation(plan: &TunePlan, field: TuneField) -> Option<&TuneRecommendedValue> {
    plan.field_statuses.iter().find_map(|status| match status {
        TuneFieldStatus::Applied { recommendation, .. }
        | TuneFieldStatus::ReportOnly { recommendation, .. }
            if recommendation.field == field =>
        {
            Some(&recommendation.value)
        }
        _ => None,
    })
}
