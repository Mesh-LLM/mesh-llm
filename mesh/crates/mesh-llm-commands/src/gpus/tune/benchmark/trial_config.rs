// Trial configuration generation.

use super::{TuneBenchmarkCandidate, TuneBenchmarkSpeculativeCandidate};
use crate::gpus::tune_apply::PreparedTunePlan;

pub(crate) fn trial_config(
    config: &mesh_llm_config::MeshConfig,
    prepared: &PreparedTunePlan,
    candidate: &TuneBenchmarkCandidate,
) -> anyhow::Result<String> {
    let mut doc = toml_edit::DocumentMut::new();
    doc["version"] = toml_edit::value(1);
    apply_trial_runtime_config(&mut doc, config)?;

    let mut table = toml_edit::Table::new();
    table["model"] = toml_edit::value(crate::gpus::tune_apply::appended_model_ref(
        &prepared.target,
    ));
    crate::gpus::tune_apply::apply_config_edits(&mut table, &prepared.plan.config_edits())?;
    apply_resolved_model_path(&mut table, prepared)?;
    apply_candidate_overrides(&mut table, candidate)?;

    let mut models = toml_edit::ArrayOfTables::new();
    models.push(table);
    doc["models"] = toml_edit::Item::ArrayOfTables(models);
    Ok(doc.to_string())
}

pub(crate) fn apply_trial_runtime_config(
    doc: &mut toml_edit::DocumentMut,
    config: &mesh_llm_config::MeshConfig,
) -> anyhow::Result<()> {
    let runtime = ensure_trial_subtable(doc.as_table_mut(), "runtime")?;

    runtime["debug"] = toml_edit::value(config.runtime.debug);
    runtime["listen_all"] = toml_edit::value(config.runtime.listen_all);
    runtime["reconcile_model_targets"] = toml_edit::value(config.runtime.reconcile_model_targets);
    runtime["reconcile_model_target_demand_upgrades"] =
        toml_edit::value(config.runtime.reconcile_model_target_demand_upgrades);

    if config.runtime.native_runtime.mesh_version.is_some()
        || config.runtime.native_runtime.skippy_abi.is_some()
        || config.runtime.native_runtime.selection.is_some()
    {
        let native_runtime = ensure_trial_subtable(runtime, "native_runtime")?;

        if let Some(mesh_version) = config.runtime.native_runtime.mesh_version.as_deref() {
            native_runtime["mesh_version"] = toml_edit::value(mesh_version);
        }
        if let Some(skippy_abi) = config.runtime.native_runtime.skippy_abi.as_deref() {
            native_runtime["skippy_abi"] = toml_edit::value(skippy_abi);
        }
        if let Some(selection) = config.runtime.native_runtime.selection.as_deref() {
            native_runtime["selection"] = toml_edit::value(selection);
        }
    }

    Ok(())
}

pub(crate) fn apply_resolved_model_path(
    table: &mut toml_edit::Table,
    prepared: &PreparedTunePlan,
) -> anyhow::Result<()> {
    let hardware = ensure_trial_subtable(table, "hardware")?;
    hardware["model_path"] = toml_edit::value(prepared.target.resolved_path.display().to_string());
    Ok(())
}

pub(crate) fn apply_candidate_overrides(
    table: &mut toml_edit::Table,
    candidate: &TuneBenchmarkCandidate,
) -> anyhow::Result<()> {
    let model_fit = ensure_trial_subtable(table, "model_fit")?;
    model_fit["ctx_size"] = toml_edit::value(i64::from(candidate.ctx_size));
    model_fit["batch"] = toml_edit::value(i64::from(candidate.batch));
    model_fit["ubatch"] = toml_edit::value(i64::from(candidate.ubatch));
    model_fit["cache_type_k"] = toml_edit::value(render_cache_type(candidate.cache_type_k));
    model_fit["cache_type_v"] = toml_edit::value(render_cache_type(candidate.cache_type_v));
    if let Some(fa) = candidate.flash_attention {
        model_fit["flash_attention"] =
            toml_edit::value(crate::gpus::tune_apply::render_flash_attention(fa));
    }
    let hardware = ensure_trial_subtable(table, "hardware")?;
    hardware["mmap"] =
        toml_edit::value(crate::gpus::tune_apply::render_bool_or_auto(candidate.mmap));
    hardware["mlock"] = toml_edit::value(candidate.mlock);
    apply_speculative_overrides(table, &candidate.speculative)?;
    Ok(())
}

pub(crate) fn apply_speculative_overrides(
    table: &mut toml_edit::Table,
    speculative: &TuneBenchmarkSpeculativeCandidate,
) -> anyhow::Result<()> {
    let spec_table = ensure_trial_subtable(table, "speculative")?;
    match speculative {
        TuneBenchmarkSpeculativeCandidate::Disabled => {
            spec_table["strategy"] = toml_edit::value("disabled");
            spec_table["mode"] = toml_edit::value("disabled");
        }
        TuneBenchmarkSpeculativeCandidate::Mtp {
            draft_model,
            draft_max_tokens,
            draft_min_tokens,
            draft_acceptance_threshold,
            draft_split_probability,
        } => {
            spec_table["strategy"] = toml_edit::value("mtp");
            spec_table["mode"] = toml_edit::value("auto");
            if let Some(draft_model) = draft_model {
                let key = if draft_model.contains(':') {
                    "draft_model"
                } else {
                    "draft_model_path"
                };
                spec_table[key] = toml_edit::value(draft_model.as_str());
                spec_table["draft_selection_policy"] = toml_edit::value("manual");
                spec_table["pairing_fault"] = toml_edit::value("fail_closed");
            }
            spec_table["draft_max_tokens"] = toml_edit::value(i64::from(*draft_max_tokens));
            spec_table["draft_min_tokens"] = toml_edit::value(i64::from(*draft_min_tokens));
            if let Some(threshold) = draft_acceptance_threshold {
                spec_table["draft_acceptance_threshold"] = toml_edit::value(*threshold);
            }
            if let Some(probability) = draft_split_probability {
                spec_table["draft_split_probability"] = toml_edit::value(*probability);
            }
        }
        TuneBenchmarkSpeculativeCandidate::Draft {
            draft_model,
            draft_max_tokens,
            draft_min_tokens,
            draft_acceptance_threshold,
            draft_split_probability,
        } => {
            spec_table["strategy"] = toml_edit::value("disabled");
            spec_table["mode"] = toml_edit::value("draft");
            let key = if draft_model.contains(':') {
                "draft_model"
            } else {
                "draft_model_path"
            };
            spec_table[key] = toml_edit::value(draft_model.as_str());
            spec_table["draft_selection_policy"] = toml_edit::value("manual");
            spec_table["pairing_fault"] = toml_edit::value("fail_closed");
            spec_table["draft_max_tokens"] = toml_edit::value(i64::from(*draft_max_tokens));
            if let Some(draft_min_tokens) = draft_min_tokens {
                spec_table["draft_min_tokens"] = toml_edit::value(i64::from(*draft_min_tokens));
            }
            if let Some(threshold) = draft_acceptance_threshold {
                spec_table["draft_acceptance_threshold"] = toml_edit::value(*threshold);
            }
            if let Some(probability) = draft_split_probability {
                spec_table["draft_split_probability"] = toml_edit::value(*probability);
            }
        }
        TuneBenchmarkSpeculativeCandidate::MtpNgram {
            ngram_min,
            ngram_max,
        } => {
            spec_table["strategy"] = toml_edit::value("mtp");
            spec_table["mode"] = toml_edit::value("auto");
            spec_table["ngram_min"] = toml_edit::value(i64::from(*ngram_min));
            spec_table["ngram_max"] = toml_edit::value(i64::from(*ngram_max));
        }
    }
    Ok(())
}

pub(crate) fn ensure_trial_subtable<'a>(
    table: &'a mut toml_edit::Table,
    key: &str,
) -> anyhow::Result<&'a mut toml_edit::Table> {
    if !table.contains_key(key) {
        table[key] = toml_edit::Item::Table(toml_edit::Table::new());
    }
    table[key]
        .as_table_mut()
        .ok_or_else(|| anyhow::anyhow!("config key `models[].{key}` is not a TOML table"))
}

pub(crate) fn render_cache_type(cache_type: crate::gpus::tune::TuneKvCacheType) -> String {
    match cache_type {
        crate::gpus::tune::TuneKvCacheType::F16 => "f16",
        crate::gpus::tune::TuneKvCacheType::Q8_0 => "q8_0",
        crate::gpus::tune::TuneKvCacheType::Q4_0 => "q4_0",
    }
    .to_string()
}
