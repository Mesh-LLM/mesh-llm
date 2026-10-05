use super::tune::{
    TuneApplyMode, TuneBoolOrAutoValue, TuneConfigEdit, TuneFieldStatus, TuneFlashAttentionValue,
    TuneGpuLayersValue, TuneKvCacheType, TunePlan,
};
use super::tune_resolver::{LocalTargetSource, ResolvedTuneTarget, TuneTargetSelection};
use anyhow::{Context, Result, anyhow, bail};
use mesh_llm_config::ConfigStore;
use std::collections::BTreeMap;
use toml_edit::{ArrayOfTables, DocumentMut, Item, Table, value};

#[derive(Clone, Debug)]
pub(crate) struct PreparedTunePlan {
    pub(crate) target: ResolvedTuneTarget,
    pub(crate) plan: TunePlan,
}

impl PreparedTunePlan {
    pub(crate) fn new(target: ResolvedTuneTarget, plan: TunePlan) -> Self {
        Self { target, plan }
    }
}

pub(crate) fn apply_prepared_tune_plans(
    store: &ConfigStore,
    prepared: &[PreparedTunePlan],
) -> Result<usize> {
    let writable_targets = collect_writable_targets(prepared)?;
    if writable_targets.is_empty() {
        return Ok(0);
    }

    store.edit_preserving(|doc| {
        let models = ensure_models_array(doc)?;
        for prepared in &writable_targets {
            let model_table = resolve_model_table(models, prepared)?;
            apply_config_edits(model_table, &prepared.plan.config_edits())?;
        }
        Ok(())
    })?;

    Ok(writable_targets.len())
}

fn collect_writable_targets(prepared: &[PreparedTunePlan]) -> Result<Vec<&PreparedTunePlan>> {
    let mut writable_targets = Vec::new();
    let mut touched_rows = BTreeMap::new();

    for prepared in prepared {
        if prepared.target.config_matches.len() > 1 {
            let configured_models = prepared
                .target
                .config_matches
                .iter()
                .map(|matched| matched.configured_model.clone())
                .collect::<Vec<_>>()
                .join(", ");
            bail!(
                "tune apply aborted: requested target `{}` collides with multiple config rows for `{}` ({configured_models})",
                prepared.target.requested_input,
                prepared.target.canonical_model_ref,
            );
        }

        if matches!(
            prepared.plan.apply_mode,
            TuneApplyMode::Review | TuneApplyMode::LaunchArgs
        ) {
            continue;
        }
        if plan_has_error(&prepared.plan) || prepared.plan.config_edits().is_empty() {
            continue;
        }

        if let Some(config_match) = prepared.target.config_matches.first()
            && let Some(first_target) = touched_rows.insert(
                config_match.row_index,
                prepared.target.requested_input.clone(),
            )
        {
            bail!(
                "tune apply aborted: requested targets `{first_target}` and `{}` both map to config row {}",
                prepared.target.requested_input,
                config_match.row_index + 1,
            );
        }

        if matches!(prepared.target.selection, TuneTargetSelection::Configured)
            && prepared.target.config_matches.is_empty()
        {
            bail!(
                "tune apply aborted: configured target `{}` no longer maps to a config row",
                prepared.target.requested_input,
            );
        }

        writable_targets.push(prepared);
    }

    Ok(writable_targets)
}

fn plan_has_error(plan: &TunePlan) -> bool {
    plan.field_statuses
        .iter()
        .any(|status| matches!(status, TuneFieldStatus::Error { .. }))
}

fn resolve_model_table<'a>(
    models: &'a mut ArrayOfTables,
    prepared: &PreparedTunePlan,
) -> Result<&'a mut Table> {
    if let Some(config_match) = prepared.target.config_matches.first() {
        return models.get_mut(config_match.row_index).ok_or_else(|| {
            anyhow!(
                "config row {} disappeared while applying tune edits",
                config_match.row_index + 1,
            )
        });
    }

    let mut table = Table::new();
    table["model"] = value(appended_model_ref(&prepared.target));
    models.push(table);
    let appended_index = models.len().saturating_sub(1);
    models.get_mut(appended_index).ok_or_else(|| {
        anyhow!(
            "failed to append config row for requested target `{}`",
            prepared.target.requested_input,
        )
    })
}

pub(crate) fn appended_model_ref(target: &ResolvedTuneTarget) -> String {
    match &target.local_source {
        LocalTargetSource::HuggingFaceCache { canonical_ref } => canonical_ref.clone(),
        LocalTargetSource::FilesystemPath { .. } => target.resolved_path.display().to_string(),
    }
}

pub(crate) fn apply_config_edits(table: &mut Table, edits: &[TuneConfigEdit]) -> Result<()> {
    for edit in edits {
        match edit {
            TuneConfigEdit::SetModelFitCacheTypeK(value_kind) => {
                ensure_subtable(table, "model_fit")?["cache_type_k"] =
                    value(render_kv_cache_type(*value_kind));
            }
            TuneConfigEdit::SetModelFitCacheTypeV(value_kind) => {
                ensure_subtable(table, "model_fit")?["cache_type_v"] =
                    value(render_kv_cache_type(*value_kind));
            }
            TuneConfigEdit::SetModelFitFlashAttention(value_kind) => {
                ensure_subtable(table, "model_fit")?["flash_attention"] =
                    value(render_flash_attention(*value_kind));
            }
            TuneConfigEdit::SetModelFitCtxSize(ctx_size) => {
                ensure_subtable(table, "model_fit")?["ctx_size"] = value(i64::from(*ctx_size));
            }
            TuneConfigEdit::SetModelFitBatch(batch) => {
                ensure_subtable(table, "model_fit")?["batch"] = value(i64::from(*batch));
            }
            TuneConfigEdit::SetModelFitUbatch(ubatch) => {
                ensure_subtable(table, "model_fit")?["ubatch"] = value(i64::from(*ubatch));
            }
            TuneConfigEdit::SetHardwareGpuLayers(gpu_layers) => {
                ensure_subtable(table, "hardware")?["gpu_layers"] =
                    value(render_gpu_layers(*gpu_layers));
            }
            TuneConfigEdit::SetHardwareFitTargetMib(fit_target_mib) => {
                ensure_subtable(table, "hardware")?["fit_target_mib"] = value(
                    i64::try_from(*fit_target_mib)
                        .context("fit_target_mib exceeded TOML integer range")?,
                );
            }
            TuneConfigEdit::SetHardwareMmap(mmap) => {
                ensure_subtable(table, "hardware")?["mmap"] = value(render_bool_or_auto(*mmap));
            }
            TuneConfigEdit::SetHardwareMlock(mlock) => {
                ensure_subtable(table, "hardware")?["mlock"] = value(*mlock);
            }
        }
    }
    Ok(())
}

fn render_kv_cache_type(value_kind: TuneKvCacheType) -> &'static str {
    match value_kind {
        TuneKvCacheType::F16 => "f16",
        TuneKvCacheType::Q8_0 => "q8_0",
        TuneKvCacheType::Q4_0 => "q4_0",
    }
}

pub(crate) fn render_flash_attention(value_kind: TuneFlashAttentionValue) -> &'static str {
    match value_kind {
        TuneFlashAttentionValue::Enabled => "enabled",
        TuneFlashAttentionValue::Disabled => "disabled",
    }
}

fn render_gpu_layers(value_kind: TuneGpuLayersValue) -> i64 {
    match value_kind {
        TuneGpuLayersValue::All => -1,
        TuneGpuLayersValue::Count(value) => i64::from(value),
    }
}

pub(crate) fn render_bool_or_auto(value_kind: TuneBoolOrAutoValue) -> toml_edit::Value {
    match value_kind {
        TuneBoolOrAutoValue::Enabled => toml_edit::Value::from(true),
        TuneBoolOrAutoValue::Disabled => toml_edit::Value::from(false),
        TuneBoolOrAutoValue::Auto => toml_edit::Value::from("auto"),
    }
}

fn ensure_models_array(doc: &mut DocumentMut) -> Result<&mut ArrayOfTables> {
    if !doc.as_table().contains_key("models") {
        doc["models"] = Item::ArrayOfTables(ArrayOfTables::new());
    }
    doc["models"]
        .as_array_of_tables_mut()
        .ok_or_else(|| anyhow!("config key `models` is not a TOML array of tables"))
}

fn ensure_subtable<'a>(table: &'a mut Table, key: &str) -> Result<&'a mut Table> {
    if !table.contains_key(key) {
        table[key] = Item::Table(Table::new());
    }
    table[key]
        .as_table_mut()
        .ok_or_else(|| anyhow!("config key `models[].{key}` is not a TOML table"))
}
