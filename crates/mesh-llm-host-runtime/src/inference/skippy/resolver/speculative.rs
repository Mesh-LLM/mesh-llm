use std::path::{Path, PathBuf};

use crate::models::find_model_path;
use anyhow::{Result, bail};
use mesh_llm_system::util::validate_draft_min_max;
use model_artifact::gguf::{scan_gguf_compact_meta, scan_gguf_tensor_names_any};
use skippy_runtime::package::{PackageGenerationInfo, PackageSpeculativeDecodingInfo};
use skippy_topology::infer_family_capability;

use super::support::{pick_owned, pick_string, pick_string_owned};
use super::types::ResolvedSpeculativeConfig;
use crate::plugin::{BoolOrAuto, SpeculativeConfig};

pub(super) fn resolve_speculative_config(
    model_config: Option<&SpeculativeConfig>,
    global_config: Option<&SpeculativeConfig>,
    model_id: &str,
    model_path: &Path,
    package_generation: Option<&PackageGenerationInfo>,
) -> Result<ResolvedSpeculativeConfig> {
    let spec_default = pick_owned(
        model_config.and_then(|config| config.spec_default.as_ref()),
        global_config.and_then(|config| config.spec_default.as_ref()),
    );
    if matches!(spec_default, Some(BoolOrAuto::Bool(true))) {
        unsupported_speculative_field("speculative.spec_default = true")?;
    }
    let has_explicit_strategy = model_config
        .and_then(|config| config.strategy.as_ref())
        .is_some()
        || global_config
            .and_then(|config| config.strategy.as_ref())
            .is_some();
    let auto_defaults_enabled =
        !matches!(spec_default, Some(BoolOrAuto::Bool(false))) || has_explicit_strategy;
    let mut draft_model_path = pick_owned(
        model_config.and_then(|config| config.draft_model.clone()),
        global_config.and_then(|config| config.draft_model.clone()),
    )
    .map(resolve_draft_model_path)
    .map(PathBuf::from);
    let supports_native_mtp = package_generation_supports_native_mtp(package_generation)
        || direct_gguf_supports_native_mtp(model_path)
        || draft_model_path
            .as_ref()
            .is_some_and(|path| path.is_file() && direct_gguf_supports_native_mtp(path));
    let strategy = pick_string_owned(
        model_config.and_then(|config| config.strategy.as_deref()),
        global_config.and_then(|config| config.strategy.as_deref()),
        Some("auto"),
    );
    let native_mtp_enabled = match strategy.as_str() {
        "auto" => {
            auto_defaults_enabled
                && package_generation_or_direct_default_supports_native_mtp(
                    package_generation,
                    model_path,
                )
        }
        "mtp" => {
            if !supports_native_mtp {
                bail!("skippy speculative.strategy = \"mtp\" requires proven native MTP support");
            }
            true
        }
        "disabled" => false,
        _ => bail!("skippy speculative.strategy must be auto, disabled, or mtp"),
    };
    let mode = pick_string_owned(
        model_config.and_then(|config| config.mode.as_deref()),
        global_config.and_then(|config| config.mode.as_deref()),
        Some("auto"),
    );
    reject_unsupported_speculative_runtime_fields(model_config, global_config)?;
    let mut mode = mode;
    let draft_max_tokens = super::support::pick_value(
        model_config.and_then(|config| config.draft_max_tokens),
        global_config.and_then(|config| config.draft_max_tokens),
        0,
    );
    let draft_min_tokens = super::support::pick_value(
        model_config.and_then(|config| config.draft_min_tokens),
        global_config.and_then(|config| config.draft_min_tokens),
        0,
    );
    let draft_n_gpu_layers = pick_owned(
        model_config.and_then(|config| config.draft_gpu_layers),
        global_config.and_then(|config| config.draft_gpu_layers),
    );
    let ngram_min = super::support::pick_value(
        model_config.and_then(|config| config.ngram_min),
        global_config.and_then(|config| config.ngram_min),
        0,
    );
    let ngram_max = super::support::pick_value(
        model_config.and_then(|config| config.ngram_max),
        global_config.and_then(|config| config.ngram_max),
        0,
    );
    let pairing_fault = normalize_pairing_fault(pick_string(
        model_config.and_then(|config| config.pairing_fault.as_deref()),
        global_config.and_then(|config| config.pairing_fault.as_deref()),
        Some("warn_disable"),
    ));
    let explicit = mode != "auto"
        || draft_model_path.is_some()
        || draft_max_tokens > 0
        || draft_min_tokens > 0
        || draft_n_gpu_layers.is_some()
        || ngram_min > 0
        || ngram_max > 0;
    if mode == "disabled" && draft_model_path.is_some() {
        bail!("skippy speculative draft source cannot be set when speculative.mode = \"disabled\"");
    }
    let effective_draft_max_tokens =
        resolved_draft_max_tokens(native_mtp_enabled, draft_max_tokens);
    validate_draft_min_max(draft_min_tokens, effective_draft_max_tokens)
        .map_err(anyhow::Error::msg)?;
    if native_mtp_enabled && draft_model_path.is_some() {
        mode = "disabled".to_string();
    } else if mode == "draft" || (mode == "auto" && draft_model_path.is_some()) {
        resolve_draft_speculative_mode(
            &mut mode,
            &mut draft_model_path,
            draft_max_tokens,
            pairing_fault.as_str(),
            model_id,
            model_path,
        )?;
    } else if mode == "ngram" || (mode == "auto" && (ngram_min > 0 || ngram_max > 0)) {
        resolve_ngram_speculative_mode(&mut mode, ngram_min, ngram_max)?;
    } else {
        mode = "disabled".to_string();
        draft_model_path = None;
    }
    Ok(ResolvedSpeculativeConfig {
        strategy,
        native_mtp_enabled,
        mode,
        draft_model_path,
        pairing_fault,
        draft_max_tokens: effective_draft_max_tokens,
        draft_min_tokens,
        explicit,
        draft_n_gpu_layers,
        ngram_min,
        ngram_max,
    })
}

fn reject_unsupported_speculative_runtime_fields(
    model_config: Option<&SpeculativeConfig>,
    global_config: Option<&SpeculativeConfig>,
) -> Result<()> {
    let unsupported_string_fields = [
        (
            model_config.and_then(|config| config.draft_hf_repo.clone()),
            global_config.and_then(|config| config.draft_hf_repo.clone()),
            "speculative.draft_hf_repo",
        ),
        (
            model_config.and_then(|config| config.draft_hf_file.clone()),
            global_config.and_then(|config| config.draft_hf_file.clone()),
            "speculative.draft_hf_file",
        ),
        (
            model_config.and_then(|config| config.draft_device.clone()),
            global_config.and_then(|config| config.draft_device.clone()),
            "speculative.draft_device",
        ),
        (
            model_config.and_then(|config| config.draft_cache_type_k.clone()),
            global_config.and_then(|config| config.draft_cache_type_k.clone()),
            "speculative.draft_cache_type_k",
        ),
        (
            model_config.and_then(|config| config.draft_cache_type_v.clone()),
            global_config.and_then(|config| config.draft_cache_type_v.clone()),
            "speculative.draft_cache_type_v",
        ),
    ];
    for (model, global, field) in unsupported_string_fields {
        if pick_owned(model, global).is_some() {
            unsupported_speculative_field(field)?;
        }
    }
    if pick_owned(
        model_config.and_then(|config| config.draft_threads),
        global_config.and_then(|config| config.draft_threads),
    )
    .is_some()
    {
        unsupported_speculative_field("speculative.draft_threads")?;
    }

    Ok(())
}

fn resolved_draft_max_tokens(native_mtp_enabled: bool, draft_max_tokens: u32) -> u32 {
    if native_mtp_enabled && draft_max_tokens == 0 {
        return 3;
    }
    draft_max_tokens
}

fn resolve_draft_model_path(raw: String) -> String {
    let raw_path = PathBuf::from(&raw);
    if raw_path.is_file() {
        return raw;
    }
    if !raw.contains(':') {
        return raw;
    }
    let candidate = find_model_path(&raw);
    if candidate.exists() {
        return candidate.to_string_lossy().into_owned();
    }
    raw
}

fn resolve_draft_speculative_mode(
    mode: &mut String,
    draft_model_path: &mut Option<PathBuf>,
    draft_max_tokens: u32,
    pairing_fault: &str,
    model_id: &str,
    model_path: &Path,
) -> Result<()> {
    if draft_model_path.is_none() {
        bail!("skippy speculative draft mode requires an explicit draft_model_path");
    }
    if draft_max_tokens == 0 {
        bail!("skippy speculative draft mode requires draft_max_tokens > 0");
    }
    *mode = "draft".to_string();
    let draft_path = draft_model_path.as_ref().expect("checked above");
    if let Some(reason) = incompatible_draft_pair_reason(model_id, model_path, draft_path) {
        match pairing_fault {
            "warn_disable" => {
                *mode = "disabled".to_string();
                *draft_model_path = None;
            }
            "fail_open" => {}
            "fail_closed" => bail!("skippy incompatible speculative draft pairing: {reason}"),
            _ => unreachable!(),
        }
    }
    Ok(())
}

const NGRAM_WINDOW_MAX: u32 = 1024;

fn resolve_ngram_speculative_mode(mode: &mut String, ngram_min: u32, ngram_max: u32) -> Result<()> {
    if ngram_min == 0 {
        bail!("skippy speculative ngram mode requires ngram_min > 0");
    }
    if ngram_max == 0 {
        bail!("skippy speculative ngram mode requires ngram_max > 0");
    }
    if ngram_min > ngram_max {
        bail!("skippy speculative ngram_min must be less than or equal to ngram_max");
    }
    if ngram_max > NGRAM_WINDOW_MAX {
        bail!("skippy speculative ngram_max must not exceed {NGRAM_WINDOW_MAX}");
    }
    *mode = "ngram".to_string();
    Ok(())
}

fn package_generation_or_direct_default_supports_native_mtp(
    generation: Option<&PackageGenerationInfo>,
    model_path: &Path,
) -> bool {
    package_generation_supports_default_native_mtp(generation)
        || direct_gguf_supports_native_mtp(model_path)
}

fn package_generation_supports_default_native_mtp(
    generation: Option<&PackageGenerationInfo>,
) -> bool {
    generation
        .and_then(|generation| generation.speculative_decoding.as_ref())
        .is_some_and(|speculative| {
            speculative
                .strategies
                .get(&speculative.default)
                .is_some_and(|strategy| {
                    strategy.strategy_type == "native-mtp"
                        && strategy.prediction_depth == Some(1)
                        && !strategy.layer_indices.is_empty()
                })
        })
}

fn package_generation_supports_native_mtp(generation: Option<&PackageGenerationInfo>) -> bool {
    generation
        .and_then(|generation| generation.speculative_decoding.as_ref())
        .is_some_and(speculative_supports_native_mtp)
}

fn speculative_supports_native_mtp(speculative: &PackageSpeculativeDecodingInfo) -> bool {
    speculative.strategies.get("mtp").is_some_and(|strategy| {
        strategy.strategy_type == "native-mtp"
            && strategy.prediction_depth == Some(1)
            && !strategy.layer_indices.is_empty()
    })
}

fn direct_gguf_supports_native_mtp(model_path: &Path) -> bool {
    scan_gguf_compact_meta(model_path).is_some_and(|meta| meta.nextn_predict_layers > 0)
        || scan_gguf_tensor_names_any(model_path, |name| name.contains(".nextn.")).unwrap_or(false)
}

fn unsupported_speculative_field(field: &str) -> Result<()> {
    bail!("skippy {field} is not supported by the embedded runtime");
}

fn normalize_pairing_fault(value: &str) -> String {
    value.replace('-', "_")
}

fn incompatible_draft_pair_reason(
    model_id: &str,
    model_path: &Path,
    draft_model_path: &Path,
) -> Option<String> {
    let target_family = infer_family_capability(model_id, 0, 0)
        .map(|capability| capability.family_id.to_string())
        .or_else(|| infer_family_from_path_string(model_path));
    let draft_family = infer_family_from_path_string(draft_model_path);
    match (target_family, draft_family) {
        (Some(target_family), Some(draft_family)) if target_family != draft_family => Some(
            format!("target family {target_family} does not match draft family {draft_family}"),
        ),
        _ => None,
    }
}

fn infer_family_from_path_string(path: &Path) -> Option<String> {
    infer_family_capability(&path.display().to_string(), 0, 0)
        .map(|capability| capability.family_id.to_string())
}
