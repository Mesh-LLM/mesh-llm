mod types;

use mesh_llm_config::MeshConfig;
use model_hf::store::{find_model_path, huggingface_identity_for_path, model_ref_for_path};
use model_ref::ModelRef;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

pub(crate) use types::{
    ConfigModelMatch, DuplicateTuneTarget, LocalTargetSource, ResolvedTuneTarget,
    TuneTargetContext, TuneTargetResolution, TuneTargetResolveError, TuneTargetResolveReason,
    TuneTargetSelection,
};

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedLocalTarget {
    requested_input: String,
    canonical_model_ref: String,
    resolved_path: PathBuf,
    local_source: LocalTargetSource,
}

#[derive(Debug, Default)]
struct ConfigResolutionIndex {
    ordered_keys: Vec<String>,
    resolved_by_key: BTreeMap<String, ResolvedLocalTarget>,
    matches_by_key: BTreeMap<String, Vec<ConfigModelMatch>>,
    duplicates: Vec<DuplicateTuneTarget>,
    errors: Vec<TuneTargetResolveError>,
}

pub(crate) fn resolve_configured_tune_targets(config: &MeshConfig) -> TuneTargetResolution {
    let index = build_config_resolution_index(config);
    let resolved = index
        .ordered_keys
        .iter()
        .filter_map(|key| {
            index
                .resolved_by_key
                .get(key)
                .map(|target| ResolvedTuneTarget {
                    requested_input: target.requested_input.clone(),
                    canonical_model_ref: target.canonical_model_ref.clone(),
                    resolved_path: target.resolved_path.clone(),
                    local_source: target.local_source.clone(),
                    config_matches: index.matches_by_key.get(key).cloned().unwrap_or_default(),
                    selection: TuneTargetSelection::Configured,
                })
        })
        .collect();
    TuneTargetResolution {
        resolved,
        duplicates: index.duplicates,
        errors: index.errors,
    }
}

pub(crate) fn resolve_explicit_tune_targets(
    config: &MeshConfig,
    inputs: &[String],
) -> TuneTargetResolution {
    resolve_explicit_tune_targets_with_probe(config, inputs, &|_| ())
}

#[cfg(test)]
pub(crate) fn resolve_explicit_tune_targets_with_probe_for_tests(
    config: &MeshConfig,
    inputs: &[String],
    remote_lookup_probe: &dyn Fn(&str),
) -> TuneTargetResolution {
    resolve_explicit_tune_targets_with_probe(config, inputs, remote_lookup_probe)
}

fn resolve_explicit_tune_targets_with_probe(
    config: &MeshConfig,
    inputs: &[String],
    _remote_lookup_probe: &dyn Fn(&str),
) -> TuneTargetResolution {
    let config_index = build_config_resolution_index(config);
    let mut seen = BTreeSet::new();
    let mut first_inputs: BTreeMap<String, String> = BTreeMap::new();
    let mut resolved = Vec::new();
    let mut duplicates = Vec::new();
    let mut errors = Vec::new();

    for input in inputs {
        match resolve_local_target(input, TuneTargetContext::ExplicitInput) {
            Ok(target) => {
                if !seen.insert(target.canonical_model_ref.clone()) {
                    let first_input = first_inputs
                        .get(&target.canonical_model_ref)
                        .cloned()
                        .unwrap_or_else(|| target.requested_input.clone());
                    duplicates.push(DuplicateTuneTarget {
                        input: target.requested_input.clone(),
                        canonical_model_ref: target.canonical_model_ref.clone(),
                        first_input,
                    });
                    continue;
                }
                first_inputs.insert(
                    target.canonical_model_ref.clone(),
                    target.requested_input.clone(),
                );
                let config_matches = config_index
                    .matches_by_key
                    .get(&target.canonical_model_ref)
                    .cloned()
                    .unwrap_or_default();
                resolved.push(ResolvedTuneTarget {
                    requested_input: target.requested_input.clone(),
                    canonical_model_ref: target.canonical_model_ref.clone(),
                    resolved_path: target.resolved_path,
                    local_source: target.local_source,
                    selection: TuneTargetSelection::Explicit {
                        configured: !config_matches.is_empty(),
                    },
                    config_matches,
                });
            }
            Err(error) => errors.push(error),
        }
    }

    TuneTargetResolution {
        resolved,
        duplicates,
        errors,
    }
}

fn build_config_resolution_index(config: &MeshConfig) -> ConfigResolutionIndex {
    let mut index = ConfigResolutionIndex::default();
    let mut first_inputs: BTreeMap<String, String> = BTreeMap::new();

    for (row_index, entry) in config.models.iter().enumerate() {
        let configured_model = entry.model.clone();
        let context = TuneTargetContext::ConfiguredRow { row_index };
        match resolve_local_target(&configured_model, context.clone()) {
            Ok(target) => {
                index
                    .matches_by_key
                    .entry(target.canonical_model_ref.clone())
                    .or_default()
                    .push(ConfigModelMatch {
                        row_index,
                        configured_model,
                    });
                if let Some(first_input) = first_inputs.get(&target.canonical_model_ref) {
                    index.duplicates.push(DuplicateTuneTarget {
                        input: target.requested_input,
                        canonical_model_ref: target.canonical_model_ref,
                        first_input: first_input.clone(),
                    });
                    continue;
                }
                first_inputs.insert(
                    target.canonical_model_ref.clone(),
                    target.requested_input.clone(),
                );
                index.ordered_keys.push(target.canonical_model_ref.clone());
                index
                    .resolved_by_key
                    .insert(target.canonical_model_ref.clone(), target);
            }
            Err(error) => index.errors.push(error),
        }
    }

    index
}

fn resolve_local_target(
    input: &str,
    context: TuneTargetContext,
) -> Result<ResolvedLocalTarget, TuneTargetResolveError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(TuneTargetResolveError {
            input: input.to_string(),
            context,
            reason: TuneTargetResolveReason::EmptyInput,
        });
    }
    if trimmed.starts_with("hf://") {
        return Err(TuneTargetResolveError {
            input: trimmed.to_string(),
            context,
            reason: TuneTargetResolveReason::RemoteRefRequiresDownload,
        });
    }

    let local_path = PathBuf::from(trimmed);
    if local_path.exists() {
        return Ok(resolved_target_for_path(trimmed, &local_path));
    }

    let installed_path = installed_model_path(trimmed);
    if installed_path.exists() {
        return Ok(resolved_target_for_path(trimmed, &installed_path));
    }

    Err(TuneTargetResolveError {
        input: trimmed.to_string(),
        context,
        reason: TuneTargetResolveReason::NotFoundLocally,
    })
}

fn resolved_target_for_path(requested_input: &str, path: &Path) -> ResolvedLocalTarget {
    let resolved_path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let canonical_model_ref = model_ref_for_path(&resolved_path);
    let local_source = match huggingface_identity_for_path(&resolved_path) {
        Some(identity) => LocalTargetSource::HuggingFaceCache {
            canonical_ref: identity.canonical_ref,
        },
        None => LocalTargetSource::FilesystemPath {
            synthetic_model_ref: canonical_model_ref.clone(),
        },
    };
    ResolvedLocalTarget {
        requested_input: requested_input.to_string(),
        canonical_model_ref,
        resolved_path,
        local_source,
    }
}

fn installed_model_path(input: &str) -> PathBuf {
    if ModelRef::parse(input).is_ok() {
        return find_model_path(input);
    }
    let installed_name = input.strip_suffix(".gguf").unwrap_or(input);
    find_model_path(installed_name)
}
