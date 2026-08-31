use crate::diagnostic::ConfigDiagnostic;
use crate::model::{
    ModelConfigDefaults, ModelConfigEntry, ModelTopologyConfig, ModelTopologyNodeSelector,
    ModelTopologyStageConfig, merge_model_topology,
};
use crate::validation_support::validation_diagnostic;

pub(crate) fn model_topology_diagnostics(
    defaults: Option<&ModelConfigDefaults>,
    model: &ModelConfigEntry,
    base_path: &str,
    inherited_diagnostic_keys: &mut std::collections::BTreeSet<(String, String)>,
) -> Vec<ConfigDiagnostic> {
    let Some(effective) = EffectiveTopology::from_sources(
        defaults.and_then(|defaults| defaults.topology.as_ref()),
        model.topology.as_ref(),
    ) else {
        return Vec::new();
    };
    let diagnostics = validate_effective_topology(&effective, base_path);
    let mut diagnostics = diagnostics
        .into_iter()
        .filter(|diagnostic| {
            let Some(path) = diagnostic.path.as_ref().map(|path| path.render()) else {
                return true;
            };
            if !path.starts_with("defaults.topology") {
                return true;
            }
            inherited_diagnostic_keys.insert((path, diagnostic.message.clone()))
        })
        .collect::<Vec<_>>();
    if !has_immutable_model_revision(&model.model) {
        diagnostics.push(validation_diagnostic(
            &format!("{base_path}.model"),
            format!(
                "{base_path}.model requires an explicit immutable revision when topology is configured"
            ),
        ));
    }
    diagnostics
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TopologyFieldSource {
    Defaults,
    Model,
}

impl TopologyFieldSource {
    fn path(self, model_base_path: &str) -> String {
        match self {
            Self::Defaults => "defaults.topology".to_string(),
            Self::Model => format!("{model_base_path}.topology"),
        }
    }
}

struct EffectiveTopology {
    config: ModelTopologyConfig,
    mode_source: TopologyFieldSource,
    manifest_sha256_source: TopologyFieldSource,
    stages_source: TopologyFieldSource,
}

impl EffectiveTopology {
    fn from_sources(
        defaults: Option<&ModelTopologyConfig>,
        model: Option<&ModelTopologyConfig>,
    ) -> Option<Self> {
        let config = merge_model_topology(defaults, model)?;
        let source_for = |model_value_is_set: bool| {
            if model_value_is_set || defaults.is_none() {
                TopologyFieldSource::Model
            } else {
                TopologyFieldSource::Defaults
            }
        };

        Some(Self {
            config,
            mode_source: source_for(model.and_then(|topology| topology.mode.as_ref()).is_some()),
            manifest_sha256_source: source_for(
                model
                    .and_then(|topology| topology.manifest_sha256.as_ref())
                    .is_some(),
            ),
            stages_source: source_for(
                model
                    .and_then(|topology| topology.stages.as_ref())
                    .is_some(),
            ),
        })
    }
}

fn has_immutable_model_revision(model_ref: &str) -> bool {
    let Some((_, revision_and_selector)) = model_ref.rsplit_once('@') else {
        return false;
    };
    let revision = revision_and_selector
        .split([':', '/'])
        .next()
        .unwrap_or_default()
        .trim();
    !revision.is_empty()
        && !matches!(
            revision.to_ascii_lowercase().as_str(),
            "main" | "master" | "latest" | "dev" | "develop" | "development"
        )
}

fn validate_effective_topology(
    effective: &EffectiveTopology,
    model_base_path: &str,
) -> Vec<ConfigDiagnostic> {
    let mode_path = effective.mode_source.path(model_base_path);
    let manifest_path = effective.manifest_sha256_source.path(model_base_path);
    let stages_path = effective.stages_source.path(model_base_path);
    let mut diagnostics = Vec::new();
    if effective.config.mode.is_none() {
        diagnostics.push(validation_diagnostic(
            &format!("{mode_path}.mode"),
            format!("{mode_path}.mode is required when topology is configured"),
        ));
    }
    match effective.config.manifest_sha256.as_deref() {
        Some(manifest) if is_sha256_hex(manifest) => {}
        Some(_) => diagnostics.push(validation_diagnostic(
            &format!("{manifest_path}.manifest_sha256"),
            format!("{manifest_path}.manifest_sha256 must be 64 lowercase hexadecimal characters"),
        )),
        None => diagnostics.push(validation_diagnostic(
            &format!("{manifest_path}.manifest_sha256"),
            format!("{manifest_path}.manifest_sha256 is required when topology is configured"),
        )),
    }
    match effective.config.stages.as_deref() {
        Some(stages) => validate_topology_stages(stages, &stages_path, &mut diagnostics),
        None => diagnostics.push(validation_diagnostic(
            &format!("{stages_path}.stages"),
            format!("{stages_path}.stages is required when topology is configured"),
        )),
    }
    diagnostics
}

fn validate_topology_stages(
    stages: &[ModelTopologyStageConfig],
    topology_path: &str,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    if stages.len() < 2 {
        diagnostics.push(validation_diagnostic(
            &format!("{topology_path}.stages"),
            format!("{topology_path}.stages requires at least two stages"),
        ));
    }
    let mut expected_start = 0;
    for (index, stage) in stages.iter().enumerate() {
        let stage_path = format!("{topology_path}.stages[{index}]");
        validate_topology_node_selector(&stage.node, &stage_path, diagnostics);
        if stage.layer_start != expected_start {
            diagnostics.push(validation_diagnostic(
                &format!("{stage_path}.layer_start"),
                format!("{stage_path}.layer_start must equal contiguous boundary {expected_start}"),
            ));
        }
        if stage.layer_end <= stage.layer_start {
            diagnostics.push(validation_diagnostic(
                &format!("{stage_path}.layer_end"),
                format!("{stage_path}.layer_end must be greater than layer_start"),
            ));
        }
        expected_start = stage.layer_end;
    }
}

fn validate_topology_node_selector(
    selector: &ModelTopologyNodeSelector,
    stage_path: &str,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    let endpoint = selector
        .endpoint_id
        .as_deref()
        .filter(|value| !value.trim().is_empty());
    let hostname = selector
        .hostname
        .as_deref()
        .filter(|value| !value.trim().is_empty());
    if endpoint.is_some() == hostname.is_some() {
        diagnostics.push(validation_diagnostic(
            &format!("{stage_path}.node"),
            format!("{stage_path}.node requires exactly one of endpoint_id or hostname"),
        ));
    }
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
