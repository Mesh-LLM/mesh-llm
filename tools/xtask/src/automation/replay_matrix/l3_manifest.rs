use super::{
    l3_contract::Config,
    manifest_preflight::Manifest,
    recorded_requests::{Selection, Trajectory},
};
use crate::command::DynResult;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};
pub(super) fn import(
    source: &Path,
    output: &Path,
    config: &Config,
) -> DynResult<(Manifest, Vec<Trajectory>, serde_json::Value)> {
    let bytes = std::fs::read(source)?;
    let document: serde_json::Value = serde_json::from_slice(&bytes)?;
    let manifest: Manifest = serde_json::from_slice(&bytes)?;
    validate(&manifest, config)?;
    let lifecycle = manifest
        .cohorts
        .get(&config.lifecycle_cohort)
        .ok_or("missing lifecycle cohort")?;
    let mut selected = Vec::new();
    for source in &config.required_sources {
        let trajectory = lifecycle
            .iter()
            .find(|trajectory| &trajectory.source_dataset == source)
            .ok_or_else(|| format!("lifecycle cohort missing required source {source}"))?;
        selected.push(serde_json::from_value(trajectory.original.clone())?);
    }
    let path = output.join("inputs/captured-trajectories.json");
    std::fs::create_dir_all(path.parent().ok_or("missing input parent")?)?;
    std::fs::write(&path, &bytes)?;
    let digest = crate::product::digest::file_sha256(&path).map_err(|error| error.error)?;
    let mut cohorts = BTreeMap::new();
    for (name, trajectories) in &manifest.cohorts {
        let originals = serde_json::Value::Array(
            trajectories
                .iter()
                .map(|trajectory| trajectory.original.clone())
                .collect(),
        );
        let mut frameworks = BTreeMap::<String, usize>::new();
        let mut turns = BTreeMap::<String, usize>::new();
        for trajectory in trajectories {
            *frameworks
                .entry(trajectory.agent_framework.clone())
                .or_default() += 1;
            *turns.entry(trajectory.agent_framework.clone()).or_default() +=
                super::recorded_requests::build(
                    trajectory,
                    &Selection {
                        model: "preflight",
                        maximum_output_tokens: 1,
                        turn_limit: None,
                        qualification_probe: true,
                    },
                )?
                .len();
        }
        let ids = trajectories
            .iter()
            .map(|trajectory| trajectory.session_id.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        cohorts.insert(name,serde_json::json!({"trajectory_count":trajectories.len(),"assistant_turns":turns.values().sum::<usize>(),"framework_trajectories":frameworks,"framework_assistant_turns":turns,"session_ids_sha256":hex::encode(Sha256::digest(ids.as_bytes())),"session_cohort_sha256":super::cohort_identity::digest(&originals)?}));
    }
    let metadata = document
        .get("metadata")
        .filter(|value| value.is_object())
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    let input = serde_json::json!({"kind":"captured","dataset":{"name":metadata.get("name").cloned().unwrap_or_else(||source.file_stem().unwrap_or_default().to_string_lossy().into_owned().into()),"revision":metadata.get("revision").cloned().unwrap_or_else(||digest.clone().into())},"source_manifest":source,"source_manifest_sha256":digest,"manifest":path,"manifest_sha256":digest,"metadata":metadata,"cohorts":cohorts});
    Ok((manifest, selected, input))
}
pub(super) fn validate(manifest: &Manifest, config: &Config) -> DynResult<()> {
    let mut expected = config
        .concurrency
        .iter()
        .map(ToString::to_string)
        .collect::<BTreeSet<_>>();
    if !expected.insert(config.lifecycle_cohort.clone())
        || expected != manifest.cohorts.keys().cloned().collect()
    {
        return Err("disk-L3 manifest cohort set differs from plan".into());
    }
    let mut sessions = BTreeSet::new();
    for (name, trajectories) in &manifest.cohorts {
        if trajectories.is_empty()
            || (name != &config.lifecycle_cohort && trajectories.len() < name.parse::<usize>()?)
        {
            return Err("empty or undersized disk-L3 cohort".into());
        }
        for trajectory in trajectories {
            if trajectory.session_id.is_empty()
                || !sessions.insert(&trajectory.session_id)
                || trajectory.source_dataset.is_empty()
                || trajectory.agent_framework.is_empty()
                || trajectory.original.get("recorded_model").is_none()
                || trajectory
                    .recorded_model
                    .as_ref()
                    .is_some_and(String::is_empty)
                || trajectory
                    .tools
                    .as_ref()
                    .is_some_and(|tools| tools.iter().any(|tool| !tool.is_object()))
            {
                return Err("invalid or duplicate captured disk-L3 session provenance".into());
            }
            super::recorded_requests::build(
                trajectory,
                &Selection {
                    model: "preflight",
                    maximum_output_tokens: 1,
                    turn_limit: None,
                    qualification_probe: true,
                },
            )?;
        }
    }
    Ok(())
}
