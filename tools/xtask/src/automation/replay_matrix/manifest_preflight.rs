use super::recorded_requests::{Selection, Trajectory, build};
use crate::command::DynResult;
use crate::repository::{check_args::Grammar, check_report::CheckReport};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Deserialize)]
pub(super) struct Manifest {
    pub cohorts: BTreeMap<String, Vec<Trajectory>>,
}

#[derive(Deserialize, Serialize)]
pub(super) struct Requirements {
    pub concurrency: Vec<usize>,
    pub minimum_worker_waves: usize,
    pub warmup_turns: usize,
    pub required_frameworks: Vec<String>,
}

pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation replay-matrix manifest-preflight --manifest PATH --requirements PATH --output PATH",
        values: &["--manifest", "--requirements", "--output"],
        flags: &["--help"],
    };
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
    let path = std::path::Path::new(parsed.last("--manifest").ok_or("missing --manifest")?);
    let manifest: Manifest = serde_json::from_slice(&std::fs::read(path)?)?;
    let requirements: Requirements = serde_json::from_slice(&std::fs::read(
        parsed
            .last("--requirements")
            .ok_or("missing --requirements")?,
    )?)?;
    let report = validate(&manifest, &requirements)?;
    let digest = crate::product::digest::file_sha256(path).map_err(|error| error.error)?;
    crate::command::write_json_file(
        std::path::Path::new(parsed.last("--output").ok_or("missing --output")?),
        &serde_json::json!({"passed":true,"manifest_sha256":digest,"cohorts":report}),
    )
}

pub(super) fn validate(
    manifest: &Manifest,
    requirements: &Requirements,
) -> DynResult<BTreeMap<String, serde_json::Value>> {
    let offered: BTreeSet<_> = requirements.concurrency.iter().copied().collect();
    if offered.is_empty()
        || offered.len() != requirements.concurrency.len()
        || offered.iter().any(|value| !(1..=256).contains(value))
        || requirements.minimum_worker_waves == 0
        || requirements.warmup_turns == 0
    {
        return Err("manifest requirements need unique concurrency in 1..=256 and positive wave/warmup budgets".into());
    }
    let mut expected: BTreeSet<_> = offered.iter().map(ToString::to_string).collect();
    expected.insert("warmup".into());
    if expected != manifest.cohorts.keys().cloned().collect() {
        return Err("trajectory manifest cohort set differs from requirements".into());
    }
    let selection = Selection {
        model: "preflight",
        maximum_output_tokens: 1,
        turn_limit: None,
        qualification_probe: true,
    };
    let mut sessions = BTreeSet::new();
    let mut report = BTreeMap::new();
    for (name, trajectories) in &manifest.cohorts {
        if trajectories.is_empty() {
            return Err(format!("{name}: empty trajectory cohort").into());
        }
        let mut turns = 0_usize;
        let mut frameworks = BTreeMap::<&str, usize>::new();
        let mut framework_turns = BTreeMap::<&str, usize>::new();
        for trajectory in trajectories {
            if !sessions.insert(&trajectory.session_id) {
                return Err("duplicate session across cohorts".into());
            }
            if trajectory.source_dataset.is_empty()
                || trajectory.agent_framework.is_empty()
                || trajectory
                    .recorded_model
                    .as_ref()
                    .is_some_and(String::is_empty)
                || trajectory.original.get("recorded_model").is_none()
                || trajectory
                    .tools
                    .as_ref()
                    .is_some_and(|tools| tools.iter().any(|tool| !tool.is_object()))
            {
                return Err("trajectory provenance must be nonempty".into());
            }
            let recorded_turns = build(trajectory, &selection)?.len();
            turns = turns
                .checked_add(recorded_turns)
                .ok_or("manifest turn count overflow")?;
            *frameworks.entry(&trajectory.agent_framework).or_default() += 1;
            let count = framework_turns
                .entry(&trajectory.agent_framework)
                .or_default();
            *count = count
                .checked_add(recorded_turns)
                .ok_or("framework turn count overflow")?;
        }
        if name == "warmup" {
            if turns < requirements.warmup_turns {
                return Err("warmup cohort has insufficient recorded turns".into());
            }
        } else {
            let concurrency: usize = name.parse()?;
            let needed = concurrency
                .checked_mul(requirements.minimum_worker_waves)
                .ok_or("cohort capacity overflow")?;
            if trajectories.len() < needed {
                return Err(format!("{name}: insufficient whole sessions for worker waves").into());
            }
            if requirements
                .required_frameworks
                .iter()
                .any(|framework| !frameworks.contains_key(framework.as_str()))
            {
                return Err(format!("{name}: missing required framework").into());
            }
        }
        use sha2::{Digest, Sha256};
        let identity = trajectories
            .iter()
            .map(|trajectory| trajectory.session_id.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        report.insert(name.clone(),serde_json::json!({"trajectory_count":trajectories.len(),"assistant_turns":turns,
            "framework_trajectories":frameworks,"framework_assistant_turns":framework_turns,"session_ids_sha256":hex::encode(Sha256::digest(identity.as_bytes()))}));
    }
    Ok(report)
}
