//! Public manual adapter; execute-run remains the single execution owner.
use super::manual_options::{GRAMMAR, Options};
use crate::{
    automation::private_state::PrivateState, command::DynResult, process,
    repository::check_report::CheckReport,
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::Duration,
};
fn resolve(repo: &Path, reference: &str) -> DynResult<String> {
    let mut environment = BTreeMap::new();
    for name in ["PATH", "HOME", "SystemRoot", "WINDIR"] {
        if let Some(value) = std::env::var_os(name) {
            environment.insert(name.into(), process::Value::Public(value));
        }
    }
    environment.insert("GIT_MASTER".into(), process::Value::Public("1".into()));
    let report = process::supervise(
        &process::ProcessSpec {
            executable: super::executable_resolution::tool("git")?,
            arguments: [
                "rev-parse",
                "--verify",
                "--end-of-options",
                &format!("{reference}^{{commit}}"),
            ]
            .into_iter()
            .map(|v| process::Value::Public(v.into()))
            .collect(),
            cwd: repo.into(),
            environment,
        },
        &process::Limits {
            execution: Duration::from_secs(30),
            graceful_shutdown: Duration::from_secs(2),
            forced_shutdown: Duration::from_secs(3),
            retained_bytes_per_stream: 4096,
            readiness: process::Readiness::None,
            completion: process::Completion::Exit,
        },
        &process::Cancellation::default(),
        Default::default(),
    )?;
    if !report.success() || !super::history_hub::terminal(&report) || report.stdout.truncated {
        return Err("manual ref resolution failed".into());
    }
    let commit = std::str::from_utf8(&report.stdout.bytes_retained)?.trim();
    if commit.len() != 40 || !commit.bytes().all(|v| v.is_ascii_hexdigit()) {
        return Err("ref resolution returned an invalid commit".into());
    }
    Ok(commit.into())
}
pub(super) fn plan(options: &Options, commits: &BTreeMap<String, String>) -> DynResult<Value> {
    let mut identities = std::collections::BTreeSet::new();
    let mut refs = Vec::new();
    for reference in &options.refs {
        let commit = commits
            .get(&reference.label)
            .ok_or("missing resolved manual ref")?;
        if !identities.insert(commit) {
            return Err("manual ref labels must resolve to distinct commits".into());
        }
        refs.push(json!({"label":reference.label,"ref":reference.reference,"commit":commit}));
    }
    let mut labels = options
        .refs
        .iter()
        .map(|v| v.label.clone())
        .collect::<Vec<_>>();
    let external = options
        .engine_config
        .as_ref()
        .map(|path| super::external_config::load(path))
        .transpose()?;
    if let Some(config) = &external {
        if config.comparison.model != options.model {
            return Err("engine comparison model differs from --model".into());
        }
        for arm in &config.arms {
            if labels.contains(&arm.label) {
                return Err("Mesh and external labels overlap".into());
            }
            labels.push(arm.label.clone());
        }
    }
    let mut order = Vec::new();
    for pass in 0..options.passes {
        let mut current = labels.clone();
        if pass % 2 == 1 {
            current.reverse();
        }
        for label in current {
            order.push(json!({"pass":pass+1,"label":label}));
        }
    }
    let commands = external
        .as_ref()
        .map(|config| {
            config
                .arms
                .iter()
                .map(|arm| {
                    super::external_command::server(
                        arm,
                        &super::external_command::executable(arm)?,
                        9337,
                    )
                })
                .collect::<DynResult<Vec<_>>>()
        })
        .transpose()?;
    let warmup_count = options.sessions;
    let measured_count = options
        .sessions
        .map(|count| {
            count
                .checked_mul(options.concurrency.len())
                .ok_or("manual selected session count overflow")
        })
        .transpose()?;
    Ok(
        json!({"schema_version":3,"kind":"manual-replay","refs":refs,"order":order,"options":options,"selection":{"measured_unique_trajectory_count":measured_count,"warmup_unique_trajectory_count":warmup_count},"workload":{"measured_requests_per_arm_pass":null,"measured_requests_total":null},"engine_config_sha256":external.as_ref().map(|config| &config.sha256),"build_commands":[["just","release-host-build"],["just","release-runtime-build",options.backend]],"server_command":["<release-binary>","serve","--model",options.model,"--log-format","json"],"external_server_commands":commands,"ordered_recorded_prefix_replay":true}),
    )
}
fn dataset(repo: &Path, options: &Options, python: &Path) -> DynResult<Value> {
    let config: Value = serde_json::from_slice(&std::fs::read(
        repo.join("evals/skippy-competitive-benchmark.json"),
    )?)?;
    let pin = &config["thoughtworks"]["dataset"];
    Ok(
        json!({"file":options.dataset,"sha256":pin["sha256"].as_str().ok_or("missing pinned dataset SHA")?,"revision":pin["revision"].as_str().ok_or("missing pinned dataset revision")?,"python":python,"timeout_seconds":options.timeout,"sessions_per_cohort":options.sessions.ok_or("missing dataset cohort count")?,"min_isl":options.min_isl,"max_isl":options.max_isl,"min_turns":options.min_turns,"frameworks":options.frameworks,"source_datasets":options.sources}),
    )
}
pub(super) fn input(
    repo: &Path,
    options: &Options,
    _staging: &Path,
    git: &Path,
    just: &Path,
    python: Option<&Path>,
) -> DynResult<Value> {
    let output = options.output.as_ref().ok_or("run requires output")?;
    let qualified =
        options.minimum_context != 0 || options.minimum_session != 0 || options.recurrent;
    let model = options.model_file.clone().unwrap_or_else(|| {
        if options.model.contains("://") {
            PathBuf::from(&options.model)
        } else {
            std::path::absolute(&options.model).unwrap_or_else(|_| PathBuf::from(&options.model))
        }
    });
    if (qualified || options.expected_model_sha256.is_some()) && !model.is_absolute()
        || qualified && options.expected_model_sha256.is_none()
    {
        return Err(
            "pinned or qualified manual URI runs need a local --model-file; qualification also needs --expected-model-sha256".into(),
        );
    }
    let worktrees = options.worktree_root.clone().unwrap_or_else(|| {
        repo.parent()
            .unwrap_or(repo)
            .join(".agentic-replay-worktrees")
    });
    let jobs=options.refs.iter().map(|reference|json!({"repo":repo,"worktree_root":worktrees,"label":reference.label,"ref":reference.reference,"backend":options.backend,"git":git,"just":just,"timeout_seconds":options.timeout,"logs":output.join("logs"),"skip_build":options.skip_build})).collect::<Vec<_>>();
    let manifest = options
        .manifest
        .clone()
        .unwrap_or_else(|| output.join("inputs/generated-manifest.json"));
    let dataset = if options.dataset.is_some() {
        Some(dataset(
            repo,
            options,
            python.ok_or("dataset reader requires --python or installed python3")?,
        )?)
    } else {
        None
    };
    let builds = if options.resume {
        let prior: Value = serde_json::from_slice(&std::fs::read(output.join("run.json"))?)?;
        Some(
            prior["builds"]
                .as_array()
                .ok_or("resume artifact missing build identities")?
                .clone(),
        )
    } else {
        None
    };
    Ok(
        json!({"manifest":manifest,"requirements":{"concurrency":options.concurrency,"minimum_worker_waves":options.worker_waves,"warmup_turns":options.warmup,"required_frameworks":options.required_frameworks},"builds":builds.unwrap_or_default(),"build_jobs":if options.resume{Vec::new()}else{jobs},"engine_config":options.engine_config,"context_qualification":if qualified{"mesh"}else{"captured"},"model":model,"model_reference":options.model,"model_sha256":options.expected_model_sha256.clone().unwrap_or_default(),"minimum_context_tokens":options.minimum_context,"minimum_session_prompt_tokens":options.minimum_session,"require_recurrent_restores":options.recurrent,"passes":options.passes,"max_output_tokens":options.max_output,"request_timeout_seconds":options.request,"startup_timeout_seconds":options.startup,"timeout_seconds":options.timeout,"output":output,"prompt_token_range":options.prompt_range,"min_cache_pct":options.min_cache,"require_output_match":options.output_match,"max_ttft_regression_pct":options.max_ttft,"resume":options.resume,"dataset":dataset,"replay_mode":options.mode,"hf_home":options.hf_home}),
    )
}
fn verify_requested_resume_refs(
    options: &Options,
    commits: &BTreeMap<String, String>,
) -> DynResult<()> {
    let output = options.output.as_ref().ok_or("resume requires output")?;
    let prior: Value = serde_json::from_slice(&std::fs::read(output.join("run.json"))?)?;
    let builds: Vec<super::run_workload::Build> = serde_json::from_value(prior["builds"].clone())?;
    let retained = builds
        .iter()
        .filter_map(|build| match build {
            super::run_workload::Build::Mesh(build) => {
                Some((build.label.as_str(), build.commit.as_str()))
            }
            super::run_workload::Build::External(_) => None,
        })
        .collect::<Vec<_>>();
    let requested = options
        .refs
        .iter()
        .map(|reference| {
            Ok((
                reference.label.as_str(),
                commits
                    .get(&reference.label)
                    .ok_or("missing resolved manual ref")?
                    .as_str(),
            ))
        })
        .collect::<DynResult<Vec<_>>>()?;
    if requested != retained {
        return Err("cannot resume: requested Mesh ref labels or resolved commits differ from retained builds".into());
    }
    Ok(())
}

pub(in crate::automation) fn run(
    root: Option<&Path>,
    args: &[String],
    running: bool,
) -> DynResult<()> {
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage)).emit();
    }
    let options = super::manual_options::parse(&parsed, running)?;
    let repo = crate::repository::RepositoryRoot::resolve(options.repo.as_deref().or(root))?;
    let mut commits = BTreeMap::new();
    for reference in &options.refs {
        commits.insert(
            reference.label.clone(),
            resolve(repo.as_path(), &reference.reference)?,
        );
    }
    if options.resume {
        verify_requested_resume_refs(&options, &commits)?;
    }
    let plan = plan(&options, &commits)?;
    if !running {
        return CheckReport::success(format!("{}\n", serde_json::to_string_pretty(&plan)?)).emit();
    }
    let state = PrivateState::create(&std::env::temp_dir(), "manual-replay-handoff")?;
    state.prepare()?;
    let result: DynResult<()> = (|| {
        let git = super::executable_resolution::tool("git")?;
        let just = super::executable_resolution::tool("just")?;
        let python = if options.dataset.is_some() {
            Some(super::history_hub::tool(
                options.python.as_ref().and_then(|v| v.to_str()),
                "python3",
            )?)
        } else {
            None
        };
        let mut input = input(
            repo.as_path(),
            &options,
            state.root(),
            &git,
            &just,
            python.as_deref(),
        )?;
        if !plan["engine_config_sha256"].is_null() {
            input["engine_config_sha256"] = plan["engine_config_sha256"].clone();
        }
        // Fail closed until the parent propagates profile/credential options through the existing owner.
        let typed: super::run_workload::Input = serde_json::from_value(input.clone())?;
        let roundtrip = serde_json::to_value(&typed)?;
        if roundtrip
            .get("replay_mode")
            .cloned()
            .unwrap_or_else(|| json!("all"))
            != input["replay_mode"]
            || options.hf_home.is_some() && roundtrip["hf_home"] != input["hf_home"]
        {
            return Err(
                "manual profile/HF-home propagation is not registered in execute-run".into(),
            );
        }
        let file = state.root().join("input.json");
        crate::command::write_json_file(&file, &input)?;
        super::run_execution::run(
            Some(repo.as_path()),
            &[
                "--input".into(),
                file.to_str().ok_or("non-Unicode manual handoff")?.into(),
            ],
        )
    })();
    state
        .finish(result)
        .map_err(|error| format!("manual replay/finalization failed: {error:?}").into())
}

#[cfg(test)]
mod requested_resume_tests {
    use super::*;
    #[test]
    fn matching_resolved_mesh_identity_is_admitted_before_retained_execution_checks() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("artifact");
        std::fs::create_dir(&path).unwrap();
        let document = json!({"builds":[{"label":"main","ref":"retained-alias","commit":"a".repeat(40),"binary":"/fixture/host","binary_sha256":"c".repeat(64),"runtime_root":"/fixture/native","runtime":"/fixture/runtime","runtime_sha256":"d".repeat(64)}]});
        std::fs::write(
            path.join("run.json"),
            serde_json::to_vec(&document).unwrap(),
        )
        .unwrap();
        let arguments = [
            "--ref",
            "main=current-alias",
            "--model",
            "hf://owner/model",
            "--trajectory-manifest",
            "capture.json",
            "--output",
            path.to_str().unwrap(),
            "--resume",
        ]
        .map(str::to_owned);
        let parsed = GRAMMAR.parse(&arguments).unwrap();
        let options = super::super::manual_options::parse(&parsed, true).unwrap();
        let commits = [("main".into(), "a".repeat(40))].into_iter().collect();
        assert!(verify_requested_resume_refs(&options, &commits).is_ok());
    }
}
