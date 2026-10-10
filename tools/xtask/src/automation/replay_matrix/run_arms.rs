use super::{
    external_config::Config,
    run_budget::Budget,
    run_workload::{Build, Input},
};
use crate::command::DynResult;

pub(super) fn prepare_config(input: &mut Input) -> DynResult<()> {
    let Some(path) = &input.engine_config else {
        if input.engine_config_sha256.is_some() {
            return Err("engine config digest requires a config path".into());
        }
        return Ok(());
    };
    let config = super::external_config::load(path)?;
    if input
        .engine_config_sha256
        .as_ref()
        .is_some_and(|digest| digest != &config.sha256)
    {
        return Err("external config changed after manual plan validation".into());
    }
    input.validated_engine_config = Some(config);
    Ok(())
}

pub(super) fn append(input: &mut Input, budget: &Budget) -> DynResult<Option<Config>> {
    let Some(_path) = &input.engine_config else {
        return Ok(None);
    };
    if !input.resume
        && input
            .builds
            .iter()
            .any(|build| matches!(build, Build::External(_)))
    {
        return Err("engine-config cannot duplicate prebuilt external identities".into());
    }
    let config = if let Some(config) = input.validated_engine_config.take() {
        config
    } else {
        prepare_config(input)?;
        input
            .validated_engine_config
            .take()
            .ok_or("missing validated engine config")?
    };
    let model = input
        .model_reference
        .as_deref()
        .unwrap_or(input.model.to_str().ok_or("non-Unicode model identity")?);
    config.admit(
        model,
        &input
            .builds
            .iter()
            .filter(|build| matches!(build, Build::Mesh(_)))
            .map(|build| build.label().to_owned())
            .collect::<Vec<_>>(),
        input.minimum_context_tokens,
    )?;
    if input
        .builds
        .iter()
        .any(|build| matches!(build, Build::External(_)))
    {
        verify_resume_config(input, &config)?;
        return Ok(Some(config));
    }
    for arm in &config.arms {
        input.builds.push(Build::External(Box::new(
            super::external_probe::verify_with_budget(
                arm,
                budget.remaining(super::external_probe::VERSION_TIMEOUT)?,
            )?,
        )));
    }
    Ok(Some(config))
}

pub(super) fn metadata(input: &Input, document: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({"kind":"prebuilt","comparison":{"model":document["config"]["model"]},
        "arms":input.builds.iter().filter_map(|build| match build { Build::External(build) => Some(&build.external_engine), Build::Mesh(_) => None }).collect::<Vec<_>>()})
}

pub(super) fn record(input: &Input, document: &serde_json::Value) -> DynResult<()> {
    let external = input
        .builds
        .iter()
        .filter_map(|build| match build {
            Build::External(build) => Some(build),
            Build::Mesh(_) => None,
        })
        .collect::<Vec<_>>();
    if external.is_empty() {
        return Ok(());
    }
    let commands = external
        .iter()
        .map(|build| {
            super::external_command::server(
                &build.external_engine,
                &build.provenance.resolved_executable,
                9337,
            )
        })
        .collect::<DynResult<Vec<_>>>()?;
    let plan = serde_json::json!({
        "engine_config":document["config"]["engine_config"], "external_server_commands":commands,
        "builds":input.builds,"order":input.builds.iter().map(|build| serde_json::json!({
            "label":build.label(),"ref":build.reference(),"commit":build.commit()})).collect::<Vec<_>>()
    });
    let path = input.output.join("plan.json");
    if input.resume {
        let retained: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
        if retained != plan {
            return Err("cannot resume: external plan evidence differs".into());
        }
        return Ok(());
    }
    crate::command::write_json_file(&path, &plan)
}

fn verify_resume_config(input: &Input, config: &Config) -> DynResult<()> {
    let retained = input
        .builds
        .iter()
        .filter_map(|build| match build {
            Build::External(build) => Some(&build.external_engine),
            Build::Mesh(_) => None,
        })
        .collect::<Vec<_>>();
    if serde_json::to_value(retained)? != serde_json::to_value(&config.arms)? {
        return Err(
            "cannot resume: external engine config differs from retained build identities".into(),
        );
    }
    Ok(())
}
#[cfg(test)]
mod resume_config_tests {
    use super::*;
    #[test]
    fn resumed_config_requires_exact_retained_arm_order_and_settings() {
        let arm:super::super::external_config::Arm=serde_json::from_value(serde_json::json!({"label":"llama","engine":"llama.cpp","executable":"/fixture/llama-server","model":"/fixture/model.gguf","context_size":32768,"max_concurrency":4,"cwd":"/fixture"})).unwrap();
        let mut provenance = serde_json::to_value(&arm).unwrap();
        provenance["resolved_executable"] = "/fixture/llama-server".into();
        provenance["version"] = "fixture-version".into();
        provenance["version_sha256"] = "a".repeat(64).into();
        let build = serde_json::json!({"label":"llama","engine":"llama.cpp","ref":"fixture-version","commit":"a".repeat(64),"version":"fixture-version","version_sha256":"a".repeat(64),"worktree":"/fixture","binary":"/fixture/llama-server","runtime_root":"external","backend":"external","served_model":"/fixture/model.gguf","external_engine":arm,"provenance":provenance});
        let input:Input=serde_json::from_value(serde_json::json!({"manifest":"/fixture/manifest.json","requirements":{"concurrency":[1],"minimum_worker_waves":1,"warmup_turns":1,"required_frameworks":[]},"builds":[build],"context_qualification":"captured","model":"/fixture/model.gguf","passes":1,"max_output_tokens":1,"request_timeout_seconds":1,"startup_timeout_seconds":2,"timeout_seconds":3,"output":"/fixture/artifact","resume":true})).unwrap();
        let mut config = Config {
            path: "/fixture/engines.json".into(),
            sha256: "b".repeat(64),
            comparison: super::super::external_config::Comparison {
                model: "/fixture/model.gguf".into(),
            },
            arms: vec![arm],
        };
        assert!(verify_resume_config(&input, &config).is_ok());
        config.arms[0].prefix_cache = false;
        assert!(verify_resume_config(&input, &config).is_err());
        config.arms.clear();
        assert!(verify_resume_config(&input, &config).is_err());
    }
}
