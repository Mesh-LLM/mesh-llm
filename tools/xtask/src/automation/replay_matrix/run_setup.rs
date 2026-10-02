use super::{
    manifest_preflight::Manifest,
    run_budget::Budget,
    run_workload::{Build, Input},
};
use crate::command::DynResult;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub(super) struct Prepared {
    pub input: Input,
    pub document: serde_json::Value,
    pub manifest: PathBuf,
    pub run_path: PathBuf,
    pub budget: Budget,
}

pub(super) fn prepare(root: Option<&Path>, mut input: Input) -> DynResult<Prepared> {
    input.validate()?;
    if (input.builds.is_empty() && input.build_jobs.is_empty() && input.engine_config.is_none())
        || (input.output.try_exists()? && !input.resume)
    {
        return Err("execute-run requires arms and an unused output directory".into());
    }
    super::run_arms::prepare_config(&mut input)?;
    let budget = Budget::new(&input);
    let reader_prepared = prepare_builds(root, &mut input)?;
    verify_model(&input)?;
    if let Some(dataset) = &input.dataset {
        if input.resume {
            super::run_input::verify(dataset)?;
        } else if !reader_prepared {
            super::run_input::prepare(root, dataset, (&input.manifest, &input.requirements))?;
        }
    }
    let bytes = std::fs::read(&input.manifest)?;
    let manifest: Manifest = serde_json::from_slice(&bytes)?;
    let metadata = super::manifest_preflight::validate(&manifest, &input.requirements)?;
    let config = super::run_arms::append(&mut input, &budget)?;
    input.verify_builds(&budget)?;
    let destination = input.output.join("inputs/captured-trajectories.json");
    std::fs::create_dir_all(destination.parent().ok_or("missing input directory")?)?;
    if !input.resume {
        std::fs::write(&destination, &bytes)?;
    }
    let manifest_sha256 = hex::encode(Sha256::digest(&bytes));
    let mut document = document(
        &input,
        &destination,
        manifest_sha256,
        serde_json::to_value(metadata)?,
    )?;
    if !input.context_qualification.is_mesh() {
        for build in &input.builds {
            document["context_preflight"][build.label()] =
                super::run_qualification::not_requested();
        }
    }
    // Do not overwrite retained plan evidence until resume admission has passed.
    if let Some(config) = &config {
        document["config"]["engine_config"] = serde_json::to_value(config)?;
    } else if input
        .builds
        .iter()
        .any(|build| matches!(build, Build::External(_)))
    {
        let metadata = super::run_arms::metadata(&input, &document);
        document["config"]["engine_config"] = metadata;
    }
    let run_path = input.output.join("run.json");
    Ok(Prepared {
        input,
        document,
        manifest: destination,
        run_path,
        budget,
    })
}

fn prepare_builds(root: Option<&Path>, input: &mut Input) -> DynResult<bool> {
    if input.build_jobs.is_empty() {
        return Ok(false);
    }
    if !input.builds.is_empty() || input.resume {
        return Err("build jobs require a fresh run without prebuilt identities".into());
    }
    verify_model(input)?;
    let reader = if let Some(dataset) = &input.dataset {
        super::run_input::prepare(root, dataset, (&input.manifest, &input.requirements))?;
        true
    } else {
        false
    };
    let manifest: Manifest = serde_json::from_slice(&std::fs::read(&input.manifest)?)?;
    super::manifest_preflight::validate(&manifest, &input.requirements)?;
    input.builds = super::run_builds::prepare(root, &input.build_jobs, &input.output)?;
    input.build_jobs.clear();
    Ok(reader)
}

fn verify_model(input: &Input) -> DynResult<()> {
    let mesh = !input.build_jobs.is_empty()
        || input
            .builds
            .iter()
            .any(|build| matches!(build, Build::Mesh(_)));
    if input.context_qualification.is_mesh() || (mesh && !input.model_sha256.is_empty()) {
        super::model_preflight::verify(
            &input.model,
            &input.model_sha256,
            if input.context_qualification.is_mesh() {
                input.minimum_context_tokens
            } else {
                input.minimum_context_tokens.max(1)
            },
        )?;
    }
    Ok(())
}

fn document(
    input: &Input,
    destination: &Path,
    digest: String,
    metadata: serde_json::Value,
) -> DynResult<serde_json::Value> {
    let mut document = serde_json::json!({"schema_version":3,"config":{
        "model":input.model,"concurrency":input.requirements.concurrency,"passes":input.passes,"replay_mode":input.replay_mode,
        "max_output_tokens":input.max_output_tokens,"warmup_turns":input.requirements.warmup_turns,
        "minimum_context_tokens":input.minimum_context_tokens,"minimum_session_prompt_tokens":input.minimum_session_prompt_tokens,
        "require_recurrent_restores":input.require_recurrent_restores},
        "inputs":{"kind":"captured","manifest":destination,"manifest_sha256":digest,"cohorts":metadata},
        "builds":input.builds,"context_preflight":{},"results":[],"order":[]});
    input.record_selection(&mut document)?;
    if let Some(dataset) = &input.dataset {
        document["inputs"]["kind"] = "thoughtworks".into();
        document["inputs"]["dataset"] =
            serde_json::json!({"revision":dataset.revision,"sha256":dataset.sha256});
        document["inputs"]["dataset_file"] = serde_json::to_value(&dataset.file)?;
        document["inputs"]["dataset_file_sha256"] = dataset.sha256.clone().into();
    }
    document["plan_sha256"] = super::cohort_identity::digest(&serde_json::to_value(input)?)?.into();
    Ok(document)
}
