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
    let captured = if input.dataset.is_none() {
        Some(captured_provenance(&input.manifest, &bytes)?)
    } else {
        None
    };
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
    if let Some(captured) = captured {
        document["inputs"]
            .as_object_mut()
            .ok_or("missing inputs object")?
            .extend(captured);
    }
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

fn captured_provenance(
    source: &Path,
    bytes: &[u8],
) -> DynResult<serde_json::Map<String, serde_json::Value>> {
    let parsed: serde_json::Value = serde_json::from_slice(bytes)?;
    let metadata = match parsed.get("metadata") {
        None => serde_json::Map::new(),
        Some(serde_json::Value::Object(metadata)) => metadata.clone(),
        Some(_) => return Err("captured manifest metadata must be an object".into()),
    };
    for key in ["name", "revision"] {
        if metadata
            .get(key)
            .is_some_and(|value| value.as_str().is_none_or(str::is_empty))
        {
            return Err(
                format!("captured manifest metadata {key} must be a nonempty string").into(),
            );
        }
    }
    let digest = hex::encode(Sha256::digest(bytes));
    let name = match metadata.get("name") {
        Some(name) => name.clone(),
        None => serde_json::to_value(
            source
                .file_stem()
                .and_then(|stem| stem.to_str())
                .ok_or("non-Unicode manifest filename")?,
        )?,
    };
    let revision = metadata
        .get("revision")
        .cloned()
        .unwrap_or(serde_json::Value::String(digest.clone()));
    let value = serde_json::json!({"dataset":{"name":name,"revision":revision},"metadata":metadata,"source_manifest":source,"source_manifest_sha256":digest});
    Ok(value
        .as_object()
        .ok_or("missing captured provenance object")?
        .clone())
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

#[cfg(test)]
mod captured_tests {
    use super::*;
    #[test]
    fn captured_metadata_has_digest_defaults_and_rejects_malformed_fields() {
        let bytes = br#"{"cohorts":{}}"#;
        let provenance = captured_provenance(Path::new("/fixture/capture.json"), bytes).unwrap();
        assert_eq!(provenance["dataset"]["name"], "capture");
        assert_eq!(
            provenance["dataset"]["revision"],
            hex::encode(Sha256::digest(bytes))
        );
        assert_eq!(provenance["metadata"], serde_json::json!({}));
        for metadata in [
            serde_json::json!(null),
            serde_json::json!([]),
            serde_json::json!({"revision":7}),
            serde_json::json!({"name":""}),
        ] {
            let bytes = serde_json::to_vec(&serde_json::json!({"metadata":metadata})).unwrap();
            assert!(captured_provenance(Path::new("/fixture/capture.json"), &bytes).is_err());
        }
    }
}
