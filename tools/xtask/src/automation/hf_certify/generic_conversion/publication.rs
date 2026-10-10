//! Existing repo provisioning followed by one complete parent-bound ordered GGUF/sidecar publication.
use super::{contract::Input, guard, identity};
use crate::{
    automation::hf_certify::{admission, bootstrap, execution, publication as model},
    command::DynResult,
    process::Cancellation,
};
use serde_json::{Value, json};
use std::{
    path::Path,
    time::{Duration, Instant},
};
fn pins(input: &Input, until: Instant, cancel: &Cancellation) -> DynResult<()> {
    for pin in [
        input.helper.as_ref(),
        input.helper_source.as_ref(),
        input.model_publisher.as_ref(),
        input.model_publisher_source.as_ref(),
    ] {
        let pin = pin.ok_or("generic repository helper/source absent")?;
        if bootstrap::execution::observe(&pin.path, until, cancel)? != pin.sha256 {
            return Err("generic repository helper/source pin mismatch".into());
        }
    }
    guard(until, cancel)
}
fn seconds(until: Instant, cancel: &Cancellation, cap: u64) -> DynResult<u64> {
    guard(until, cancel)?;
    until
        .checked_duration_since(Instant::now())
        .and_then(|d| d.checked_sub(Duration::from_secs(3)))
        .map(|d| d.as_secs().min(cap))
        .filter(|n| *n > 0)
        .ok_or_else(|| "generic publication cleanup reserve".into())
}
pub(super) fn execute(
    input: &Input,
    root: &Path,
    until: Instant,
    cancel: &Cancellation,
    evidence: &mut Value,
) -> DynResult<()> {
    if !input.publish_confirmed || input.dry_run {
        return Err("generic publication explicit confirmation required".into());
    }
    let roster = identity::observe(input, true, root, "before-publication", until, cancel)?;
    let artifact = |name: &str| -> DynResult<model::PublisherArtifact> {
        let file = roster
            .files
            .get(name)
            .ok_or("generic publication roster missing")?;
        Ok(model::PublisherArtifact {
            path: file.path.clone(),
            path_in_repo: name.into(),
            sha256: file.sha256.clone(),
            byte_size: *roster
                .sizes
                .get(name)
                .ok_or("generic observed artifact size absent")?,
        })
    };
    let shard_names = input.shards(
        roster
            .effective_splits
            .ok_or("generic publication effective roster absent")?,
    );
    let shards = shard_names
        .iter()
        .map(|name| artifact(name))
        .collect::<DynResult<Vec<_>>>()?;
    let sidecars = roster
        .files
        .keys()
        .filter(|name| name.as_str() != "__binary__" && !shard_names.contains(name))
        .map(|name| artifact(name))
        .collect::<DynResult<Vec<_>>>()?;
    model::validate_artifacts(&shards, &sidecars)?;
    let credential = input
        .credential_file
        .as_ref()
        .ok_or("generic explicit credential file")?;
    let out = root.join("repository");
    let timeout = seconds(until, cancel, 1200)?;
    let args = vec![
        "ensure-repo".into(),
        "--confirm".into(),
        "--repo".into(),
        input.target_repo.clone(),
        "--credential-file".into(),
        credential.display().to_string(),
        "--output-directory".into(),
        out.display().to_string(),
        "--timeout-seconds".into(),
        timeout.to_string(),
    ];
    pins(input, until, cancel)?;
    let process = execution::run_process(
        &input
            .helper
            .as_ref()
            .ok_or("generic repository helper")?
            .path,
        args,
        root,
        "repository",
        until,
        cancel,
    )?;
    let mut repository = json!({"process":model::process_observation(&process),"receipt":null});
    if let Ok(bytes) = admission::read(&out.join("repository.json"), 1048576) {
        repository["receipt"] = serde_json::from_slice(&bytes)?;
    }
    evidence["repository"] = repository.clone();
    pins(input, until, cancel)?;
    if !execution::clean(&process) {
        return Err(
            "generic repository helper incomplete; confirmed/uncertain child receipt retained"
                .into(),
        );
    }
    let receipt = &repository["receipt"];
    if !super::receipt::repository(
        receipt,
        &input.target_repo,
        &super::helper_contract::repo_hash(input, &out, timeout)?,
    ) {
        return Err("generic repository receipt contradictory/unconfirmed".into());
    }
    let request = model::Request {
        helper: input
            .model_publisher
            .clone()
            .ok_or("generic model publisher")?,
        helper_source: input
            .model_publisher_source
            .clone()
            .ok_or("generic publisher source")?,
        input: model::PublisherInput {
            schema_version: 1,
            repo: input.target_repo.clone(),
            parent_commit: receipt["repository"]["observed_parent"]
                .as_str()
                .ok_or("repository parent")?
                .into(),
            shards,
            sidecars,
            credential_file: Some(credential.clone()),
            execution_timeout_ms: seconds(until, cancel, 86400)? * 1000,
        },
    };
    let output = root.join("model-publication");
    std::fs::create_dir(&output)?;
    let mut publication = Value::Null;
    let result = model::execute(&request, &output, until, cancel, &mut publication);
    evidence["model_publication"] = publication;
    result?;
    let after = identity::observe(input, true, root, "after-publication", until, cancel)?;
    if after.files != roster.files
        || after.sizes != roster.sizes
        || after.effective_splits != roster.effective_splits
    {
        return Err("generic published local folder byte custody differs".into());
    }
    guard(until, cancel)?;
    evidence["publication_completed"] = json!(true);
    Ok(())
}
