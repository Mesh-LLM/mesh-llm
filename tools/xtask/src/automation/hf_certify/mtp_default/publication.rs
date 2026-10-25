use super::super::{admission, bootstrap, execution, publication as publisher};
use super::{check, contract::Input};
use crate::{command::DynResult, process::Cancellation};
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    path::Path,
    time::{Duration, Instant},
};
#[derive(Serialize)]
struct RepoOptions<'a> {
    repo: &'a str,
    credential_file: &'a Path,
    output_directory: &'a Path,
    timeout_seconds: u64,
    confirm: bool,
}
fn ensure(
    input: &Input,
    root: &Path,
    deadline: Instant,
    cancel: &Cancellation,
    evidence: &mut Value,
) -> DynResult<String> {
    check(deadline, cancel)?;
    if bootstrap::execution::observe(&input.repository_helper_source.path, deadline, cancel)?
        != input.repository_helper_source.sha256
    {
        return Err("repository helper source pin refused".into());
    }
    if bootstrap::execution::observe(&input.repository_helper.path, deadline, cancel)?
        != input.repository_helper.sha256
    {
        return Err("repository helper pin refused".into());
    }
    let output = root.join("repository");
    let timeout = deadline
        .checked_duration_since(Instant::now())
        .and_then(|d| d.checked_sub(Duration::from_secs(3)))
        .map(|d| d.as_secs().min(1200))
        .filter(|s| *s > 0)
        .ok_or("repository allowance exhausted")?;
    let request = RepoOptions {
        repo: &input.composite_repo,
        credential_file: &input.credential_file,
        output_directory: &output,
        timeout_seconds: timeout,
        confirm: true,
    };
    let args = vec![
        "ensure-repo".into(),
        "--confirm".into(),
        "--repo".into(),
        input.composite_repo.clone(),
        "--credential-file".into(),
        input
            .credential_file
            .to_str()
            .ok_or("credential Unicode")?
            .into(),
        "--output-directory".into(),
        output.to_str().ok_or("repository Unicode")?.into(),
        "--timeout-seconds".into(),
        timeout.to_string(),
    ];
    let raw = execution::run_process(
        &input.repository_helper.path,
        args,
        root,
        "ensure-repository",
        deadline,
        cancel,
    )?;
    evidence["repository_process"] = publisher::process_observation(&raw);
    let receipt: Value =
        serde_json::from_slice(&admission::read(&output.join("repository.json"), 1048576)?)?;
    evidence["repository_receipt"] = receipt.clone();
    if !execution::clean(&raw)
        || receipt["schema_version"] != 1
        || receipt["status"] != "REPOSITORY_READY"
        || receipt["request_sha256"] != admission::digest(&serde_json::to_vec(&request)?)
        || receipt["repository"]["repo"] != input.composite_repo
        || receipt["repository"]["completed"] != true
        || !receipt["repository"]["error"].is_null()
    {
        return Err("repository provisioning receipt refused".into());
    }
    let parent = receipt["repository"]["observed_parent"]
        .as_str()
        .filter(|s| super::contract::pin(s, 40))
        .ok_or("repository parent pin")?
        .to_string();
    if bootstrap::execution::observe(&input.repository_helper.path, deadline, cancel)?
        != input.repository_helper.sha256
        || bootstrap::execution::observe(&input.repository_helper_source.path, deadline, cancel)?
            != input.repository_helper_source.sha256
    {
        return Err("repository helper/source changed".into());
    }
    check(deadline, cancel)?;
    Ok(parent)
}
pub(super) fn ordered(plan: &Value) -> DynResult<Vec<Value>> {
    let rows = plan["entries"]
        .as_array()
        .filter(|r| r.len() >= 2 && r.len() <= 128)
        .ok_or("complete ordered compose entries")?;
    let mut names = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for row in rows {
        let path = std::path::PathBuf::from(row["local"].as_str().ok_or("entry local")?);
        let hash = row["sha256"]
            .as_str()
            .filter(|h| super::contract::pin(h, 64))
            .ok_or("entry hash")?;
        let name = row["path_in_repo"].as_str().ok_or("entry name")?;
        if !path.is_absolute() || !names.insert(name) {
            return Err("ordered entry path/name refusal".into());
        }
        let metadata = std::fs::symlink_metadata(&path)?;
        if !metadata.is_file() || metadata.len() == 0 {
            return Err("ordered GGUF regular file refused".into());
        }
        out.push(json!({"path":path,"path_in_repo":name,"sha256":hash,"byte_size":metadata.len()}));
    }
    Ok(out)
}
pub(super) fn execute(
    input: &Input,
    root: &Path,
    plan: &Value,
    deadline: Instant,
    cancel: &Cancellation,
    evidence: &mut Value,
) -> DynResult<()> {
    check(deadline, cancel)?;
    let shards = ordered(plan)?;

    let mut sidecars = Vec::new();
    for row in &input.sidecars {
        sidecars.push(json!({"path":row.artifact.path,"path_in_repo":row.path_in_repo,"sha256":row.artifact.sha256,"byte_size":std::fs::symlink_metadata(&row.artifact.path)?.len()}));
    }
    let card = root.join("README.md");
    let bytes=format!("# {}\n\nNative Nemotron MTP composition. Target and converted checkpoint bytes are supplied/pinned and verified by the native attachment owner.\n\nCheckpoint: {} @ {}\nTokenizer source: {} @ {}\nMesh source: {}\nMTP block: {}\n\nNo measured performance or real-family acceptance is claimed by this card.\n",input.composite_basename,input.checkpoint.repo,input.checkpoint.revision,input.tokenizer_source.repo,input.tokenizer_source.revision,input.bootstrap.mesh_commit,input.mtp_block).into_bytes();
    if input.sidecars.iter().any(|s| s.path_in_repo == "README.md") {
        return Err("derived card conflicts with supplied sidecar".into());
    }
    std::fs::write(&card, &bytes)?;
    sidecars.push(json!({"path":card,"path_in_repo":"README.md","sha256":admission::digest(&bytes),"byte_size":bytes.len()}));
    let phase = root.join("publication");
    std::fs::create_dir(&phase)?;
    let mut request: publisher::Request = serde_json::from_value(
        json!({"helper":input.publisher_helper,"helper_source":input.publisher_source,"input":{"schema_version":1,"repo":input.composite_repo,"parent_commit":"0".repeat(40),"shards":shards,"sidecars":sidecars,"credential_file":input.credential_file,"execution_timeout_ms":86400000}}),
    )?;
    publisher::validate_artifacts(&request.input.shards, &request.input.sidecars)?;
    request.input.parent_commit = ensure(input, root, deadline, cancel, evidence)?;
    publisher::execute(
        &request,
        &phase,
        deadline,
        cancel,
        &mut evidence["ordered_publication"],
    )?;
    check(deadline, cancel)
}
