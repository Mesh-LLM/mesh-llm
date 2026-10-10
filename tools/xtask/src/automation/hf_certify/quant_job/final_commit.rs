//! One common immutable commit, reacquisition and independent native verification.
use super::*;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
#[derive(Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Artifact {
    path: String,
    sha256: String,
    byte_size: u64,
}
pub(super) fn window_artifacts(
    input: &window::contract::Input,
    root: &Path,
    row: &Value,
    roster: &mut Vec<Value>,
) -> DynResult<()> {
    if row["window_uploaded"] != true {
        return Err("quant window not verified".into());
    }
    let (record, commit) = if let Some(r) = &input.resume {
        (r.record.path.clone(), r.commit.clone())
    } else {
        (
            root.join("window-record.json"),
            row["record_commit"]
                .as_str()
                .ok_or("record commit absent")?
                .into(),
        )
    };
    let bytes = admission::read(&record, 1048576)?;
    let v: Value = serde_json::from_slice(&bytes)?;
    if !bootstrap::contract::hex(&commit, 40)
        || v["context_sha256"] != input.context_sha256()?
        || v["ordinal"] != input.ordinal
        || v["relative_path"] != input.remote_path()
        || v["window_uploaded"] != true
    {
        return Err("quant window record correlation refused".into());
    }
    let a = Artifact {
        path: input.remote_path(),
        sha256: v["artifact"]["sha256"]
            .as_str()
            .ok_or("window hash")?
            .into(),
        byte_size: v["artifact"]["byte_size"].as_u64().ok_or("window size")?,
    };
    if !bootstrap::contract::hex(&a.sha256, 64)
        || !(24..=1024_u64.pow(4)).contains(&a.byte_size)
        || roster.iter().any(|r| r["path"] == a.path)
    {
        return Err("quant window artifact invalid/duplicate".into());
    }
    roster.push(serde_json::to_value(a)?);
    roster.push(json!({"path":input.record_path(),"sha256":admission::digest(&bytes),"byte_size":bytes.len()}));
    Ok(())
}
pub(super) fn publish_and_verify(
    input: &Input,
    root: &Path,
    until: Instant,
    cancel: &Cancellation,
    roster: &mut Vec<Value>,
    evidence: &mut Value,
) -> DynResult<PathBuf> {
    let w = &input.window_template;
    if roster.len() != w.expected_splits as usize * 2 {
        return Err("quant complete window roster absent".into());
    }
    for (label, path, a) in [
        ("recipe-upload", "tensor-policy.recipe", &w.recipe),
        ("manifest-upload", "quantization-manifest.json", &w.manifest),
    ] {
        let mut detail = json!({});
        let published = window::helper::upload(
            w,
            &a.path,
            path,
            false,
            window::helper::Context {
                root,
                until,
                cancel,
                evidence: &mut detail,
                label,
            },
        )?;
        if published["identity"]["sha256"] != a.sha256 {
            return Err("quant durable recipe/manifest bytes changed".into());
        }
        roster.push(
            json!({"path":path,"sha256":a.sha256,"byte_size":published["identity"]["byte_size"]}),
        );
        evidence[label] = evidence::retain(root, label, &detail)?;
    }
    let plan = json!({"schema_version":1,"context_sha256":w.context_sha256()?,"expected_splits":w.expected_splits,"artifacts":roster});
    let plan_path = root.join("quantization-roster.json");
    admission::publish(&plan_path, &plan)?;
    let plan_bytes = admission::read(&plan_path, 1048576)?;
    let published = window::helper::upload(
        w,
        &plan_path,
        "quantization-roster.json",
        false,
        window::helper::Context {
            root,
            until,
            cancel,
            evidence,
            label: "final-roster-upload",
        },
    )?;
    let commit = window::helper::commit(&published)?;
    roster.push(json!({"path":"quantization-roster.json","sha256":admission::digest(&plan_bytes),"byte_size":plan_bytes.len()}));
    verify_common(input, root, until, cancel, roster, evidence, &commit)
}
fn verify_common(
    input: &Input,
    root: &Path,
    until: Instant,
    cancel: &Cancellation,
    roster: &[Value],
    evidence: &mut Value,
    commit: &str,
) -> DynResult<PathBuf> {
    let w = &input.window_template;
    let request = json!({"schema_version":1,"repo":w.target_repo,"commit":commit,
        "gguf_prefix":w.target_prefix,"basename":w.basename,"expected_splits":w.expected_splits,"artifacts":roster});
    let path = root.join("common-commit-request.json");
    admission::publish(&path, &request)?;
    let digest = admission::digest(&admission::read(&path, 1048576)?);
    let output = root.join("common-commit");
    let args = vec![
        "verify-quant-commit".into(),
        "--input".into(),
        path.to_string_lossy().into(),
        "--credential-file".into(),
        w.credential_file.to_string_lossy().into(),
        "--output-directory".into(),
        output.to_string_lossy().into(),
        "--timeout-seconds".into(),
        seconds(until, cancel)?.to_string(),
    ];
    window::pin(&w.helper_source, until, cancel)?;
    let result = child(
        &w.helper,
        args,
        root,
        "common-commit",
        until,
        cancel,
        evidence,
    );
    let receipt: Value =
        serde_json::from_slice(&admission::read(&output.join("commit.json"), 1048576)?)?;
    evidence["common_commit"] = evidence::retain(root, "common-commit", &receipt)?;
    result?;
    window::pin(&w.helper_source, until, cancel)?;
    let v = &receipt["verification"];
    let expected: Vec<Artifact> = roster
        .iter()
        .cloned()
        .map(serde_json::from_value)
        .collect::<Result<_, _>>()?;
    let actual: Vec<Artifact> = serde_json::from_value(v["verified"].clone())?;
    let artifact_root = PathBuf::from(v["artifact_root"].as_str().ok_or("artifact root absent")?);
    if receipt["request_sha256"] != digest
        || receipt["status"] != "QUANT_COMMIT_BYTES_VERIFIED"
        || v["repo"] != w.target_repo
        || v["commit"] != commit
        || v["completed"] != true
        || !v["error"].is_null()
        || actual != expected
        || artifact_root != output.join("artifacts")
        || artifact_root.canonicalize()? != artifact_root
    {
        return Err("quant full common-commit verification refused".into());
    }
    for a in actual {
        window::pin(
            &admission::Artifact {
                path: artifact_root.join(a.path),
                sha256: a.sha256,
            },
            until,
            cancel,
        )?;
    }
    evidence["full_roster_verified"] = json!(true);
    evidence["final_commit"] = json!(commit);
    check(until, cancel)?;
    Ok(artifact_root)
}
pub(super) fn native_verify(
    input: &Input,
    artifacts: &Path,
    root: &Path,
    until: Instant,
    cancel: &Cancellation,
    evidence: &mut Value,
) -> DynResult<()> {
    let w = &input.window_template;
    let mut manifest: Value = serde_json::from_slice(&admission::read(&w.manifest.path, 1048576)?)?;
    manifest["target"] = json!(artifacts);
    let path = root.join("immutable-verification-manifest.json");
    admission::publish(&path, &manifest)?;
    window::pin(&input.loader, until, cancel)?;
    let value = observed(
        &w.tool,
        vec![
            "verify-job".into(),
            "--manifest".into(),
            path.to_string_lossy().into(),
            "--llama-load".into(),
            "--llama-cli".into(),
            input.loader.path.to_string_lossy().into(),
            "--check-tensors".into(),
            "--json".into(),
        ],
        root,
        "native-verify",
        until,
        cancel,
        evidence,
    )?;
    evidence["verify_job"] = value.clone();
    window::pin(&input.loader, until, cancel)?;
    let a = &value["artifact"];
    let first = artifacts.join(w.remote_path());
    if a["complete"] != true
        || a["expected_splits"] != w.expected_splits
        || a["completed_count"] != w.expected_splits
        || a["root"] != artifacts.to_string_lossy().as_ref()
        || a["prefix"] != w.target_prefix
        || a["basename"] != w.basename
        || value["llama_load"]["success"] != true
        || value["llama_load"]["status_code"] != 0
        || value["llama_load"]["llama_cli"] != input.loader.path.to_string_lossy().as_ref()
        || value["llama_load"]["model"] != first.to_string_lossy().as_ref()
    {
        return Err("quant native full verification/load refused".into());
    }
    evidence["verify_job"]["completed"] = json!(true);
    check(until, cancel)
}
