//! Independent source-complete package verification and same-commit publication.
use super::*;
use std::{collections::BTreeMap, path::PathBuf};
fn relative(s: &str) -> bool {
    !s.is_empty()
        && !Path::new(s).is_absolute()
        && Path::new(s)
            .components()
            .all(|c| matches!(c, std::path::Component::Normal(_)))
}
fn files(
    root: &Path,
    at: &Path,
    out: &mut BTreeMap<String, PathBuf>,
    until: Instant,
    cancel: &Cancellation,
) -> DynResult<()> {
    for entry in std::fs::read_dir(at)? {
        check(until, cancel)?;
        let path = entry?.path();
        let m = std::fs::symlink_metadata(&path)?;
        if m.is_dir() {
            files(root, &path, out, until, cancel)?;
        } else if m.is_file() {
            out.insert(
                path.strip_prefix(root)?
                    .to_str()
                    .ok_or("package path encoding")?
                    .into(),
                path,
            );
        } else {
            return Err("package foreign/symlink entry refused".into());
        }
        if out.len() > 4096 {
            return Err("package publication roster exceeds bound".into());
        }
    }
    Ok(())
}
fn admitted(
    input: &Input,
    root: &Path,
    value: &Value,
    until: Instant,
    cancel: &Cancellation,
) -> DynResult<BTreeMap<String, admission::Artifact>> {
    let w = &input.window_template;
    if value["schema_version"] != 2
        || value["format"] != "gguf"
        || value["source_model"]["repo"] != w.target_repo
        || value["source_model"]["revision"] != input_commit(input, root)?
        || value["source_model"]["primary_file"] != w.remote_path()
    {
        return Err("package immutable quant source binding refused".into());
    }
    let common: Value = serde_json::from_slice(&admission::read(
        &root
            .parent()
            .ok_or("package parent")?
            .join("common-commit/commit.json"),
        1048576,
    )?)?;
    let verified = common["verification"]["verified"]
        .as_array()
        .ok_or("common quant artifacts")?;
    let expected: BTreeMap<_, _> = verified
        .iter()
        .filter(|a| a["path"].as_str().is_some_and(|s| s.ends_with(".gguf")))
        .map(|a| {
            (
                a["path"].as_str().unwrap().to_owned(),
                (a["sha256"].clone(), a["byte_size"].clone()),
            )
        })
        .collect();
    let supplied = value["source_model"]["files"]
        .as_array()
        .ok_or("package source files")?;
    let mut actual_sources = BTreeMap::new();
    for a in supplied {
        let name = a["path"].as_str().ok_or("package source path")?;
        if actual_sources
            .insert(
                name.to_owned(),
                (a["sha256"].clone(), a["byte_size"].clone()),
            )
            .is_some()
        {
            return Err("package duplicate source refused".into());
        }
    }
    if actual_sources != expected || expected.len() != w.expected_splits as usize {
        return Err("package complete quant source bytes refused".into());
    }
    let entries = value["artifact_catalog"]["entries"]
        .as_array()
        .ok_or("package artifacts absent")?;
    if entries.is_empty() || entries.len() >= 4096 {
        return Err("package artifact count refused".into());
    }
    let mut actual = BTreeMap::new();
    files(root, root, &mut actual, until, cancel)?;
    let mut declared = BTreeMap::new();
    for a in entries {
        let name = a["path"].as_str().ok_or("package artifact path absent")?;
        let hash = a["sha256"].as_str().ok_or("package artifact SHA absent")?;
        if !relative(name)
            || !bootstrap::contract::hex(hash, 64)
            || declared.contains_key(name)
            || a["byte_size"].as_u64() != Some(std::fs::symlink_metadata(root.join(name))?.len())
        {
            return Err("package artifact identity refused".into());
        }
        declared.insert(
            name.to_owned(),
            admission::Artifact {
                path: root.join(name),
                sha256: hash.into(),
            },
        );
    }
    let bytes = admission::read(&root.join("model-package.json"), 1048576)?;
    declared.insert(
        "model-package.json".into(),
        admission::Artifact {
            path: root.join("model-package.json"),
            sha256: admission::digest(&bytes),
        },
    );
    if actual.keys().collect::<Vec<_>>() != declared.keys().collect::<Vec<_>>() {
        return Err("package extra/missing file refused".into());
    }
    for a in declared.values() {
        window::pin(a, until, cancel)?;
    }
    Ok(declared)
}
// The coordinator stores the immutable quant commit alongside, never inside the package bytes.
fn input_commit(_input: &Input, root: &Path) -> DynResult<String> {
    let v: Value = serde_json::from_slice(&admission::read(
        &root
            .parent()
            .ok_or("package parent")?
            .join("common-commit/commit.json"),
        1048576,
    )?)?;
    Ok(v["verification"]["commit"]
        .as_str()
        .ok_or("quant common commit absent")?
        .into())
}
pub(super) fn execute(
    input: &Input,
    p: &contract::Package,
    artifacts: &Path,
    root: &Path,
    until: Instant,
    cancel: &Cancellation,
    evidence: &mut Value,
) -> DynResult<()> {
    for a in [&p.writer, &p.writer_source, &p.generation_defaults] {
        window::pin(a, until, cancel)?;
    }
    let w = &input.window_template;
    let commit = evidence["final_commit"]
        .as_str()
        .ok_or("quant commit absent")?
        .to_owned();
    let package_root = root.join("package");
    let first = artifacts.join(w.remote_path());
    child(
        &p.writer,
        vec![
            "write-package".into(),
            first.to_string_lossy().into(),
            "--out-dir".into(),
            package_root.to_string_lossy().into(),
            "--model-id".into(),
            format!("{}:{}", w.target_repo, w.remote_path()),
            "--source-repo".into(),
            w.target_repo.clone(),
            "--source-revision".into(),
            commit,
            "--source-file".into(),
            w.remote_path(),
            "--generation-defaults".into(),
            p.generation_defaults.path.to_string_lossy().into(),
            "--max-artifact-bytes".into(),
            p.max_artifact_bytes.to_string(),
        ],
        root,
        "package-write",
        until,
        cancel,
        evidence,
    )?;
    let verified = observed(
        &p.writer,
        vec![
            "verify-package-v2".into(),
            package_root.to_string_lossy().into(),
            "--source".into(),
            first.to_string_lossy().into(),
            "--source-file".into(),
            w.remote_path(),
        ],
        root,
        "package-verify",
        until,
        cancel,
        evidence,
    )?;
    let manifest: Value = serde_json::from_slice(&admission::read(
        &package_root.join("model-package.json"),
        1048576,
    )?)?;
    let roster = admitted(input, &package_root, &manifest, until, cancel)?;
    if verified["source_completeness_verified"] != true
        || verified["package_id"] != manifest["package_id"]
        || verified["checked_source_files"] != w.expected_splits
        || verified["checked_artifacts"]
            != manifest["artifact_catalog"]["entries"]
                .as_array()
                .ok_or("entries")?
                .len()
        || verified["checked_tensors"].as_u64().unwrap_or(0) == 0
    {
        return Err("independent package source verification refused".into());
    }
    evidence["package"] = json!({"completed":false,"verification":verified,"final_commit":null});
    let final_commit = publish_package(input, p, roster, root, until, cancel, evidence)?;
    for a in [&p.writer, &p.writer_source, &p.generation_defaults] {
        window::pin(a, until, cancel)?;
    }
    check(until, cancel)?;
    evidence["package"]["completed"] = json!(true);
    evidence["package"]["final_commit"] = json!(final_commit);
    Ok(())
}
fn publish_package(
    input: &Input,
    p: &contract::Package,
    mut roster: BTreeMap<String, admission::Artifact>,
    root: &Path,
    until: Instant,
    cancel: &Cancellation,
    evidence: &mut Value,
) -> DynResult<String> {
    let mut request: window::contract::Input =
        serde_json::from_value(serde_json::to_value(&input.window_template)?)?;
    request.target_repo = p.target_repo.clone();
    let manifest_artifact = roster
        .remove("model-package.json")
        .ok_or("manifest artifact")?;
    let ordered: Vec<_> = roster
        .iter()
        .map(|(s, a)| (s.clone(), a.clone()))
        .chain(std::iter::once((
            "model-package.json".into(),
            manifest_artifact,
        )))
        .collect();
    evidence::repository(
        input,
        &p.target_repo,
        window::helper::Context {
            root,
            until,
            cancel,
            evidence,
            label: "package-repository",
        },
    )?;
    use sha2::Digest as _;
    let mut observations = sha2::Sha256::new();
    let mut observation_count = 0_u32;
    let mut final_commit = String::new();
    for (i, (name, a)) in ordered.iter().enumerate() {
        let mut detail = json!({});
        let pubvalue = window::helper::upload(
            &request,
            &a.path,
            name,
            false,
            window::helper::Context {
                root,
                until,
                cancel,
                evidence: &mut detail,
                label: &format!("package-upload-{i:04}"),
            },
        )?;
        if pubvalue["identity"]["sha256"] != a.sha256 {
            return Err("package uploaded bytes differ".into());
        }
        let reference = evidence::retain(root, &format!("package-upload-{i:04}"), &detail)?;
        observations.update(serde_json::to_vec(&reference)?);
        observation_count += 1;
        final_commit = window::helper::commit(&pubvalue)?;
    }
    for (i, (name, a)) in ordered.iter().enumerate() {
        let mut detail = json!({});
        window::helper::verify(
            &request,
            a,
            &final_commit,
            name,
            window::helper::Context {
                root,
                until,
                cancel,
                evidence: &mut detail,
                label: &format!("package-common-{i:04}"),
            },
        )?;
        let reference = evidence::retain(root, &format!("package-common-{i:04}"), &detail)?;
        observations.update(serde_json::to_vec(&reference)?);
        observation_count += 1;
    }
    evidence["package"]["observations"] = json!({"root":root,"files":observation_count,"sha256":observations.finalize().iter().map(|b|format!("{b:02x}")).collect::<String>()});
    Ok(final_commit)
}
