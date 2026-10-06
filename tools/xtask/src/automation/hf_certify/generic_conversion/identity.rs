//! Complete local file custody in the existing supervised self-worker boundary.
use super::{contract::Input, guard};
use crate::{
    automation::hf_certify::{
        admission::{self, Artifact},
        execution,
    },
    command::DynResult,
    process::Cancellation,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::Instant,
};
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Request {
    input: Input,
    output_roster: bool,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Receipt {
    pub request_sha256: String,
    pub files: BTreeMap<String, Artifact>,
    pub sizes: BTreeMap<String, u64>,
    pub effective_splits: Option<usize>,
}
fn walk(root: &Path, directory: &Path, rows: &mut BTreeMap<String, PathBuf>) -> DynResult<()> {
    if directory.strip_prefix(root)?.components().count() > 32 {
        return Err("generic file roster directory depth bound".into());
    }
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        if rows.len() >= 1024 {
            return Err("generic complete file roster exceeds1024".into());
        }
        if kind.is_dir() {
            walk(root, &entry.path(), rows)?;
        } else if kind.is_file() {
            let path = entry.path();
            let name = path
                .strip_prefix(root)?
                .to_str()
                .ok_or("generic source UTF8 path")?
                .replace('\\', "/");
            if name.len() > 4096 {
                return Err("generic relative path bound".into());
            }
            rows.insert(name, path);
        } else {
            return Err("generic file roster refuses links and nonregular inputs".into());
        }
    }
    Ok(())
}
pub(super) fn worker(args: &[String]) -> DynResult<()> {
    let [a, input, b, output] = args else {
        return Err("generic identity closed flags".into());
    };
    if a != "--input" || b != "--output" {
        return Err("generic identity closed flags".into());
    }
    let request: Request =
        serde_json::from_slice(&admission::read(Path::new(input), 8 * 1048576)?)?;
    request.input.validate()?;
    let mut files = BTreeMap::new();
    let mut sizes = BTreeMap::new();
    let mut effective_splits = None;
    if !request.input.upload_only {
        let root = request.input.source.canonicalize()?;
        if root != request.input.source {
            return Err("generic source must be canonical".into());
        }
        let mut rows = BTreeMap::new();
        walk(&root, &root, &mut rows)?;
        let expected = request
            .input
            .source_files
            .iter()
            .map(|p| (p.path.clone(), p.sha256.clone()))
            .collect::<BTreeMap<_, _>>();
        if rows.len() != expected.len() {
            return Err("generic source complete roster mismatch".into());
        }
        for path in rows.values() {
            let observed = admission::observe(path, false)?;
            if expected.get(path) != Some(&observed.sha256) {
                return Err("generic source byte pin mismatch".into());
            }
        }
        let mut binary = request
            .input
            .binary
            .clone()
            .ok_or("generic supplied binary")?;
        admission::admit(&mut binary, false)?;
        files.insert("__binary__".into(), binary);
    }
    if request.output_roster {
        let root = request.input.artifact_directory();
        let mut rows = BTreeMap::new();
        walk(&root, &root, &mut rows)?;
        let manifest_bytes =
            admission::read(&root.join("skippy-convert-manifest.json"), 8 * 1048576)?;
        let manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes)?;
        let actual = manifest["expected_splits"]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .filter(|n| *n >= request.input.expected_splits && *n <= 1024)
            .ok_or("generic effective split count refused")?;
        effective_splits = Some(actual);
        if manifest["output_basename"] != request.input.output_basename
            || manifest["target_prefix"] != request.input.target_prefix
        {
            return Err("generic converted manifest identity mismatch".into());
        }
        crate::automation::hf_converted_artifact::validate_converted_artifact_manifest(
            &root, &manifest,
        )?;
        for required in request
            .input
            .shards(actual)
            .into_iter()
            .chain(["README.md".into(), "skippy-convert-manifest.json".into()])
        {
            if !rows.contains_key(&required) {
                return Err("generic complete output shard/sidecar roster missing".into());
            }
        }
        for (name, path) in rows {
            sizes.insert(name.clone(), std::fs::metadata(&path)?.len());
            let observed =
                admission::observe(&path, path.extension().is_some_and(|s| s == "gguf"))?;
            if name == "skippy-convert-manifest.json"
                && observed.sha256 != admission::digest(&manifest_bytes)
            {
                return Err("generic admitted manifest byte custody changed".into());
            }
            files.insert(name, observed);
        }
    }
    admission::publish(
        Path::new(output),
        &Receipt {
            request_sha256: admission::digest(&serde_json::to_vec(&request)?),
            files,
            sizes,
            effective_splits,
        },
    )
}
pub(super) fn observe(
    input: &Input,
    output_roster: bool,
    root: &Path,
    label: &str,
    until: Instant,
    cancel: &Cancellation,
) -> DynResult<Receipt> {
    guard(until, cancel)?;
    let request = Request {
        input: input.clone(),
        output_roster,
    };
    let path = root.join(format!("{label}-input.json"));
    let output = root.join(format!("{label}-identity.json"));
    admission::publish(&path, &request)?;
    let args = vec![
        "automation".into(),
        "hf-certify".into(),
        "generic-identity-worker".into(),
        "--input".into(),
        path.display().to_string(),
        "--output".into(),
        output.display().to_string(),
    ];
    let process =
        execution::run_process(&std::env::current_exe()?, args, root, label, until, cancel)?;
    if !execution::clean(&process) {
        return Err("generic identity child incomplete".into());
    }
    let result: Receipt = serde_json::from_slice(&admission::read(&output, 8 * 1048576)?)?;
    if result.request_sha256 != admission::digest(&serde_json::to_vec(&request)?) {
        return Err("generic identity correlation mismatch".into());
    }
    guard(until, cancel)?;
    Ok(result)
}
