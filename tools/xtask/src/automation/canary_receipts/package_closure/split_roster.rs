//! Exact v2 native recipe and source-owned split-certified architecture projection.
use super::{policy_document, process, producer_receipt::Context, source};
use crate::{automation::canary_receipts::Digest, command::DynResult};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    context: Context,
    root: PathBuf,
    check: bool,
}
#[derive(Serialize)]
struct Roster {
    schema_version: u8,
    native_recipe: Recipe,
    architectures: Vec<String>,
}
#[derive(Serialize)]
struct Recipe {
    llama_upstream_sha: String,
    skippy_abi: String,
    patch_queue_sha256: Digest,
}

pub(super) fn execute(input: &Input) -> DynResult<Value> {
    input.context.validate()?;
    let result = admit(&input.root, input.check)?;
    input.context.validate()?;
    process::check()?;
    Ok(result)
}

pub(super) fn admit(root: &Path, check: bool) -> DynResult<Value> {
    if !root.is_absolute() {
        return Err("split roster source must be absolute".into());
    }
    let root = root.canonicalize()?;
    let bytes = render(
        &root,
        &policy_document::read(&root, "ci/llama-canary/family-certified.json")?,
    )?;
    let output_path =
        root.join("crates/mesh-llm-host-runtime/src/inference/skippy/split-certified.json");
    let parent = output_path
        .parent()
        .ok_or("roster output has no parent")?
        .canonicalize()?;
    if !parent.starts_with(&root)
        || fs::symlink_metadata(&output_path)
            .is_ok_and(|metadata| metadata.file_type().is_symlink())
    {
        return Err("split roster output escapes selected source".into());
    }
    output(&root, &output_path, &bytes, check)?;
    process::check()?;
    Ok(
        serde_json::json!({"status":"split_roster_admitted","check":check,"sha256":Digest::of_bytes(&bytes)}),
    )
}

fn frame(hash: &mut Sha256, bytes: &[u8]) -> DynResult<()> {
    hash.update(u64::try_from(bytes.len())?.to_le_bytes());
    hash.update(bytes);
    Ok(())
}
fn queue(root: &Path) -> DynResult<Digest> {
    let (root, members) =
        source::ordered_patch_members(&root.join("third_party/llama.cpp/patches"))?;
    let mut hash = Sha256::new();
    hash.update(b"mesh-llm-skippy-patch-queue-v2\0");
    hash.update(u64::try_from(members.len())?.to_le_bytes());
    for path in members {
        process::check()?;
        frame(
            &mut hash,
            path.strip_prefix(&root)?
                .to_str()
                .ok_or("non-UTF8 patch name")?
                .as_bytes(),
        )?;
        frame(&mut hash, &fs::read(path)?)?;
    }
    Ok(Digest::try_from(hex::encode(hash.finalize()))?)
}
fn abi(root: &Path) -> DynResult<String> {
    let text = fs::read_to_string(root.join("crates/skippy-ffi/src/lib.rs"))?;
    let mut versions = BTreeMap::new();
    for line in text.lines() {
        for name in ["MAJOR", "MINOR", "PATCH"] {
            if let Some(value) = line
                .strip_prefix(&format!("pub const ABI_VERSION_{name}: u32 = "))
                .and_then(|line| line.strip_suffix(';'))
                && versions.insert(name, value.parse::<u32>()?).is_some()
            {
                return Err("duplicate Skippy ABI version component".into());
            }
        }
    }
    if versions.len() != 3 {
        return Err("incomplete Skippy ABI version".into());
    }
    Ok(format!(
        "{}.{}.{}",
        versions["MAJOR"], versions["MINOR"], versions["PATCH"]
    ))
}
fn architectures(manifest: &Value) -> DynResult<Vec<String>> {
    let profiles = manifest["policy"]["profiles"]
        .as_object()
        .ok_or("missing family certification profiles")?;
    let models = manifest["models"]
        .as_array()
        .ok_or("missing family certification models")?;
    let mut result = BTreeSet::new();
    for model in models {
        if !model.is_object() {
            return Err("family certification row must be object".into());
        }
        let class = model["class"].as_str().ok_or("missing workload class")?;
        let profile = model["profile"]
            .as_str()
            .ok_or("missing certification profile")?;
        if [
            "embedding",
            "rerank",
            "encoder_decoder",
            "ocr",
            "speech_synthesis",
            "speech_recognition",
        ]
        .contains(&class)
        {
            if !["workload-smoke", "workload-oracle"].contains(&profile) {
                return Err("non-chat row cannot claim split-certified profile".into());
            }
            continue;
        }
        if class != "causal_generation" || ["workload-smoke", "workload-oracle"].contains(&profile)
        {
            return Err("causal row has invalid workload class/profile".into());
        }
        let Some(policy) = profiles.get(profile).and_then(Value::as_object) else {
            continue;
        };
        if policy.get("status").and_then(Value::as_str) != Some("certified") {
            continue;
        }
        let Some(lanes) = policy.get("required_lanes").and_then(Value::as_array) else {
            continue;
        };
        if !["single-step", "chain", "state-handoff"]
            .iter()
            .all(|lane| lanes.iter().any(|value| value.as_str() == Some(lane)))
        {
            continue;
        }
        let architecture = model["architecture"]
            .as_str()
            .filter(|value| !value.is_empty())
            .ok_or("certified row missing architecture")?;
        result.insert(architecture.to_owned());
    }
    if result.is_empty() {
        return Err("family manifest produced no split-certified architectures".into());
    }
    Ok(result.into_iter().collect())
}
fn render(root: &Path, manifest: &[u8]) -> DynResult<Vec<u8>> {
    let pin = fs::read_to_string(root.join("third_party/llama.cpp/upstream.txt"))?;
    source::revision(pin.trim())?;
    let roster = Roster {
        schema_version: 2,
        native_recipe: Recipe {
            llama_upstream_sha: pin.trim().to_owned(),
            skippy_abi: abi(root)?,
            patch_queue_sha256: queue(root)?,
        },
        architectures: architectures(&serde_json::from_slice(manifest)?)?,
    };
    let mut bytes = serde_json::to_vec_pretty(&roster)?;
    bytes.push(b'\n');
    Ok(bytes)
}
#[cfg(test)]
#[path = "split_roster_tests.rs"]
mod tests;

fn output(root: &Path, path: &Path, bytes: &[u8], check: bool) -> DynResult<()> {
    const MAXIMUM: usize = 8 * 1024 * 1024;
    if bytes.len() > MAXIMUM {
        return Err("split roster exceeds 8 MiB".into());
    }
    let before = match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_file() => Some(metadata),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && !check => None,
        _ => {
            return Err(
                "split roster output must be regular; missing allowed only for write".into(),
            );
        }
    };
    let mut options = fs::OpenOptions::new();
    options
        .read(check)
        .write(!check)
        .create_new(before.is_none());
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let mut file = options.open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file()
        || !path.canonicalize()?.starts_with(root)
        || before
            .as_ref()
            .is_some_and(|before| !policy_document::same_file(before, &metadata))
        || !policy_document::same_file(&metadata, &fs::symlink_metadata(path)?)
    {
        return Err("split roster opened output is not contained regular source".into());
    }
    process::check()?;
    if check {
        check_output(root, path, &mut file, &metadata, bytes)?;
    } else {
        file.set_len(0)?;
        for chunk in bytes.chunks(65536) {
            process::check()?;
            file.write_all(chunk)?;
        }
    }
    if !path.canonicalize()?.starts_with(root)
        || !policy_document::same_file(&file.metadata()?, &fs::symlink_metadata(path)?)
    {
        return Err("split roster pathname changed during operation".into());
    }
    process::check()?;
    Ok(())
}

fn check_output(
    root: &Path,
    path: &Path,
    file: &mut fs::File,
    metadata: &fs::Metadata,
    bytes: &[u8],
) -> DynResult<()> {
    const MAXIMUM: usize = 8 * 1024 * 1024;
    if metadata.len() > MAXIMUM as u64 {
        return Err("split roster exceeds 8 MiB".into());
    }
    let mut actual = Vec::new();
    let mut chunk = [0u8; 65536];
    loop {
        process::check()?;
        let count = file.read(&mut chunk)?;
        if count == 0 {
            break;
        }
        if count > MAXIMUM.saturating_sub(actual.len()) {
            return Err("split roster exceeds 8 MiB".into());
        }
        actual.extend_from_slice(&chunk[..count]);
    }
    if !policy_document::same_file(metadata, &file.metadata()?)
        || !policy_document::same_file(metadata, &fs::symlink_metadata(path)?)
        || !path.canonicalize()?.starts_with(root)
    {
        return Err("split roster changed during bounded check".into());
    }
    if actual != bytes {
        return Err("split-certified roster is stale for current native recipe".into());
    }
    Ok(())
}
