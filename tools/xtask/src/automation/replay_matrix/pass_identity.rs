use crate::command::DynResult;
use std::{collections::BTreeMap, path::Path};

pub(super) fn capture(directory: &Path) -> DynResult<BTreeMap<String, String>> {
    let mut identities = BTreeMap::new();
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| "non-Unicode pass artifact")?;
        if (name == "workload.json" && directory.join("server.command.json").try_exists()?)
            || (name.starts_with("c-")
                && (name.ends_with(".json") || name.ends_with("-requests.jsonl")))
            || name.ends_with("-probes.jsonl")
            || (name.ends_with(".json")
                && name
                    .trim_end_matches(".json")
                    .bytes()
                    .all(|byte| byte.is_ascii_digit()))
            || matches!(
                name.as_str(),
                "warmup.json"
                    | "warmup-requests.jsonl"
                    | "runtime.json"
                    | "model-identity.json"
                    | "lifecycle.json"
                    | "server.command.json"
            )
        {
            if !kind.is_file() {
                return Err("pass evidence must be regular files".into());
            }
            let digest =
                crate::product::digest::file_sha256(&entry.path()).map_err(|error| error.error)?;
            identities.insert(name, digest);
        }
    }
    Ok(identities)
}

pub(super) fn verify(directory: &Path, expected: &serde_json::Value) -> DynResult<()> {
    let expected: BTreeMap<String, String> = serde_json::from_value(expected.clone())?;
    if expected.is_empty() || capture(directory)? != expected {
        return Err("cannot resume: retained pass artifact bytes changed".into());
    }
    Ok(())
}
