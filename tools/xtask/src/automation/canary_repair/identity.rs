use super::Input;
use crate::{automation::canary_receipts::Digest, command::DynResult};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

const MARKERS: [&str; 4] = [
    ".mesh-llm-upstream-sha",
    ".mesh-llm-patch-digest",
    ".mesh-llm-patched-sha",
    ".mesh-llm-prepare-schema",
];
const BINARIES: [&str; 3] = ["skippy", "skippy-correctness", "skippy-package-builder"];

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub(super) struct SourceMismatch(&'static str);

#[derive(Serialize)]
pub(super) struct Identity {
    backend: String,
    recipe: BTreeMap<String, String>,
    binaries: BTreeMap<PathBuf, Digest>,
}

fn recipe(input: &Input) -> DynResult<BTreeMap<String, String>> {
    let mut result = BTreeMap::new();
    for marker in MARKERS {
        let path = input.root.join(".deps/llama.cpp").join(marker);
        if !fs::symlink_metadata(&path)?.is_file() || fs::metadata(&path)?.len() > 256 {
            return Err("invalid prepared native identity marker".into());
        }
        result.insert(marker.into(), fs::read_to_string(path)?.trim().into());
    }
    let rust = abi(
        &input.root.join("skippy/crates/skippy-ffi/src/lib.rs"),
        "pub const ABI_VERSION_",
        ": u32 = ",
    )?;
    let native = abi(
        &input.root.join(".deps/llama.cpp/include/skippy/common.h"),
        "#define SKIPPY_ABI_VERSION_",
        " ",
    )?;
    if rust != native {
        return Err(SourceMismatch("native and Rust ABI versions differ").into());
    }
    result.insert("skippy_abi".into(), rust);
    let pin = fs::read_to_string(input.root.join("skippy/llama_cpp/upstream.txt"))?;
    if result[MARKERS[0]] != pin.trim() {
        return Err(SourceMismatch("prepared runtime differs from requested pin").into());
    }
    Ok(result)
}

fn abi(path: &Path, prefix: &str, delimiter: &str) -> DynResult<String> {
    let text = fs::read_to_string(path)?;
    let mut components = BTreeMap::new();
    for line in text.lines() {
        let Some(rest) = line.trim().strip_prefix(prefix) else {
            continue;
        };
        for name in ["MAJOR", "MINOR", "PATCH"] {
            if let Some(value) = rest
                .strip_prefix(name)
                .and_then(|s| s.strip_prefix(delimiter))
            {
                let value: u32 = value.trim().trim_end_matches(';').parse()?;
                if components.insert(name, value).is_some() {
                    return Err("duplicate native ABI component".into());
                }
            }
        }
    }
    if components.len() != 3 {
        return Err("incomplete native ABI version".into());
    }
    Ok(format!(
        "{}.{}.{}",
        components["MAJOR"], components["MINOR"], components["PATCH"]
    ))
}

pub(super) fn capture(input: &Input) -> DynResult<Identity> {
    Ok(Identity {
        backend: input.backend.clone(),
        recipe: recipe(input)?,
        binaries: BINARIES
            .iter()
            .map(|name| {
                let path = input.root.join("target/debug").join(name);
                Ok((path.clone(), Digest::of_file(&path)?))
            })
            .collect::<DynResult<_>>()?,
    })
}

impl Identity {
    pub(super) fn verify(&self, input: &Input) -> DynResult<()> {
        if self.backend != input.backend || self.recipe != recipe(input)? {
            return Err(
                "prepared runtime changed after compilation; run repair-check again".into(),
            );
        }
        for (path, expected) in &self.binaries {
            if &Digest::of_file(path)? != expected {
                return Err(format!(
                    "repair binary changed after compilation: {}",
                    path.display()
                )
                .into());
            }
        }
        Ok(())
    }
    pub(super) fn binaries(&self) -> impl Iterator<Item = &PathBuf> {
        self.binaries.keys()
    }
}
