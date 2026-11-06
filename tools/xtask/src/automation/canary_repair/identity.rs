use super::Input;
use crate::{automation::canary_receipts::Digest, command::DynResult};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};
#[cfg(unix)]
use std::{ffi::OsStr, os::unix::ffi::OsStrExt};

const MARKERS: [&str; 4] = [
    ".mesh-llm-upstream-sha",
    ".mesh-llm-patch-digest",
    ".mesh-llm-patched-sha",
    ".mesh-llm-prepare-schema",
];
const BINARIES: [&str; 3] = ["skippy", "skippy-correctness", "skippy-package-builder"];

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub(super) struct SourceMismatch(String);

#[derive(Serialize)]
pub(super) struct Identity {
    backend: String,
    recipe: BTreeMap<String, String>,
    source: BTreeMap<String, Digest>,
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
        return Err(SourceMismatch("native and Rust ABI versions differ".into()).into());
    }
    result.insert("skippy_abi".into(), rust);
    let pin = fs::read_to_string(input.root.join("skippy/llama_cpp/upstream.txt"))?;
    if result[MARKERS[0]] != pin.trim() {
        return Err(SourceMismatch("prepared runtime differs from requested pin".into()).into());
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
        source: source(input)?,
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
        if self.backend != input.backend
            || self.recipe != recipe(input)?
            || self.source != source(input)?
        {
            return Err(SourceMismatch(
                "candidate source changed after compilation; run repair-check again".into(),
            )
            .into());
        }
        for (path, expected) in &self.binaries {
            if &Digest::of_file(path)? != expected {
                return Err(SourceMismatch(format!(
                    "repair binary changed after compilation: {}",
                    path.display()
                ))
                .into());
            }
        }
        Ok(())
    }
    pub(super) fn binaries(&self) -> impl Iterator<Item = &PathBuf> {
        self.binaries.keys()
    }
}

fn source(input: &Input) -> DynResult<BTreeMap<String, Digest>> {
    [
        (&input.root, "mesh"),
        (&input.root.join(".deps/llama.cpp"), "llama.cpp"),
    ]
    .into_iter()
    .map(|(root, name)| Ok((name.into(), source_digest(root)?)))
    .collect()
}

fn git(root: &Path, args: &[&str]) -> DynResult<Vec<u8>> {
    const MAX_STATUS: u64 = 8 * 1024 * 1024;
    let mut child = Command::new("/usr/bin/git")
        .args([
            "--no-optional-locks",
            "--no-replace-objects",
            "-c",
            "core.fsmonitor=false",
        ])
        .args(args)
        .current_dir(root)
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_NO_LAZY_FETCH", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
    let mut output = Vec::new();
    child
        .stdout
        .take()
        .ok_or("Git stdout missing")?
        .take(MAX_STATUS + 1)
        .read_to_end(&mut output)?;
    if output.len() as u64 > MAX_STATUS {
        child.kill()?;
        child.wait()?;
        return Err("candidate source status exceeds 8 MiB".into());
    }
    if !child.wait()?.success() {
        return Err("candidate source Git inspection failed".into());
    }
    Ok(output)
}

fn source_digest(root: &Path) -> DynResult<Digest> {
    let head = git(root, &["rev-parse", "HEAD"])?;
    let status = git(
        root,
        &["status", "--porcelain=v1", "--untracked-files=all", "-z"],
    )?;
    let mut material = Vec::new();
    material.extend_from_slice(&head);
    material.extend_from_slice(&status);
    let mut records = status
        .split(|byte| *byte == 0)
        .filter(|part| !part.is_empty());
    let mut count = 0;
    let mut total_bytes = 0_u64;
    while let Some(record) = records.next() {
        count += 1;
        if count > 65_536 || record.len() < 4 || record[2] != b' ' {
            return Err("candidate source status is malformed or too large".into());
        }
        if [record[0], record[1]]
            .iter()
            .any(|code| matches!(code, b'R' | b'C'))
        {
            records
                .next()
                .ok_or("candidate source rename lacks its old path")?;
        }
        #[cfg(unix)]
        let path = root.join(OsStr::from_bytes(&record[3..]));
        #[cfg(windows)]
        let path = root.join(std::str::from_utf8(&record[3..])?);
        let metadata = match fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        };
        total_bytes = total_bytes.saturating_add(metadata.len());
        if total_bytes > 512 * 1024 * 1024 {
            return Err("candidate changed source exceeds 512 MiB".into());
        }
        let digest = if metadata.is_symlink() {
            Digest::of_bytes(fs::read_link(&path)?.as_os_str().as_encoded_bytes())
        } else if metadata.is_file() {
            Digest::of_file(&path)?
        } else {
            return Err("candidate changed source is not a regular file or link".into());
        };
        material.extend_from_slice(digest.as_str().as_bytes());
    }
    Ok(Digest::of_bytes(&material))
}
