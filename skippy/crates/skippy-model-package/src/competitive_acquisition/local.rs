use super::{
    contract::{Case, check, digest, tree},
    export, read,
};
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::BTreeMap,
    io::Read as _,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    source_directory: PathBuf,
    source_sha256: String,
    output_directory: PathBuf,
    export_sha256: String,
    cases: Vec<Case>,
    timeout_seconds: u64,
}
pub(super) fn pins(root: &Path, deadline: Instant) -> Result<BTreeMap<String, String>> {
    let mut pending = vec![(root.to_path_buf(), 0)];
    let mut rows = BTreeMap::new();
    let mut bytes = 0_u64;
    while let Some((directory, depth)) = pending.pop() {
        check(deadline)?;
        if depth > 32 {
            bail!("source depth refused");
        }
        for entry in std::fs::read_dir(directory)? {
            let entry = entry?;
            let kind = entry.file_type()?;
            if kind.is_dir() {
                pending.push((entry.path(), depth + 1));
            } else if kind.is_file() {
                let relative = entry
                    .path()
                    .strip_prefix(root)?
                    .to_str()
                    .ok_or_else(|| anyhow::anyhow!("source UTF8 path"))?
                    .to_string();
                if !super::contract::path(&relative) {
                    bail!("source path grammar");
                }
                let mut f = read::open(&entry.path(), 1024 * 1024 * 1024 * 1024, false)?;
                use sha2::Digest as _;
                let mut h = sha2::Sha256::new();
                let mut b = [0; 65536];
                loop {
                    check(deadline)?;
                    let n = f.read(&mut b)?;
                    if n == 0 {
                        break;
                    }
                    bytes = bytes
                        .checked_add(n as u64)
                        .filter(|v| *v <= 1024 * 1024 * 1024 * 1024)
                        .ok_or_else(|| anyhow::anyhow!("source byte bound"))?;
                    h.update(&b[..n]);
                }
                rows.insert(
                    relative,
                    h.finalize().iter().map(|b| format!("{b:02x}")).collect(),
                );
            } else {
                bail!("source symlink/nonregular refused");
            }
            if rows.len() + pending.len() > 16384 {
                bail!("source roster refused");
            }
        }
    }
    Ok(rows)
}
pub(super) fn run(path: &Path) -> Result<()> {
    let mut fd = read::open(path, 1024 * 1024, false)?;
    let bytes = read::read(&mut fd, 1024 * 1024)?;
    let input: Request = serde_json::from_slice(&bytes)?;
    if !input.source_directory.is_absolute()
        || input.source_directory.canonicalize()? != input.source_directory
        || !input.output_directory.is_absolute()
        || !(1..=3600).contains(&input.timeout_seconds)
        || !super::contract::pin(&input.source_sha256, 64)
        || !super::contract::pin(&input.export_sha256, 64)
    {
        bail!("local export input refused");
    }
    let parent = input
        .output_directory
        .parent()
        .ok_or_else(|| anyhow::anyhow!("output parent"))?;
    if parent.canonicalize()? != parent
        || !parent.is_dir()
        || input.output_directory.starts_with(&input.source_directory)
        || input.source_directory.starts_with(&input.output_directory)
    {
        bail!("local source/output overlap refused");
    }
    match std::fs::symlink_metadata(&input.output_directory) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
        _ => bail!("local output must be fresh"),
    };
    let deadline = Instant::now() + Duration::from_secs(input.timeout_seconds);
    let before = pins(&input.source_directory, deadline)?;
    if tree(&before)? != input.source_sha256 {
        bail!("local source pin mismatch");
    }
    let receipt = export::fast(
        &input.source_directory,
        &input.output_directory,
        &input.export_sha256,
        &input.cases,
        deadline,
    )?;
    if pins(&input.source_directory, deadline)? != before
        || read::read(&mut fd, 1024 * 1024)? != bytes
    {
        bail!("local export final source/input custody refused");
    }
    check(deadline)?;
    super::publish(
        &input.output_directory.join("export.json"),
        &serde_json::to_vec_pretty(
            &json!({"schema_version":1,"status":"EXPORTED_SINGLE","request_sha256":digest(&bytes),"source_tree_sha256":input.source_sha256,"export":receipt,"acquisition_performed":false,"real_family_qualified":false}),
        )?,
    )
}
