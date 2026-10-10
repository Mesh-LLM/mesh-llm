//! Hashing is exclusively performed by a parent-supervised current-exe worker.
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    io::{Read, Write},
    path::{Path, PathBuf},
};
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub current: PathBuf,
    pub released: PathBuf,
    pub model: PathBuf,
    #[serde(default)]
    pub evidence_logs: Option<PathBuf>,
}
pub(super) fn bytes(path: &Path, limit: u64) -> DynResult<Vec<u8>> {
    if !std::fs::symlink_metadata(path)?.is_file() {
        return Err("compatibility input must be regular".into());
    }
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err("compatibility opened input must be regular".into());
    }
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err("compatibility input limit".into());
    }
    Ok(bytes)
}
pub(super) fn fresh(path: &Path, bytes: &[u8]) -> DynResult<()> {
    match std::fs::symlink_metadata(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
        _ => return Err("compatibility output must be fresh".into()),
    }
    let mut file = tempfile::NamedTempFile::new_in(path.parent().ok_or("output parent")?)?;
    file.write_all(bytes)?;
    file.as_file().sync_all()?;
    file.persist_noclobber(path)?;
    Ok(())
}
fn binary(path: &Path) -> DynResult<serde_json::Value> {
    let path = path.canonicalize()?;
    if !std::fs::metadata(&path)?.is_file() {
        return Err("compatibility binary must be regular".into());
    }
    let runtime = path
        .parent()
        .ok_or("binary parent")?
        .join("native-runtimes");
    if !std::fs::symlink_metadata(&runtime)?.is_dir() {
        return Err(
            "each compatibility bundle requires its adjacent native-runtimes directory".into(),
        );
    }
    Ok(
        json!({"path":path,"sha256":crate::product::digest::file_sha256(&path).map_err(|e|e.error)?,"runtime_root":runtime,"runtime_tree_sha256":crate::product::digest::tree_sha256(&runtime).map_err(|e|e.error)?,"runtime_scope":"adjacent_offered_package_bytes_not_loaded_attestation"}),
    )
}
pub(super) fn worker(args: &[String]) -> DynResult<()> {
    let [a, input, b, output] = args else {
        return Err("identity-worker requires --input PATH --output PATH".into());
    };
    if a != "--input" || b != "--output" {
        return Err("identity-worker closed flags".into());
    }
    match std::fs::symlink_metadata(output) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
        _ => return Err("identity receipt must be fresh".into()),
    }
    let input: Input = serde_json::from_slice(&bytes(Path::new(input), 65536)?)?;
    let current = binary(&input.current)?;
    let released = binary(&input.released)?;
    if current["sha256"] == released["sha256"] {
        return Err("compatibility requires distinct current/released binary bytes".into());
    }
    let model = input.model.canonicalize()?;
    if !std::fs::metadata(&model)?.is_file() {
        return Err("compatibility model must be regular".into());
    }
    let out = json!({"schema_version":1,"request_sha256":hash(&serde_json::to_vec(&input)?),"current":current,"released":released,"model":{"path":model,"sha256":crate::product::digest::file_sha256(&model).map_err(|e|e.error)?}});
    let mut out = out;
    if let Some(root) = &input.evidence_logs {
        let mut logs = serde_json::Map::new();
        for name in [
            "current-provider",
            "released-client-free",
            "released-client-paid",
            "released-provider",
            "current-client",
        ] {
            for file in ["stdout.log", "stderr.log", "config.toml"] {
                let path = root.join(name).join(file);
                if path.exists() {
                    let contents = bytes(&path, 1048576)?;
                    logs.insert(format!("{name}/{file}"),json!({"sha256":hash(&contents),"bytes":contents.len(),"scope":"captured_sanitized_log_or_owned_config"}));
                }
            }
        }
        out["logs"] = logs.into();
    }
    fresh(Path::new(output), &serde_json::to_vec_pretty(&out)?)
}
pub(super) fn hash(bytes: &[u8]) -> String {
    use sha2::Digest as _;
    hex::encode(sha2::Sha256::digest(bytes))
}
