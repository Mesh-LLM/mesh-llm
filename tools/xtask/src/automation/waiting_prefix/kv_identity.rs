//! File hashing and GGUF admission inside a separately supervised native worker.
use super::{adaptive_identity as io, options};
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub schema_version: u64,
    pub binary: PathBuf,
    pub model: PathBuf,
}
fn file(path: &Path) -> DynResult<Value> {
    if !path.is_absolute() {
        return Err("restart identity requires absolute local paths".into());
    }
    let path = path.canonicalize()?;
    let before = std::fs::metadata(&path)?;
    if !before.is_file() || before.len() == 0 {
        return Err("restart identity requires nonempty regular files".into());
    }
    let digest = crate::product::digest::file_sha256(&path).map_err(|e| e.error)?;
    let after = std::fs::metadata(&path)?;
    if before.len() != after.len() || before.modified()? != after.modified()? {
        return Err("restart file changed during hashing".into());
    }
    Ok(json!({"path":path,"sha256":digest,"size_bytes":after.len()}))
}
pub(super) fn evidence(input: &Input) -> DynResult<Value> {
    if input.schema_version != 1 {
        return Err("restart identity schema refused".into());
    }
    let binary = file(&input.binary)?;
    let model = file(&input.model)?;
    let dimensions = crate::automation::replay_matrix::model_preflight::verify(
        Path::new(model["path"].as_str().ok_or("model path absent")?),
        model["sha256"].as_str().ok_or("model SHA absent")?,
        1,
    )?;
    let checked = file(&input.model)?;
    if checked != model {
        return Err("restart model changed across native metadata admission".into());
    }
    let adjacent = Path::new(binary["path"].as_str().ok_or("binary path absent")?)
        .parent()
        .ok_or("binary parent absent")?
        .join("native-runtimes");
    let memory = if std::env::consts::OS == "linux" {
        io::bounded(Path::new("/proc/meminfo"), 65536)
            .ok()
            .and_then(|bytes| linux_memory(&bytes))
    } else {
        None
    };
    Ok(
        json!({"schema_version":1,"request_sha256":io::digest(&serde_json::to_vec(input)?),"binary":binary,"model":model,"model_metadata":dimensions,"adjacent_runtime_root":adjacent,"runtime_policy":"host-default-loader-authority-not-package-loaded-attestation","hardware":{"platform":std::env::consts::OS,"architecture":std::env::consts::ARCH,"cpu_core_count":std::thread::available_parallelism().ok().map(|v|v.get()),"physical_memory_bytes":memory}}),
    )
}
pub(super) fn linux_memory(bytes: &[u8]) -> Option<u64> {
    if bytes.len() > 65536 {
        return None;
    }
    let text = std::str::from_utf8(bytes).ok()?;
    let mut matches = text
        .lines()
        .filter_map(|line| line.strip_prefix("MemTotal:"));
    let line = matches.next()?;
    if matches.next().is_some() {
        return None;
    }
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() != 2 || fields[1] != "kB" {
        return None;
    }
    fields[0].parse::<u64>().ok()?.checked_mul(1024)
}
pub(super) fn run(args: &[String]) -> DynResult<()> {
    let flags = options(args, &["--input", "--output"], &["--input", "--output"])?;
    let input: Input = serde_json::from_slice(&io::bounded(Path::new(flags["--input"]), 65536)?)?;
    let output = Path::new(flags["--output"]);
    match std::fs::symlink_metadata(output) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        _ => return Err("restart identity output must be fresh".into()),
    };
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let value = evidence(&input)?;
    if interrupt.cancellation().is_cancelled() {
        return Err("restart identity interrupted".into());
    }
    io::fresh(output, &serde_json::to_vec_pretty(&value)?)?;
    interrupt.finish()?;
    Ok(())
}
