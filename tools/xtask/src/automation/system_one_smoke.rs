//! System One smoke aggregate qualification receipt.
use crate::command::DynResult;
use serde::Serialize;
use serde_json::json;
use std::{fs, io::Write, path::Path};

const USAGE: &str = "cargo xtool automation system-one-smoke {report OUTPUT STATUS CONTRACT READ ARTIFACT BACKEND ARTIFACT_PATH CACHE_CHECKED CERTIFIED REQUIRE_QUALIFIED SKIP_CONTRACT REASONS}";

fn nullable(value: &str) -> Option<&str> {
    (!value.is_empty()).then_some(value)
}

fn enabled(value: &str) -> bool {
    matches!(value, "1" | "true")
}

fn report(args: &[String]) -> DynResult<()> {
    let [
        output,
        status,
        contract,
        read,
        artifact,
        backend,
        artifact_path,
        checked,
        certified,
        required,
        skipped,
        reasons,
    ] = args
    else {
        return Err(USAGE.into());
    };
    if !matches!(status.as_str(), "pass" | "fail" | "unqualified")
        || !matches!(contract.as_str(), "pass" | "fail" | "error" | "skipped")
        || !matches!(
            read.as_str(),
            "pass" | "fail" | "error" | "skipped" | "unqualified"
        )
    {
        return Err("invalid smoke qualification status".into());
    }
    let receipt = json!({
        "schema_version": 1,
        "status": status,
        "contract": {"status": contract},
        "full_model_read": {
            "status": read,
            "artifact": artifact,
            "backend": nullable(backend),
            "artifact_path": nullable(artifact_path),
            "artifact_cache_checked": enabled(checked),
            "certified_backends": certified.split(',').filter(|s| !s.is_empty()).collect::<Vec<_>>(),
            "require_qualified": enabled(required),
        },
        "contract_part_skipped": enabled(skipped),
        "reasons": reasons.split("; ").filter(|s| !s.is_empty()).collect::<Vec<_>>(),
    });
    publish(output, &receipt, true, true)
}

fn publish(path: &str, value: &impl Serialize, parents: bool, stdout: bool) -> DynResult<()> {
    if path.is_empty() {
        return Err("output path is required".into());
    }
    let path = Path::new(path);
    if let Ok(metadata) = fs::symlink_metadata(path)
        && !metadata.file_type().is_file()
    {
        return Err("output must be a regular file".into());
    }
    if parents && let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    let mut options = fs::OpenOptions::new();
    options.write(true).create(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let mut file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err("output must be a regular file".into());
    }
    file.set_len(0)?;
    file.write_all(&bytes)?;
    if stdout {
        std::io::stdout().lock().write_all(&bytes)?;
    }
    Ok(())
}

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    match args {
        [verb, rest @ ..] if verb == "report" => report(rest),
        [help] if matches!(help.as_str(), "--help" | "-h") => {
            println!("{USAGE}");
            Ok(())
        }
        _ => Err(USAGE.into()),
    }
}
