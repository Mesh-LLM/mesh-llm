//! Admit the digest-bound import report before packaging a standalone Skippy CLI.
use crate::{native_policy, repository::check_report::CheckReport};
use serde::Deserialize;
use std::{fs::OpenOptions, io::Read, path::Path};

const USAGE: &str = "cargo xtool product skippy-cli-input REPORT BINARY TARGET";
const MAX_REPORT_BYTES: u64 = 1024 * 1024;

#[derive(Deserialize)]
struct ImportReport {
    binary: String,
    binary_sha256: String,
    format: String,
    policy: String,
    imports: Vec<String>,
    rejected_imports: Vec<String>,
}

pub(super) fn run(args: &[String]) -> CheckReport {
    let [report, binary, target] = args else {
        return CheckReport::usage(USAGE, "expected report, binary and target");
    };
    let (name, format) = match target.as_str() {
        "darwin-aarch64" => ("skippy", "macho"),
        "linux-x86_64" | "linux-aarch64" => ("skippy", "elf"),
        "windows-x86_64" => ("skippy.exe", "pe"),
        _ => return CheckReport::usage(USAGE, "unsupported Skippy CLI target"),
    };
    match admit(Path::new(report), Path::new(binary), name, format) {
        Ok(()) => CheckReport::success(String::new()),
        Err(error) => CheckReport::failure(String::new(), format!("{error}\n")),
    }
}

fn admit(report_path: &Path, binary: &Path, name: &str, format: &str) -> Result<(), String> {
    let report: ImportReport = serde_json::from_slice(&read_report(report_path)?)
        .map_err(|error| format!("invalid Skippy host import-policy report: {error}"))?;
    if binary.file_name().and_then(|value| value.to_str()) != Some(name)
        || report.binary != name
        || report.format != format
        || report.policy != "mesh-llm-dynamic-host-v2"
        || !report.rejected_imports.is_empty()
        || !native_policy::rejected_host_imports(&report.imports).is_empty()
    {
        return Err("invalid or rejected Skippy host import-policy report".into());
    }
    if report.binary_sha256 != native_policy::host_binary_sha256(binary)? {
        return Err("Skippy host import-policy report does not bind the current binary".into());
    }
    Ok(())
}

fn read_report(path: &Path) -> Result<Vec<u8>, String> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(0x0020_0000); // FILE_FLAG_OPEN_REPARSE_POINT
    }
    let file = options.open(path).map_err(|error| error.to_string())?;
    let metadata = file.metadata().map_err(|error| error.to_string())?;
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        if metadata.file_attributes() & 0x400 != 0 {
            return Err("Skippy import report cannot follow a reparse point".into());
        }
    }
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > MAX_REPORT_BYTES {
        return Err("Skippy import report requires a nonempty regular file at most 1 MiB".into());
    }
    let mut bytes = Vec::new();
    file.take(MAX_REPORT_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    if bytes.len() as u64 > MAX_REPORT_BYTES || bytes.len() as u64 != metadata.len() {
        return Err("Skippy import report changed or exceeded its bound while reading".into());
    }
    Ok(bytes)
}
