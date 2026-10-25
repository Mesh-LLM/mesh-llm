//! Closed worker entrypoints with fresh atomic file receipts and exact request correlation.
use super::{evidence_io, identity_worker, measurement_worker, probes};
use crate::{automation::command_interrupt::Interrupt, command::DynResult, process::Cancellation};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
};
pub(super) const IDENTITY_BYTES: usize = 512 * 1024;

pub(super) fn run(verb: &str, args: &[String]) -> DynResult<()> {
    match verb {
        "measurement-worker" => measurement_worker(args),
        "identity-worker" => identity_worker(args),
        _ => Err("unknown event benchmark worker verb".into()),
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Receipt<T> {
    pub schema_version: u64,
    pub request_sha256: String,
    pub data: T,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ExpectedFile {
    pub path: PathBuf,
    pub sha256: String,
}
#[derive(Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "kebab-case", deny_unknown_fields)]
pub(super) enum IdentityRequest {
    Admit {
        input: identity_worker::Input,
    },
    Revalidate {
        binaries: [ExpectedFile; 2],
        model: ExpectedFile,
    },
}
#[derive(Serialize, Deserialize)]
#[serde(tag = "result", rename_all = "kebab-case", deny_unknown_fields)]
pub(super) enum IdentityData {
    Admitted {
        identity: Box<identity_worker::Evidence>,
        runtime_packages: [Vec<PathBuf>; 2],
        thermal_state: serde_json::Value,
    },
    Revalidated,
}
pub(super) fn request_sha256<T: Serialize>(value: &T) -> DynResult<String> {
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(value)?)))
}
pub(super) fn correlated<T: DeserializeOwned, R: Serialize>(
    path: &Path,
    request: &R,
) -> DynResult<T> {
    let receipt: Receipt<T> = evidence_io::read(path, IDENTITY_BYTES)?;
    if receipt.schema_version != 1 || receipt.request_sha256 != request_sha256(request)? {
        return Err("worker receipt request identity mismatch".into());
    }
    Ok(receipt.data)
}
pub(super) fn flags(args: &[String]) -> DynResult<(PathBuf, PathBuf)> {
    if args.len() != 4 {
        return Err("worker requires exactly --input PATH --output PATH".into());
    }
    let mut input = None;
    let mut output = None;
    for pair in args.as_chunks::<2>().0 {
        let destination = match pair[0].as_str() {
            "--input" => &mut input,
            "--output" => &mut output,
            _ => return Err("unknown worker flag".into()),
        };
        let path = PathBuf::from(&pair[1]);
        if destination.is_some() || !path.is_absolute() || pair[1].len() > 16 * 1024 {
            return Err("worker paths must be unique bounded absolute paths".into());
        }
        *destination = Some(path);
    }
    let (input, output) = (
        input.ok_or("missing worker input")?,
        output.ok_or("missing worker output")?,
    );
    if input == output {
        return Err("worker requires a fresh distinct receipt path".into());
    }
    match fs::symlink_metadata(&output) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
        Ok(_) => return Err("worker receipt destination already exists".into()),
    }
    Ok((input, output))
}
fn packages(root: &Path) -> DynResult<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for (index, entry) in fs::read_dir(root)?.enumerate() {
        if index >= 64 {
            return Err("adjacent runtime root exceeds 64 entries".into());
        }
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_symlink() {
            return Err("adjacent runtime package entries cannot be symlinks".into());
        }
        if !kind.is_dir() {
            continue;
        }
        let path = fs::canonicalize(entry.path())?;
        if path.parent() != Some(root)
            || !fs::symlink_metadata(path.join("manifest.json"))?
                .file_type()
                .is_file()
        {
            return Err("runtime package must have a local regular manifest".into());
        }
        paths.push(path);
    }
    if paths.is_empty() {
        return Err("no adjacent local native runtime packages".into());
    }
    paths.sort();
    Ok(paths)
}
fn check_file(expected: &ExpectedFile, cancel: &Cancellation) -> DynResult<()> {
    if cancel.is_cancelled() {
        return Err("identity revalidation interrupted".into());
    }
    if expected.sha256.len() != 64 || !expected.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("invalid expected file digest".into());
    }
    let path = identity_worker::regular(&expected.path)?;
    if path != expected.path
        || crate::product::digest::file_sha256(&path).map_err(|failure| failure.error)?
            != expected.sha256
    {
        return Err("trial binary/model identity changed".into());
    }
    if cancel.is_cancelled() {
        return Err("identity revalidation interrupted".into());
    }
    Ok(())
}
fn identity(request: &IdentityRequest, cancel: &Cancellation) -> DynResult<IdentityData> {
    match request {
        IdentityRequest::Admit { input } => {
            let identity = identity_worker::execute(input, cancel)?;
            let runtime_packages = [
                packages(&identity.binaries[0].adjacent_runtime_root)?,
                packages(&identity.binaries[1].adjacent_runtime_root)?,
            ];
            let thermal_state = if std::env::consts::OS == "linux" {
                probes::linux_thermal(Path::new("/sys/class/thermal"))
            } else {
                serde_json::json!({"available":false,"source":"not-captured-by-identity-worker"})
            };
            if cancel.is_cancelled() {
                return Err("identity worker interrupted".into());
            }
            Ok(IdentityData::Admitted {
                identity: Box::new(identity),
                runtime_packages,
                thermal_state,
            })
        }
        IdentityRequest::Revalidate { binaries, model } => {
            for binary in binaries {
                check_file(binary, cancel)?;
            }
            check_file(model, cancel)?;
            Ok(IdentityData::Revalidated)
        }
    }
}
pub(super) fn identity_worker(args: &[String]) -> DynResult<()> {
    let (input, output) = flags(args)?;
    let request: IdentityRequest = evidence_io::read(&input, IDENTITY_BYTES)?;
    let interrupt = Interrupt::install()?;
    let result = identity(&request, &interrupt.cancellation());
    interrupt.finish()?;
    let data = result?;
    evidence_io::publish(
        &output,
        &Receipt {
            schema_version: 1,
            request_sha256: request_sha256(&request)?,
            data,
        },
        IDENTITY_BYTES,
    )
}
pub(super) fn measurement_worker(args: &[String]) -> DynResult<()> {
    let (input, output) = flags(args)?;
    let request: measurement_worker::Input = evidence_io::read(&input, evidence_io::INPUT_BYTES)?;
    request.validate()?;
    let interrupt = Interrupt::install()?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let mut evidence = runtime.block_on(measurement_worker::execute(
        &request,
        &interrupt.cancellation(),
    ))?;
    evidence.request_sha256 = Some(request_sha256(&request)?);
    let failed = evidence.error.is_some();
    // Preserve failed measurement evidence before returning its nonzero status.
    evidence_io::publish(&output, &evidence, evidence_io::RECEIPT_BYTES)?;
    interrupt.finish()?;
    if failed {
        Err("benchmark measurement worker failed; retained atomic receipt".into())
    } else {
        Ok(())
    }
}
#[cfg(test)]
pub(super) fn measurement_receipt(
    path: &Path,
    input: &measurement_worker::Input,
) -> DynResult<measurement_worker::Evidence> {
    measurement_receipt_hash(path, &request_sha256(input)?, &input.prompt_sha256)
}
/// The parent retains these hashes before launch; never reconstruct them from child-writable input.
pub(super) fn measurement_receipt_hash(
    path: &Path,
    expected_request_sha256: &str,
    expected_prompt_sha256: &str,
) -> DynResult<measurement_worker::Evidence> {
    let evidence: measurement_worker::Evidence =
        evidence_io::read(path, evidence_io::RECEIPT_BYTES)?;
    if evidence.schema_version != 1
        || evidence.request_sha256.as_deref() != Some(expected_request_sha256)
        || evidence.prompt_sha256 != expected_prompt_sha256
    {
        return Err("measurement receipt request identity mismatch".into());
    }
    Ok(evidence)
}
#[cfg(test)]
#[path = "worker_frontends_tests.rs"]
mod tests;
