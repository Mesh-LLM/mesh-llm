use crate::command::DynResult;
use crate::repository::check_args::Grammar;
use crate::repository::check_report::CheckReport;
use serde::Deserialize;
use std::collections::BTreeSet;
use std::path::Path;

const GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool automation replay-matrix publication-verify --publication-dir <path> --base-sha <sha> --checkout-sha <sha> --run-id <id> --run-attempt <number>",
    values: &[
        "--publication-dir",
        "--base-sha",
        "--checkout-sha",
        "--run-id",
        "--run-attempt",
    ],
    flags: &["--help"],
};

#[derive(Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct Status {
    schema_version: u32,
    base_sha: String,
    run_id: String,
    run_attempt: u64,
    resolution: Resolution,
    patch_bytes: u64,
    patch_sha256: String,
    body_bytes: u64,
    body_sha256: String,
}

#[derive(Deserialize, serde::Serialize)]
enum Resolution {
    #[serde(rename = "fix-verified")]
    FixVerified,
    #[serde(rename = "needs-attention")]
    NeedsAttention,
}

impl Resolution {
    fn name(&self) -> &'static str {
        match self {
            Self::FixVerified => "fix-verified",
            Self::NeedsAttention => "needs-attention",
        }
    }
}

pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("usage: {}\n", GRAMMAR.usage)).emit();
    }
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error("unexpected positional arguments").emit();
    }
    let required = |key| parsed.last(key).ok_or_else(|| format!("missing {key}"));
    let directory = Path::new(required("--publication-dir")?);
    let base = required("--base-sha")?;
    let checkout = required("--checkout-sha")?;
    let run = required("--run-id")?;
    let attempt = required("--run-attempt")?;
    let status = verify(directory)?;
    if base.len() != 40
        || !base
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
        || status.base_sha != base
        || checkout != base
    {
        return Err("repair publication does not match the trusted workflow checkout".into());
    }
    if run.is_empty() || !run.bytes().all(|byte| byte.is_ascii_digit()) || status.run_id != run {
        return Err("repair publication run_id is invalid".into());
    }
    if attempt.is_empty()
        || !attempt.bytes().all(|byte| byte.is_ascii_digit())
        || attempt.starts_with('0')
        || attempt.parse::<u64>()? != status.run_attempt
        || status.run_attempt == 0
    {
        return Err("repair publication run_attempt is invalid".into());
    }
    CheckReport::success(format!(
        "{}\t{}\t{}\n",
        status.run_id,
        status.run_attempt,
        status.resolution.name()
    ))
    .emit()
}

fn verify(directory: &Path) -> DynResult<Status> {
    check_directory(directory, true)?;
    let status: Status = serde_json::from_slice(&std::fs::read(directory.join("status.json"))?)?;
    if status.schema_version != 1 {
        return Err("repair publication status schema is invalid".into());
    }
    bound_file(
        &directory.join("repair.patch"),
        status.patch_bytes,
        &status.patch_sha256,
        64 * 1024 * 1024,
    )?;
    bound_file(
        &directory.join("pr-body.md"),
        status.body_bytes,
        &status.body_sha256,
        1024 * 1024,
    )?;
    let body = std::fs::read_to_string(directory.join("pr-body.md"))?;
    if body.contains('\0') {
        return Err("repair PR body contains a NUL byte".into());
    }
    let patch = std::fs::read(directory.join("repair.patch"))?;
    if !patch.starts_with(b"From ") || !patch.windows(10).any(|window| window == b"\nSubject: ") {
        return Err("repair patch is not a format-patch artifact".into());
    }
    Ok(status)
}

fn bound_file(path: &Path, bytes: u64, digest: &str, limit: u64) -> DynResult<()> {
    let actual = std::fs::metadata(path)?.len();
    if actual == 0 || actual > limit || actual != bytes {
        return Err(
            "repair publication size is outside the allowed bounds or does not match status".into(),
        );
    }
    let actual_digest = crate::product::digest::file_sha256(path).map_err(|error| error.error)?;
    if digest != actual_digest {
        return Err("repair publication digest does not match status".into());
    }
    Ok(())
}

fn check_directory(directory: &Path, status_present: bool) -> DynResult<()> {
    if !std::fs::symlink_metadata(directory)?.file_type().is_dir() {
        return Err("repair publication must be a non-symlink directory".into());
    }
    let names = std::fs::read_dir(directory)?
        .map(|entry| entry.map(|entry| entry.file_name()))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let mut expected: BTreeSet<_> = ["pr-body.md", "repair.patch"]
        .map(std::ffi::OsString::from)
        .into_iter()
        .collect();
    if status_present {
        expected.insert("status.json".into());
    }
    if names != expected {
        return Err("repair publication must contain exactly its three data files".into());
    }
    for name in expected {
        if !std::fs::symlink_metadata(directory.join(&name))?
            .file_type()
            .is_file()
        {
            return Err("repair publication files must be regular and not symlinks".into());
        }
    }
    Ok(())
}

pub(super) fn prepare_status(
    directory: &Path,
    base_sha: &str,
    run_id: &str,
    attempt: &str,
) -> DynResult<()> {
    if base_sha.len() != 40
        || !base_sha
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
        || (run_id != "local"
            && (run_id.is_empty() || !run_id.bytes().all(|byte| byte.is_ascii_digit())))
        || attempt.is_empty()
        || attempt.starts_with('0')
        || !attempt.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err("invalid repair publication identity".into());
    }
    check_directory(directory, false)?;
    let patch = directory.join("repair.patch");
    let body = directory.join("pr-body.md");
    let (patch_bytes, patch_sha256) = prepare_identity(&patch, 64 * 1024 * 1024)?;
    let (body_bytes, body_sha256) = prepare_identity(&body, 1024 * 1024)?;
    let status = Status {
        schema_version: 1,
        base_sha: base_sha.into(),
        run_id: run_id.into(),
        run_attempt: attempt.parse()?,
        resolution: Resolution::FixVerified,
        patch_bytes,
        patch_sha256,
        body_bytes,
        body_sha256,
    };
    bound_file(
        &patch,
        status.patch_bytes,
        &status.patch_sha256,
        64 * 1024 * 1024,
    )?;
    bound_file(&body, status.body_bytes, &status.body_sha256, 1024 * 1024)?;
    let path = directory.join("status.json");
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)?;
    use std::io::Write;
    writeln!(file, "{}", serde_json::to_string(&status)?)?;
    if let Err(error) = verify(directory) {
        std::fs::remove_file(path)?;
        return Err(error);
    }
    Ok(())
}

fn prepare_identity(path: &Path, limit: u64) -> DynResult<(u64, String)> {
    let bytes = std::fs::metadata(path)?.len();
    if bytes == 0 || bytes > limit {
        return Err("repair publication size is outside the allowed bounds".into());
    }
    Ok((
        bytes,
        crate::product::digest::file_sha256(path).map_err(|error| error.error)?,
    ))
}
