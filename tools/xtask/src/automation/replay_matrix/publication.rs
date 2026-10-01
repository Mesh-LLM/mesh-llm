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

#[derive(Deserialize)]
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

#[derive(Deserialize)]
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
    if !std::fs::symlink_metadata(directory)?.file_type().is_dir() {
        return Err("repair publication must be a non-symlink directory".into());
    }
    let names = std::fs::read_dir(directory)?
        .map(|entry| entry.map(|entry| entry.file_name()))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let expected = ["pr-body.md", "repair.patch", "status.json"]
        .map(std::ffi::OsString::from)
        .into_iter()
        .collect();
    if names != expected {
        return Err("repair publication must contain exactly its three data files".into());
    }
    for name in ["pr-body.md", "repair.patch", "status.json"] {
        if !std::fs::symlink_metadata(directory.join(name))?
            .file_type()
            .is_file()
        {
            return Err("repair publication files must be regular and not symlinks".into());
        }
    }
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
