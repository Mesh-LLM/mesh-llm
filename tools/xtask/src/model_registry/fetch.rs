//! `models fetch <manifest> --cadence <cadence> --root <dir>`: materialize a
//! suite's cadence-admitted artifacts under one root, each in a directory
//! named by its artifact id, through the pinned HTTPS transfer, and hold
//! every file to the manifest's size and SHA-256. A file that already
//! verifies is kept; one that fails verification is reported and left for the
//! operator, because the transfer publishes only fresh paths. The layout is
//! what `models resolve --artifact-id <id> --verify-root <root>/<id>` checks.

use super::fields::{ModelError, ModelResult, fail};
use super::manifest::{PinnedFile, Resolved, Selection, admitted_ids, resolve};
use crate::ci_plan::catalog::{os_error_text, python_path_display};
use crate::ci_plan::document::Json;
use crate::command_interrupt::Interrupt;
use crate::repository::check_args::Grammar;
use crate::repository::check_report::CheckReport;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const PROGRAM: &str = "models fetch";
const GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool models fetch <manifest> --cadence <cadence> --root <dir>
    [--artifact-id <id>] [--timeout-secs <1..86400>] [--dry-run]",
    values: &["--cadence", "--root", "--artifact-id", "--timeout-secs"],
    flags: &["--dry-run"],
};
const DEFAULT_TIMEOUT_SECS: u64 = 3600;
/// The pinned transfer's byte cap; a larger model is split in the registry.
const TRANSFER_CAP: u64 = 64 * 1024 * 1024 * 1024;

struct Request<'a> {
    manifest: &'a str,
    cadence: &'a str,
    root: &'a Path,
    artifact_id: Option<&'a str>,
    timeout: Duration,
    dry_run: bool,
}

/// One manifest file and where it lives under the root.
struct Item<'a> {
    artifact: &'a str,
    file: &'a PinnedFile,
    directory: PathBuf,
}

enum State {
    Verified,
    Missing,
    Corrupt(String),
}

pub(super) fn run(args: &[String]) -> CheckReport {
    if args.len() == 1 && matches!(args[0].as_str(), "-h" | "--help") {
        return CheckReport::success(format!(
            "usage: {}\n\nFetches every artifact the cadence admits, or one with --artifact-id, into\n<root>/<artifact id>/<file>, verified against the manifest. --dry-run lists what\nwould be fetched. --timeout-secs bounds each file's transfer (default {DEFAULT_TIMEOUT_SECS}).\n",
            GRAMMAR.usage
        ));
    }
    let parsed = match super::argv::parse(&GRAMMAR, PROGRAM, args) {
        Ok(parsed) => parsed,
        Err(report) => return report,
    };
    let missing = [
        ("manifest", parsed.positionals.is_empty()),
        ("--cadence", parsed.last("--cadence").is_none()),
        ("--root", parsed.last("--root").is_none()),
    ]
    .iter()
    .filter(|(_, absent)| *absent)
    .map(|(name, _)| *name)
    .collect::<Vec<_>>();
    if !missing.is_empty() {
        let message = format!(
            "the following arguments are required: {}",
            missing.join(", ")
        );
        return super::argv::usage(&GRAMMAR, PROGRAM, &message);
    }
    if let [_, extra @ ..] = parsed.positionals.as_slice()
        && !extra.is_empty()
    {
        let message = format!("unrecognized arguments: {}", extra.join(" "));
        return super::argv::usage(&GRAMMAR, PROGRAM, &message);
    }
    let timeout = match parsed.last("--timeout-secs") {
        None => DEFAULT_TIMEOUT_SECS,
        Some(text) => match text.parse::<u64>() {
            Ok(seconds) if (1..=86_400).contains(&seconds) => seconds,
            _ => {
                return super::argv::usage(
                    &GRAMMAR,
                    PROGRAM,
                    "argument --timeout-secs: expected an integer from 1 to 86400",
                );
            }
        },
    };
    let request = Request {
        manifest: &parsed.positionals[0],
        cadence: parsed.last("--cadence").unwrap_or_default(),
        root: Path::new(parsed.last("--root").unwrap_or_default()),
        artifact_id: parsed.last("--artifact-id"),
        timeout: Duration::from_secs(timeout),
        dry_run: parsed.flag("--dry-run"),
    };
    let mut stdout = String::new();
    match fetch_into(&request, &mut stdout) {
        Ok(()) => CheckReport::success(stdout),
        Err(error) => CheckReport {
            stdout,
            stderr: format!("test-model fetch error: {error}\n"),
            code: 2,
        },
    }
}

fn fetch_into(request: &Request<'_>, stdout: &mut String) -> ModelResult<()> {
    let bytes =
        std::fs::read(request.manifest).map_err(|error| io_error(&error, request.manifest))?;
    let manifest = Json::parse(&bytes).map_err(|error| ModelError(error.to_string()))?;
    let ids = match request.artifact_id {
        Some(id) => vec![id],
        None => admitted_ids(&manifest, request.cadence)?,
    };
    if ids.is_empty() {
        return fail(format!(
            "no artifact in {} is allowed at cadence {}",
            request.manifest, request.cadence
        ));
    }
    let artifacts = ids
        .iter()
        .map(|id| {
            resolve(
                &manifest,
                &Selection {
                    artifact_id: Some(id),
                    cadence: request.cadence,
                },
            )
        })
        .collect::<ModelResult<Vec<_>>>()?;
    let classified = artifacts
        .iter()
        .flat_map(|artifact| items(request.root, artifact))
        .map(|item| {
            let state = state(&item);
            (item, state)
        })
        .collect::<Vec<_>>();

    let mut wanted = Vec::new();
    let mut corrupt = 0_usize;
    for (item, state) in &classified {
        match state {
            State::Verified => stdout.push_str(&line("cached", item, None)),
            State::Missing => {
                stdout.push_str(&line("missing", item, None));
                wanted.push(item);
            }
            State::Corrupt(reason) => {
                corrupt += 1;
                stdout.push_str(&line("corrupt", item, Some(reason)));
            }
        }
    }
    let pending = wanted.iter().map(|item| item.file.size_bytes).sum::<u64>();
    stdout.push_str(&format!(
        "{} file(s) cached, {} to fetch ({}), {corrupt} corrupt\n",
        classified.len() - wanted.len() - corrupt,
        wanted.len(),
        gib(pending)
    ));
    if corrupt != 0 {
        return fail("corrupt files must be removed before they can be fetched again");
    }
    if request.dry_run {
        return Ok(());
    }
    if let Some(item) = wanted
        .iter()
        .find(|item| item.file.size_bytes > TRANSFER_CAP)
    {
        return fail(format!(
            "{} exceeds the {} transfer cap; split it in the registry",
            shown(item),
            gib(TRANSFER_CAP)
        ));
    }

    let interrupt = Interrupt::install().map_err(|reason| ModelError(reason.to_string()))?;
    let cancellation = interrupt.cancellation();
    let mut failed = Vec::new();
    for item in &wanted {
        if cancellation.is_cancelled() {
            break;
        }
        stdout.push_str(&line("fetching", item, None));
        let deadline = Instant::now() + request.timeout;
        let output = item.directory.join(&item.file.name);
        match super::projector_download::acquire_pinned(
            &item.file.url,
            &output,
            &item.file.sha256,
            item.file.size_bytes,
            deadline,
            &cancellation,
        ) {
            Ok(_) => stdout.push_str(&line("fetched", item, None)),
            Err(error) => {
                stdout.push_str(&line("failed", item, Some(&error.to_string())));
                failed.push(shown(item));
            }
        }
    }
    interrupt
        .finish()
        .map_err(|_| ModelError("fetch interrupted; rerun to continue".to_owned()))?;
    if !failed.is_empty() {
        return fail(format!(
            "{} file(s) not fetched: {}",
            failed.len(),
            failed.join(", ")
        ));
    }
    Ok(())
}

fn items<'a>(root: &Path, artifact: &'a Resolved<'a>) -> impl Iterator<Item = Item<'a>> {
    let directory = root.join(artifact.id);
    artifact.files.iter().map(move |file| Item {
        artifact: artifact.id,
        file,
        directory: directory.clone(),
    })
}

/// Verified, absent, or present but not the pinned bytes.
fn state(item: &Item<'_>) -> State {
    if !item.directory.join(&item.file.name).exists() {
        return State::Missing;
    }
    match super::resolve::verify(&item.directory.display().to_string(), item.file) {
        Ok(_) => State::Verified,
        Err(error) => State::Corrupt(error.to_string()),
    }
}

fn shown(item: &Item<'_>) -> String {
    format!("{}/{}", item.artifact, item.file.name)
}

fn line(state: &str, item: &Item<'_>, detail: Option<&str>) -> String {
    let detail = detail.map_or_else(String::new, |text| format!(": {text}"));
    format!(
        "{state:<9}{} ({}){detail}\n",
        shown(item),
        gib(item.file.size_bytes)
    )
}

fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / f64::from(1_u32 << 30))
}

fn io_error(error: &std::io::Error, path: &str) -> ModelError {
    ModelError(os_error_text(error, &python_path_display(Path::new(path))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repository::check_report::CheckReport;

    fn manifest(dir: &Path, id: &str, cadence: &str, name: &str, bytes: &[u8]) -> PathBuf {
        use sha2::{Digest, Sha256};
        let digest = hex::encode(Sha256::digest(bytes));
        let text = format!(
            r#"{{"schema_version":1,"manifest_kind":"test-model-artifacts","suite":"s","artifacts":[{{"id":"{id}","repo":"acme/Tiny-GGUF","revision":"deadbeef","selector":"Q4_K_M","model_ref":"acme/Tiny-GGUF:Q4_K_M","cadences":["{cadence}"],"files":["{name}"],"urls":["https://huggingface.co/acme/Tiny-GGUF/resolve/deadbeef/{name}"],"file_integrity":{{"{name}":{{"size_bytes":{},"blob_id":"{digest}"}}}}}}]}}"#,
            bytes.len()
        );
        let path = dir.join("manifest.json");
        std::fs::write(&path, text).unwrap();
        path
    }

    fn run_args(args: &[&str]) -> CheckReport {
        run(&args.iter().map(|arg| (*arg).to_owned()).collect::<Vec<_>>())
    }

    #[test]
    fn dry_run_reports_cached_and_missing_files_without_fetching() {
        let dir = tempfile::tempdir().unwrap();
        let bytes = b"GGUFtiny model bytes";
        let manifest = manifest(dir.path(), "tiny", "manual", "Tiny-Q4_K_M.gguf", bytes);
        let root = dir.path().join("models");
        let args = |root: &Path| {
            vec![
                manifest.to_string_lossy().into_owned(),
                "--cadence".into(),
                "manual".into(),
                "--root".into(),
                root.to_string_lossy().into_owned(),
                "--dry-run".into(),
            ]
        };

        let report = run(&args(&root));
        assert_eq!(report.code, 0, "{}", report.stderr);
        assert!(report.stdout.contains("missing  tiny/Tiny-Q4_K_M.gguf"));
        assert!(report.stdout.contains("0 file(s) cached, 1 to fetch"));
        assert!(!root.exists(), "a dry run writes nothing");

        std::fs::create_dir_all(root.join("tiny")).unwrap();
        std::fs::write(root.join("tiny/Tiny-Q4_K_M.gguf"), bytes).unwrap();
        let report = run(&args(&root));
        assert_eq!(report.code, 0, "{}", report.stderr);
        assert!(report.stdout.contains("cached   tiny/Tiny-Q4_K_M.gguf"));
        assert!(report.stdout.contains("1 file(s) cached, 0 to fetch"));
    }

    #[test]
    fn a_file_with_other_bytes_is_corrupt_and_blocks_the_fetch() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = manifest(
            dir.path(),
            "tiny",
            "manual",
            "Tiny-Q4_K_M.gguf",
            b"GGUFright",
        );
        let root = dir.path().join("models");
        std::fs::create_dir_all(root.join("tiny")).unwrap();
        std::fs::write(root.join("tiny/Tiny-Q4_K_M.gguf"), b"GGUFwrong").unwrap();
        let report = run_args(&[
            manifest.to_str().unwrap(),
            "--cadence",
            "manual",
            "--root",
            root.to_str().unwrap(),
        ]);
        assert_eq!(report.code, 2);
        assert!(report.stdout.contains("corrupt  tiny/Tiny-Q4_K_M.gguf"));
        assert!(report.stderr.contains("corrupt files must be removed"));
    }

    #[test]
    fn the_cadence_must_admit_an_artifact() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = manifest(dir.path(), "tiny", "nightly", "Tiny-Q4_K_M.gguf", b"GGUF");
        let report = run_args(&[
            manifest.to_str().unwrap(),
            "--cadence",
            "manual",
            "--root",
            dir.path().to_str().unwrap(),
            "--dry-run",
        ]);
        assert_eq!(report.code, 2);
        assert!(report.stderr.contains("is allowed at cadence manual"));
    }

    #[test]
    fn options_are_required_and_bounded() {
        assert!(run_args(&["m.json"]).stderr.contains("--cadence, --root"));
        let report = run_args(&[
            "m.json",
            "--cadence",
            "manual",
            "--root",
            "r",
            "--timeout-secs",
            "0",
        ]);
        assert!(report.stderr.contains("1 to 86400"));
    }
}
