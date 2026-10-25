//! `models resolve`: the Rust owner of `scripts/resolve-test-model-manifest.py`.
//! Selects one cadence-authorized artifact, optionally verifies its files on
//! disk, and emits a JSON summary or single-line GitHub step outputs.

use super::fields::{ModelError, ModelResult, fail};
use super::json_bytes::{ASCII, ASCII_COMPACT, dumps};
use super::manifest::{PinnedFile, Resolved, Selection, resolve};
use crate::ci_plan::catalog::{os_error_text, python_path_display};
use crate::ci_plan::document::Json;
use crate::repository::check_args::Grammar;
use crate::repository::check_report::CheckReport;
use sha2::{Digest, Sha256};
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::Path;

const PROGRAM: &str = "resolve-test-model-manifest.py";
const GRAMMAR: Grammar = Grammar {
    usage: "resolve-test-model-manifest.py [-h] [--artifact-id ARTIFACT_ID]
                                      --cadence CADENCE
                                      [--require-single-file]
                                      [--github-output GITHUB_OUTPUT]
                                      [--github-output-prefix GITHUB_OUTPUT_PREFIX]
                                      [--verify-root VERIFY_ROOT]
                                      manifest",
    values: &[
        "--artifact-id",
        "--cadence",
        "--github-output",
        "--github-output-prefix",
        "--verify-root",
    ],
    flags: &["--require-single-file", "--print-serving-file"],
};

/// One resolution, as the legacy argv or the restore step describes it.
pub(super) struct Request<'a> {
    pub(super) manifest: &'a str,
    pub(super) selection: Selection<'a>,
    pub(super) require_single_file: bool,
    pub(super) print_serving_file: bool,
    pub(super) github_output: Option<&'a str>,
    pub(super) output_prefix: &'a str,
    pub(super) verify_root: Option<&'a str>,
}

pub(super) fn run(args: &[String]) -> CheckReport {
    if args.len() == 1 && matches!(args[0].as_str(), "-h" | "--help") {
        let usage = GRAMMAR.usage.replace(
            "[--require-single-file]",
            "[--require-single-file] [--print-serving-file]",
        );
        return CheckReport::success(format!(
            "usage: {usage}\n\n--print-serving-file selects a cadence-admitted GGUF serving filename.\nIt does not inspect model bytes or local caches and cannot combine with --github-output or --verify-root.\n"
        ));
    }
    let parsed = match super::argv::parse(&GRAMMAR, PROGRAM, args) {
        Ok(parsed) => parsed,
        Err(report) => return report,
    };
    let missing = [
        ("manifest", parsed.positionals.is_empty()),
        ("--cadence", parsed.last("--cadence").is_none()),
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
    execute(&Request {
        manifest: &parsed.positionals[0],
        selection: Selection {
            artifact_id: parsed.last("--artifact-id"),
            cadence: parsed.last("--cadence").unwrap_or_default(),
        },
        require_single_file: parsed.flag("--require-single-file"),
        print_serving_file: parsed.flag("--print-serving-file"),
        github_output: parsed.last("--github-output"),
        output_prefix: parsed.last("--github-output-prefix").unwrap_or_default(),
        verify_root: parsed.last("--verify-root"),
    })
}

/// Resolves `request`; stdout keeps verified lines written before a failure.
pub(super) fn execute(request: &Request<'_>) -> CheckReport {
    let mut stdout = String::new();
    match resolve_into(request, &mut stdout) {
        Ok(()) => CheckReport::success(stdout),
        Err(error) => CheckReport {
            stdout,
            stderr: format!("test-model manifest error: {error}\n"),
            code: 2,
        },
    }
}

// Insert before resolve_into in existing model_registry/resolve.rs.
/// Select a name from a cadence-admitted manifest with reviewed integrity fields.
/// This does not verify immutable revisions, local blobs or cache discovery.
fn serving_file(files: &[PinnedFile]) -> ModelResult<&PinnedFile> {
    files
        .iter()
        .filter_map(|file| {
            crate::model_registry::serving_entry::rank(&file.name).map(|rank| (rank, file))
        })
        .min_by(|(ar, a), (br, b)| {
            ar.cmp(br)
                .then(a.size_bytes.cmp(&b.size_bytes))
                .then(a.name.cmp(&b.name))
        })
        .map(|(_, file)| file)
        .ok_or_else(|| ModelError("artifact has no serving GGUF or first shard".into()))
}

fn resolve_into(request: &Request<'_>, stdout: &mut String) -> ModelResult<()> {
    if request.print_serving_file
        && (request.github_output.is_some() || request.verify_root.is_some())
    {
        return fail("--print-serving-file cannot combine with --github-output or --verify-root");
    }
    let bytes =
        std::fs::read(request.manifest).map_err(|error| io_error(&error, request.manifest))?;
    let manifest = Json::parse(&bytes).map_err(|error| ModelError(error.to_string()))?;
    let artifact = resolve(&manifest, &request.selection)?;
    if request.require_single_file && artifact.files.len() != 1 {
        return fail(format!(
            "artifact {} must contain exactly one file",
            artifact.id
        ));
    }
    if request.print_serving_file {
        stdout.push_str(&serving_file(&artifact.files)?.name);
        stdout.push('\n');
        return Ok(());
    }
    if let Some(root) = request.verify_root {
        for file in &artifact.files {
            stdout.push_str(&verify(root, file)?);
        }
    }
    let summary = summary(&artifact);
    match request.github_output {
        Some(path) => write_outputs(path, request.output_prefix, &summary),
        None if request.verify_root.is_none() => {
            let mut sorted = summary;
            sorted.sort_by_key(|(key, _)| *key);
            let fields = sorted
                .into_iter()
                .map(|(key, value)| (key.to_owned(), Json::String(value)));
            stdout.push_str(&dumps(&Json::Object(fields.collect()), ASCII));
            stdout.push('\n');
            Ok(())
        }
        None => Ok(()),
    }
}

fn io_error(error: &std::io::Error, path: &str) -> ModelError {
    ModelError(os_error_text(error, &python_path_display(Path::new(path))))
}

/// `_summary`: the step outputs, in legacy insertion order.
fn summary(artifact: &Resolved<'_>) -> Vec<(&'static str, String)> {
    let text = |key: &str| {
        super::fields::get(artifact.row, key)
            .and_then(Json::as_str)
            .unwrap_or_default()
            .to_owned()
    };
    let names = artifact
        .files
        .iter()
        .map(|file| Json::String(file.name.clone()))
        .collect();
    let mut result = vec![
        ("artifact_id", artifact.id.to_owned()),
        ("repo", text("repo")),
        ("revision", text("revision")),
        ("selector", text("selector")),
        ("model_ref", text("model_ref")),
        ("files_json", dumps(&Json::Array(names), ASCII_COMPACT)),
    ];
    if let [file] = artifact.files.as_slice() {
        result.push(("file", file.name.clone()));
        result.push(("url", file.url.clone()));
        result.push(("sha256", file.sha256.clone()));
        result.push(("size_bytes", file.size_bytes.to_string()));
    }
    let quantizations = super::fields::get(artifact.row, "quantizations").and_then(Json::as_array);
    if let Some(items) = quantizations
        && items
            .iter()
            .all(|item| item.as_str().is_some_and(|text| !text.is_empty()))
    {
        result.push((
            "quantizations_json",
            dumps(&Json::Array(items.to_vec()), ASCII_COMPACT),
        ));
    }
    result
}

/// `root / name`, displayed as `pathlib` prints it.
fn joined(root: &str, name: &str) -> String {
    python_path_display(&Path::new(root).join(name))
}

/// Size first, then a streamed SHA-256, against the pinned record.
pub(super) fn verify(root: &str, file: &PinnedFile) -> ModelResult<String> {
    verify_with_guard(root, file, &mut || Ok(()))
}

/// The parity owner supplies cancellation/deadline admission between file chunks.
/// Existing resolver callers keep their established adapter and output contract.
pub(super) fn verify_with_guard(
    root: &str,
    file: &PinnedFile,
    guard: &mut impl FnMut() -> std::io::Result<()>,
) -> ModelResult<String> {
    guard().map_err(|error| ModelError(error.to_string()))?;
    let shown = joined(root, &file.name);
    let path = Path::new(root).join(&file.name);
    if !path.is_file() {
        return fail(format!("artifact file is missing: {shown}"));
    }
    let actual_size = path
        .metadata()
        .map_err(|error| io_error(&error, &shown))?
        .len();
    if actual_size != file.size_bytes {
        return fail(format!(
            "artifact size mismatch for {shown}: expected {}, got {actual_size}",
            file.size_bytes
        ));
    }
    let actual =
        stream_sha256(&path, file.size_bytes, guard).map_err(|error| io_error(&error, &shown))?;
    if actual != file.sha256 {
        return fail(format!(
            "artifact SHA-256 mismatch for {shown}: expected {}, got {actual}",
            file.sha256
        ));
    }
    guard().map_err(|error| ModelError(error.to_string()))?;
    Ok(format!(
        "verified immutable test artifact: {shown} ({actual_size} bytes)\n"
    ))
}

fn stream_sha256(
    path: &Path,
    maximum: u64,
    guard: &mut impl FnMut() -> std::io::Result<()>,
) -> std::io::Result<String> {
    guard()?;
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NONBLOCK);
    }
    let file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err(std::io::Error::other("artifact must remain a regular file"));
    }
    let mut reader = file.take(
        maximum
            .checked_add(1)
            .ok_or_else(|| std::io::Error::other("artifact size overflow"))?,
    );
    let mut digest = Sha256::new();
    let mut chunk = vec![0_u8; 1024 * 1024];
    loop {
        guard()?;
        match reader.read(&mut chunk)? {
            0 => {
                guard()?;
                return Ok(hex::encode(digest.finalize()));
            }
            read => digest.update(&chunk[..read]),
        }
    }
}

/// `[A-Za-z][A-Za-z0-9_]*_`, or empty for unprefixed outputs.
fn valid_prefix(prefix: &str) -> bool {
    let mut chars = prefix.chars();
    prefix.is_empty()
        || (chars
            .next()
            .is_some_and(|first| first.is_ascii_alphabetic())
            && prefix.ends_with('_')
            && chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_'))
}

/// Appends `name=value` lines; a multi-line value stops the write there.
fn write_outputs(path: &str, prefix: &str, summary: &[(&str, String)]) -> ModelResult<()> {
    if !valid_prefix(prefix) {
        return fail(
            "github output prefix must end in '_' and contain only ASCII letters, digits, and underscores",
        );
    }
    let mut output = OpenOptions::new()
        .append(true)
        .create(true)
        .open(path)
        .map_err(|error| io_error(&error, path))?;
    for (name, value) in summary {
        if value.contains(['\n', '\r']) {
            return fail(format!("output {name} must be single-line"));
        }
        writeln!(output, "{prefix}{name}={value}").map_err(|error| io_error(&error, path))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parity_hash_guard_refuses_cancellation_between_chunks_before_verification() {
        let scratch = tempfile::tempdir().unwrap();
        let file = scratch.path().join("model.gguf");
        std::fs::write(&file, vec![42_u8; 3 * 1024 * 1024]).unwrap();
        let cancellation = crate::process::Cancellation::default();
        let mut calls = 0;
        let mut guard = || {
            calls += 1;
            if calls == 3 {
                cancellation.cancel();
            }
            if cancellation.is_cancelled() {
                Err(std::io::Error::new(
                    std::io::ErrorKind::Interrupted,
                    "cancelled",
                ))
            } else {
                Ok(())
            }
        };
        let result = stream_sha256(&file, 3 * 1024 * 1024, &mut guard);
        assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::Interrupted);
        assert_eq!(calls, 3);
        let actual = stream_sha256(&file, 3 * 1024 * 1024, &mut || Ok(())).unwrap();
        assert_eq!(
            actual,
            hex::encode(Sha256::digest(vec![42_u8; 3 * 1024 * 1024]))
        );
    }

    #[test]
    fn parity_verified_claim_is_refused_when_guard_cancels_after_complete_hash() {
        let scratch = tempfile::tempdir().unwrap();
        let bytes = vec![42_u8; 3 * 1024 * 1024];
        std::fs::write(scratch.path().join("model.gguf"), &bytes).unwrap();
        let record = PinnedFile {
            name: "model.gguf".into(),
            size_bytes: bytes.len() as u64,
            sha256: hex::encode(Sha256::digest(&bytes)),
            url: "https://fixture.invalid/model.gguf".into(),
        };
        let cancellation = crate::process::Cancellation::default();
        let mut calls = 0;
        let mut guard = || {
            calls += 1;
            // Three bounded chunks have been consumed before the EOF guard.
            if calls == 7 {
                cancellation.cancel();
            }
            if cancellation.is_cancelled() {
                Err(std::io::Error::new(
                    std::io::ErrorKind::Interrupted,
                    "cancelled",
                ))
            } else {
                Ok(())
            }
        };
        let refused = verify_with_guard(scratch.path().to_str().unwrap(), &record, &mut guard);
        assert!(refused.is_err());
        assert_eq!(calls, 7);
        let healthy = verify(scratch.path().to_str().unwrap(), &record).unwrap();
        assert!(healthy.contains("verified immutable test artifact:"));
    }

    #[test]
    fn migration_models_output_prefix_is_a_safe_identifier() {
        for good in ["", "dense_", "A_", "a1_b_"] {
            assert!(valid_prefix(good), "{good:?}");
        }
        for bad in ["_", "1x_", "dense", "a-b_", "a_\nurl=x_", "é_"] {
            assert!(!valid_prefix(bad), "{bad:?}");
        }
    }

    #[test]
    fn migration_models_joined_paths_display_like_pathlib() {
        assert_eq!(joined("./", "fixture.bin"), "fixture.bin");
        assert_eq!(joined("/", "sub"), "/sub");
        assert_eq!(joined("/tmp/x/", "./a.bin"), "/tmp/x/a.bin");
    }
    // Insert inside existing model_registry::resolve::tests module.
    #[test]
    fn serving_entry_selects_first_shard_before_smaller_later_shard_or_projector() {
        let file = |name: &str, size_bytes| PinnedFile {
            name: name.into(),
            url: String::new(),
            size_bytes,
            sha256: "a".repeat(64),
        };
        let mut files = vec![
            file("nested/Model-Q4_K_M-00002-of-00002.gguf", 1),
            file("nested/mmproj.gguf", 0),
            file("nested/Model-Q4_K_M-00001-of-00002.gguf", 100),
            file("single.gguf", 2),
        ];
        assert_eq!(
            serving_file(&files).unwrap().name,
            "nested/Model-Q4_K_M-00001-of-00002.gguf"
        );
        files.reverse();
        assert_eq!(
            serving_file(&files).unwrap().name,
            "nested/Model-Q4_K_M-00001-of-00002.gguf"
        );
        assert!(
            serving_file(&[file("model-00002-of-00002.gguf", 1), file("mmproj.gguf", 2)]).is_err()
        );
        assert_eq!(
            serving_file(&[file("single.gguf", 2)]).unwrap().name,
            "single.gguf"
        );
    }

    #[test]
    fn serving_cli_requires_cadence_authority_and_refuses_conflicting_output_before_selection() {
        for flag in ["-h", "--help"] {
            let help = run(&[flag.into()]);
            assert_eq!(help.code, 0);
            assert!(help.stderr.is_empty());
            assert!(help.stdout.contains("[--print-serving-file]"));
            assert!(
                help.stdout
                    .contains("does not inspect model bytes or local caches")
            );
        }
        let missing = run(&[]);
        assert_eq!(missing.code, 2);
        assert!(missing.stdout.is_empty());
        assert!(!missing.stderr.contains("--print-serving-file"));
        assert!(
            missing
                .stderr
                .ends_with("the following arguments are required: manifest, --cadence\n")
        );
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("manifest.json");
        let revision = "a".repeat(40);
        let artifact = serde_json::json!({"manifest_kind":"test-model-artifacts","artifacts":[{"id":"fixture","repo":"org/model","revision":revision,"selector":"Q4","model_ref":"org/model:Q4","cadences":["manual"],"files":["later-00002-of-00002.gguf","model-00001-of-00002.gguf"],"urls":[format!("https://huggingface.co/org/model/resolve/{revision}/later-00002-of-00002.gguf"),format!("https://huggingface.co/org/model/resolve/{revision}/model-00001-of-00002.gguf")],"file_integrity":{"later-00002-of-00002.gguf":{"size_bytes":1,"blob_id":"b".repeat(64)},"model-00001-of-00002.gguf":{"size_bytes":100,"blob_id":"c".repeat(64)}}}]});
        std::fs::write(&path, serde_json::to_vec(&artifact).unwrap()).unwrap();
        let args = |cadence: &str| {
            vec![
                "--cadence".into(),
                cadence.into(),
                "--print-serving-file".into(),
                path.to_str().unwrap().into(),
            ]
        };
        let accepted = run(&args("manual"));
        assert_eq!(accepted.code, 0, "{}", accepted.stderr);
        assert_eq!(accepted.stdout, "model-00001-of-00002.gguf\n");
        let refused = run(&args("pull-request"));
        assert_eq!(refused.code, 2);
        assert!(refused.stdout.is_empty());
        let mut conflict = args("manual");
        conflict.extend([
            "--github-output".into(),
            directory.path().join("outputs").to_str().unwrap().into(),
        ]);
        assert_eq!(run(&conflict).code, 2);
        assert!(!directory.path().join("outputs").exists());
        directory.close().unwrap();
    }
}
