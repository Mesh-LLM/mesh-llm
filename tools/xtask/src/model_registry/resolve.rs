//! `models resolve`: the Rust owner of `scripts/resolve-test-model-manifest.py`.
//! Selects one cadence-authorized artifact, optionally verifies its files on
//! disk, and emits a JSON summary or single-line GitHub step outputs.

use super::fields::{ModelError, ModelResult, fail};
use super::manifest::{PinnedFile, Resolved, Selection, resolve};
use super::python_json::{ASCII, ASCII_COMPACT, dumps};
use crate::ci_plan::catalog::{os_error_text, python_path_display};
use crate::ci_plan::document::Json;
use crate::repository::check_args::Grammar;
use crate::repository::check_report::CheckReport;
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
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
    flags: &["--require-single-file"],
};

/// One resolution, as the legacy argv or the restore step describes it.
pub(super) struct Request<'a> {
    pub(super) manifest: &'a str,
    pub(super) selection: Selection<'a>,
    pub(super) require_single_file: bool,
    pub(super) github_output: Option<&'a str>,
    pub(super) output_prefix: &'a str,
    pub(super) verify_root: Option<&'a str>,
}

pub(super) fn run(args: &[String]) -> CheckReport {
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

fn resolve_into(request: &Request<'_>, stdout: &mut String) -> ModelResult<()> {
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
fn verify(root: &str, file: &PinnedFile) -> ModelResult<String> {
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
    let actual = stream_sha256(&path).map_err(|error| io_error(&error, &shown))?;
    if actual != file.sha256 {
        return fail(format!(
            "artifact SHA-256 mismatch for {shown}: expected {}, got {actual}",
            file.sha256
        ));
    }
    Ok(format!(
        "verified immutable test artifact: {shown} ({actual_size} bytes)\n"
    ))
}

fn stream_sha256(path: &Path) -> std::io::Result<String> {
    let mut reader = File::open(path)?;
    let mut digest = Sha256::new();
    let mut chunk = vec![0_u8; 1024 * 1024];
    loop {
        match reader.read(&mut chunk)? {
            0 => return Ok(hex::encode(digest.finalize())),
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
}
