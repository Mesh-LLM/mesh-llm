//! `models pin`: register Hugging Face files as new artifacts. Each revision
//! is resolved to its commit and each file's size and LFS SHA-256 are read
//! from the Hub's tree API over the pinned HTTPS transport, so a row never
//! enters the registry with integrity typed by hand. With no arguments it
//! registers every artifact the registry's `derived_suites` lack (see
//! `derived`); with `<repo>@<revision> <file>` it registers that one file.
//! `models generate` then projects the suites.

use super::derived::Entry;
use super::fields::{ModelError, ModelResult, fail, get, is_identifier, is_lower_hex};
use super::json_bytes::{PRETTY, dumps};
use crate::ci_plan::catalog::{os_error_text, python_path_display};
use crate::ci_plan::document::Json;
use crate::command_interrupt::Interrupt;
use crate::repository::check_args::Grammar;
use crate::repository::check_report::CheckReport;
use std::path::Path;
use std::time::{Duration, Instant};

const PROGRAM: &str = "models pin";
const GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool models pin [<repo>@<revision> <file> --id <id> --family <family>
    --selector <text> --suite <suite>... --tag <tag>... [--cadence <cadence>...]
    [--notes <text>] [--expect-sha256 <hex>]] [--registry <path>]
    [--timeout-secs <1..3600>] [--dry-run]",
    values: &[
        "--id",
        "--family",
        "--selector",
        "--suite",
        "--tag",
        "--cadence",
        "--notes",
        "--expect-sha256",
        "--registry",
        "--timeout-secs",
    ],
    flags: &["--dry-run"],
};
const DEFAULT_REGISTRY: &str = "ci/model-artifacts/registry.json";
const DEFAULT_TIMEOUT_SECS: u64 = 600;
const API: &str = "https://huggingface.co/api/models";
/// A tree listing or a revision document; a model directory is far smaller.
const REPLY_CAP: u64 = 16 * 1024 * 1024;
const SINGLE_OPTIONS: [&str; 7] = [
    "--id",
    "--family",
    "--selector",
    "--suite",
    "--tag",
    "--cadence",
    "--notes",
];

struct Request<'a> {
    registry: &'a Path,
    cadences: Vec<&'a str>,
    timeout: Duration,
    dry_run: bool,
}

/// What the Hub reports for one file at one commit.
struct Pinned {
    commit: String,
    size_bytes: u64,
    sha256: String,
}

pub(super) fn run(root: &Path, args: &[String]) -> CheckReport {
    if args.len() == 1 && matches!(args[0].as_str(), "-h" | "--help") {
        return CheckReport::success(format!(
            "usage: {}\n\nWith no file, registers every artifact the registry's derived_suites lack.\nWith <repo>@<revision> <file>, registers that file. Either way each revision\nis resolved to its commit and the file's size and LFS SHA-256 come from the\nHub, the rows are appended to the registry (default {DEFAULT_REGISTRY} under\nthe repository root) and any new suite is declared. Cadence defaults to\nmanual. --dry-run prints the rows instead. Run `models generate` afterwards.\n",
            GRAMMAR.usage
        ));
    }
    let parsed = match super::argv::parse(&GRAMMAR, PROGRAM, args) {
        Ok(parsed) => parsed,
        Err(report) => return report,
    };
    let single = !parsed.positionals.is_empty();
    if !single
        && let Some(name) = SINGLE_OPTIONS
            .iter()
            .chain(["--expect-sha256"].iter())
            .find(|name| !parsed.all(name).is_empty())
    {
        let message = format!("argument {name}: only with <repo>@<revision> <file>");
        return super::argv::usage(&GRAMMAR, PROGRAM, &message);
    }
    let timeout = match parsed.last("--timeout-secs") {
        None => DEFAULT_TIMEOUT_SECS,
        Some(text) => match text.parse::<u64>() {
            Ok(seconds) if (1..=3600).contains(&seconds) => seconds,
            _ => {
                return super::argv::usage(
                    &GRAMMAR,
                    PROGRAM,
                    "argument --timeout-secs: expected an integer from 1 to 3600",
                );
            }
        },
    };
    let registry = parsed
        .last("--registry")
        .map_or_else(|| root.join(DEFAULT_REGISTRY), |path| root.join(path));
    let mut cadences = parsed.all("--cadence");
    if cadences.is_empty() {
        cadences.push("manual");
    }
    let request = Request {
        registry: &registry,
        cadences,
        timeout: Duration::from_secs(timeout),
        dry_run: parsed.flag("--dry-run"),
    };
    let entries = if single {
        match single_entry(&parsed) {
            Ok(entry) => Some(vec![entry]),
            Err(report) => return report,
        }
    } else {
        None
    };
    let mut stdout = String::new();
    match pin_into(root, &request, entries, &mut stdout) {
        Ok(()) => CheckReport::success(stdout),
        Err(error) => CheckReport {
            stdout,
            stderr: format!("test-model pin error: {error}\n"),
            code: 2,
        },
    }
}

/// The one artifact named on the command line.
fn single_entry(parsed: &crate::repository::check_args::ParsedArgs) -> Result<Entry, CheckReport> {
    let required = [
        ("<file>", parsed.positionals.len() < 2),
        ("--id", parsed.last("--id").is_none()),
        ("--family", parsed.last("--family").is_none()),
        ("--selector", parsed.last("--selector").is_none()),
        ("--suite", parsed.all("--suite").is_empty()),
        ("--tag", parsed.all("--tag").is_empty()),
    ];
    let missing = required
        .iter()
        .filter(|(_, absent)| *absent)
        .map(|(name, _)| *name)
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        let message = format!(
            "the following arguments are required: {}",
            missing.join(", ")
        );
        return Err(super::argv::usage(&GRAMMAR, PROGRAM, &message));
    }
    if let [_, _, extra @ ..] = parsed.positionals.as_slice()
        && !extra.is_empty()
    {
        let message = format!("unrecognized arguments: {}", extra.join(" "));
        return Err(super::argv::usage(&GRAMMAR, PROGRAM, &message));
    }
    let Some((repo, revision)) = parsed.positionals[0].split_once('@') else {
        return Err(super::argv::usage(
            &GRAMMAR,
            PROGRAM,
            "argument <repo>@<revision>: expected owner/name@revision",
        ));
    };
    let mut tags = parsed
        .all("--tag")
        .iter()
        .map(|tag| (*tag).to_owned())
        .collect::<Vec<_>>();
    tags.sort();
    tags.dedup();
    Ok(Entry {
        suites: parsed
            .all("--suite")
            .iter()
            .map(|suite| (*suite).to_owned())
            .collect(),
        id: parsed.last("--id").unwrap_or_default().to_owned(),
        family: parsed.last("--family").unwrap_or_default().to_owned(),
        tags,
        repo: repo.to_owned(),
        revision: revision.to_owned(),
        file: parsed.positionals[1].clone(),
        selector: parsed.last("--selector").unwrap_or_default().to_owned(),
        expected: parsed.last("--expect-sha256").map(str::to_owned),
        notes: parsed.last("--notes").map(str::to_owned),
    })
}

fn pin_into(
    root: &Path,
    request: &Request<'_>,
    entries: Option<Vec<Entry>>,
    stdout: &mut String,
) -> ModelResult<()> {
    let shown = python_path_display(request.registry);
    let bytes = std::fs::read(request.registry).map_err(|error| io_error(&error, &shown))?;
    let registry = Json::parse(&bytes).map_err(|error| ModelError(error.to_string()))?;
    let object = registry
        .as_object()
        .ok_or_else(|| ModelError("registry must be an object".to_owned()))?;
    let entries = match entries {
        Some(entries) => entries,
        None => super::derived::missing(root, object)?,
    };
    if entries.is_empty() {
        stdout.push_str("every derived artifact is registered; nothing to pin\n");
        return Ok(());
    }
    validate(request, &entries)?;
    let rows = get(object, "artifacts")
        .and_then(Json::as_array)
        .unwrap_or_default();
    let taken = rows
        .iter()
        .filter_map(Json::as_object)
        .filter_map(|row| get(row, "id").and_then(Json::as_str))
        .find(|id| entries.iter().any(|entry| entry.id == *id));
    if let Some(id) = taken {
        return fail(format!("artifact id is already registered: {id}"));
    }
    let declared = get(object, "cadences")
        .and_then(Json::as_array)
        .unwrap_or_default();
    if let Some(cadence) = request
        .cadences
        .iter()
        .find(|cadence| !declared.iter().any(|item| item.as_str() == Some(cadence)))
    {
        return fail(format!("registry does not declare cadence {cadence}"));
    }

    let rows = pin_entries(request, &entries, stdout)?;
    if request.dry_run {
        for row in &rows {
            stdout.push_str(&dumps(row, PRETTY));
            stdout.push('\n');
        }
        return Ok(());
    }
    let suites = entries
        .iter()
        .flat_map(|entry| entry.suites.iter().map(String::as_str))
        .collect::<Vec<_>>();
    let text = dumps(&with_rows(&registry, &suites, rows), PRETTY) + "\n";
    std::fs::write(request.registry, text).map_err(|error| io_error(&error, &shown))?;
    stdout.push_str(&format!(
        "registered {} artifact(s) in {shown}; run `cargo xtool models generate`\n",
        entries.len()
    ));
    Ok(())
}

fn validate(request: &Request<'_>, entries: &[Entry]) -> ModelResult<()> {
    let segment = |text: &str| {
        !text.is_empty()
            && text != "."
            && text != ".."
            && text
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    };
    if let Some(cadence) = request
        .cadences
        .iter()
        .find(|cadence| !is_identifier(cadence))
    {
        return fail(format!("cadence must be a lowercase identifier: {cadence}"));
    }
    for entry in entries {
        match entry.repo.split_once('/') {
            Some((owner, name)) if segment(owner) && segment(name) => {}
            _ => return fail(format!("repo must be owner/name: {}", entry.repo)),
        }
        if !segment(&entry.revision) {
            return fail(format!("revision is invalid: {}", entry.revision));
        }
        if entry.file.is_empty()
            || entry.file.starts_with('/')
            || entry.file.contains('\\')
            || entry.file.split('/').any(|part| !segment(part))
        {
            return fail(format!("file path is unsafe: {}", entry.file));
        }
        if !is_identifier(&entry.id) {
            return fail(format!("id must be a lowercase identifier: {}", entry.id));
        }
        if let Some(suite) = entry.suites.iter().find(|suite| !is_identifier(suite)) {
            return fail(format!("suite must be a lowercase identifier: {suite}"));
        }
        if let Some(tag) = entry.tags.iter().find(|tag| !is_identifier(tag)) {
            return fail(format!("tag must be a lowercase identifier: {tag}"));
        }
        for (name, text) in [("family", &entry.family), ("selector", &entry.selector)] {
            if text.is_empty() || text.lines().count() != 1 {
                return fail(format!("{name} must be a single-line value"));
            }
        }
        if entry
            .expected
            .as_deref()
            .is_some_and(|expected| !is_lower_hex(expected, 64..=64))
        {
            return fail("--expect-sha256 must be 64 lowercase hex digits");
        }
    }
    let mut ids = entries
        .iter()
        .map(|entry| entry.id.as_str())
        .collect::<Vec<_>>();
    ids.sort_unstable();
    if let Some(pair) = ids.windows(2).find(|pair| pair[0] == pair[1]) {
        return fail(format!("artifact id is repeated: {}", pair[0]));
    }
    Ok(())
}

/// Every entry's row, with the Hub asked under one interruption scope.
fn pin_entries(
    request: &Request<'_>,
    entries: &[Entry],
    stdout: &mut String,
) -> ModelResult<Vec<Json>> {
    let interrupt = Interrupt::install().map_err(|reason| ModelError(reason.to_string()))?;
    let cancellation = interrupt.cancellation();
    let deadline = Instant::now() + request.timeout;
    let document = |url: String| -> ModelResult<Json> {
        let bytes =
            super::projector_download::fetch_trusted(&url, deadline, &cancellation, REPLY_CAP)
                .map_err(|error| ModelError(format!("{url}: {error}")))?;
        Json::parse(&bytes).map_err(|error| ModelError(format!("{url}: {error}")))
    };
    let rows = entries
        .iter()
        .map(|entry| {
            let pinned = pin_file(entry, &document)?;
            stdout.push_str(&format!(
                "{} {}@{} {} ({} bytes, sha256 {})\n",
                entry.id, entry.repo, pinned.commit, entry.file, pinned.size_bytes, pinned.sha256
            ));
            Ok(row(entry, &request.cadences, &pinned))
        })
        .collect::<ModelResult<Vec<_>>>();
    interrupt
        .finish()
        .map_err(|_| ModelError("pin interrupted".to_owned()))?;
    rows
}

/// The commit, size and LFS digest the Hub reports, checked against the
/// expected digest when one was given.
fn pin_file(entry: &Entry, document: &dyn Fn(String) -> ModelResult<Json>) -> ModelResult<Pinned> {
    let commit = if is_lower_hex(&entry.revision, 40..=40) {
        entry.revision.clone()
    } else {
        let info = document(format!("{API}/{}/revision/{}", entry.repo, entry.revision))?;
        match info.get("sha").and_then(Json::as_str) {
            Some(sha) if is_lower_hex(sha, 40..=40) => sha.to_owned(),
            _ => {
                return fail(format!(
                    "{} has no commit for {}",
                    entry.repo, entry.revision
                ));
            }
        }
    };
    let (directory, name) = entry
        .file
        .rsplit_once('/')
        .map_or(("", entry.file.as_str()), |(directory, name)| {
            (directory, name)
        });
    let listing = document(format!(
        "{API}/{}/tree/{commit}{}{directory}",
        entry.repo,
        if directory.is_empty() { "" } else { "/" }
    ))?;
    let found = listing
        .as_array()
        .unwrap_or_default()
        .iter()
        .filter_map(Json::as_object)
        .find(|item| {
            get(item, "type").and_then(Json::as_str) == Some("file")
                && get(item, "path").and_then(Json::as_str) == Some(entry.file.as_str())
        });
    let Some(found) = found else {
        return fail(format!(
            "{}@{commit} has no file {name} in {directory:?}",
            entry.repo
        ));
    };
    let Some(lfs) = get(found, "lfs").and_then(Json::as_object) else {
        return fail(format!(
            "{} is not an LFS object; nothing to pin",
            entry.file
        ));
    };
    let sha256 = match get(lfs, "oid").and_then(Json::as_str) {
        Some(oid) if is_lower_hex(oid, 64..=64) => oid.to_owned(),
        _ => return fail(format!("{} has no LFS SHA-256", entry.file)),
    };
    let size_bytes = get(lfs, "size")
        .and_then(Json::as_int)
        .and_then(|size| u64::try_from(size).ok())
        .filter(|size| *size > 0)
        .ok_or_else(|| ModelError(format!("{} has no LFS size", entry.file)))?;
    if let Some(expected) = &entry.expected
        && *expected != sha256
    {
        return fail(format!(
            "{}: expected sha256 {expected}, the Hub reports {sha256}",
            entry.file
        ));
    }
    Ok(Pinned {
        commit,
        size_bytes,
        sha256,
    })
}

fn text(value: &str) -> Json {
    Json::String(value.to_owned())
}

fn strings<'a>(values: impl Iterator<Item = &'a str>) -> Json {
    Json::Array(values.map(text).collect())
}

fn row(entry: &Entry, cadences: &[&str], pinned: &Pinned) -> Json {
    let file = Json::Object(vec![
        ("path".to_owned(), text(&entry.file)),
        (
            "size_bytes".to_owned(),
            Json::Number(serde_json::Number::from(pinned.size_bytes)),
        ),
        ("sha256".to_owned(), text(&pinned.sha256)),
    ]);
    let artifact = Json::Object(vec![
        ("repo".to_owned(), text(&entry.repo)),
        ("revision".to_owned(), text(&pinned.commit)),
        ("selector".to_owned(), text(&entry.selector)),
        ("files".to_owned(), Json::Array(vec![file])),
    ]);
    let mut entries = vec![
        ("id".to_owned(), text(&entry.id)),
        ("family".to_owned(), text(&entry.family)),
        (
            "suites".to_owned(),
            strings(entry.suites.iter().map(String::as_str)),
        ),
        ("cadences".to_owned(), strings(cadences.iter().copied())),
        (
            "capability_tags".to_owned(),
            strings(entry.tags.iter().map(String::as_str)),
        ),
        ("artifact".to_owned(), artifact),
    ];
    entries.extend(
        entry
            .notes
            .as_deref()
            .map(|notes| ("notes".to_owned(), text(notes))),
    );
    Json::Object(entries)
}

/// The registry with `rows` appended and every suite they name declared.
fn with_rows(registry: &Json, suites: &[&str], rows: Vec<Json>) -> Json {
    let entries = registry.as_object().unwrap_or_default();
    Json::Object(
        entries
            .iter()
            .map(|(key, value)| match (key.as_str(), value) {
                ("suites", Json::Array(declared)) => {
                    let mut added = suites
                        .iter()
                        .filter(|suite| !declared.iter().any(|item| item.as_str() == Some(suite)))
                        .copied()
                        .collect::<Vec<_>>();
                    added.dedup();
                    (
                        key.clone(),
                        Json::Array(
                            declared
                                .iter()
                                .cloned()
                                .chain(added.into_iter().map(text))
                                .collect(),
                        ),
                    )
                }
                ("artifacts", Json::Array(existing)) => (
                    key.clone(),
                    Json::Array(
                        existing
                            .iter()
                            .cloned()
                            .chain(rows.iter().cloned())
                            .collect(),
                    ),
                ),
                _ => (key.clone(), value.clone()),
            })
            .collect(),
    )
}

fn io_error(error: &std::io::Error, shown: &str) -> ModelError {
    ModelError(os_error_text(error, shown))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SMALL: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/models/registry-small.json"
    );

    fn entry(id: &str) -> Entry {
        Entry {
            suites: vec!["performance-core".to_owned()],
            id: id.to_owned(),
            family: "tiny".to_owned(),
            tags: vec!["dense".to_owned(), "gguf".to_owned()],
            repo: "acme/Tiny-GGUF".to_owned(),
            revision: "main".to_owned(),
            file: "Tiny-Q4_K_M.gguf".to_owned(),
            selector: "Q4_K_M".to_owned(),
            expected: None,
            notes: Some("a note".to_owned()),
        }
    }

    fn request(registry: &Path) -> Request<'_> {
        Request {
            registry,
            cadences: vec!["manual"],
            timeout: Duration::from_secs(1),
            dry_run: false,
        }
    }

    fn owned(args: &[&str]) -> Vec<String> {
        args.iter().map(|arg| (*arg).to_owned()).collect()
    }

    #[test]
    fn the_registry_round_trips_byte_for_byte_before_a_row_is_added() {
        let bytes = std::fs::read(SMALL).unwrap();
        let registry = Json::parse(&bytes).unwrap();
        assert_eq!(
            dumps(&registry, PRETTY) + "\n",
            String::from_utf8(bytes).unwrap()
        );
    }

    #[test]
    fn a_row_is_appended_and_its_new_suite_declared() {
        let bytes = std::fs::read(SMALL).unwrap();
        let registry = Json::parse(&bytes).unwrap();
        let pinned = Pinned {
            commit: "0123456789abcdef0123456789abcdef01234567".to_owned(),
            size_bytes: 12,
            sha256: "ab".repeat(32),
        };
        let rows = vec![row(&entry("tiny-q4-perf"), &["manual"], &pinned)];
        let updated = with_rows(&registry, &["performance-core", "performance-core"], rows);
        let suites = updated.get("suites").and_then(Json::as_array).unwrap();
        assert_eq!(
            suites
                .iter()
                .filter(|suite| suite.as_str() == Some("performance-core"))
                .count(),
            1
        );
        let rows = updated.get("artifacts").and_then(Json::as_array).unwrap();
        let last = rows.last().and_then(Json::as_object).unwrap();
        assert_eq!(get(last, "id").and_then(Json::as_str), Some("tiny-q4-perf"));
        assert_eq!(
            last.iter().map(|(key, _)| key.as_str()).collect::<Vec<_>>(),
            [
                "id",
                "family",
                "suites",
                "cadences",
                "capability_tags",
                "artifact",
                "notes"
            ]
        );
        let files = get(last, "artifact")
            .and_then(|artifact| artifact.get("files"))
            .and_then(Json::as_array)
            .unwrap();
        assert_eq!(files[0].get("size_bytes").and_then(Json::as_int), Some(12));
        assert_eq!(
            rows.len(),
            registry
                .get("artifacts")
                .and_then(Json::as_array)
                .unwrap()
                .len()
                + 1
        );
    }

    #[test]
    fn the_request_is_validated_before_the_hub_is_asked() {
        let registry = Path::new(SMALL);
        let bad = |entry: Entry| validate(&request(registry), &[entry]).unwrap_err().0;
        assert!(bad(entry("Perf")).contains("lowercase identifier"));
        let mut escape = entry("ok");
        escape.file = "../escape.gguf".to_owned();
        assert!(bad(escape).contains("unsafe"));
        let mut owner = entry("ok");
        owner.repo = "no-owner".to_owned();
        assert!(bad(owner).contains("owner/name"));
        let mut sha = entry("ok");
        sha.expected = Some("abc".to_owned());
        assert!(bad(sha).contains("64 lowercase hex"));
        let twice = [entry("ok"), entry("ok")];
        assert!(
            validate(&request(registry), &twice)
                .unwrap_err()
                .0
                .contains("repeated")
        );
        assert!(validate(&request(registry), &[entry("ok")]).is_ok());
    }

    #[test]
    fn a_taken_id_is_refused_without_a_network_call() {
        let bytes = std::fs::read(SMALL).unwrap();
        let first = Json::parse(&bytes).unwrap();
        let id = first.get("artifacts").and_then(Json::as_array).unwrap()[0]
            .get("id")
            .and_then(Json::as_str)
            .unwrap()
            .to_owned();
        let mut stdout = String::new();
        let error = pin_into(
            Path::new("."),
            &request(Path::new(SMALL)),
            Some(vec![entry(&id)]),
            &mut stdout,
        )
        .unwrap_err();
        assert!(error.0.contains("already registered"), "{}", error.0);
    }

    #[test]
    fn a_registry_without_derived_suites_has_nothing_to_pin() {
        let mut stdout = String::new();
        pin_into(
            Path::new("."),
            &request(Path::new(SMALL)),
            None,
            &mut stdout,
        )
        .unwrap();
        assert!(stdout.contains("nothing to pin"));
    }

    #[test]
    fn single_mode_options_are_required_and_exclusive() {
        let report = run(
            Path::new("."),
            &owned(&["acme/Tiny-GGUF@main", "Tiny.gguf"]),
        );
        assert!(
            report
                .stderr
                .contains("--id, --family, --selector, --suite, --tag")
        );
        let complete = [
            "acme/Tiny-GGUF",
            "Tiny.gguf",
            "--id",
            "tiny",
            "--family",
            "tiny",
            "--selector",
            "Q4",
            "--suite",
            "s",
            "--tag",
            "gguf",
        ];
        let report = run(Path::new("."), &owned(&complete));
        assert!(
            report.stderr.contains("owner/name@revision"),
            "{}",
            report.stderr
        );
        let report = run(Path::new("."), &owned(&["--suite", "s"]));
        assert!(
            report.stderr.contains("only with <repo>@<revision> <file>"),
            "{}",
            report.stderr
        );
    }
}
