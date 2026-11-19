//! `release notes-base <target-tag> [--unpromoted <file>]`:
//! `scripts/select-release-notes-base.py`. Reads candidate tags from stdin and
//! prints the newest stable `vX.Y.Z` tag older than the target, or nothing
//! when there is none. A candidate that was cut but never promoted carries a
//! stable-looking tag without being a release, so the tags listed in the
//! `--unpromoted` file are never selected.

use crate::prepared_input::text_io::decode_utf8;
use crate::repository::check_report::CheckReport;
use crate::repository::text::strip;
use std::cmp::Ordering;

const USAGE: &str = "usage: select-release-notes-base.py <target-tag> [--unpromoted <file>]\n";

/// One version component: an unbounded Python `int` of ASCII digits, kept
/// as its digits without leading zeros so comparison never overflows.
#[derive(Clone, PartialEq, Eq)]
struct Component(String);

impl Component {
    fn new(digits: &str) -> Self {
        let trimmed = digits.trim_start_matches('0');
        Self(if trimmed.is_empty() { "0" } else { trimmed }.to_owned())
    }
}

impl Ord for Component {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .len()
            .cmp(&other.0.len())
            .then_with(|| self.0.cmp(&other.0))
    }
}

impl PartialOrd for Component {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

type Version = [Component; 3];

/// `[0-9]+` at the start of `text`: the digits and the remainder.
fn digits(text: &str) -> Option<(&str, &str)> {
    let end = text
        .find(|ch: char| !ch.is_ascii_digit())
        .unwrap_or(text.len());
    (end > 0).then(|| text.split_at(end))
}

/// `(?P<major>[0-9]+)\.(?P<minor>[0-9]+)\.(?P<patch>[0-9]+)` and the rest.
fn version_prefix(text: &str) -> Option<(Version, &str)> {
    let (major, rest) = digits(text)?;
    let (minor, rest) = digits(rest.strip_prefix('.')?)?;
    let (patch, rest) = digits(rest.strip_prefix('.')?)?;
    let version = [major, minor, patch].map(Component::new);
    Some((version, rest))
}

/// `TARGET_TAG.fullmatch`: optional `v`, then an optional `-[0-9A-Za-z.-]+`.
fn target_version(tag: &str) -> Option<Version> {
    let (version, rest) = version_prefix(tag.strip_prefix('v').unwrap_or(tag))?;
    let valid_suffix = |suffix: &str| {
        !suffix.is_empty()
            && suffix
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '.' || ch == '-')
    };
    (rest.is_empty() || rest.strip_prefix('-').is_some_and(valid_suffix)).then_some(version)
}

/// `STABLE_TAG.fullmatch`: exactly `vX.Y.Z`.
fn stable_version(tag: &str) -> Option<Version> {
    let (version, rest) = version_prefix(tag.strip_prefix('v')?)?;
    rest.is_empty().then_some(version)
}

/// `select_release_notes_base`: `max((version, tag))` below the target.
fn select<'a>(
    target: &Version,
    lines: impl Iterator<Item = &'a str>,
    unpromoted: &[&str],
) -> Option<&'a str> {
    lines
        .map(strip)
        .filter(|tag| !unpromoted.contains(tag))
        .filter_map(|tag| stable_version(tag).map(|version| (version, tag)))
        .filter(|(version, _)| version < target)
        .max()
        .map(|(_, tag)| tag)
}

/// Iterating `sys.stdin`: text split after each `\n` only.
fn stdin_lines(text: &str) -> impl Iterator<Item = &str> {
    text.split_inclusive('\n')
}

pub(crate) fn run(
    args: &[String],
    read_stdin: impl FnOnce() -> std::io::Result<Vec<u8>>,
) -> CheckReport {
    let (target, unpromoted) = match args {
        [target] => (target, None),
        [target, option, path] if option == "--unpromoted" => (target, Some(path)),
        _ => {
            return CheckReport {
                stdout: String::new(),
                stderr: USAGE.to_owned(),
                code: 2,
            };
        }
    };
    let Some(version) = target_version(strip(target)) else {
        return usage_failure(format!("invalid release tag: {target}"));
    };
    let text = match read_stdin() {
        Ok(bytes) => match decode_utf8(bytes) {
            Ok(text) => text,
            // `UnicodeDecodeError` is a `ValueError`, so main reports it.
            Err(message) => return usage_failure(message),
        },
        Err(error) => return usage_failure(error.to_string()),
    };
    let unpromoted = match unpromoted.map(std::fs::read_to_string).transpose() {
        Ok(text) => text.unwrap_or_default(),
        Err(error) => return usage_failure(format!("unpromoted candidates: {error}")),
    };
    let unpromoted: Vec<&str> = unpromoted.lines().map(strip).collect();
    let stdout = select(&version, stdin_lines(&text), &unpromoted)
        .map(|tag| format!("{tag}\n"))
        .unwrap_or_default();
    CheckReport::success(stdout)
}

fn usage_failure(message: String) -> CheckReport {
    CheckReport {
        stdout: String::new(),
        stderr: format!("{message}\n"),
        code: 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pick(target: &str, input: &str) -> CheckReport {
        run(&[target.to_owned()], || Ok(input.as_bytes().to_vec()))
    }

    #[test]
    fn migration_release_notes_base_selects_previous_stable() {
        let tags = "v0.9.9\nv1.0.0-rc.1\nv1.0.0\n 1.1.0\nv1.2.0\nv01.1.0\nv1.1.0\n";
        assert_eq!(pick("v1.2.0", tags).stdout, "v1.1.0\n");
        assert_eq!(pick(" 1.2.0-rc.1 ", tags).stdout, "v1.1.0\n");
        assert_eq!(pick("v1.0.0", tags).stdout, "v0.9.9\n");
        assert_eq!(pick("v0.9.9", tags).stdout, "");
        assert_eq!(pick("v1.0.0\r", "v0.1.0\rv0.2.0\n").stdout, "");
    }

    #[test]
    fn release_notes_base_skips_candidates_that_were_never_promoted() {
        let directory = tempfile::tempdir().unwrap();
        let list = directory.path().join("unpromoted");
        std::fs::write(&list, "v1.1.0\nv1.0.0-rc.1\n").unwrap();
        let pick = |target: &str, list: &std::path::Path| {
            let args = [target, "--unpromoted", list.to_str().unwrap()].map(str::to_owned);
            run(&args, || Ok(b"v0.9.9\nv1.0.0\nv1.1.0\n".to_vec()))
        };
        assert_eq!(pick("v1.1.1", &list).stdout, "v1.0.0\n");
        assert_eq!(pick("v1.0.0", &list).stdout, "v0.9.9\n");
        let absent = pick("v1.1.1", &directory.path().join("absent"));
        assert_eq!((absent.code, absent.stdout.as_str()), (2, ""));
        assert!(absent.stderr.starts_with("unpromoted candidates: "));
    }

    #[test]
    fn migration_release_notes_base_rejects_invalid_targets() {
        for target in ["x", "v1.2", "v1.2.3-", "vv1.2.3", "1.2.3+b", "-h"] {
            let report = pick(target, "");
            assert_eq!(report.code, 2, "{target}");
            assert_eq!(report.stderr, format!("invalid release tag: {target}\n"));
        }
    }

    #[test]
    fn migration_release_notes_base_compares_unbounded_components() {
        let tags = "v99999999999999999999999.0.0\nv100000000000000000000000.0.0\n";
        assert_eq!(
            pick("v100000000000000000000001.0.0", tags).stdout,
            "v100000000000000000000000.0.0\n"
        );
    }
}
