//! `ci-ops sccache-stats`: the Rust owner of
//! `.github/actions/capture-sccache-stats/capture.py`. Runs
//! `sccache --show-stats --stats-format json`, keeps only the required
//! counters (paths, URLs and raw sccache stderr never reach the logs or the
//! evidence), classifies the cache observation, writes the evidence JSON and
//! optional step outputs, and matches the legacy streams and statuses.

use crate::ci_operations::build_cache_tree::io_text;
use crate::ci_operations::build_cache_values::resolve;
use crate::ci_operations::sccache_argv::{Args, parse};
use crate::ci_operations::sccache_evidence::{Failure, assess, decode, sanitize};
use crate::ci_operations::sccache_render::{evidence_text, github_output_text, summary_text};
use crate::repository::check_report::CheckReport;
use std::io::Write as _;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};

const SHOW_STATS: [&str; 3] = ["--show-stats", "--stats-format", "json"];

pub(crate) fn run(args: &[String]) -> CheckReport {
    let args = match parse(args) {
        Ok(args) => args,
        Err(report) => return report,
    };
    capture(&args).map_or_else(
        |message| CheckReport::failure(String::new(), format!("ERROR: {message}\n")),
        CheckReport::success,
    )
}

/// `ARTIFACT_NAME_PATTERN.fullmatch`: `[A-Za-z0-9][A-Za-z0-9._-]{0,127}`.
fn valid_artifact_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    bytes.first().is_some_and(u8::is_ascii_alphanumeric)
        && bytes.len() <= 128
        && bytes
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

/// `shutil.which("sccache") is not None` on POSIX.
fn sccache_on_path() -> bool {
    let path = std::env::var_os("PATH").unwrap_or_else(|| ":/bin:/usr/bin".into());
    std::env::split_paths(&path).any(|directory| {
        let candidate = directory.join("sccache");
        is_executable_file(&candidate)
    })
}

fn is_executable_file(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path)
        .is_ok_and(|metadata| metadata.is_file() && metadata.permissions().mode() & 0o111 != 0)
}

/// `subprocess.run(..., capture_output=True, text=True)`: stderr is
/// captured and discarded; stdout uses universal newlines.
fn run_sccache() -> Result<String, Failure> {
    let output = Command::new("sccache")
        .args(SHOW_STATS)
        .stdin(Stdio::inherit())
        .output()
        .map_err(|error| io_text(&error, Path::new("sccache")))?;
    if !output.status.success() {
        let code = output.status.code().unwrap_or_else(|| {
            use std::os::unix::process::ExitStatusExt;
            -output.status.signal().unwrap_or(0)
        });
        return Err(format!(
            "sccache {} failed with exit code {code}",
            SHOW_STATS.join(" ")
        ));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    Ok(text.replace("\r\n", "\n").replace('\r', "\n"))
}

/// `pathlib.Path(text)`: drops empty and `.` components.
fn pure_path(text: &str) -> PathBuf {
    let path = Path::new(text);
    let mut normalized = PathBuf::new();
    for component in path.components() {
        if component != Component::CurDir {
            normalized.push(component);
        }
    }
    if normalized.as_os_str().is_empty() {
        normalized.push(".");
    }
    normalized
}

fn capture(args: &Args) -> Result<String, Failure> {
    if !valid_artifact_name(&args.artifact_name) {
        return Err("artifact name must contain only letters, numbers, dots, \
                    underscores, and hyphens"
            .to_owned());
    }
    if !sccache_on_path() {
        return Err("sccache is required to capture build-cache evidence".to_owned());
    }
    let payload = decode(run_sccache()?)?;
    let counters = sanitize(&payload)?;
    let assessment = assess(args.expectation, args.minimum_hit_rate, &counters)?;

    let output = pure_path(&args.output);
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent).map_err(|error| io_text(&error, parent))?;
    }
    std::fs::write(&output, evidence_text(&counters, &assessment))
        .map_err(|error| io_text(&error, &output))?;
    let stats_file = resolve(&output);
    if let Some(destination) = &args.github_output {
        let destination = pure_path(destination);
        let text = github_output_text(&stats_file.to_string_lossy(), &counters, &assessment);
        std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&destination)
            .and_then(|mut file| file.write_all(text.as_bytes()))
            .map_err(|error| io_text(&error, &destination))?;
    }
    Ok(summary_text(&counters, &assessment))
}

#[cfg(test)]
mod tests {
    use super::{pure_path, valid_artifact_name};

    #[test]
    fn migration_ci_operations_artifact_names_follow_the_legacy_pattern() {
        for accepted in ["a", "sccache-test-1", "A.b_c-9", &"x".repeat(128)] {
            assert!(valid_artifact_name(accepted), "{accepted}");
        }
        for rejected in ["", "../a", ".a", "-a", "a/b", "a b", "é", &"x".repeat(129)] {
            assert!(!valid_artifact_name(rejected), "{rejected}");
        }
    }

    #[test]
    fn migration_ci_operations_pure_path_matches_pathlib() {
        assert_eq!(pure_path("a//./b.json").to_string_lossy(), "a/b.json");
        assert_eq!(pure_path("").to_string_lossy(), ".");
    }
}
