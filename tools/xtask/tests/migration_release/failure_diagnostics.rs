use std::fmt::Write as _;
use std::fs::File;
use std::io::Read as _;
use std::path::Path;
use std::process::Output;

const LIMIT: u64 = 4096;

fn safe_lines(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(&bytes[..bytes.len().min(4096)]);
    let mut result = String::new();
    for line in text.lines().take(32) {
        let safe = match line {
            "release-notes: no PR entries after the link pass; nothing to regroup"
            | "release-notes: publishing deterministic notes"
            | "release-notes: refusing to edit a published release without approval"
            | "release-notes: preview with DRY_RUN=true, or publish with RELEASE_NOTES_APPROVED=true"
            | "release-notes: AGENT_MODEL unset; skipping agent review"
            | "fatal"
            | "HTTP 502" => line,
            _ => "[redacted non-control line]",
        };
        let _ = writeln!(result, "{safe}");
    }
    if bytes.len() > 4096 {
        result.push_str("[output exceeds diagnostic byte limit]\n");
    }
    result
}

fn staged_body(path: &Path) -> String {
    let Ok(file) = File::open(path) else {
        return if path.exists() {
            "exists, unreadable".to_owned()
        } else {
            "absent".to_owned()
        };
    };
    let mut bytes = Vec::new();
    if file.take(LIMIT).read_to_end(&mut bytes).is_err() {
        return "exists, read failed".to_owned();
    }
    let text = String::from_utf8_lossy(&bytes);
    let mut result = String::from("exists; redacted contents:\n");
    for line in text.lines().take(32) {
        if line == "## What's Changed" || line == "**Full Changelog**: compare" {
            let _ = writeln!(result, "{line}");
        } else if line.starts_with('*') {
            result.push_str("[redacted release entry]\n");
        } else if line.is_empty() {
            result.push('\n');
        } else {
            result.push_str("[redacted body line]\n");
        }
    }
    if path.metadata().is_ok_and(|metadata| metadata.len() > LIMIT) {
        result.push_str("[body exceeds diagnostic byte limit]\n");
    }
    result
}

pub fn failure_snapshot(stage: &Path, output: &Output, bodies: &[&str]) -> String {
    let mut result = format!(
        "stage={}; stage_exists={}; status={}; code={:?}; stdout ({} bytes):\n{}stderr ({} bytes):\n{}",
        stage.display(),
        stage.exists(),
        output.status,
        output.status.code(),
        output.stdout.len(),
        safe_lines(&output.stdout),
        output.stderr.len(),
        safe_lines(&output.stderr),
    );
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt as _;
        let _ = writeln!(result, "signal={:?}", output.status.signal());
    }
    for relative in bodies {
        let _ = writeln!(result, "{relative}: {}", staged_body(&stage.join(relative)));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::{Stage, TestResult};
    #[cfg(unix)]
    use std::process::Command;

    #[cfg(unix)]
    #[test]
    fn diagnostic_distinguishes_signaled_child_and_redacts_release_author() -> TestResult {
        let stage = Stage::new("diagnostic-signal")?;
        stage.write(
            "body.md",
            b"## What's Changed\n* fix by @private in https://github.com/o/r/pull/1\n",
        )?;
        let output = Command::new("/bin/sh")
            .arg("-c")
            .arg("printf 'api_key=private-value\\n'; printf 'token=private-value\\n' >&2; kill -TERM $$")
            .output()?;

        let diagnostic = failure_snapshot(stage.path(), &output, &["body.md", "missing.md"]);

        assert!(diagnostic.contains("signal=Some(15)"), "{diagnostic}");
        assert!(diagnostic.contains("body.md: exists; redacted contents:"));
        assert!(diagnostic.contains("missing.md: absent"));
        assert!(diagnostic.contains("stdout (") && diagnostic.contains("stderr ("));
        assert!(!diagnostic.contains("private-value") && !diagnostic.contains("@private"));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn failed_status_assertion_reports_stage_and_early_return_without_author() -> TestResult {
        let stage = Stage::new("diagnostic-exit-zero")?;
        stage.write(
            "work/body.md",
            b"## What's Changed\n* fix by @private in https://github.com/o/r/pull/1\n",
        )?;
        let output = Command::new("/bin/sh")
            .arg("-c")
            .arg("printf 'release-notes: no PR entries after the link pass; nothing to regroup\\n'")
            .output()?;

        let diagnostic = failure_snapshot(
            stage.path(),
            &output,
            &["work/body.md", "work/notes.deterministic.md"],
        );
        assert!(diagnostic.contains(&format!("stage={}", stage.path().display())));
        assert!(diagnostic.contains("stage_exists=true"));
        assert!(
            diagnostic.contains("status=exit status: 0") && diagnostic.contains("code=Some(0)")
        );
        assert!(diagnostic.contains("no PR entries after the link pass"));
        assert!(diagnostic.contains("work/body.md: exists; redacted contents:"));
        assert!(diagnostic.contains("work/notes.deterministic.md: absent"));
        assert!(!diagnostic.contains("@private"));
        Ok(())
    }
}
