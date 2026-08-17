use std::collections::BTreeSet;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) type DynError = Box<dyn Error>;
pub(crate) type DynResult<T> = Result<T, DynError>;

pub(crate) fn write_json_file<T: serde::Serialize>(path: &Path, value: &T) -> DynResult<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

pub(crate) fn print_json<T: serde::Serialize>(value: &T) -> DynResult<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

pub(crate) fn sourced_script_stdout(
    repo_root: &Path,
    script_relative_path: &str,
    expression: &str,
    envs: &[(&str, &str)],
    extra_args: &[&str],
) -> DynResult<String> {
    let script_path = repo_root.join(script_relative_path);
    let command = format!("source \"$1\"; {expression}");
    let mut bash = Command::new("bash");
    bash.current_dir(repo_root)
        .arg("-lc")
        .arg(command)
        .arg("bash")
        .arg(script_path);
    for extra_arg in extra_args {
        bash.arg(extra_arg);
    }
    for (key, value) in envs {
        bash.env(key, value);
    }

    let output = run_command(&mut bash)?;
    if !output.status.success() {
        return Err(format!(
            "script command failed: {}",
            trimmed_stderr_or_stdout(&output)
        )
        .into());
    }
    Ok(trimmed_stdout(&output))
}

pub(crate) fn run_command(command: &mut Command) -> DynResult<Output> {
    Ok(command.output()?)
}

pub(crate) fn trimmed_stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

pub(crate) fn trimmed_stderr_or_stdout(output: &Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    if !stderr.is_empty() {
        stderr
    } else {
        trimmed_stdout(output)
    }
}

pub(crate) fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    std::env::temp_dir().join(format!(".tmp-{prefix}-{}-{nanos}", std::process::id()))
}

pub(crate) fn ensure_eq(expected: &str, actual: &str, context: &str) -> DynResult<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(format!("{context}: expected `{expected}`, got `{actual}`").into())
    }
}

pub(crate) fn ensure_eq_option(
    expected: Option<&str>,
    actual: Option<&str>,
    context: &str,
) -> DynResult<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(format!("{context}: expected {:?}, got {:?}", expected, actual).into())
    }
}

pub(crate) fn ensure_nonempty_option(value: &Option<String>, context: &str) -> DynResult<()> {
    match value.as_deref() {
        Some(value) if !value.is_empty() => Ok(()),
        _ => Err(format!("{context}: missing value").into()),
    }
}

pub(crate) fn ensure_set_eq(
    expected: &BTreeSet<String>,
    actual: &BTreeSet<String>,
    context: &str,
) -> DynResult<()> {
    if expected == actual {
        return Ok(());
    }

    let missing = expected
        .difference(actual)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    let extra = actual
        .difference(expected)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    Err(format!(
        "{context}: workspace crate list drift detected; missing [{}], extra [{}]",
        missing, extra
    )
    .into())
}

pub(crate) fn ensure_status(expected: i32, actual: Option<i32>, context: &str) -> DynResult<()> {
    match actual {
        Some(status) if status == expected => Ok(()),
        Some(status) => {
            Err(format!("{context}: expected exit code {expected}, got {status}").into())
        }
        None => Err(format!("{context}: process terminated by signal").into()),
    }
}

pub(crate) fn ensure_contains(haystack: &str, needle: &str, context: &str) -> DynResult<()> {
    if haystack.contains(needle) {
        Ok(())
    } else {
        Err(format!("{context}: missing `{needle}`").into())
    }
}

pub(crate) fn ensure_not_contains(haystack: &str, needle: &str, context: &str) -> DynResult<()> {
    if haystack.contains(needle) {
        Err(format!("{context}: unexpected `{needle}`").into())
    } else {
        Ok(())
    }
}

pub(crate) fn ensure_contains_normalized(
    haystack: &str,
    needle: &str,
    context: &str,
) -> DynResult<()> {
    let normalized_haystack = normalize_whitespace(haystack);
    let normalized_needle = normalize_whitespace(needle);
    if normalized_haystack.contains(&normalized_needle) {
        Ok(())
    } else {
        Err(format!("{context}: missing `{needle}`").into())
    }
}

pub(crate) fn manifest_section<'a>(manifest: &'a str, section: &str) -> Option<&'a str> {
    let header = format!("[{section}]");
    let mut section_start = None;

    let mut offset = 0;
    for raw_line in manifest.split_inclusive('\n') {
        let line_start = offset;
        offset += raw_line.len();
        let line = raw_line.strip_suffix('\n').unwrap_or(raw_line);
        let trimmed = line.trim();
        if trimmed == header {
            section_start = Some(offset);
            continue;
        }
        if section_start.is_some() && trimmed.starts_with('[') {
            return section_start.map(|start| &manifest[start..line_start]);
        }
    }

    section_start.map(|start| &manifest[start..])
}

pub(crate) fn package_section_uses_workspace_version(section: &str) -> bool {
    section.lines().any(|line| {
        let compact = line
            .split('#')
            .next()
            .unwrap_or_default()
            .chars()
            .filter(|ch| !ch.is_whitespace())
            .collect::<String>();
        compact == "version.workspace=true" || compact == "version={workspace=true}"
    })
}

pub(crate) fn workflow_job_section<'a>(workflow: &'a str, job_name: &str) -> Option<&'a str> {
    let header = format!("  {job_name}:");
    let mut section_start = None;

    let mut offset = 0;
    for raw_line in workflow.split_inclusive('\n') {
        let line_start = offset;
        offset += raw_line.len();
        let line = raw_line.strip_suffix('\n').unwrap_or(raw_line);
        if line == header {
            section_start = Some(offset);
            continue;
        }
        if section_start.is_some()
            && line.starts_with("  ")
            && !line.starts_with("    ")
            && line.trim_end().ends_with(':')
        {
            return section_start.map(|start| &workflow[start..line_start]);
        }
    }

    section_start.map(|start| &workflow[start..])
}

pub(crate) fn display_relative(repo_root: &Path, path: &Path) -> String {
    path.strip_prefix(repo_root)
        .unwrap_or(path)
        .display()
        .to_string()
}

pub(crate) fn normalize_whitespace(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}
