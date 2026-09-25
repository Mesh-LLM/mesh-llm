use crate::command::DynResult;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

#[derive(Clone)]
pub(super) struct Candidate {
    pub(super) id: String,
    pub(super) path: String,
    pub(super) source_block: String,
    pub(super) executable: bool,
}

pub(super) fn is_instruction(path: &str) -> bool {
    path == "AGENTS.md"
        || path.ends_with("/AGENTS.md")
        || (path.starts_with(".agents/skills/")
            && (path.ends_with(".md")
                || path.ends_with("/agents/openai.yaml")
                || path.ends_with("/scripts/collect-release-inventory.py")))
        || path == ".agents/agents/release-validation.md"
        || path == ".github/instructions/pr.instructions.md"
        || path == "ci/llama-canary/agent-repair-prompt.md"
}

fn is_source(path: &str) -> bool {
    path == "Justfile"
        || [
            ".github/",
            "just/",
            "scripts/",
            "tools/",
            "ci/",
            "evals/",
            ".agents/",
            ".omo/prompts/",
        ]
        .iter()
        .any(|prefix| path.starts_with(prefix))
}

fn executable(path: &str) -> bool {
    path == "Justfile"
        || path == "scripts/hooks/commit-msg"
        || matches!(
            path,
            "tools/xtask/src/ci_validation/producers.rs"
                | "tools/xtask/src/ci_validation/windows_runtime.rs"
                | "tools/xtask/src/ci_validation/crate_coverage.rs"
                | "tools/xtask/src/automation_parity/legacy.rs"
        )
        || is_instruction(path)
        || [
            ".yml",
            ".yaml",
            ".sh",
            ".ps1",
            ".just",
            ".cmake",
            "CMakeLists.txt",
            "/prompt.txt",
            "/prompt.md",
        ]
        .iter()
        .any(|suffix| path.ends_with(suffix))
}

fn js_token(line: &str) -> bool {
    line.split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_' || ch == '.'))
        .any(|word| {
            matches!(
                word,
                "python"
                    | "python3"
                    | "pip"
                    | "pip3"
                    | "uv"
                    | "uvx"
                    | "py"
                    | "Python3_EXECUTABLE"
                    | "python_bin"
            ) || word.starts_with("python3.")
        })
        || line.contains(".py")
        || line.contains("$sdk_python")
        || line.contains("setup-python")
}

fn github_command(line: &str, in_run: bool, selected: Option<&str>) -> bool {
    let text = line.trim();
    if text.starts_with("- uses: actions/setup-python@")
        || text.starts_with("uses: actions/setup-python@")
    {
        return true;
    }
    if !in_run {
        return js_token(line) || text == "ci/requirements-ci-python.txt";
    }
    if let Some(name) = selected
        && (text.starts_with(&format!("\"${name}\" "))
            || text.starts_with(&format!("${name} "))
            || text.starts_with(&format!("\"${{{name}}}\" ")))
    {
        return true;
    }
    js_token(line)
        || text == "ci/requirements-ci-python.txt"
        || (text.starts_with('"') || text.starts_with('$'))
            && text
                .split_whitespace()
                .next()
                .is_some_and(|word| word.contains("python"))
}

fn shell_execution(text: &str) -> bool {
    let text = text.trim();
    if text.starts_with('#')
        || text.starts_with("//")
        || text.starts_with("<!--")
        || text.starts_with("for ")
        || text.starts_with("foreach ")
        || text.contains("command -v ")
        || text.contains("Get-Command ")
        || (text.starts_with("echo ") || text.starts_with("printf ")) && !text.contains("| python")
        || text.contains("hashFiles(")
        || text.contains("grep -E '")
        || text.starts_with("pkill -f ")
    {
        return false;
    }
    if let Some((name, value)) = text.split_once('=')
        && name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
        && !value.contains("$(python")
        && !value.contains("| python")
        && !value.contains(" python3 ")
        && !value.contains(" python ")
        && !value.contains("$(uv ")
        && !value.contains("$(pip ")
        && !value.contains("; python")
    {
        return false;
    }
    if text.starts_with("if [[ ") && !text.contains(" -c ") && !text.contains(" <<") {
        return false;
    }
    true
}

fn source_execution(path: &str, line: &str) -> bool {
    let text = line.trim();
    if is_instruction(path)
        && !text.starts_with("python")
        && !text.starts_with("pip")
        && !text.starts_with("uv ")
        && !text.starts_with("- `python")
        && !text.starts_with("- `pip")
        && !text.starts_with("- `uv ")
        && !text.starts_with("scripts/")
    {
        return false;
    }
    if text.starts_with('"')
        && text.contains(".py")
        && !text.contains(" -c ")
        && !text.contains("$(")
        && !text.contains("$SCRIPT")
    {
        return false;
    }
    shell_execution(text)
}

fn scan_github(path: &str, text: &str) -> Vec<Candidate> {
    let mut rows = Vec::new();
    let mut occurrences = BTreeMap::new();
    let mut block: Option<(&str, usize)> = None;
    let mut selected: Option<&str> = None;
    let mut heredoc: Option<&str> = None;
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(end) = heredoc {
            if trimmed == end {
                heredoc = None;
            }
            continue;
        }
        let indent = line.len() - line.trim_start().len();
        if block.is_some_and(|(_, depth)| !trimmed.is_empty() && indent <= depth) {
            block = None;
            selected = None;
        }
        let run_value = trimmed.strip_prefix("run: ");
        let run = run_value.is_some() || trimmed == "run: " || trimmed == "run:";
        let in_run = run || block.is_some_and(|(kind, _)| kind == "run");
        if let Some(value) = run_value {
            if value == "|" || value == ">" || value.starts_with("|-") || value.starts_with(">-") {
                block = Some(("run", indent));
            }
        } else if trimmed == "cache-dependency-path: |" || trimmed == "cache-dependency-path: >" {
            block = Some(("cache", indent));
        }
        if in_run
            && let Some((name, value)) = trimmed.split_once('=')
            && name
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
            && (value.trim_matches(['"', '\'']) == "python"
                || value.trim_matches(['"', '\'']) == "python3")
        {
            selected = Some(name);
        }
        if github_command(line, in_run, selected) {
            let mut row = candidate(path, "candidate", line, &mut occurrences);
            row.executable = in_run
                && shell_execution(trimmed)
                && !trimmed.starts_with("- uses:")
                && !trimmed.starts_with("uses:")
                && !trimmed.starts_with("python-version:")
                && !trimmed.starts_with("cache:")
                && !trimmed.starts_with("NOTE:")
                && !trimmed.starts_with("PIP_CACHE_DIR:")
                && !trimmed.starts_with("ci/requirements-ci-python.txt")
                && !trimmed.contains("- uses: actions/setup-python@");
            rows.push(row);
            if in_run && let Some((_, end)) = trimmed.rsplit_once("<<") {
                let delimiter = end
                    .split_whitespace()
                    .next()
                    .unwrap_or("")
                    .trim_matches(['"', '\'']);
                if !delimiter.is_empty()
                    && delimiter
                        .chars()
                        .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
                {
                    heredoc = Some(delimiter);
                }
            }
        }
    }
    rows
}

fn candidate(
    path: &str,
    kind: &str,
    line: &str,
    occurrences: &mut BTreeMap<String, usize>,
) -> Candidate {
    let normalized = line.split_whitespace().collect::<Vec<_>>().join(" ");
    let hash = Sha256::digest(normalized.as_bytes());
    let digest = hex::encode(hash);
    let base = format!("{path}#{kind}:{}", &digest[..16]);
    let occurrence = occurrences.entry(base.clone()).or_default();
    *occurrence += 1;
    Candidate {
        id: format!("{base}:{occurrence}"),
        path: path.to_owned(),
        source_block: line.trim().to_owned(),
        executable: true,
    }
}

pub(super) fn scan_source(path: &str, text: &str) -> Vec<Candidate> {
    if path.starts_with(".github/") && (path.ends_with(".yml") || path.ends_with(".yaml")) {
        return scan_github(path, text);
    }
    let mut occurrences = BTreeMap::new();
    text.lines()
        .filter_map(|line| {
            if path.ends_with(".py") {
                let kind = if line.contains("spec_from_file_location(")
                    || line.contains("run_path(")
                    || line.contains("import_module(")
                {
                    Some("dynamic-import")
                } else if [
                    "subprocess.run",
                    "subprocess.Popen",
                    "subprocess.check_call",
                    "subprocess.check_output",
                    "os.system",
                    "os.popen",
                    "sys.executable",
                ]
                .iter()
                .any(|token| line.contains(token))
                {
                    Some("subprocess-or-interpreter")
                } else {
                    None
                };
                kind.map(|kind| candidate(path, kind, line, &mut occurrences))
            } else if is_source(path)
                && executable(path)
                && js_token(line)
                && !line.trim().starts_with("set -e")
                && !line.contains("grep -E '")
                && (!line.trim().starts_with("#") || line.trim().contains(".py"))
                && (path.starts_with(".agents/")
                    || !path.ends_with(".md")
                    || !line.trim().starts_with('`'))
            {
                let mut row = candidate(path, "candidate", line, &mut occurrences);
                row.executable = source_execution(path, line);
                Some(row)
            } else {
                None
            }
        })
        .collect()
}

pub(super) fn scan_paths(root: &Path, paths: &[String]) -> DynResult<Vec<Candidate>> {
    let mut observed = Vec::new();
    for path in paths {
        if (is_source(path) && executable(path)) || path.ends_with(".py") || is_instruction(path) {
            let source = fs::read_to_string(root.join(path))?;
            observed.extend(scan_source(path, &source));
        }
    }
    Ok(observed)
}
