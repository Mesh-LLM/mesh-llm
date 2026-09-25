pub(super) fn tokens(line: &str) -> impl Iterator<Item = &str> {
    line.split(|ch: char| !(ch.is_ascii_alphanumeric() || matches!(ch, '/' | '.' | '_' | '-')))
        .filter(|word| {
            ["scripts/", "just/", "evals/", "tools/"]
                .iter()
                .any(|prefix| word.starts_with(prefix))
                && [".py", ".sh", ".ps1", ".just"]
                    .iter()
                    .any(|suffix| word.ends_with(suffix))
        })
}

fn executable_line(line: &str) -> bool {
    let line = line
        .trim_start()
        .trim_start_matches(['@', '-'])
        .trim_start();
    !line.starts_with('#')
        && !line.starts_with("echo ")
        && !line.starts_with("printf ")
        && !line.starts_with("command -v ")
}

pub(super) fn source_lines(path: &str, text: &str) -> Vec<(usize, String)> {
    if !path.ends_with(".yml") && !path.ends_with(".yaml") {
        let mut heredoc = None;
        let mut lines = Vec::new();
        for (index, line) in text.lines().enumerate() {
            let trimmed = line.trim();
            if let Some(end) = heredoc {
                if trimmed == end {
                    heredoc = None;
                }
                continue;
            }
            if executable_line(line) {
                lines.push((index + 1, trimmed.to_owned()));
            }
            heredoc = delimiter(trimmed);
        }
        return lines;
    }
    let mut run_indent = None;
    let mut heredoc = None;
    let mut result = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if let Some(end) = heredoc {
            if trimmed == end {
                heredoc = None;
            }
            continue;
        }
        let indent = line.len() - line.trim_start().len();
        if run_indent.is_some_and(|depth| !trimmed.is_empty() && indent <= depth) {
            run_indent = None;
        }
        if trimmed.starts_with("run: |") {
            run_indent = Some(indent);
        }
        if (trimmed.starts_with("uses: ")
            || trimmed.starts_with("- uses: ")
            || trimmed.starts_with("run: ") && !trimmed.starts_with("run: |")
            || run_indent.is_some() && executable_line(trimmed))
            && !trimmed.starts_with('#')
        {
            result.push((index + 1, trimmed.to_owned()));
            if run_indent.is_some() {
                heredoc = delimiter(trimmed);
            }
        }
    }
    result
}

fn delimiter(line: &str) -> Option<&str> {
    let (_, tail) = line.rsplit_once("<<")?;
    let end = tail
        .split_whitespace()
        .next()?
        .trim_matches(['\'', '"', '-']);
    if !end.is_empty()
        && end
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
    {
        Some(end)
    } else {
        None
    }
}

pub(super) fn local_target(line: &str) -> Option<String> {
    let value = line.trim_start_matches("- ").strip_prefix("uses: ")?;
    if value.contains("${{") || value.contains('@') || value.split_whitespace().count() != 1 {
        return None;
    }
    if let Some(path) = value.strip_prefix("./.github/workflows/") {
        return Some(format!(".github/workflows/{path}"));
    }
    value
        .strip_prefix("./.github/actions/")
        .map(|path| format!(".github/actions/{path}/action.yml"))
}

pub(super) fn protected_target(line: &str) -> bool {
    line.trim_start_matches("- ")
        .starts_with("uses: Mesh-LLM/mesh-llm/.github/workflows/")
}

pub(super) fn selection(line: &str) -> Option<&str> {
    line.trim_start_matches("- ").strip_prefix("uses: ")
}
