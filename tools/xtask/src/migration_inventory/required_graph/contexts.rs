use super::sources::tokens;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Context {
    Launch,
    Reference,
    Provision,
    Optional,
    Unknown,
}

pub(super) fn selected_interpreter_call(line: &str) -> bool {
    line.split_whitespace()
        .next()
        .is_some_and(|word| word.starts_with("\"$python") && word.ends_with('"'))
}

pub(super) fn classify(block: &str, child: &str) -> Context {
    let line = block
        .trim_start_matches("run: ")
        .trim_start_matches('@')
        .trim();
    if line.starts_with('#') || line.starts_with("printf ") || line.starts_with("echo ") {
        return Context::Reference;
    }
    if line.starts_with("command -v ") || line.contains("cache-dependency-path:") {
        return Context::Provision;
    }
    if line.split_once('=').is_some_and(|(name, value)| {
        name.chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
            && !value.contains("$(")
    }) {
        return Context::Reference;
    }
    if line.starts_with("if ") && line.contains(child) && !line.contains("python") {
        return Context::Unknown;
    }
    if let Some((_, substitution)) = line.split_once("$(")
        && substitution.starts_with(child)
    {
        return if line.starts_with("if ") {
            Context::Optional
        } else {
            Context::Launch
        };
    }
    let words = line.split_whitespace().collect::<Vec<_>>();
    let position = words
        .iter()
        .position(|word| word.trim_matches(['\'', '"', '\\', '(', ')', ';']) == child);
    let Some(index) = position else {
        return Context::Reference;
    };
    let executable = words[..index]
        .iter()
        .rev()
        .find(|word| !matches!(**word, "|" | "!" | "&" | "exec"));
    let launch = matches!(
        executable,
        Some(&"python" | &"python3" | &"bash" | &"sh" | &"pwsh" | &"powershell")
    ) || (index > 0 && selected_interpreter_call(line))
        || index == 0
        || words.first() == Some(&"env")
        || (line.starts_with("if ")
            && words[..index]
                .iter()
                .any(|word| *word == "python3" || *word == "python"))
        || (index >= 2 && words[index - 2] == "-m" && words[index - 1] == "unittest")
        || words[..index]
            .iter()
            .any(|word| word.starts_with("$(python") || word.starts_with("${python"));
    if launch {
        if line.starts_with("if ") || line.contains("|| true") {
            Context::Optional
        } else {
            Context::Launch
        }
    } else if executable.is_some_and(|word| word.contains('$'))
        || line.contains(" -c ")
        || line.contains(" <<")
        || tokens(line).count() > 1
    {
        Context::Unknown
    } else {
        Context::Reference
    }
}
