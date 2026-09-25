use std::collections::BTreeMap;
use std::path::Path;

fn quoted_at_end(value: &str) -> Option<&str> {
    let value = value.trim().trim_end_matches([')', ',', ' ']);
    let quote = value.chars().last()?;
    if !matches!(quote, '\'' | '"') {
        return None;
    }
    let start = value[..value.len() - 1].rfind(quote)?;
    Some(&value[start + 1..value.len() - 1])
}

pub(super) fn path_target(parent: &str, expression: &str) -> (Option<String>, bool) {
    let Some(value) = quoted_at_end(expression) else {
        return (None, false);
    };
    if !value.ends_with(".py") && !value.ends_with(".sh") && !value.ends_with(".ps1") {
        return (None, false);
    }
    if expression.contains("__file__") {
        if !expression.contains(" / ") {
            return (None, false);
        }
        let path = Path::new(parent);
        let directory = if expression.contains("parents[1]") {
            path.parent().and_then(Path::parent)
        } else {
            path.parent()
        }
        .unwrap_or_else(|| Path::new(""));
        return (
            Some(directory.join(value).to_string_lossy().into_owned()),
            true,
        );
    }
    if value.starts_with("scripts/") || value.starts_with("evals/") {
        return (Some(value.to_owned()), !expression.contains("root /"));
    }
    (None, false)
}

pub(super) fn subprocess_target(
    parent: &str,
    line: &str,
    bindings: &BTreeMap<String, String>,
) -> (Option<String>, bool) {
    let Some((_, argv)) = line.split_once("subprocess.") else {
        return (None, false);
    };
    let Some(first) = argv.trim_start().split_once('(').map(|(_, args)| args) else {
        return (None, false);
    };
    let Some(first) = first.trim_start().strip_prefix('[') else {
        return (None, false);
    };
    let Some((command, remaining)) = first.split_once(',') else {
        return (None, false);
    };
    let command = command.trim().trim_matches(['\'', '"']);
    if command.is_empty() || command.contains('(') {
        return (None, false);
    }
    if command == "bash" || command == "sh" || command == "python3" || command == "python" {
        let argument = remaining.trim_start();
        if let Some(name) = argument
            .strip_prefix("str(")
            .and_then(|value| value.split_once(')').map(|(name, _)| name))
            && let Some(target) = bindings.get(name.trim())
        {
            return (Some(target.clone()), false);
        }
        if argument.starts_with("str(") {
            let expression = argument.trim_start_matches("str(");
            if expression.contains("__file__") {
                let expression = expression
                    .rsplit_once(" / ")
                    .map_or(expression, |(_, value)| value);
                let expression = expression
                    .split_once(')')
                    .map_or(expression, |(value, _)| value);
                return path_target(parent, &format!("__file__ / {expression}"));
            }
            return path_target(parent, expression);
        }
        let target = argument
            .split_once(',')
            .map_or(argument, |(path, _)| path)
            .trim()
            .trim_end_matches(']')
            .trim_matches(['\'', '"']);
        return (
            (target.starts_with("scripts/") || target.starts_with("evals/"))
                .then(|| target.to_owned()),
            !argument.contains("root /"),
        );
    }
    (None, false)
}

pub(super) fn root_bindings(text: &str) -> BTreeMap<String, String> {
    text.lines()
        .filter_map(|line| {
            let (name, value) = line.trim().split_once(" = root / ")?;
            let name = name.trim();
            if !name
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
            {
                return None;
            }
            let components = value
                .split(" / ")
                .map(|part| part.trim().trim_matches(['\'', '"']))
                .collect::<Vec<_>>();
            if components.is_empty()
                || !components.iter().all(|part| {
                    !part.is_empty()
                        && part
                            .chars()
                            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
                })
            {
                return None;
            }
            Some((name.to_owned(), components.join("/")))
        })
        .collect()
}
