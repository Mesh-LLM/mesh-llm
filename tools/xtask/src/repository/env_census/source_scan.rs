//! Line-level Rust source recognisers matching the legacy census regexes:
//! `MUTATION_RE`, `FUNCTION_RE`, `SERIAL_ATTR_RE` and the adjacent comment
//! block walk.

use crate::repository::python_text::{is_space, strip};

/// `(?:std::)?env::(?:set_var|remove_var)\s*\(` anywhere in the line.
pub(super) fn is_mutation(line: &str) -> bool {
    ["env::set_var", "env::remove_var"].iter().any(|call| {
        line.match_indices(call).any(|(index, _)| {
            line[index + call.len()..]
                .trim_start_matches(is_space)
                .starts_with('(')
        })
    })
}

/// Zero-based indices of every mutation line.
pub(super) fn mutation_lines(lines: &[&str]) -> Vec<usize> {
    lines
        .iter()
        .enumerate()
        .filter(|(_, line)| is_mutation(line))
        .map(|(index, _)| index)
        .collect()
}

fn is_word(ch: char) -> bool {
    ch.is_alphanumeric() || ch == '_'
}

/// `\bfn\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s*<[^>{}]*>)?\s*\(`: the first
/// function name declared on the line.
pub(super) fn function_name(line: &str) -> Option<&str> {
    line.match_indices("fn").find_map(|(index, _)| {
        if line[..index].chars().next_back().is_some_and(is_word) {
            return None;
        }
        let after = &line[index + 2..];
        let name_start = after.trim_start_matches(is_space);
        if name_start.len() == after.len()
            || !name_start.starts_with(|ch: char| ch.is_ascii_alphabetic() || ch == '_')
        {
            return None;
        }
        let name_len = name_start.len()
            - name_start
                .trim_start_matches(|ch: char| ch.is_ascii_alphanumeric() || ch == '_')
                .len();
        let (name, mut rest) = name_start.split_at(name_len);
        rest = rest.trim_start_matches(is_space);
        if let Some(generics) = rest.strip_prefix('<') {
            let close = generics.find(['>', '{', '}'])?;
            if !generics[close..].starts_with('>') {
                return None;
            }
            rest = generics[close + 1..].trim_start_matches(is_space);
        }
        rest.starts_with('(').then_some(name)
    })
}

/// The nearest function declared at or above `line_index`.
pub(super) fn nearest_function<'a>(
    lines: &[&'a str],
    line_index: usize,
) -> Option<(usize, &'a str)> {
    (0..=line_index)
        .rev()
        .find_map(|index| function_name(lines[index]).map(|name| (index, name)))
}

/// `^\s*#\[(?:serial|serial_test::serial)\]\s*$`.
pub(super) fn is_serial_attribute(line: &str) -> bool {
    matches!(strip(line), "#[serial]" | "#[serial_test::serial]")
}

/// The comment block directly above `line_index`, in source order.
pub(super) fn preceding_comment_block(lines: &[&str], line_index: usize) -> String {
    let mut block = Vec::new();
    let mut in_block_comment = false;
    for line in lines[..line_index].iter().rev() {
        let stripped = strip(line);
        if stripped.ends_with("*/") {
            in_block_comment = true;
        }
        if !(in_block_comment
            || stripped.starts_with("//")
            || stripped.starts_with("/*")
            || stripped.starts_with('*'))
        {
            break;
        }
        block.push(*line);
        if stripped.starts_with("/*") {
            in_block_comment = false;
        }
    }
    block.reverse();
    block.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_repository_census_recognises_mutations() {
        // Split literals keep this file out of the census it implements.
        assert!(is_mutation(
            &["unsafe { std::env::set_var", " (\"A\", \"1\") };"].concat()
        ));
        assert!(is_mutation(&["env::remove_var", "(key)"].concat()));
        assert!(!is_mutation("env::set_var"));
        assert!(!is_mutation("env::var(\"A\")"));
    }

    #[test]
    fn migration_repository_census_recognises_functions() {
        assert_eq!(function_name("pub fn with<T: Into<String>>(x: T) {"), None);
        assert_eq!(function_name("fn update<T>() {"), Some("update"));
        assert_eq!(
            function_name("    async fn drop(&mut self) {"),
            Some("drop")
        );
        assert_eq!(function_name("fn  set ("), Some("set"));
        assert_eq!(function_name("refn x()"), None);
        assert_eq!(function_name("fn(x)"), None);
        assert_eq!(function_name("fn a b() fn c()"), Some("c"));
    }

    #[test]
    fn migration_repository_census_comment_block_is_adjacent_only() {
        let lines = [
            "// far",
            "code();",
            "/* SAFETY: a",
            " * b */",
            "// c",
            "set();",
        ];
        assert_eq!(
            preceding_comment_block(&lines, 5),
            "/* SAFETY: a\n * b */\n// c"
        );
        assert_eq!(preceding_comment_block(&lines, 1), "// far");
        assert!(is_serial_attribute("  #[serial_test::serial] "));
        assert!(!is_serial_attribute("// #[serial]"));
    }
}
