//! The import lists `verify-host-dependencies.py` extracts from inspection
//! tool output, matched as its `re` patterns match: `readelf -d` `(NEEDED)`
//! rows, indented `otool -L` rows, and PE `DLL Name:`/`Name:` rows. Each
//! result is `sorted(set(...))`.

use crate::repository::text::{is_space, splitlines, strip};
use std::collections::BTreeSet;

/// `re.IGNORECASE` equality of an input character with a lowercase ASCII
/// pattern character, including sre's non-ASCII folds onto `i`, `s`, `k`.
pub(super) fn fold_eq(pattern: char, ch: char) -> bool {
    ch.to_ascii_lowercase() == pattern
        || matches!(
            (pattern, ch),
            ('i', '\u{130}' | '\u{131}') | ('s', '\u{17f}') | ('k', '\u{212a}')
        )
}

/// Whether `chars[at..]` starts with `pattern` under `fold_eq`.
pub(super) fn starts_with_folded(chars: &[char], at: usize, pattern: &str) -> bool {
    let mut index = at;
    for expected in pattern.chars() {
        match chars.get(index) {
            Some(ch) if fold_eq(expected, *ch) => index += 1,
            _ => return false,
        }
    }
    true
}

/// `sorted(set(re.findall(r"\(NEEDED\).*\[([^\]]+)\]", output)))`.
pub(super) fn elf_imports(output: &str) -> Vec<String> {
    let bytes = output.as_bytes();
    let mut found = BTreeSet::new();
    let mut search = 0;
    while let Some(offset) = output[search..].find("(NEEDED)") {
        let start = search + offset;
        match needed_value(bytes, start + "(NEEDED)".len()) {
            Some((value_start, close)) => {
                found.insert(output[value_start..close].to_owned());
                search = close + 1;
            }
            None => search = start + 1,
        }
    }
    found.into_iter().collect()
}

/// The greedy `.*\[([^\]]+)\]` after `(NEEDED)`: the rightmost `[` on the
/// same line whose next `]` (on any line) closes a nonempty value.
fn needed_value(bytes: &[u8], after: usize) -> Option<(usize, usize)> {
    let line_end = bytes[after..]
        .iter()
        .position(|byte| *byte == b'\n')
        .map_or(bytes.len(), |offset| after + offset);
    (after..line_end).rev().find_map(|open| {
        if bytes[open] != b'[' {
            return None;
        }
        let close = bytes[open + 1..]
            .iter()
            .position(|byte| *byte == b']')
            .map(|offset| open + 1 + offset)?;
        (close > open + 1).then_some((open + 1, close))
    })
}

/// The indented rows of `otool -L`, without their compatibility suffix.
pub(super) fn macho_imports(output: &str) -> Vec<String> {
    let found: BTreeSet<String> = splitlines(output)
        .into_iter()
        .filter(|line| line.chars().next().is_some_and(is_space))
        .map(|line| {
            let stripped = strip(line);
            stripped
                .split_once(" (compatibility version")
                .map_or(stripped, |(value, _)| value)
                .to_owned()
        })
        .filter(|value| !value.is_empty())
        .collect();
    found.into_iter().collect()
}

/// `re.search(r"(?:DLL Name:|Name:)\s*(\S+\.dll)\b", line, re.IGNORECASE)`
/// over each line.
pub(super) fn pe_imports(output: &str) -> Vec<String> {
    let found: BTreeSet<String> = splitlines(output)
        .into_iter()
        .filter_map(pe_import)
        .collect();
    found.into_iter().collect()
}

fn pe_import(line: &str) -> Option<String> {
    let chars: Vec<char> = line.chars().collect();
    (0..chars.len()).find_map(|start| {
        let after = if starts_with_folded(&chars, start, "dll name:") {
            start + "dll name:".len()
        } else if starts_with_folded(&chars, start, "name:") {
            start + "name:".len()
        } else {
            return None;
        };
        dll_after(&chars, after)
    })
}

/// `\s*(\S+\.dll)\b` at `index`: the longest non-space run's rightmost
/// `.dll` that ends at a word boundary and follows at least one character.
fn dll_after(chars: &[char], index: usize) -> Option<String> {
    let begin = (index..chars.len())
        .find(|at| !is_space(chars[*at]))
        .unwrap_or(chars.len());
    let run_end = (begin..chars.len())
        .find(|at| is_space(chars[*at]))
        .unwrap_or(chars.len());
    (begin + 5..=run_end).rev().find_map(|end| {
        let boundary = chars.get(end).is_none_or(|ch| !is_word(*ch));
        (boundary && starts_with_folded(chars, end - 4, ".dll"))
            .then(|| chars[begin..end].iter().collect())
    })
}

/// sre's Unicode `\w`.
fn is_word(ch: char) -> bool {
    ch.is_alphanumeric() || ch == '_'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_native_policy_elf_needed_rows_match_python_findall() {
        let output = " 0x1 (NEEDED) Shared library: [libc.so.6]\n \
                      0x1 (NEEDED) [a] x [libm.so.6]\n (NEEDED) []\n [b]\n (NEEDED) [lib\nz]";
        assert_eq!(elf_imports(output), ["lib\nz", "libc.so.6", "libm.so.6"]);
    }

    #[test]
    fn migration_native_policy_macho_rows_skip_the_header() {
        let output =
            "bin:\n\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0)\n\t@rpath/x\n \n";
        assert_eq!(
            macho_imports(output),
            ["/usr/lib/libSystem.B.dylib", "@rpath/x"]
        );
    }

    #[test]
    fn migration_native_policy_pe_rows_match_python_search() {
        let output =
            "  DLL Name: KERNEL32.dll\n  name:  a.dll.dll-x\n Name: .dll\n x Name: ws2_32.DLL,y";
        assert_eq!(
            pe_imports(output),
            ["KERNEL32.dll", "a.dll.dll", "ws2_32.DLL"]
        );
    }
}
