//! Ratchet check that product code routes console output through the app's
//! format-aware event facility instead of raw print macros.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use crate::command::{DynResult, write_json_file};

const ALLOWLIST_RELATIVE_PATH: &str = "tools/xtask/data/console_print_allowlist.json";
const REGEN_FLAG: &str = "--regen";
const REGEN_COMMAND: &str = "cargo run -p xtask -- repo-consistency no-console-print --regen";

/// Macros the ratchet forbids in product crates. `eprintln!` contains
/// `println!`, so matches must be boundary-checked (see `is_macro_boundary`).
pub(crate) const FORBIDDEN_CONSOLE_MACROS: [&str; 4] =
    ["println!", "eprintln!", "print!", "eprint!"];

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ConsolePrintHit {
    pub line: usize,
    pub macro_name: &'static str,
}

/// One ratchet-approved console print occurrence. The ratchet approves exact
/// occurrences rather than per-file counts so that retiring one legacy print
/// can never free up allowance for a new one elsewhere in the file.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct AllowedOccurrence {
    line: usize,
    macro_name: String,
}

/// Finds every forbidden console print macro occurrence in a source file.
/// Whole-line comments are skipped; string literal mentions are intentionally
/// counted so the regenerated baseline stays stable and conservative. An
/// invocation counts even when whitespace or comments separate the macro name
/// from `!`, including across line breaks, because such spellings compile too.
pub(crate) fn find_console_prints(source: &str) -> Vec<ConsolePrintHit> {
    let lines: Vec<&str> = source.lines().collect();
    let mut hits = Vec::new();
    for (index, raw_line) in lines.iter().enumerate() {
        if is_comment_only_line(raw_line) {
            continue;
        }
        for macro_name in FORBIDDEN_CONSOLE_MACROS {
            let bare_name = &macro_name[..macro_name.len() - 1];
            for (byte_offset, _matched_name) in raw_line.match_indices(bare_name) {
                let end = byte_offset + bare_name.len();
                if !is_macro_boundary(raw_line, byte_offset)
                    || !resolves_to_invocation(&lines, index, end)
                {
                    continue;
                }
                hits.push(ConsolePrintHit {
                    line: index + 1,
                    macro_name,
                });
            }
        }
    }
    hits.sort_by(|a, b| {
        a.line
            .cmp(&b.line)
            .then_with(|| a.macro_name.cmp(b.macro_name))
    });
    hits
}

/// Whole-line comments carry no code, so every line whose first token is `//`
/// (including doc and inner-doc lines) is skipped wholesale.
fn is_comment_only_line(raw_line: &str) -> bool {
    raw_line.trim_start().starts_with("//")
}

/// A macro match is real only when nothing identifier-like precedes it. This
/// rejects the `println` inside `eprintln!` and identifiers such as
/// `my_println`. The byte offset comes from `match_indices`, so the slice
/// always starts on a character boundary.
fn is_macro_boundary(raw_line: &str, byte_offset: usize) -> bool {
    match raw_line[..byte_offset].chars().next_back() {
        None => true,
        Some(previous) => !previous.is_alphanumeric() && previous != '_',
    }
}

/// Decides whether a boundary-checked macro name ending at byte offset `end`
/// in `lines[start_line]` is an invocation whose `!` may be separated from the
/// name by whitespace or comments. Trivia follows the Rust parser's treatment:
/// newlines and line comments are skipped, block comments (which nest) may span
/// lines. Any other token first — a digit or identifier character extending
/// the name (as in `println2`), the `=` of `!=`, anything else — means there
/// is no invocation. Hits belong to the line carrying the macro name.
fn resolves_to_invocation(lines: &[&str], start_line: usize, end: usize) -> bool {
    let mut line_index = start_line;
    let mut rest = &lines[start_line][end..];
    let mut comment_depth = 0u32;

    loop {
        if comment_depth > 0 {
            if let Some(closed_at) = scan_block_comment_payload(rest, comment_depth) {
                rest = &rest[closed_at..];
                comment_depth = 0;
                continue;
            }
            // The block comment continues on the next line.
        } else {
            let leading_ws = rest.len() - rest.trim_start().len();
            if leading_ws > 0 {
                rest = &rest[leading_ws..];
                continue;
            }
            match rest.as_bytes().first().copied() {
                None => {}                                 // Blank line: trivia.
                Some(b'/') if rest.starts_with("//") => {} // Line comment runs to end of line.
                Some(b'/') if rest.starts_with("/*") => {
                    comment_depth = 1;
                    rest = &rest[2..];
                    continue;
                }
                Some(b'!') => return !rest.starts_with("!="), // Bare `!`, not `!=`.
                Some(_) => return false,
            }
        }
        if !advance_to_next_line(lines, &mut line_index) {
            return false; // EOF with no bare `!` in sight.
        }
        rest = lines[line_index];
    }
}

/// Scans block-comment payload starting at nesting level `depth`. Returns the
/// byte offset just past the closing `*/` that closes the outermost level, or
/// None when more input is needed. Rust block comments nest; `/*` and `*/` are
/// ASCII, so stepping character by character stays safe on any UTF-8 payload.
fn scan_block_comment_payload(rest: &str, mut depth: u32) -> Option<usize> {
    let mut previous: Option<char> = None;
    for (offset, ch) in rest.char_indices() {
        match (previous, ch) {
            (Some('/'), '*') => depth += 1,
            (Some('*'), '/') => depth -= 1,
            _ => {}
        }
        previous = Some(ch);
        if depth == 0 {
            return Some(offset + ch.len_utf8());
        }
    }
    None
}

/// Moves `line_index` to the next physical line; false when there is none.
fn advance_to_next_line(lines: &[&str], line_index: &mut usize) -> bool {
    if *line_index + 1 >= lines.len() {
        return false;
    }
    *line_index += 1;
    true
}

/// Collects relative paths (slash separated, deterministic order) of every
/// `.rs` file under `crates/`, excluding `build.rs` where print macros are a
/// cargo directive mechanism rather than product console output. Paths carry
/// the `crates/` prefix so they stay stable as repo-relative allowlist keys.
fn collect_rs_files(crates_dir: &Path) -> std::io::Result<Vec<String>> {
    let mut files = Vec::new();
    collect_rs_files_recursive(crates_dir, "crates/", &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_rs_files_recursive(
    dir: &Path,
    prefix: &str,
    out: &mut Vec<String>,
) -> std::io::Result<()> {
    let mut entries = fs::read_dir(dir)?.collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let file_name = entry.file_name();
        let name = file_name.to_string_lossy().into_owned();
        if entry.file_type()?.is_dir() {
            let child_prefix = format!("{prefix}{name}/");
            collect_rs_files_recursive(&entry.path(), &child_prefix, out)?;
        } else if name == "build.rs" || !name.ends_with(".rs") {
            continue;
        } else {
            out.push(format!("{prefix}{name}"));
        }
    }
    Ok(())
}

/// Gates CI: every console print occurrence in a scanned file must have an
/// exact ratchet approval (same file, line, and macro). Prints at unapproved
/// locations fail, as do approvals whose occurrence moved or disappeared — so
/// retiring one legacy print can never hide a new one. New files with prints
/// fail outright, and allowlist entries for deleted files are reported as
/// stale debt to drop via `--regen`.
pub(crate) fn check_no_console_prints(repo_root: &Path) -> DynResult<()> {
    let allowlist_path = repo_root.join(ALLOWLIST_RELATIVE_PATH);
    let raw_allowlist = fs::read_to_string(&allowlist_path).map_err(|error| {
        format!(
            "missing console print ratchet at {}: run `{REGEN_COMMAND}` to generate it ({error})",
            allowlist_path.display()
        )
    })?;
    let allowed: BTreeMap<String, Vec<AllowedOccurrence>> = serde_json::from_str(&raw_allowlist)
        .map_err(|error| {
            format!(
                "invalid console print ratchet at {}: {error}",
                allowlist_path.display()
            )
        })?;

    let crates_dir = repo_root.join("crates");
    let files = collect_rs_files(&crates_dir).map_err(|error| {
        format!(
            "failed to list Rust sources under {}: {error}",
            crates_dir.display()
        )
    })?;
    let mut seen = BTreeSet::new();
    let mut new_violations: Vec<String> = Vec::new();
    let mut drift_violations: Vec<String> = Vec::new();

    for file in &files {
        seen.insert(file.as_str());
        let source = read_source(repo_root, file)?;
        let hits = find_console_prints(&source);
        if hits.is_empty() && !allowed.contains_key(file.as_str()) {
            continue;
        }
        match allowed.get(file.as_str()) {
            None => {
                for hit in &hits {
                    new_violations.push(format!("{file}:{} {}", hit.line, hit.macro_name));
                }
            }
            Some(approved) => claim_approved_occurrences(
                file,
                &hits,
                approved,
                &mut new_violations,
                &mut drift_violations,
            ),
        }
    }

    for stale_path in allowed.keys().filter(|path| !seen.contains(path.as_str())) {
        drift_violations.push(format!(
            "{stale_path}: stale allowlist entry (no console prints remain); remove it with `--regen`"
        ));
    }

    if new_violations.is_empty() && drift_violations.is_empty() {
        return Ok(());
    }
    let mut sections = Vec::new();
    if !new_violations.is_empty() {
        sections.push(format!(
            "forbidden console print macros found in product code:\n{}",
            new_violations.join("\n")
        ));
    }
    if !drift_violations.is_empty() {
        sections.push(format!(
            "console print ratchet is out of sync with the tree:\n{}",
            drift_violations.join("\n")
        ));
    }
    Err(format!(
        "{}\n\nRoute output through mesh_llm_events::emit_event instead; retire legacy debt line by \
line and regenerate the ratchet with `{REGEN_COMMAND}`.",
        sections.join("\n\n")
    )
    .into())
}

/// Reads one scanned source file, mapping I/O failures to ratchet errors.
fn read_source(repo_root: &Path, file: &str) -> DynResult<String> {
    let path = repo_root.join(file);
    Ok(fs::read_to_string(&path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?)
}

/// Multiset-matches the observed hits against the file's approved occurrences.
/// Each hit must claim a distinct approval with the same line and macro; an
/// unclaimed hit is a new print, an unspent approval is drift.
fn claim_approved_occurrences(
    file: &str,
    hits: &[ConsolePrintHit],
    approved: &[AllowedOccurrence],
    new_violations: &mut Vec<String>,
    drift_violations: &mut Vec<String>,
) {
    let mut approved_used = vec![false; approved.len()];
    for hit in hits {
        let claimed = (0..approved.len()).find(|index| {
            !approved_used[*index]
                && approved[*index].line == hit.line
                && approved[*index].macro_name == hit.macro_name
        });
        match claimed {
            Some(index) => approved_used[index] = true,
            None => new_violations.push(format!("{file}:{} {}", hit.line, hit.macro_name)),
        }
    }
    for (index, occurrence) in approved.iter().enumerate() {
        if !approved_used[index] {
            drift_violations.push(format!(
                "{file}:{} {}: approved occurrence is missing or was replaced",
                occurrence.line, occurrence.macro_name
            ));
        }
    }
}

/// Entry point for `xtask repo-consistency no-console-print [--regen]`. The
/// plain invocation gates CI; `--regen` rewrites the ratchet from the current
/// tree and always succeeds so the reduced baseline can be committed.
pub(crate) fn check_no_console_print_command(rest: &[String]) -> DynResult<()> {
    let repo_root = crate::repo_consistency::repo_root()?;
    if rest.iter().any(|arg| arg == REGEN_FLAG) {
        regenerate_allowlist(&repo_root)?;
    } else {
        check_no_console_prints(&repo_root)?;
    }
    println!("repo consistency checks passed: no-console-print");
    Ok(())
}

fn regenerate_allowlist(repo_root: &Path) -> DynResult<()> {
    let crates_dir = repo_root.join("crates");
    let files = collect_rs_files(&crates_dir).map_err(|error| {
        format!(
            "failed to list Rust sources under {}: {error}",
            crates_dir.display()
        )
    })?;
    let mut allowed: BTreeMap<String, Vec<AllowedOccurrence>> = BTreeMap::new();
    for file in &files {
        let source = fs::read_to_string(repo_root.join(file))
            .map_err(|error| format!("failed to read {file}: {error}"))?;
        let hits = find_console_prints(&source);
        if !hits.is_empty() {
            allowed.insert(
                file.clone(),
                hits.iter()
                    .map(|hit| AllowedOccurrence {
                        line: hit.line,
                        macro_name: hit.macro_name.to_string(),
                    })
                    .collect(),
            );
        }
    }
    let allowlist_path = repo_root.join(ALLOWLIST_RELATIVE_PATH);
    write_json_file(&allowlist_path, &allowed)?;
    let total: usize = allowed.values().map(Vec::len).sum();
    println!(
        "console print ratchet regenerated at {}: {} file(s), {} legacy hit(s)",
        allowlist_path.display(),
        allowed.len(),
        total
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_each_macro_with_line_numbers_and_sorts_hits() {
        let source = r#"fn main() {
    eprintln!("boom");
    println!(ok);
    do_print_stuff();
    print!();
}"#;
        assert_eq!(
            find_console_prints(source),
            vec![
                ConsolePrintHit {
                    line: 2,
                    macro_name: "eprintln!"
                },
                ConsolePrintHit {
                    line: 3,
                    macro_name: "println!"
                },
                ConsolePrintHit {
                    line: 5,
                    macro_name: "print!"
                },
            ]
        );
    }

    #[test]
    fn ignores_comment_lines_and_prefixed_identifiers() {
        let source = r#"// println!("documented, not executed")
/// eprintln! doc comment too
fn f() { my_println!(x); custom_print(y); }"#;
        assert_eq!(find_console_prints(source), Vec::<ConsolePrintHit>::new());
    }

    #[test]
    fn counts_string_literal_mentions_to_keep_baseline_stable() {
        let source = r#"const HINT: &str = "avoid println! here";
fn f() { eprintln!("{HINT}"); }"#;
        assert_eq!(
            find_console_prints(source),
            vec![
                ConsolePrintHit {
                    line: 1,
                    macro_name: "println!"
                },
                ConsolePrintHit {
                    line: 2,
                    macro_name: "eprintln!"
                },
            ]
        );
    }

    #[test]
    fn finds_macros_when_trivia_separates_name_from_bang() {
        let source = r#"fn f(a, b, c, d) {
    println !(a);
    eprintln /* why */ !(b);
    print   !(c);
    eprint // note stays on the name's line
    !(d);
}"#;
        assert_eq!(
            find_console_prints(source),
            vec![
                ConsolePrintHit {
                    line: 2,
                    macro_name: "println!"
                },
                ConsolePrintHit {
                    line: 3,
                    macro_name: "eprintln!"
                },
                ConsolePrintHit {
                    line: 4,
                    macro_name: "print!"
                },
                ConsolePrintHit {
                    line: 5,
                    macro_name: "eprint!"
                },
            ]
        );

        let block_comment_across_lines = r#"fn g(x) {
    println /* spans
lines */ !(x);
}"#;
        assert_eq!(
            find_console_prints(block_comment_across_lines),
            vec![ConsolePrintHit {
                line: 2,
                macro_name: "println!"
            }]
        );
    }

    #[test]
    fn ignores_non_invocation_spellings_of_forbidden_names() {
        let source = r#"fn f(x) -> bool {
    let extended = println2 !(x);
    let ok = print != other;
    x"#;
        assert_eq!(find_console_prints(source), Vec::<ConsolePrintHit>::new());
    }

    #[test]
    fn ratchet_fails_for_unapproved_print_locations() {
        let repo_root = temp_repo_with_files(&[(
            "crates/demo/src/lib.rs",
            "fn main() {\n    println!(\"one\");\n    eprintln!(\"two\");\n}\n",
        )]);
        write_allowlist(
            &repo_root,
            r#"{"crates/demo/src/lib.rs": [{"line": 2, "macro_name": "println!"}]}"#,
        );
        let error = check_no_console_prints(&repo_root).unwrap_err().to_string();
        assert!(
            error.contains("forbidden console print macros found"),
            "{error}"
        );
        // The approved occurrence is not reported; only the new one is.
        assert!(!error.contains(":2 println!"), "{error}");
        assert!(
            error.contains("crates/demo/src/lib.rs:3 eprintln!"),
            "{error}"
        );
    }

    #[test]
    fn retiring_an_approved_print_cannot_hide_a_new_one() {
        let repo_root = temp_repo_with_files(&[(
            "crates/demo/src/lib.rs",
            "fn main() {\n    eprintln!(\"swapped\");\n}\n",
        )]);
        write_allowlist(
            &repo_root,
            r#"{"crates/demo/src/lib.rs": [{"line": 2, "macro_name": "println!"}]}"#,
        );
        let error = check_no_console_prints(&repo_root).unwrap_err().to_string();
        // Old count-based ratchets pass this (1 print <= allowance of 1); the
        // occurrence ratchet must flag both sides of the swap.
        assert!(
            error.contains("crates/demo/src/lib.rs:2 eprintln!"),
            "{error}"
        );
        assert!(
            error.contains(
                "crates/demo/src/lib.rs:2 println!: approved occurrence is missing or was replaced"
            ),
            "{error}"
        );
    }

    #[test]
    fn ratchet_fails_when_an_approved_occurrence_is_removed() {
        let repo_root = temp_repo_with_files(&[(
            "crates/demo/src/lib.rs",
            "fn main() {\n    println!(\"one\");\n}\n",
        )]);
        write_allowlist(
            &repo_root,
            r#"{"crates/demo/src/lib.rs": [{"line": 2, "macro_name": "println!"}, {"line": 3, "macro_name": "eprintln!"}]}"#,
        );
        let error = check_no_console_prints(&repo_root).unwrap_err().to_string();
        assert!(
            error.contains("console print ratchet is out of sync with the tree"),
            "{error}"
        );
        assert!(
            !error.contains("forbidden console print macros found"),
            "{error}"
        );
        assert!(
            error.contains(
                "crates/demo/src/lib.rs:3 eprintln!: approved occurrence is missing or was replaced"
            ),
            "{error}"
        );
    }

    #[test]
    fn ratchet_passes_when_occurrences_match_exactly() {
        let repo_root = temp_repo_with_files(&[(
            "crates/demo/src/lib.rs",
            "fn main() {\n    println!(\"one\");\n}\n",
        )]);
        write_allowlist(
            &repo_root,
            r#"{"crates/demo/src/lib.rs": [{"line": 2, "macro_name": "println!"}]}"#,
        );
        check_no_console_prints(&repo_root).expect("matching occurrences must pass");

        let duplicates = temp_repo_with_files(&[(
            "crates/twin/src/lib.rs",
            "fn main() {\n    println!(\"a\"); println!(\"b\");\n}\n",
        )]);
        write_allowlist(
            &duplicates,
            r#"{"crates/twin/src/lib.rs": [{"line": 2, "macro_name": "println!"}, {"line": 2, "macro_name": "println!"}]}"#,
        );
        check_no_console_prints(&duplicates)
            .expect("duplicate approvals for same-line prints must pass");
    }

    #[test]
    fn ratchet_fails_for_new_files_and_stale_entries() {
        let repo_root = temp_repo_with_files(&[
            (
                "crates/legacy/src/lib.rs",
                "fn f() { println!(\"old\"); }\n",
            ),
            (
                "crates/fresh/src/lib.rs",
                "fn g() { eprintln!(\"new\"); }\n",
            ),
        ]);
        write_allowlist(
            &repo_root,
            r#"{"crates/legacy/src/lib.rs": [{"line": 1, "macro_name": "println!"}], "crates/gone/src/lib.rs": [{"line": 5, "macro_name": "print!"}]}"#,
        );
        let error = check_no_console_prints(&repo_root).unwrap_err().to_string();
        // The approved legacy occurrence passes; only the fresh file and the
        // entry for a deleted file are reported.
        assert!(!error.contains("crates/legacy/src/lib.rs:1"), "{error}");
        assert!(
            error.contains("crates/fresh/src/lib.rs:1 eprintln!"),
            "{error}"
        );
        assert!(error.contains("stale allowlist entry"), "{error}");
    }

    fn temp_repo_with_files(files: &[(&str, &str)]) -> std::path::PathBuf {
        let dir = crate::command::unique_temp_dir("no-console-print-test");
        for (relative_path, contents) in files {
            let path = dir.join(relative_path);
            fs::create_dir_all(path.parent().expect("parent dir")).unwrap();
            fs::write(&path, contents).unwrap();
        }
        dir
    }

    fn write_allowlist(repo_root: &Path, raw_json: &str) {
        fs::create_dir_all(repo_root.join("tools/xtask/data")).unwrap();
        fs::write(repo_root.join(ALLOWLIST_RELATIVE_PATH), raw_json).unwrap();
    }
}
