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

/// Finds every forbidden console print macro occurrence in a source file.
/// Comment lines are skipped; string literal mentions are intentionally
/// counted so the regenerated baseline stays stable and conservative.
pub(crate) fn find_console_prints(source: &str) -> Vec<ConsolePrintHit> {
    let mut hits = Vec::new();
    for (index, raw_line) in source.lines().enumerate() {
        if raw_line.trim_start().starts_with("//") {
            continue;
        }
        for macro_name in FORBIDDEN_CONSOLE_MACROS {
            for (byte_offset, _match_text) in raw_line.match_indices(macro_name) {
                if is_macro_boundary(raw_line, byte_offset) {
                    hits.push(ConsolePrintHit {
                        line: index + 1,
                        macro_name,
                    });
                }
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

/// A macro match is real only when nothing identifier-like precedes it. This
/// rejects `println!` inside `eprintln!` and identifiers such as
/// `my_println!`. The byte offset comes from `match_indices`, so the slice
/// always starts on a character boundary.
fn is_macro_boundary(raw_line: &str, byte_offset: usize) -> bool {
    match raw_line[..byte_offset].chars().next_back() {
        None => true,
        Some(previous) => !previous.is_alphanumeric() && previous != '_',
    }
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

/// Gates CI: every candidate file must stay within its ratchet allowance, new
/// files with print macros fail, and allowlist entries that no longer match a
/// real file are reported as stale debt to drop via `--regen`.
pub(crate) fn check_no_console_prints(repo_root: &Path) -> DynResult<()> {
    let allowlist_path = repo_root.join(ALLOWLIST_RELATIVE_PATH);
    let raw_allowlist = fs::read_to_string(&allowlist_path).map_err(|error| {
        format!(
            "missing console print ratchet at {}: run `{REGEN_COMMAND}` to generate it ({error})",
            allowlist_path.display()
        )
    })?;
    let allowed: BTreeMap<String, u32> = serde_json::from_str(&raw_allowlist).map_err(|error| {
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
    let mut violations = Vec::new();

    for file in &files {
        seen.insert(file.as_str());
        let path = repo_root.join(file);
        let source = fs::read_to_string(&path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        let hits = find_console_prints(&source);
        let allowance = allowed.get(file.as_str()).copied().unwrap_or(0);
        if (hits.len() as u32) > allowance {
            for hit in &hits {
                violations.push(format!("{file}:{} {}", hit.line, hit.macro_name));
            }
        }
    }

    for stale_path in allowed.keys().filter(|path| !seen.contains(path.as_str())) {
        violations.push(format!(
            "{stale_path}: stale allowlist entry (no console prints remain); remove it with `--regen`"
        ));
    }

    if violations.is_empty() {
        return Ok(());
    }
    Err(format!(
        "forbidden console print macros found in product code:\n{}\n\nRoute output through \
mesh_llm_events::emit_event instead; retire legacy debt line by line and regenerate the ratchet \
with `{REGEN_COMMAND}`.",
        violations.join("\n")
    )
    .into())
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
    let mut counts = BTreeMap::new();
    for file in &files {
        let source = fs::read_to_string(repo_root.join(file))
            .map_err(|error| format!("failed to read {file}: {error}"))?;
        let hits = find_console_prints(&source);
        if !hits.is_empty() {
            counts.insert(file.clone(), hits.len() as u32);
        }
    }
    let allowlist_path = repo_root.join(ALLOWLIST_RELATIVE_PATH);
    write_json_file(&allowlist_path, &counts)?;
    let total: u32 = counts.values().sum();
    println!(
        "console print ratchet regenerated at {}: {} file(s), {} legacy hit(s)",
        allowlist_path.display(),
        counts.len(),
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
    fn ratchet_fails_when_a_file_exceeds_its_allowed_count() {
        let repo_root = temp_repo_with_files(&[(
            "crates/demo/src/lib.rs",
            "fn main() {\n    println!(\"one\");\n    eprintln!(\"two\");\n}\n",
        )]);
        fs::create_dir_all(repo_root.join("tools/xtask/data")).unwrap();
        fs::write(
            repo_root.join("tools/xtask/data/console_print_allowlist.json"),
            r#"{"crates/demo/src/lib.rs": 1}"#,
        )
        .unwrap();
        let error = check_no_console_prints(&repo_root).unwrap_err().to_string();
        assert!(
            error.contains("forbidden console print macros found"),
            "{error}"
        );
        assert!(
            error.contains("crates/demo/src/lib.rs:2 println!"),
            "{error}"
        );
        assert!(
            error.contains("crates/demo/src/lib.rs:3 eprintln!"),
            "{error}"
        );
    }

    #[test]
    fn ratchet_passes_within_allowed_counts() {
        let repo_root = temp_repo_with_files(&[(
            "crates/demo/src/lib.rs",
            "fn main() {\n    println!(\"one\");\n}\n",
        )]);
        fs::create_dir_all(repo_root.join("tools/xtask/data")).unwrap();
        fs::write(
            repo_root.join("tools/xtask/data/console_print_allowlist.json"),
            r#"{"crates/demo/src/lib.rs": 1}"#,
        )
        .unwrap();
        check_no_console_prints(&repo_root).expect("within-allowlist tree must pass");
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
        fs::create_dir_all(repo_root.join("tools/xtask/data")).unwrap();
        fs::write(
            repo_root.join("tools/xtask/data/console_print_allowlist.json"),
            r#"{"crates/legacy/src/lib.rs": 1, "crates/gone/src/lib.rs": 4}"#,
        )
        .unwrap();
        let error = check_no_console_prints(&repo_root).unwrap_err().to_string();
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
}
