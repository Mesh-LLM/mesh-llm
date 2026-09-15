//! Check that product code routes console output through the app's
//! format-aware event facility instead of writing to the terminal itself.
//!
//! This is a plain gate, not a ratchet: there is no allowlist and no way to
//! approve an individual call site. Exemptions are category rules that live in
//! `scope`, so an exception always names a surface that legitimately owns
//! terminal output rather than a line someone wanted to keep.

use std::fs;
use std::path::Path;

use crate::command::DynResult;

mod scope;

/// The retired ratchet's data file. The gate fails if it reappears so a future
/// change cannot quietly reintroduce per-location approvals.
const RETIRED_ALLOWLIST_RELATIVE_PATH: &str = "tools/xtask/data/console_print_allowlist.json";

/// Macros the gate forbids in product crates. `eprintln!` contains
/// `println!`, so matches must be boundary-checked (see `is_macro_boundary`).
pub(crate) const FORBIDDEN_CONSOLE_MACROS: [&str; 4] =
    ["println!", "eprintln!", "print!", "eprint!"];

/// Direct terminal handles the gate forbids outside the surfaces that own
/// console output. Retiring the print macros makes `writeln!(io::stdout(), ..)`
/// the obvious way to reintroduce exactly the debt the macros carried.
pub(crate) const DIRECT_TERMINAL_HANDLES: [(&str, &str); 2] =
    [("stdout", "stdout()"), ("stderr", "stderr()")];

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ConsolePrintHit {
    pub line: usize,
    pub macro_name: &'static str,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct DirectHandleHit {
    pub line: usize,
    pub handle: &'static str,
}

/// Finds every forbidden console print macro occurrence in a source file.
/// Whole-line comments are skipped; string literal mentions are intentionally
/// counted so the gate stays conservative. An invocation counts even when
/// whitespace or comments separate the macro name from `!`, including across
/// line breaks, because such spellings compile too.
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

/// Finds every direct terminal handle acquisition in a source file. Whole-line
/// comments are skipped so prose about the rule does not trip it. Capability
/// probes such as `io::stdout().is_terminal()` read nothing and write nothing,
/// so they are not handles for this purpose.
pub(crate) fn find_direct_terminal_handles(source: &str) -> Vec<DirectHandleHit> {
    let lines: Vec<&str> = source.lines().collect();
    let mut hits = Vec::new();
    for (index, raw_line) in lines.iter().enumerate() {
        if is_comment_only_line(raw_line) {
            continue;
        }
        for (function, handle) in DIRECT_TERMINAL_HANDLES {
            for (byte_offset, _matched) in raw_line.match_indices(function) {
                let end = byte_offset + function.len();
                if !is_identifier_boundary(raw_line, byte_offset, end) {
                    continue;
                }
                let direct_call = terminal_call_end(&lines, index, end);
                let alias_import = resolves_to_alias(&lines, index, end);
                if direct_call.is_some_and(|cursor| is_capability_probe_at(&lines, cursor))
                    || direct_call.is_none() && !alias_import
                {
                    continue;
                }
                hits.push(DirectHandleHit {
                    line: index + 1,
                    handle,
                });
            }
        }
    }
    hits.sort_by(|a, b| a.line.cmp(&b.line).then_with(|| a.handle.cmp(b.handle)));
    hits
}

fn is_identifier_boundary(line: &str, start: usize, end: usize) -> bool {
    let identifier = |ch: char| ch.is_alphanumeric() || ch == '_';
    !line[..start].chars().next_back().is_some_and(identifier)
        && !line[end..].chars().next().is_some_and(identifier)
}

#[derive(Clone, Copy)]
struct SourceCursor {
    line: usize,
    byte: usize,
}

/// Skip Rust whitespace and comments, including nested block comments.
fn skip_trivia(lines: &[&str], cursor: &mut SourceCursor) {
    let mut block_depth = 0u32;
    loop {
        if cursor.line >= lines.len() {
            return;
        }
        let line = lines[cursor.line];
        if cursor.byte >= line.len() {
            cursor.line += 1;
            cursor.byte = 0;
            continue;
        }
        let rest = &line[cursor.byte..];
        if block_depth > 0 {
            if rest.starts_with("/*") {
                block_depth += 1;
                cursor.byte += 2;
            } else if rest.starts_with("*/") {
                block_depth -= 1;
                cursor.byte += 2;
            } else {
                cursor.byte += rest.chars().next().expect("non-empty rest").len_utf8();
            }
            continue;
        }
        if rest.starts_with("//") {
            cursor.line += 1;
            cursor.byte = 0;
        } else if rest.starts_with("/*") {
            block_depth = 1;
            cursor.byte += 2;
        } else if rest.chars().next().expect("non-empty rest").is_whitespace() {
            cursor.byte += rest.chars().next().expect("non-empty rest").len_utf8();
        } else {
            return;
        }
    }
}

fn consume(lines: &[&str], cursor: &mut SourceCursor, token: &str) -> bool {
    skip_trivia(lines, cursor);
    let Some(rest) = lines
        .get(cursor.line)
        .and_then(|line| line.get(cursor.byte..))
    else {
        return false;
    };
    if !rest.starts_with(token) {
        return false;
    }
    cursor.byte += token.len();
    true
}

/// Return the cursor after a trivia-tolerant empty call to stdout/stderr.
fn terminal_call_end(lines: &[&str], line: usize, byte: usize) -> Option<SourceCursor> {
    let mut cursor = SourceCursor { line, byte };
    if consume(lines, &mut cursor, "(") && consume(lines, &mut cursor, ")") {
        Some(cursor)
    } else {
        None
    }
}

/// A renamed direct import can hide the canonical function name at the call
/// site, so the import itself is enough to fail the conservative gate.
fn resolves_to_alias(lines: &[&str], line: usize, byte: usize) -> bool {
    let mut cursor = SourceCursor { line, byte };
    if !consume(lines, &mut cursor, "as") {
        return false;
    }
    skip_trivia(lines, &mut cursor);
    lines
        .get(cursor.line)
        .and_then(|source| source.get(cursor.byte..))
        .and_then(|rest| rest.chars().next())
        .is_some_and(|ch| ch.is_alphabetic() || ch == '_')
}

/// True when the handle is immediately consumed by a read-only capability
/// question rather than kept for writing.
fn is_capability_probe_at(lines: &[&str], mut cursor: SourceCursor) -> bool {
    consume(lines, &mut cursor, ".")
        && consume(lines, &mut cursor, "is_terminal")
        && consume(lines, &mut cursor, "(")
        && consume(lines, &mut cursor, ")")
}

/// Collects relative paths (slash separated, deterministic order) of every
/// product `.rs` file under `crates/`, using the explicit scope rules in
/// `scope`. Build scripts use print macros for Cargo directives. Paths carry
/// the `crates/` prefix so reported violations are repo-relative.
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
            let path = format!("{prefix}{name}");
            if scope::is_product_source(&path) {
                out.push(path);
            }
        }
    }
    Ok(())
}

/// Gates CI: no console print macro may appear in product code, and no product
/// crate outside the console-owning surfaces may take a terminal handle. There
/// is no per-location approval — an exception is a category rule in `scope`.
pub(crate) fn check_no_console_prints(repo_root: &Path) -> DynResult<()> {
    check_retired_allowlist_absent(repo_root)?;

    let crates_dir = repo_root.join("crates");
    let files = collect_rs_files(&crates_dir).map_err(|error| {
        format!(
            "failed to list Rust sources under {}: {error}",
            crates_dir.display()
        )
    })?;
    let mut macro_violations: Vec<String> = Vec::new();
    let mut handle_violations: Vec<String> = Vec::new();

    for file in &files {
        let source = read_source(repo_root, file)?;
        let product_source = scope::without_test_modules(&source);
        for hit in find_console_prints(&product_source) {
            macro_violations.push(format!("{file}:{} {}", hit.line, hit.macro_name));
        }
        if scope::owns_console_output(file) {
            continue;
        }
        for hit in find_direct_terminal_handles(&product_source) {
            handle_violations.push(format!("{file}:{} {}", hit.line, hit.handle));
        }
    }

    if macro_violations.is_empty() && handle_violations.is_empty() {
        return Ok(());
    }
    let mut sections = Vec::new();
    if !macro_violations.is_empty() {
        sections.push(format!(
            "forbidden console print macros found in product code:\n{}",
            macro_violations.join("\n")
        ));
    }
    if !handle_violations.is_empty() {
        sections.push(format!(
            "direct terminal handles found outside the console output facility:\n{}",
            handle_violations.join("\n")
        ));
    }
    Err(format!("{}\n\n{}", sections.join("\n\n"), CONVERSION_GUIDANCE).into())
}

const CONVERSION_GUIDANCE: &str = "\
Convert each site to the facility that owns the stream:
  - operational or diagnostic output -> mesh_llm_events::emit_event, or tracing
    (`tracing::info!` / `warn!` / `error!`) for runtime diagnostics;
  - human-facing CLI prose, tables, and prompts -> mesh_llm_events::console_out
    / console_err, which discard while a JSON sink or the TUI owns the terminal;
  - the machine-readable payload a --json command exists to produce ->
    mesh_llm_events::machine_out.
Writing to io::stdout() / io::stderr() directly bypasses all three: it corrupts
the interactive dashboard and puts free-form text on the stream while a JSON
sink is installed. Only the console output facility itself may hold a terminal
handle; see scope::CONSOLE_OUTPUT_OWNERS.";

/// The gate replaced a ratchet whose approvals lived in a JSON file. Failing
/// when that file returns keeps a revert or a stray merge from silently
/// restoring per-location approvals nothing reads any more.
fn check_retired_allowlist_absent(repo_root: &Path) -> DynResult<()> {
    let retired = repo_root.join(RETIRED_ALLOWLIST_RELATIVE_PATH);
    if !retired.exists() {
        return Ok(());
    }
    Err(format!(
        "stale console print allowlist at {}: the ratchet was retired and this file is no longer \
read. Delete it; console prints are now gated outright, not approved per location.",
        retired.display()
    )
    .into())
}

/// Reads one scanned source file, mapping I/O failures to ratchet errors.
fn read_source(repo_root: &Path, file: &str) -> DynResult<String> {
    let path = repo_root.join(file);
    Ok(fs::read_to_string(&path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?)
}

/// Entry point for `xtask repo-consistency no-console-print`.
pub(crate) fn check_no_console_print_command(rest: &[String]) -> DynResult<()> {
    if !rest.is_empty() {
        return Err("usage: cargo run -p xtask -- repo-consistency no-console-print".into());
    }
    let repo_root = crate::repo_consistency::repo_root()?;
    scope::check_exempt_crates(&repo_root)?;
    check_no_console_prints(&repo_root)?;
    println!("repo consistency checks passed: no-console-print");
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
    fn finds_direct_terminal_handles_and_skips_capability_probes() {
        let source = "use std::io::Write;\n\
             // io::stdout() in prose is not a handle\n\
             fn f() {\n\
             \x20   let interactive = std::io::stdout().is_terminal();\n\
             \x20   let _ = writeln!(std::io::stderr(), \"hi\");\n\
             \x20   let out = io::stdout();\n\
             }\n";
        assert_eq!(
            find_direct_terminal_handles(source),
            vec![
                DirectHandleHit {
                    line: 5,
                    handle: "stderr()"
                },
                DirectHandleHit {
                    line: 6,
                    handle: "stdout()"
                },
            ]
        );
    }

    #[test]
    fn finds_imported_aliased_and_multiline_terminal_handles() {
        let source = r#"use std::io::stdout;
use std::io::{stderr as terminal_error};
fn f() {
    stdout /* split */
      (
      );
    terminal_error();
    std::io::
      stderr
      /* split */ ()
      .is_terminal();
}"#;
        assert_eq!(
            find_direct_terminal_handles(source),
            vec![
                DirectHandleHit {
                    line: 2,
                    handle: "stderr()"
                },
                DirectHandleHit {
                    line: 4,
                    handle: "stdout()"
                },
            ]
        );
    }

    #[test]
    fn no_console_print_command_rejects_trailing_arguments() {
        let error = check_no_console_print_command(&["--regen".to_owned()])
            .unwrap_err()
            .to_string();
        assert_eq!(
            error,
            "usage: cargo run -p xtask -- repo-consistency no-console-print"
        );
    }

    #[test]
    fn gate_fails_for_every_console_print_with_no_way_to_approve_one() {
        let repo_root = temp_repo_with_files(&[(
            "crates/demo/src/lib.rs",
            "fn main() {\n    println!(\"one\");\n    eprintln!(\"two\");\n}\n",
        )]);
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
        assert!(error.contains("mesh_llm_events::console_out"), "{error}");
    }

    #[test]
    fn gate_passes_for_product_code_with_no_console_output() {
        let repo_root =
            temp_repo_with_files(&[("crates/demo/src/lib.rs", "fn main() {\n    let _ = 1;\n}\n")]);
        check_no_console_prints(&repo_root).expect("clean product code must pass");
    }

    #[test]
    fn gate_fails_when_the_retired_allowlist_reappears() {
        let repo_root = temp_repo_with_files(&[
            ("crates/demo/src/lib.rs", "fn main() {}\n"),
            (RETIRED_ALLOWLIST_RELATIVE_PATH, "{}\n"),
        ]);
        let error = check_no_console_prints(&repo_root).unwrap_err().to_string();
        assert!(error.contains("stale console print allowlist"), "{error}");
    }

    #[test]
    fn gate_fails_for_direct_terminal_handles_outside_the_output_facility() {
        let repo_root = temp_repo_with_files(&[(
            "crates/demo/src/lib.rs",
            "fn f() {\n    let _ = writeln!(std::io::stderr(), \"bypass\");\n}\n",
        )]);
        let error = check_no_console_prints(&repo_root).unwrap_err().to_string();
        assert!(
            error.contains("direct terminal handles found outside the console output facility"),
            "{error}"
        );
        assert!(
            error.contains("crates/demo/src/lib.rs:2 stderr()"),
            "{error}"
        );
    }

    #[test]
    fn console_output_owners_may_hold_terminal_handles() {
        let owner = scope::CONSOLE_OUTPUT_OWNERS
            .iter()
            .find(|path| path.starts_with("crates/mesh-llm-events/"))
            .expect("an events-crate owner");
        let repo_root =
            temp_repo_with_files(&[(owner, "fn f() {\n    let mut out = std::io::stderr();\n}\n")]);
        check_no_console_prints(&repo_root).expect("the output facility owns terminal access");
    }

    #[test]
    fn gate_respects_product_scope() {
        let repo_root = temp_repo_with_files(&[
            (
                "crates/demo/src/lib.rs",
                "#[cfg(test)] mod t { fn f() { print!(\"test\"); } }\nfn f() { println!(\"product\"); }\n",
            ),
            (
                "crates/demo/tests/integration.rs",
                "fn f() { println!(\"test\"); }",
            ),
            (
                "crates/demo/src/bin/tool.rs",
                "fn main() { println!(\"tool\"); }",
            ),
            (
                "crates/skippy-bench/src/lib.rs",
                "fn f() { println!(\"bench\"); }",
            ),
        ]);
        let error = check_no_console_prints(&repo_root).unwrap_err().to_string();
        assert!(
            error.contains("crates/demo/src/lib.rs:2 println!"),
            "{error}"
        );
        for out_of_scope in [
            "crates/demo/src/lib.rs:1",
            "crates/demo/tests/integration.rs",
            "crates/demo/src/bin/tool.rs",
            "crates/skippy-bench/src/lib.rs",
        ] {
            assert!(!error.contains(out_of_scope), "{out_of_scope}: {error}");
        }
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
