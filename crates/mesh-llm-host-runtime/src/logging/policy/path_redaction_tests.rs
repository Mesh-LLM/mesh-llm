//! Path API and boundary-aware free-form logging redaction contracts.

use super::{sanitize_path, sanitize_paths_in_text, sanitize_paths_with_home};

#[test]
fn path_sanitization_hides_home_dir() {
    let home = dirs::home_dir().expect("test runner has a home directory");
    let test_path = home.join("some/deep/path/file.log");
    let sanitized = sanitize_path(&test_path);
    let expected_prefix = format!("~{}", std::path::MAIN_SEPARATOR);
    assert!(sanitized.starts_with(&expected_prefix));
    assert!(!sanitized.contains(&home.display().to_string()));
}

#[test]
fn sibling_path_and_text_are_unchanged() {
    let home = dirs::home_dir().expect("test runner has a home directory");
    let sibling = std::path::PathBuf::from(format!("{}-backup", home.display())).join("log.txt");

    let sanitized = sanitize_path(&sibling);
    assert_eq!(sanitized, sibling.display().to_string());

    let text = format!("Error in {}", sibling.display());
    assert_eq!(sanitize_paths_in_text(&text), text);
}

#[test]
fn the_home_directory_itself_collapses_to_a_tilde() {
    let home = dirs::home_dir().expect("test runner has a home directory");
    assert_eq!(sanitize_path(&home), "~");
}

#[test]
fn sanitize_paths_in_text_replaces_home() {
    let home = dirs::home_dir().expect("test runner has a home directory");
    let text = format!("Error in {}", home.join("mesh-llm/logs/app.log").display());
    let sanitized = sanitize_paths_in_text(&text);
    let expected = format!("~{}mesh-llm", std::path::MAIN_SEPARATOR);
    assert!(sanitized.contains(&expected));
    assert!(!sanitized.contains(&home.display().to_string()));
}

#[test]
fn bare_home_in_quotes_whitespace_and_punctuation_is_redacted() {
    let home = dirs::home_dir().expect("test runner has a home directory");
    for (before, after) in [
        ("", ""),
        ("\"", "\""),
        ("'", "'"),
        ("home=", " next"),
        ("\t", "\t"),
        ("\n", "\n"),
        ("(", ")"),
        ("", ","),
        ("", ";"),
        ("", "."),
        ("", ":"),
        ("", "!"),
    ] {
        let text = format!("{before}{}{after}", home.display());
        assert_eq!(sanitize_paths_in_text(&text), format!("{before}~{after}"));
    }
}

#[test]
fn bare_home_in_json_string_values_is_redacted() {
    let home = dirs::home_dir().expect("test runner has a home directory");
    // Sanitize the raw values before JSON encoding so Windows separators are
    // not escaped before the text redaction boundary sees them.
    let sanitized = sanitize_paths_in_text(&format!("\"{}\"", home.display()));
    assert_eq!(sanitized, "\"~\"");
    let text = format!(
        r#"{{"home":"{}","again":"{}"}}"#,
        home.display(),
        home.display()
    );
    assert_eq!(sanitize_paths_in_text(&text), r#"{"home":"~","again":"~"}"#);
}

#[test]
fn deterministic_unix_and_windows_home_boundaries() {
    for home in ["/home/alice", r"C:\Users\alice", "C:/Users/alice"] {
        for after in [
            "", "/logs", r"\logs", "\"", "'", " next", "\t", "\n", "\u{2003}", ")", "]", "}", ",",
            ";", ".", ". next", ":", "!", "?",
        ] {
            let text = format!("home={home}{after}");
            assert_eq!(
                sanitize_paths_with_home(&text, home),
                format!("home=~{after}"),
                "{text:?}"
            );
        }
        for after in [
            "-backup/log.txt",
            "_backup",
            ".backup",
            "2",
            "é",
            "-backup",
            ".config/file",
        ] {
            let text = format!("home={home}{after}");
            assert_eq!(sanitize_paths_with_home(&text, home), text);
        }
        let text = format!("{home}-backup {home}/logs \"{home}\" {home}.backup {home}.");
        assert_eq!(
            sanitize_paths_with_home(&text, home),
            format!("{home}-backup ~/logs \"~\" {home}.backup ~.")
        );
        let text = format!("{home}/{home}/{home}");
        assert_eq!(sanitize_paths_with_home(&text, home), "~/~/~");
    }
}

#[test]
fn empty_home_does_not_insert_redactions() {
    assert_eq!(sanitize_paths_with_home("unchanged", ""), "unchanged");
}
