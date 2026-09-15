//! Path API and boundary-aware free-form logging redaction contracts.

use super::{
    sanitize_json_paths_with_home, sanitize_path, sanitize_paths_in_json_text,
    sanitize_paths_in_text, sanitize_paths_with_home,
};

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
    let text = serde_json::to_string(&serde_json::json!({
        "home": home.to_string_lossy(),
        "again": home.to_string_lossy(),
    }))
    .unwrap();
    let sanitized: serde_json::Value =
        serde_json::from_str(&sanitize_paths_in_json_text(&text)).unwrap();
    assert_eq!(sanitized, serde_json::json!({"home": "~", "again": "~"}));
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

#[test]
fn serialized_windows_and_unix_paths_are_redacted_in_nested_values() {
    for home in [
        r"C:\Users\alice",
        "C:/Users/alice",
        "/home/alice",
        "/home/a\"lice",
    ] {
        let sibling = format!("{home}-backup\\log.txt");
        let input = serde_json::json!({
            "home": home,
            "nested": [{
                "descendant": format!("{home}\\logs\\app.log"),
                "forward": format!("{home}/logs/app.log"),
                "message": format!("home=\"{home}\" next {home}\t{home}-backup"),
                "sibling": sibling,
            }],
            "number": 42,
            "flag": true,
            "empty": null,
        });
        let text = serde_json::to_string(&input).unwrap();
        let sanitized = sanitize_json_paths_with_home(&text, home);
        let actual: serde_json::Value = serde_json::from_str(&sanitized).unwrap();
        assert_eq!(
            actual,
            serde_json::json!({
                "home": "~",
                "nested": [{
                    "descendant": "~\\logs\\app.log",
                    "forward": "~/logs/app.log",
                    "message": format!("home=\"~\" next ~\t{home}-backup"),
                    "sibling": sibling,
                }],
                "number": 42,
                "flag": true,
                "empty": null,
            }),
            "serialized input: {text}"
        );
    }
}

#[test]
fn json_detail_fallback_and_private_paths_are_preserved() {
    let home = r"C:\Users\alice";
    let text = format!("Error in {home}\\logs; {home}-backup");
    assert_eq!(
        sanitize_json_paths_with_home(&text, home),
        format!("Error in ~\\logs; {home}-backup")
    );
    let text = serde_json::to_string(&serde_json::json!({"path": "/private/tmp/log"})).unwrap();
    let actual: serde_json::Value =
        serde_json::from_str(&sanitize_json_paths_with_home(&text, home)).unwrap();
    assert_eq!(actual, serde_json::json!({"path": "/tmp/log"}));
    assert_eq!(sanitize_json_paths_with_home(&text, ""), text);
}

#[test]
fn home_in_url_query_values_is_redacted_before_query_separators() {
    for home in ["/home/alice", r"C:\Users\alice"] {
        for separator in ["&", "&amp;"] {
            let text = format!("https://example.com/?path={home}{separator}next=1");
            assert_eq!(
                sanitize_paths_with_home(&text, home),
                format!("https://example.com/?path=~{separator}next=1")
            );
        }
    }
}
