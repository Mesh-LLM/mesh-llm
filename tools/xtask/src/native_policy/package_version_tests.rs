use super::*;

#[test]
fn workspace_first_match_when_sections_repeat() {
    let source = b"version = \"outside\"\n[workspace.package]\nversion = \"first\" tail\nversion = \"second\"\n[workspace.package]\nversion = \"third\"";
    let actual = workspace_version(source).unwrap();
    assert_eq!(actual.to_string(), "first");
}

#[test]
fn workspace_reenters_when_any_header_ends_eligibility() {
    let source = b"[workspace.package]\n[broken\nversion = \"wrong\"\n[workspace.package] # comment\nversion = \"also wrong\"\n[workspace.package]\nversion = \"right\"";
    let actual = workspace_version(source).unwrap();
    assert_eq!(actual.to_string(), "right");
}

#[test]
fn workspace_prefix_when_not_valid_toml() {
    let source = " [workspace.package]\u{a0}\r\nversion =\u{2003}\"not semver\\escape\"garbage";
    let actual = workspace_version(source.as_bytes()).unwrap();
    assert_eq!(actual.to_string(), "not semver\\escape");
}

#[test]
fn workspace_skips_when_prefix_is_malformed() {
    let source = b"[workspace.package]\nversion = \"\"\nversion_extra = \"wrong\"\nversion = 'wrong'\nversion = \"unterminated\nversion=\"right\"";
    let actual = workspace_version(source).unwrap();
    assert_eq!(actual.to_string(), "right");
}

#[test]
fn workspace_missing_when_empty_or_wrong_section() {
    for source in [
        "",
        "\n",
        "[package]\nversion=\"wrong\"",
        "[workspace.package] # comment\nversion=\"wrong\"",
    ] {
        let actual = workspace_version(source.as_bytes());
        assert!(
            matches!(actual, Err(VersionError::WorkspaceMissing)),
            "{source:?}"
        );
    }
}

#[test]
fn workspace_matches_when_final_newline_varies() {
    for ending in ["", "\n", "\r\n", "\r"] {
        let source = format!("[workspace.package]\r\nversion=\"line\"{ending}");
        let actual = workspace_version(source.as_bytes()).unwrap();
        assert_eq!(actual.to_string(), "line");
    }
}

#[test]
fn workspace_keeps_embedded_separators_when_python_file_iteration_would() {
    let source = "[workspace.package]\nversion=\"left\u{2028}right\u{1e}end\"";
    let actual = workspace_version(source.as_bytes()).unwrap();
    assert_eq!(actual.to_string(), "left\u{2028}right\u{1e}end");
}

#[test]
fn abi_last_match_when_declarations_repeat() {
    let source = "pub const ABI_VERSION_PATCH: u32 = 3;\npub const ABI_VERSION_MAJOR: u32 = 1;\npub const ABI_VERSION_MINOR: u32 = 2;\npub const ABI_VERSION_MAJOR: u32 = 004; trailing\npub const ABI_VERSION_MINOR: u32 = 005;\npub const ABI_VERSION_PATCH: u32 = 006;";
    let actual = abi_version(source.as_bytes()).unwrap();
    assert_eq!(actual.to_string(), "004.005.006");
}

#[test]
fn abi_ignores_when_declarations_are_not_exact_prefixes() {
    let source = "pub const ABI_VERSION_MAJOR: u32 = 1;\npub const ABI_VERSION_MINOR: u32 = 2;\npub const ABI_VERSION_PATCH: u32 = 3;\npub const ABI_VERSION_MAJOR: u32 = 9 ;\npub const ABI_VERSION_MINOR: u64 = 9;\npub const ABI_VERSION_PATCH: u32 = ９;\npub const ABI_VERSION_PATCH: u32 = +9;\npub const ABI_VERSION_MAJOR:  u32 = 9;\n// pub const ABI_VERSION_MINOR: u32 = 9;";
    let actual = abi_version(source.as_bytes()).unwrap();
    assert_eq!(actual.to_string(), "1.2.3");
}

#[test]
fn abi_preserves_when_digit_text_exceeds_integer_range() {
    let digits = "0".repeat(2) + &"9".repeat(5000);
    let source = format!(
        "pub const ABI_VERSION_MAJOR: u32 = {digits};\npub const ABI_VERSION_MINOR: u32 = 00;\npub const ABI_VERSION_PATCH: u32 = 000;"
    );
    let actual = abi_version(source.as_bytes()).unwrap();
    assert_eq!(actual.to_string(), format!("{digits}.00.000"));
}

#[test]
fn abi_strips_when_c0_and_unicode_whitespace_wrap_crlf_lines() {
    for ending in ["", "\n", "\r\n", "\r"] {
        let source = format!(
            " pub const ABI_VERSION_MAJOR: u32 = 01;\u{a0}\r\npub const ABI_VERSION_MINOR: u32 = 02;\r\u{2003}pub const ABI_VERSION_PATCH: u32 = 03;{ending}"
        );
        let actual = abi_version(source.as_bytes()).unwrap();
        assert_eq!(actual.to_string(), "01.02.03");
    }
}

#[test]
fn abi_missing_when_components_absent_in_lookup_order() {
    for (source, expected) in [
        ("", AbiComponent::Major),
        ("pub const ABI_VERSION_PATCH: u32 = 3;", AbiComponent::Major),
        ("pub const ABI_VERSION_MAJOR: u32 = 1;", AbiComponent::Minor),
        (
            "pub const ABI_VERSION_MAJOR: u32 = 1;\npub const ABI_VERSION_MINOR: u32 = 2;",
            AbiComponent::Patch,
        ),
    ] {
        let actual = abi_version(source.as_bytes());
        assert!(
            matches!(actual, Err(VersionError::AbiMissing(component)) if component == expected)
        );
    }
}

#[test]
fn source_rejected_when_utf8_is_invalid() {
    let source = b"\xff";
    let actual = workspace_version(source);
    assert!(matches!(actual, Err(VersionError::Utf8(_))));
}

#[test]
fn abi_rejected_when_utf8_is_invalid() {
    let source = b"\xff";
    let actual = abi_version(source);
    assert!(matches!(actual, Err(VersionError::Utf8(_))));
}

#[test]
fn source_read_error_when_path_is_directory() {
    let directory = tempfile::tempdir().unwrap();
    let actual = read_source(directory.path());
    assert!(matches!(actual, Err(VersionError::Read { path, .. }) if path == directory.path()));
}
