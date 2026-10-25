use super::*;

const INPUT: &str = include_str!("../../../tests/fixtures/release/swift_manifest/template.swift");
const EXPECTED: &str = include_str!("../../../tests/fixtures/release/swift_manifest/updated.swift");

fn values() -> ReleaseValues<'static> {
    ReleaseValues {
        tag: ReleaseTag("not-a-version"),
        checksum: SwiftChecksum("opaque-checksum"),
    }
}

#[test]
fn migration_release_swift_manifest_updates_when_both_fields_exist() {
    let input = INPUT.as_bytes();
    let result = update(input, &values()).unwrap();
    assert_eq!(result, EXPECTED);
}

#[test]
fn migration_release_swift_manifest_preserves_file_when_checksum_missing() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("Package.swift");
    let original = b"sentinel\r\nlet remoteFFIXCFrameworkURL = \"old\"\r\n";
    std::fs::write(&path, original).unwrap();
    let result = update_file(&path, &values());
    assert!(matches!(
        result,
        Err(ManifestError::Missing(Field::Checksum))
    ));
    assert_eq!(std::fs::read(&path).unwrap(), original);
}

#[test]
fn migration_release_swift_manifest_reports_url_when_both_fields_missing() {
    let input = b"sentinel";
    let result = update(input, &values());
    assert!(matches!(result, Err(ManifestError::Missing(Field::Url))));
}

#[test]
fn migration_release_swift_manifest_updates_first_match_when_duplicates_exist() {
    let input = format!("// {INPUT}{INPUT}");
    let result = update(input.as_bytes(), &values()).unwrap();
    assert_eq!(result, format!("// {EXPECTED}{INPUT}"));
}

#[test]
fn migration_release_swift_manifest_matches_substring_when_keyword_has_prefix() {
    let input = format!("outlet {}", INPUT.trim_start_matches("let "));
    let result = update(input.as_bytes(), &values()).unwrap();
    assert_eq!(result, format!("out{EXPECTED}"));
}

#[test]
fn migration_release_swift_manifest_skips_nonmatches_when_names_have_suffixes() {
    let invalid = [
        "letremoteFFIXCFrameworkURL = \"bad\"",
        "let remoteFFIXCFrameworkURLSuffix = \"bad\"",
        "let remoteFFIXCFrameworkURL: String = \"bad\"",
        "let remoteFFIXCFrameworkURL = 'bad'",
        "let\u{200b}remoteFFIXCFrameworkURL = \"bad\"",
    ]
    .join("\n");
    let input = format!("{invalid}\n{INPUT}");
    let result = update(input.as_bytes(), &values()).unwrap();
    assert_eq!(result, format!("{invalid}\n{EXPECTED}"));
}

#[test]
fn migration_release_swift_manifest_normalizes_newlines_when_update_succeeds() {
    let input = format!("sentinel\r\n{}tail\r", INPUT.replace('\n', "\r\n"));
    let result = update(input.as_bytes(), &values()).unwrap();
    assert_eq!(result, format!("sentinel\n{EXPECTED}tail\n"));
}

#[test]
fn migration_release_swift_manifest_matches_python_space_when_fields_span_lines() {
    let spaces =
        " \t\n\r\u{b}\u{c}\u{85}\u{a0}\u{1680}\u{2000}\u{2028}\u{2029}\u{202f}\u{205f}\u{3000}";
    let input = format!(
        "let{spaces}remoteFFIXCFrameworkURL{spaces}={spaces}\"old\"\nlet{spaces}remoteFFIXCFrameworkChecksum{spaces}={spaces}\"old\"\n"
    );
    let result = update(input.as_bytes(), &values()).unwrap();
    assert_eq!(result, EXPECTED);
}

#[test]
fn migration_release_swift_manifest_inserts_literal_text_when_values_contain_escapes() {
    let values = ReleaseValues {
        tag: ReleaseTag("a\\1\"\r\n/$"),
        checksum: SwiftChecksum("\\g<1>\"\r\n$"),
    };
    let result = update(INPUT.as_bytes(), &values).unwrap();
    assert_eq!(
        result,
        "let remoteFFIXCFrameworkURL = \"https://github.com/Mesh-LLM/mesh-llm/releases/download/a\\1\"\r\n/$/MeshLLMFFI.xcframework.zip\"\nlet remoteFFIXCFrameworkChecksum = \"\\g<1>\"\r\n$\"\n"
    );
}

#[test]
fn migration_release_swift_manifest_stops_at_quote_when_quote_is_backslash_prefixed() {
    let input =
        "let remoteFFIXCFrameworkURL = \"a\\\"tail\"\nlet remoteFFIXCFrameworkChecksum = \"old\"\n";
    let result = update(input.as_bytes(), &values()).unwrap();
    assert_eq!(result, EXPECTED.replacen("zip\"\n", "zip\"tail\"\n", 1));
}

#[test]
fn migration_release_swift_manifest_searches_updated_text_when_tag_injects_checksum_field() {
    let values = ReleaseValues {
        tag: ReleaseTag("\" let remoteFFIXCFrameworkChecksum = \"injected\""),
        checksum: SwiftChecksum("replacement"),
    };
    let result = update(INPUT.as_bytes(), &values).unwrap();
    assert_eq!(
        result,
        "let remoteFFIXCFrameworkURL = \"https://github.com/Mesh-LLM/mesh-llm/releases/download/\" let remoteFFIXCFrameworkChecksum = \"replacement\"/MeshLLMFFI.xcframework.zip\"\nlet remoteFFIXCFrameworkChecksum = \"__MESH_SWIFT_RELEASE_CHECKSUM__\"\n"
    );
}

#[test]
fn migration_release_swift_manifest_rejects_utf8_when_update_input_is_invalid() {
    let input = b"\xff";
    let result = update(input, &values());
    assert!(matches!(result, Err(ManifestError::Utf8(_))));
}
