use super::*;

const VALID: &str = include_str!("../../../tests/fixtures/release/swift_manifest/updated.swift");

fn values() -> ReleaseValues<'static> {
    ReleaseValues {
        tag: ReleaseTag("not-a-version"),
        checksum: SwiftChecksum("opaque-checksum"),
    }
}

#[test]
fn migration_release_swift_manifest_verifies_when_values_match() {
    let input = VALID.as_bytes();
    let result = verify(input, &values());
    assert!(result.is_ok());
}

#[test]
fn migration_release_swift_manifest_verifies_first_when_later_definitions_differ() {
    let input = format!(
        "// {VALID}let remoteFFIXCFrameworkURL = \"wrong\"\nlet remoteFFIXCFrameworkChecksum = \"wrong\""
    );
    let result = verify(input.as_bytes(), &values());
    assert!(result.is_ok());
}

#[test]
fn migration_release_swift_manifest_reports_missing_before_placeholders() {
    let input = b"let remoteFFIXCFrameworkURL = \"__MESH_SWIFT_RELEASE_TAG__\"";
    let result = verify(input, &values());
    assert!(matches!(
        result,
        Err(ManifestError::Missing(Field::Checksum))
    ));
}

#[test]
fn migration_release_swift_manifest_reports_missing_url_when_only_checksum_exists() {
    let input = b"let remoteFFIXCFrameworkChecksum = \"__MESH_SWIFT_RELEASE_CHECKSUM__\"";
    let result = verify(input, &values());
    assert!(matches!(result, Err(ManifestError::Missing(Field::Url))));
}

#[test]
fn migration_release_swift_manifest_reports_url_placeholder_when_both_are_placeholders() {
    let input = include_bytes!("../../../tests/fixtures/release/swift_manifest/template.swift");
    let result = verify(input, &values());
    assert!(matches!(result, Err(ManifestError::UrlPlaceholder)));
}

#[test]
fn migration_release_swift_manifest_reports_checksum_placeholder_before_url_mismatch() {
    let input = b"let remoteFFIXCFrameworkURL = \"wrong\"\nlet remoteFFIXCFrameworkChecksum = \"prefix__MESH_SWIFT_RELEASE_CHECKSUM__suffix\"";
    let result = verify(input, &values());
    assert!(matches!(result, Err(ManifestError::ChecksumPlaceholder)));
}

#[test]
fn migration_release_swift_manifest_reports_url_mismatch_before_checksum_mismatch() {
    let input = VALID
        .replace("not-a-version", "wrong")
        .replace("opaque-checksum", "wrong");
    let result = verify(input.as_bytes(), &values());
    assert!(matches!(result, Err(ManifestError::UrlMismatch { .. })));
}

#[test]
fn migration_release_swift_manifest_reports_checksum_mismatch_when_url_matches() {
    let input = VALID.replace("opaque-checksum", "wrong");
    let result = verify(input.as_bytes(), &values());
    assert!(matches!(
        result,
        Err(ManifestError::ChecksumMismatch { .. })
    ));
}

#[test]
fn migration_release_swift_manifest_normalizes_values_when_reading_crlf() {
    let values = ReleaseValues {
        tag: ReleaseTag("not-a-version"),
        checksum: SwiftChecksum("opaque\nchecksum"),
    };
    let input = VALID.replace("opaque-checksum", "opaque\r\nchecksum");
    let result = verify(input.as_bytes(), &values);
    assert!(result.is_ok());
}

#[test]
fn migration_release_swift_manifest_rejects_utf8_when_verification_input_is_invalid() {
    let input = [VALID.as_bytes(), b"\xff"].concat();
    let result = verify(&input, &values());
    assert!(matches!(result, Err(ManifestError::Utf8(_))));
}
