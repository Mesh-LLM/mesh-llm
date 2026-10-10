use super::{PrivacyError, verify_files};
use std::{fs, path::PathBuf};

const TEMPLATE: &[u8] = include_bytes!("fixtures/PrivacyInfo.xcprivacy");

fn fixture() -> (tempfile::TempDir, PathBuf, PathBuf) {
    let temp = tempfile::tempdir().unwrap();
    let template = temp.path().join("template.xcprivacy");
    let framework = temp.path().join("MeshLLMFFI.xcframework");
    fs::write(&template, TEMPLATE).unwrap();
    fs::create_dir(&framework).unwrap();
    (temp, template, framework)
}

#[test]
fn retains_native_lint_obligation_when_template_policy_passes() {
    let (_temp, template, _framework) = fixture();
    let result = verify_files(&template, None).unwrap();
    assert_eq!(result.pending_native_lint.template, template);
    assert!(result.pending_native_lint.embedded.is_empty());
}

#[test]
fn finds_exact_bytes_when_nested_in_arbitrary_directories() {
    let (_temp, template, framework) = fixture();
    let directory = framework.join("macos/MeshLLMFFI.framework/Versions/A/Resources");
    fs::create_dir_all(&directory).unwrap();
    let embedded = directory.join("PrivacyInfo.xcprivacy");
    fs::write(&embedded, TEMPLATE).unwrap();
    let result = verify_files(&template, Some(&framework)).unwrap();
    assert_eq!(result.pending_native_lint.embedded, [embedded]);
}

#[test]
fn rejects_template_before_framework_discovery_when_both_invalid() {
    let (temp, template, _framework) = fixture();
    fs::write(&template, b"invalid plist").unwrap();
    let missing = temp.path().join("missing");
    let result = verify_files(&template, Some(&missing));
    assert!(matches!(result, Err(PrivacyError::Plist(_))));
}

#[test]
fn rejects_framework_when_missing_after_valid_template() {
    let (temp, template, _framework) = fixture();
    let missing = temp.path().join("missing");
    let result = verify_files(&template, Some(&missing));
    assert!(matches!(result, Err(PrivacyError::MissingFramework(path)) if path == missing));
}

#[test]
fn rejects_framework_when_no_matching_name_is_embedded() {
    let (_temp, template, framework) = fixture();
    fs::write(framework.join("privacyinfo.xcprivacy"), TEMPLATE).unwrap();
    let result = verify_files(&template, Some(&framework));
    assert!(matches!(result, Err(PrivacyError::NotEmbedded(path)) if path == framework));
}

#[test]
fn rejects_embedding_when_equal_policy_has_different_bytes() {
    let (_temp, template, framework) = fixture();
    let mut bytes = TEMPLATE.to_vec();
    bytes.push(b'\n');
    let embedded = framework.join("PrivacyInfo.xcprivacy");
    fs::write(&embedded, bytes).unwrap();
    let result = verify_files(&template, Some(&framework));
    assert!(matches!(result, Err(PrivacyError::EmbeddedDifference(path)) if path == embedded));
}

#[test]
fn counts_matching_directory_when_find_would_match_its_name() {
    let (_temp, template, framework) = fixture();
    let embedded = framework.join("PrivacyInfo.xcprivacy");
    fs::create_dir(&embedded).unwrap();
    let result = verify_files(&template, Some(&framework));
    assert!(matches!(result, Err(PrivacyError::EmbeddedDifference(path)) if path == embedded));
}

#[cfg(unix)]
#[test]
fn ignores_directory_symlink_when_find_does_not_follow_it() {
    let (temp, template, framework) = fixture();
    let outside = temp.path().join("outside");
    fs::create_dir(&outside).unwrap();
    fs::write(outside.join("PrivacyInfo.xcprivacy"), TEMPLATE).unwrap();
    std::os::unix::fs::symlink(&outside, framework.join("Resources")).unwrap();
    let result = verify_files(&template, Some(&framework));
    assert!(matches!(result, Err(PrivacyError::NotEmbedded(_))));
}

#[cfg(unix)]
#[test]
fn accepts_matching_file_symlink_when_cmp_reads_target_bytes() {
    let (_temp, template, framework) = fixture();
    let embedded = framework.join("PrivacyInfo.xcprivacy");
    std::os::unix::fs::symlink(&template, &embedded).unwrap();
    let result = verify_files(&template, Some(&framework)).unwrap();
    assert_eq!(result.pending_native_lint.embedded, [embedded]);
}

#[cfg(unix)]
#[test]
fn does_not_follow_root_directory_symlink_when_find_uses_physical_walk() {
    let (temp, template, framework) = fixture();
    fs::write(framework.join("PrivacyInfo.xcprivacy"), TEMPLATE).unwrap();
    let linked = temp.path().join("linked.xcframework");
    std::os::unix::fs::symlink(&framework, &linked).unwrap();
    let result = verify_files(&template, Some(&linked));
    assert!(matches!(result, Err(PrivacyError::NotEmbedded(_))));
}
