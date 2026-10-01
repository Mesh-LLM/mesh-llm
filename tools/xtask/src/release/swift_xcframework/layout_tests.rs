use super::{
    Mode,
    fixtures::{Fixture, set},
};
use std::fs;

#[test]
fn accepts_complete_versioned_framework_when_full_and_host_only() {
    for (fixture, mode) in [
        (Fixture::full(), Mode::Full),
        (Fixture::host(), Mode::HostOnly),
    ] {
        fixture.materialize();
        fixture.write(false);
        let result = fixture.verify(Some(mode));
        assert_eq!(result.unwrap(), fixture.entries.len());
    }
}

#[test]
fn lipo_mismatch_when_catalyst_binary_lacks_x86_64() {
    let fixture = Fixture::full();
    fixture.materialize();
    fixture.write(true);
    fs::write(fixture.framework(2).join("MeshLLMFFI"), "arm64").unwrap();
    let result = fixture.verify(Some(Mode::Full));
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("lipo architectures")
    );
}

#[test]
fn containment_rejects_framework_when_slice_symlink_escapes() {
    let fixture = Fixture::host();
    fixture.materialize();
    fixture.write(false);
    let slice = fixture.root.join("macos-");
    let outside = fixture.directory.path().join("outside");
    fs::rename(&slice, &outside).unwrap();
    std::os::unix::fs::symlink(&outside, &slice).unwrap();
    let result = fixture.verify(None);
    assert!(result.unwrap_err().to_string().contains("escapes its root"));
}

#[test]
fn links_reject_wrong_or_missing_target_when_each_macos_link_changes() {
    for relative in [
        "Versions/Current",
        "MeshLLMFFI",
        "Headers",
        "Modules",
        "Resources",
    ] {
        for wrong in [false, true] {
            let fixture = Fixture::host();
            fixture.materialize();
            fixture.write(false);
            let path = fixture.framework(0).join(relative);
            fs::remove_file(&path).unwrap();
            if wrong {
                std::os::unix::fs::symlink("wrong", &path).unwrap();
            }
            let result = fixture.verify(None);
            let error = result.unwrap_err().to_string();
            assert!(error.contains(if wrong {
                "unexpected symlink target"
            } else {
                "missing symlink"
            }));
        }
    }
}

#[test]
fn versioned_existence_when_required_file_or_directory_is_missing() {
    for relative in [
        "MeshLLMFFI",
        "Headers",
        "Modules/module.modulemap",
        "Resources/Info.plist",
        "Resources/PrivacyInfo.xcprivacy",
    ] {
        let fixture = Fixture::host();
        fixture.materialize();
        fixture.write(false);
        let path = fixture.framework(0).join("Versions/A").join(relative);
        if path.is_dir() {
            fs::remove_dir(&path).unwrap();
        } else {
            fs::remove_file(&path).unwrap();
        }
        let result = fixture.verify(None);
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("layout is incomplete")
        );
    }
}

#[test]
fn binary_stem_when_framework_has_multiple_dots() {
    let mut fixture = Fixture::host();
    set(
        &mut fixture.entries[0],
        "LibraryPath",
        plist::Value::String("Mesh.FFI.framework".into()),
    );
    fixture.materialize();
    fixture.write(true);
    let result = fixture.verify(Some(Mode::HostOnly));
    assert_eq!(result.unwrap(), 1);
}

#[test]
fn preceding_native_failure_when_later_declaration_is_invalid() {
    let mut fixture = Fixture::full();
    fixture.materialize();
    set(
        &mut fixture.entries[1],
        "SupportedArchitectures",
        plist::Value::Boolean(false),
    );
    fixture.write(false);
    let document = super::input::Document::read(&fixture.root).unwrap();
    let entries = document.entries(None).unwrap();
    let result = super::verify(
        super::Verification {
            entries: &entries,
            root: &fixture.root,
            mode: None,
        },
        |_| Err(super::Error::Contract("first-native-failure".into())),
    );
    assert_eq!(result.unwrap_err().to_string(), "first-native-failure");
}

#[test]
fn verifier_is_read_only_when_real_fixture_tree_succeeds() {
    let fixture = Fixture::host();
    fixture.materialize();
    fixture.write(false);
    let before = snapshot(&fixture.root);
    let result = fixture.verify(None);
    assert_eq!(result.unwrap(), 1);
    assert_eq!(snapshot(&fixture.root), before);
}

pub(super) fn snapshot(root: &std::path::Path) -> Vec<(std::path::PathBuf, Vec<u8>)> {
    let mut files = Vec::new();
    for entry in fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if path.is_symlink() {
            files.push((
                path.clone(),
                fs::read_link(&path)
                    .unwrap()
                    .as_os_str()
                    .as_encoded_bytes()
                    .to_vec(),
            ));
        } else if path.is_dir() {
            files.extend(snapshot(&path));
        } else {
            files.push((path.clone(), fs::read(&path).unwrap()));
        }
    }
    files.sort();
    files
}
