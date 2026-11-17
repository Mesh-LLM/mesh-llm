use super::*;

fn environment(directory: &Path, name: &str) -> BTreeMap<OsString, OsString> {
    BTreeMap::from([
        (
            "CARGO_MANIFEST_DIR".into(),
            directory.as_os_str().to_owned(),
        ),
        ("CARGO_CRATE_NAME".into(), name.into()),
    ])
}

#[test]
fn only_the_matching_nonprimary_proc_macro_library_is_selected() {
    let directory = tempfile::tempdir().unwrap();
    for (library, name, expected) in [
        ("", "macro_fixture", false),
        ("[lib]\nproc-macro=false\n", "macro_fixture", false),
        ("[lib]\nproc-macro=true\n", "macro_fixture", true),
        ("[lib]\nproc_macro=true\n", "macro_fixture", true),
        ("[lib]\nproc-macro=true\nname='renamed'\n", "renamed", true),
        ("[lib]\nproc-macro=true\n", "build_script_build", false),
        ("[lib]\nproc-macro=true\n", "different_target", false),
    ] {
        fs::write(
            directory.path().join("Cargo.toml"),
            format!("[package]\nname='macro-fixture'\n{library}"),
        )
        .unwrap();
        let environment = environment(directory.path(), name);
        assert_eq!(
            proc_macro_dependency(&environment).unwrap(),
            expected,
            "{library}: {name}"
        );
    }
}

#[test]
fn primary_targets_and_non_cargo_probes_never_need_manifest_inspection() {
    let directory = tempfile::tempdir().unwrap();
    let mut environment = environment(directory.path(), "macro_fixture");
    // No manifest exists: a primary target must remain outside this policy.
    for primary in ["1", ""] {
        environment.insert("CARGO_PRIMARY_PACKAGE".into(), primary.into());
        assert!(!proc_macro_dependency(&environment).unwrap());
    }
    assert!(!proc_macro_dependency(&BTreeMap::new()).unwrap());
    environment.remove(std::ffi::OsStr::new("CARGO_PRIMARY_PACKAGE"));
    environment.insert("CARGO_MANIFEST_DIR".into(), "".into());
    assert!(!proc_macro_dependency(&environment).unwrap());
}

#[test]
fn unavailable_or_invalid_dependency_metadata_fails_instead_of_guessing() {
    let directory = tempfile::tempdir().unwrap();
    let environment = environment(directory.path(), "macro_fixture");
    let path = directory.path().join("Cargo.toml");
    assert!(proc_macro_dependency(&environment).is_err());
    fs::write(&path, "not valid TOML = [").unwrap();
    assert!(proc_macro_dependency(&environment).is_err());
    fs::write(&path, vec![b' '; 1024 * 1024 + 1]).unwrap();
    assert!(proc_macro_dependency(&environment).is_err());
}
