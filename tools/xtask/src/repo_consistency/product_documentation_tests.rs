use super::*;

fn fixture() -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    for product in ["mesh", "skippy"] {
        fs::create_dir_all(root.path().join(product).join("crates")).unwrap();
    }
    root
}

fn crate_files(
    root: &Path,
    product: &str,
    metadata: &str,
    readme: Option<&str>,
) -> std::path::PathBuf {
    let directory = root.join(product).join("crates/example");
    fs::create_dir_all(&directory).unwrap();
    fs::write(directory.join("Cargo.toml"), metadata).unwrap();
    if let Some(source) = readme {
        fs::write(directory.join("README.md"), source).unwrap();
    }
    directory
}

const VALID: &str = "[package]\nname='example'\ndescription='Fixture documentation'\n";

#[test]
fn current_product_crates_satisfy_documentation_contract() {
    check(&crate::repo_consistency::repo_root().unwrap()).unwrap();
}

#[test]
fn independently_reports_missing_metadata_readme_and_link() {
    let root = fixture();
    crate_files(root.path(), "mesh", "[package]\nname='example'\n", None);
    crate_files(
        root.path(),
        "skippy",
        VALID,
        Some("[missing](../absent/README.md)"),
    );
    let errors = problems(root.path()).unwrap().join("\n");
    assert!(errors.contains("missing non-empty package description"));
    assert!(errors.contains("unreadable crate README"));
    assert!(errors.contains("broken local link"));
}

#[test]
fn accepts_existing_relative_root_encoded_and_unicode_paths() {
    let root = fixture();
    fs::write(root.path().join("guide space.md"), "guide").unwrap();
    let directory = crate_files(
        root.path(),
        "skippy",
        VALID,
        Some(
            "[manifest](Cargo.toml#package) [root](/guide%20space.md?q=1#s) [angle](<notes space.md>) [unicode](résumé.md)",
        ),
    );
    fs::write(directory.join("notes space.md"), "notes").unwrap();
    fs::write(directory.join("résumé.md"), "notes").unwrap();
    check(root.path()).unwrap();
}

#[test]
fn ignores_external_anchors_and_both_fenced_example_styles() {
    let root = fixture();
    crate_files(
        root.path(),
        "mesh",
        VALID,
        Some(
            "[external](https://example.com/missing) [network](//example.com/missing) [mail](mailto:fixture@example.com) [anchor](#section)\n```md\n[missing](not-file)\n```\n~~~md\n[missing](also-not-file)\n~~~\n[real](Cargo.toml)",
        ),
    );
    check(root.path()).unwrap();
}

#[test]
fn refuses_escaped_existing_target_and_noncanonical_readme_metadata() {
    let outside = tempfile::tempdir().unwrap();
    fs::write(outside.path().join("outside.md"), "outside").unwrap();
    let root = fixture();
    let directory = crate_files(
        root.path(),
        "mesh",
        "[package]\nname='example'\ndescription='ok'\nreadme=false\n",
        Some("[escape](../../../../outside.md)"),
    );
    assert_eq!(root.path().parent(), outside.path().parent());
    fs::write(
        directory.join("README.md"),
        format!(
            "[escape](../../../../{}/outside.md)",
            outside.path().file_name().unwrap().to_string_lossy()
        ),
    )
    .unwrap();
    assert!(check(root.path()).is_err());
    assert!(
        problems(root.path())
            .unwrap()
            .iter()
            .any(|error| error.contains("readme must be README.md"))
    );
    assert!(
        problems(root.path())
            .unwrap()
            .iter()
            .any(|error| error.contains("leaves repository"))
    );
}

#[cfg(unix)]
#[test]
fn refuses_existing_symlink_link_outside_repository() {
    let root = fixture();
    let outside = tempfile::tempdir().unwrap();
    fs::write(outside.path().join("outside.md"), "outside").unwrap();
    let directory = crate_files(root.path(), "mesh", VALID, Some("[escape](escape.md)"));
    std::os::unix::fs::symlink(
        outside.path().join("outside.md"),
        directory.join("escape.md"),
    )
    .unwrap();
    assert!(
        problems(root.path())
            .unwrap()
            .iter()
            .any(|error| error.contains("leaves repository"))
    );
}

#[test]
fn malformed_manifest_and_absent_product_directory_are_actionable() {
    let root = fixture();
    crate_files(root.path(), "mesh", "[package", Some("# README"));
    fs::remove_dir(root.path().join("skippy/crates")).unwrap();
    let errors = problems(root.path()).unwrap().join("\n");
    assert!(errors.contains("cannot read package metadata"));
    assert!(errors.contains("missing product crates directory"));
}

#[test]
fn percent_decoding_handles_literal_percent_and_utf8_without_panicking() {
    assert_eq!(decode_path("r%C3%A9sum%C3%A9.md").unwrap(), "résumé.md");
    assert_eq!(decode_path("%résumé.md").unwrap(), "%résumé.md");
    assert!(decode_path("%FF.md").is_err());
}
