//! Replacement commits and rollback boundaries of the actual Unix installer.
use super::fixture::{Fixture, stderr};
use std::{fs, os::unix::fs::PermissionsExt};

fn previous(fixture: &Fixture) {
    let install = fixture.root.join("install");
    fs::create_dir_all(install.join("native-runtimes/old-runtime")).unwrap();
    for (name, bytes) in [
        ("mesh-llm", "old host\n"),
        ("product-manifest.json", "old manifest\n"),
        ("native-runtimes/old-runtime/library", "old runtime\n"),
        ("unrelated.txt", "unrelated\n"),
    ] {
        fs::write(install.join(name), bytes).unwrap();
    }
}

fn unchanged(fixture: &Fixture) {
    let install = fixture.root.join("install");
    for (name, bytes) in [
        ("mesh-llm", "old host\n"),
        ("product-manifest.json", "old manifest\n"),
        ("native-runtimes/old-runtime/library", "old runtime\n"),
        ("unrelated.txt", "unrelated\n"),
    ] {
        assert_eq!(fs::read_to_string(install.join(name)).unwrap(), bytes);
    }
    no_staging(fixture);
}

fn no_staging(fixture: &Fixture) {
    for entry in fs::read_dir(fixture.root.join("install")).unwrap() {
        assert!(
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".mesh-llm-")
        );
    }
}

fn bundle(fixture: &Fixture) {
    let root = fixture.root.join("bundle");
    fs::create_dir_all(root.join("native-runtimes/new-runtime/lib")).unwrap();
    fs::write(root.join("mesh-llm"), "new host\n").unwrap();
    fs::set_permissions(root.join("mesh-llm"), fs::Permissions::from_mode(0o755)).unwrap();
    fs::write(root.join("product-manifest.json"), "new manifest\n").unwrap();
    fs::write(
        root.join("native-runtimes/new-runtime/lib/libllama.so"),
        "new runtime\n",
    )
    .unwrap();
}

const INSTALL: &str = "install_bundle \"$FIXTURE_ROOT/bundle\"";

#[test]
fn installer_invalid_bundle_preserves_previous_product_before_mutation() {
    for invalid in [
        "missing-host",
        "nonexecutable-host",
        "partial-composed",
        "current-legacy",
    ] {
        let fixture = Fixture::new();
        previous(&fixture);
        bundle(&fixture);
        let root = fixture.root.join("bundle");
        match invalid {
            "missing-host" => fs::remove_file(root.join("mesh-llm")).unwrap(),
            "nonexecutable-host" => {
                fs::set_permissions(root.join("mesh-llm"), fs::Permissions::from_mode(0o644))
                    .unwrap();
            }
            "partial-composed" => fs::remove_dir_all(root.join("native-runtimes")).unwrap(),
            _ => {
                fs::remove_dir_all(root.join("native-runtimes")).unwrap();
                fs::remove_file(root.join("product-manifest.json")).unwrap();
                fs::write(
                    root.join("mesh-llm"),
                    "#!/bin/bash\nprintf 'mesh-llm 0.75.0\\n'\n",
                )
                .unwrap();
            }
        }
        let report = fixture.run(INSTALL);
        assert!(!report.success(), "{invalid}: {report:?}");
        assert!(!stderr(&report).is_empty());
        unchanged(&fixture);
    }
}

#[test]
fn installer_precontract_archive_remains_installable() {
    let fixture = Fixture::new();
    fs::create_dir(fixture.root.join("bundle")).unwrap();
    let host = fixture.root.join("bundle/mesh-llm");
    fs::write(&host, "#!/bin/bash\nprintf 'mesh-llm 0.74.0\\n'\n").unwrap();
    fs::set_permissions(&host, fs::Permissions::from_mode(0o755)).unwrap();
    let report = fixture.run(INSTALL);
    assert!(report.success(), "{report:?}");
    assert!(stderr(&report).contains("supported legacy MeshLLM 0.74.0"));
    assert_eq!(
        fs::read(fixture.root.join("install/mesh-llm")).unwrap(),
        fs::read(host).unwrap()
    );
    no_staging(&fixture);
}

#[test]
fn installer_commits_new_runtime_tree_and_preserves_unrelated_files() {
    let fixture = Fixture::new();
    previous(&fixture);
    bundle(&fixture);
    fs::write(fixture.root.join("install/llama-server"), "stale\n").unwrap();
    let report = fixture.run(INSTALL);
    assert!(report.success(), "{report:?}");
    let install = fixture.root.join("install");
    assert_eq!(
        fs::read_to_string(install.join("mesh-llm")).unwrap(),
        "new host\n"
    );
    assert_eq!(
        fs::read_to_string(install.join("product-manifest.json")).unwrap(),
        "new manifest\n"
    );
    assert_eq!(
        fs::read_to_string(install.join("native-runtimes/new-runtime/lib/libllama.so")).unwrap(),
        "new runtime\n"
    );
    assert_eq!(
        fs::read_to_string(install.join("unrelated.txt")).unwrap(),
        "unrelated\n"
    );
    assert!(!install.join("native-runtimes/old-runtime").exists());
    assert!(!install.join("llama-server").exists());
    no_staging(&fixture);
}

#[test]
fn installer_failed_staging_and_interrupted_replacement_restore_previous_product() {
    for failure in ["mktemp", "copy", "move", "term"] {
        let fixture = Fixture::new();
        previous(&fixture);
        bundle(&fixture);
        let injection = match failure {
            "mktemp" => "mktemp() { return 41; }".to_owned(),
            "copy" => "cp() { return 37; }".to_owned(),
            mode => format!(
                "failed_runtime_move=0\nmv() {{ if [[ \"$failed_runtime_move\" == 0 && \"$1\" == *\".mesh-llm-stage.\"*/native-runtimes ]]; then failed_runtime_move=1; {}; return 42; fi; command mv \"$@\"; }}",
                if mode == "term" {
                    "kill -TERM \"$$\""
                } else {
                    ":"
                }
            ),
        };
        let report = fixture.run(&format!("{injection}\n{INSTALL}"));
        assert!(!report.success(), "{failure}: {report:?}");
        unchanged(&fixture);
    }
}
