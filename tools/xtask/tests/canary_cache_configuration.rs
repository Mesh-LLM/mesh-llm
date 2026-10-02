//! Actual Rust CLI fixtures for existing, offline runner cache configuration.
use std::{
    fs,
    path::Path,
    process::{Command, Output},
};

fn run(root: &Path, values: &[(&str, &str)]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
    command
        .args(["ci-ops", "configure-canary-cache"])
        .env_remove("XDG_CACHE_HOME")
        .env_remove("HF_HOME")
        .env_remove("HF_CACHE")
        .env_remove("HF_HUB_CACHE")
        .env_remove("HF_TOKEN")
        .env_remove("HF_TOKEN_PATH")
        .env("HOME", root)
        .env("GITHUB_ENV", root.join("github-env"));
    for (name, value) in values {
        command.env(name, value);
    }
    command.output().unwrap()
}

#[test]
fn mounted_home_wins_and_existing_credentials_are_masked_before_export() {
    let root = tempfile::tempdir().unwrap();
    let home = root.path().join("mounted cache");
    fs::create_dir_all(home.join("hub")).unwrap();
    let token_path = home.join("token");
    let output = run(
        root.path(),
        &[
            ("HF_HOME", home.to_str().unwrap()),
            ("HF_CACHE", "/obsolete"),
            ("HF_TOKEN", "fixture%token"),
            ("HF_TOKEN_PATH", token_path.to_str().unwrap()),
        ],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).starts_with("::add-mask::fixture%25token\n"));
    let exported = fs::read_to_string(root.path().join("github-env")).unwrap();
    assert!(exported.contains(&format!(
        "HF_HOME={}\nHF_HUB_CACHE={}/hub\nHF_HUB_OFFLINE=1\n",
        home.display(),
        home.display()
    )));
    assert!(exported.contains("HF_TOKEN=fixture%token\n"));
    assert!(!token_path.exists());
}

#[test]
fn missing_mount_and_multiline_credentials_have_no_exports_or_secret_logs() {
    let root = tempfile::tempdir().unwrap();
    let missing = root.path().join("missing");
    let output = run(
        root.path(),
        &[
            ("HF_HOME", missing.to_str().unwrap()),
            ("HF_TOKEN", "private-fixture"),
        ],
    );
    assert!(!output.status.success());
    assert!(!missing.exists());
    assert!(!root.path().join("github-env").exists());
    assert!(!String::from_utf8_lossy(&output.stdout).contains("private-fixture"));
    assert!(!String::from_utf8_lossy(&output.stderr).contains("private-fixture"));
    fs::create_dir_all(root.path().join("hub")).unwrap();
    let output = run(
        root.path(),
        &[
            ("HF_HOME", root.path().to_str().unwrap()),
            ("HF_TOKEN", "private\nINJECT=1"),
        ],
    );
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("invalid multiline value for HF_TOKEN")
    );
    assert!(output.stdout.is_empty());
    assert!(!root.path().join("github-env").exists());
}

#[test]
fn legacy_xdg_and_home_relative_configuration_preserve_existing_cache() {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir_all(root.path().join("huggingface/hub")).unwrap();
    for values in [
        vec![(
            "HF_CACHE",
            root.path().join("huggingface").to_str().unwrap().to_owned(),
        )],
        vec![("XDG_CACHE_HOME", root.path().to_str().unwrap().to_owned())],
        vec![("HF_HOME", "~/huggingface".into())],
    ] {
        let borrowed = values
            .iter()
            .map(|(name, value)| (*name, value.as_str()))
            .collect::<Vec<_>>();
        assert!(run(root.path(), &borrowed).status.success());
    }
    let exports = fs::read_to_string(root.path().join("github-env")).unwrap();
    assert_eq!(exports.matches("HF_HUB_OFFLINE=1\n").count(), 3);
}

#[test]
fn conflicting_existing_hub_override_fails_before_export() {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir_all(root.path().join("hub")).unwrap();
    fs::create_dir_all(root.path().join("other")).unwrap();
    let other = root.path().join("other");
    let output = run(
        root.path(),
        &[
            ("HF_HOME", root.path().to_str().unwrap()),
            ("HF_HUB_CACHE", other.to_str().unwrap()),
        ],
    );
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("HF_HUB_CACHE must resolve"));
    assert!(!root.path().join("github-env").exists());
}

#[cfg(unix)]
#[test]
fn symlinked_mount_identity_is_accepted_without_rewriting_exports() {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir_all(root.path().join("mount/hub")).unwrap();
    std::os::unix::fs::symlink(root.path().join("mount"), root.path().join("alias")).unwrap();
    let alias = root.path().join("alias");
    let hub = root.path().join("mount/hub");
    let output = run(
        root.path(),
        &[
            ("HF_HOME", alias.to_str().unwrap()),
            ("HF_HUB_CACHE", hub.to_str().unwrap()),
        ],
    );
    assert!(output.status.success());
    assert!(
        fs::read_to_string(root.path().join("github-env"))
            .unwrap()
            .contains(&format!("HF_HOME={}\n", alias.display()))
    );
}
