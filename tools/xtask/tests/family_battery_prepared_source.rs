#![cfg(unix)]
use std::{fs, os::unix::fs::PermissionsExt, path::PathBuf, process::Command};

fn words(path: &std::path::Path) -> Vec<String> {
    fs::read(path)
        .unwrap()
        .split(|byte| *byte == 0)
        .filter(|word| !word.is_empty())
        .map(|word| String::from_utf8(word.to_vec()).unwrap())
        .collect()
}

fn write_owner(owner: &std::path::Path) {
    fs::write(
        owner,
        concat!(
            "#!/bin/sh\n",
            "if [ \"$2\" = canary-receipts ]; then\n",
            "  printf '%s\\0' \"$@\" > \"$PROVENANCE_ARGS\"\n",
            "  if [ \"$SOURCE_STATUS\" != 0 ]; then exit \"$SOURCE_STATUS\"; fi\n",
            "  printf '%s\n' aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n",
            "else\n",
            "  printf '%s\\0' \"$@\" > \"$VERIFY_ARGS\"\n",
            "  exit \"$VERIFY_STATUS\"\n",
            "fi\n"
        ),
    )
    .unwrap();
    fs::set_permissions(owner, fs::Permissions::from_mode(0o700)).unwrap();
}

fn assert_evidence_arguments(verify: &std::path::Path) {
    assert_eq!(
        words(verify),
        [
            "automation",
            "workload-oracle-evidence",
            "verify",
            "--evidence",
            "evidence with spaces.json",
            "--class",
            "embedding",
            "--smoke-lane",
            "embedding-smoke",
            "--oracle-lane",
            "embedding-oracle",
            "--model-id",
            "model id",
            "--model-path",
            "/selected/model with spaces.gguf",
            "--candidate-executable",
            "/selected/candidate",
            "--oracle-executable",
            "/selected/oracle",
            "--pinned-patch-sha",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "--projector-path",
            "/selected/projector"
        ]
    );
}

#[test]
fn selected_source_provenance_gate_precedes_evidence_and_preserves_owner_failure() {
    let source = fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../scripts/skippy-family-battery.sh"),
    )
    .unwrap();
    let function = source
        .split("verify_workload_oracle() {")
        .nth(1)
        .unwrap()
        .split("\nrun_workload_certify() {")
        .next()
        .unwrap();
    let shell = format!(
        "verify_workload_oracle() {{{function}\nautomation=(\"$OWNER\")\nverify_workload_oracle 'evidence with spaces.json' embedding embedding-smoke embedding-oracle 'model id' '/selected/model with spaces.gguf' '/selected/candidate' '/selected/oracle' '/selected/projector'\n"
    );
    let state = tempfile::tempdir().unwrap();
    let selected = state.path().join("selected source with spaces");
    fs::create_dir(&selected).unwrap();
    let owner = state.path().join("frozen owner with spaces");
    write_owner(&owner);
    let provenance = state.path().join("provenance args");
    let verify = state.path().join("verify args");
    for (source_status, verify_status, expected_status) in [(31, 0, 31), (0, 23, 23), (0, 0, 0)] {
        if verify.exists() {
            fs::remove_file(&verify).unwrap();
        }
        let output = Command::new("/bin/bash")
            .args(["-c", &shell])
            .env("ROOT", &selected)
            .env("OWNER", &owner)
            .env("PATH", "")
            .env("PROVENANCE_ARGS", &provenance)
            .env("VERIFY_ARGS", &verify)
            .env("SOURCE_STATUS", source_status.to_string())
            .env("VERIFY_STATUS", verify_status.to_string())
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(expected_status));
        assert!(output.stdout.is_empty());
        assert!(output.stderr.is_empty());
        assert_eq!(
            words(&provenance),
            [
                "automation",
                "canary-receipts",
                "prepared-source",
                "--root",
                selected.to_str().unwrap()
            ]
        );
        if source_status != 0 {
            assert!(!verify.exists());
            continue;
        }
        assert_evidence_arguments(&verify);
    }
}
