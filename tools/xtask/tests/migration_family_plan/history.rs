use super::*;

/// The original source bytes paired with real-* captures, never current policy.
pub(super) fn run_historical(args: &[&str]) -> Output {
    let directory = tempfile::tempdir().expect("historical selected source");
    let selected = directory.path().canonicalize().expect("selected root");
    fs::create_dir_all(selected.join("ci/llama-canary")).unwrap();
    fs::create_dir_all(selected.join("tools/xtask/tests/fixtures/family_evidence")).unwrap();
    fs::write(selected.join("Cargo.toml"), b"").unwrap();
    fs::write(selected.join("tools/xtask/Cargo.toml"), b"").unwrap();
    let manifest = fixture("historical-manifest", "json");
    let historical: Value = serde_json::from_slice(&fixture("real-4", "stdout")).unwrap();
    assert_eq!(
        historical["manifest_sha256"],
        hex::encode(Sha256::digest(&manifest))
    );
    fs::write(
        selected.join("ci/llama-canary/family-certified.json"),
        manifest,
    )
    .unwrap();
    fs::write(
        selected.join("tools/xtask/tests/fixtures/family_evidence/real-4.stdout"),
        fixture("real-4", "stdout"),
    )
    .unwrap();
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(root())
        .args([
            "--repo-root",
            selected.to_str().unwrap(),
            "ci",
            "family-plan",
        ])
        .args(args)
        .output()
        .expect("historical plan execution")
}

#[test]
fn current_canonical_roster_identity_and_evidence_are_preserved_per_family() {
    let manifest_bytes = fs::read(root().join("ci/llama-canary/family-certified.json")).unwrap();
    let manifest: Value = serde_json::from_slice(&manifest_bytes).unwrap();
    let output = run(&["--shard-count", "256"]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let plan: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        plan["manifest_sha256"],
        hex::encode(Sha256::digest(&manifest_bytes))
    );
    let models = manifest["models"].as_array().unwrap();
    let selected = plan["selected_models"].as_array().unwrap();
    assert_eq!(selected.len(), models.len());
    let matrix = plan["github_matrix"]["include"].as_array().unwrap();
    assert_eq!(matrix.len(), models.len());
    for model in models {
        let family = model["family"].as_str().unwrap();
        let rows: Vec<_> = selected
            .iter()
            .filter(|row| row["family"] == family)
            .collect();
        assert_eq!(rows.len(), 1, "{family}");
        let row = rows[0];
        for field in ["artifact", "evidence", "class", "architecture", "profile"] {
            assert_eq!(row[field], model[field], "{family}.{field}");
        }
        for (field, value) in model["execution"].as_object().unwrap() {
            assert_eq!(
                &row["execution"][field], value,
                "{family}.execution.{field}"
            );
        }
        assert_eq!(
            matrix
                .iter()
                .filter(|row| row["families"] == family)
                .count(),
            1,
            "{family}"
        );
    }
    let supplied = tempfile::NamedTempFile::new().unwrap();
    fs::write(supplied.path(), output.stdout).unwrap();
    let verified = run(&[
        "--verify-plan",
        supplied.path().to_str().unwrap(),
        "--shard-count",
        "0",
    ]);
    assert_eq!(verified.status.code(), Some(0));
    assert!(verified.stdout.is_empty() && verified.stderr.is_empty());
}
