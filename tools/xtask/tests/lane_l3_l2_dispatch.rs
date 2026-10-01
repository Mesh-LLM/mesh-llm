use std::process::Command;

fn xtask() -> Command {
    Command::new(env!("CARGO_BIN_EXE_xtask"))
}

#[test]
fn authority_dispatch_rejects_loopback_without_disclosing_endpoint() {
    let output = xtask()
        .args(["ci-ops", "authority-audit", "endpoint", "L3_ENDPOINT"])
        .env("L3_ENDPOINT", "http://[::ffff:127.0.0.1]:80/private-token")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert_eq!(
        String::from_utf8(output.stderr).unwrap(),
        "L3_ENDPOINT: loopback\n"
    );
}

#[test]
fn authority_dispatch_accepts_remote_endpoint_without_output() {
    let output = xtask()
        .args(["ci-ops", "authority-audit", "endpoint", "L3_ENDPOINT"])
        .env("L3_ENDPOINT", "http://[2001:db8::1]:80/cache")
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(output.stdout.is_empty());
    assert!(output.stderr.is_empty());
}

#[test]
fn authority_dispatch_rejects_depot_auth_environment() {
    let temporary = tempfile::tempdir().unwrap();
    let output = xtask()
        .args(["ci-ops", "authority-audit", "docker"])
        .arg(temporary.path().join("config.json"))
        .args(["--depot-selected", "true"])
        .env("DOCKER_AUTH_CONFIG", "{}")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert_eq!(
        String::from_utf8(output.stderr).unwrap(),
        "DOCKER_AUTH_CONFIG/config.json: authentication\n"
    );
}

#[test]
fn registry_dispatch_writes_negative_report_before_enforcement() {
    let temporary = tempfile::tempdir().unwrap();
    let inputs = temporary.path().join("observations");
    std::fs::create_dir(&inputs).unwrap();
    for source in ["upstream", "depot"] {
        for sample in 1..=5 {
            let item = serde_json::json!({
                "source": source,
                "sample": sample,
                "elapsed_ms": if source == "upstream" { 20_000 } else { 14_000 },
                "digest": format!("sha256:{}", "a".repeat(64)),
            });
            std::fs::write(
                inputs.join(format!("{source}-{sample}.json")),
                serde_json::to_vec(&item).unwrap(),
            )
            .unwrap();
        }
    }
    let report = temporary.path().join("summary.json");
    let output = xtask()
        .args(["ci-ops", "registry-pulls"])
        .arg(&inputs)
        .args(["--enforce", "--json-out"])
        .arg(&report)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let document: serde_json::Value =
        serde_json::from_slice(&std::fs::read(report).unwrap()).unwrap();
    assert_eq!(document["eligible"], false);
    assert_eq!(document["samples_per_source"]["upstream"], 5);
    assert!(
        String::from_utf8(output.stdout)
            .unwrap()
            .contains("| Adoption gate | fail |")
    );
}
