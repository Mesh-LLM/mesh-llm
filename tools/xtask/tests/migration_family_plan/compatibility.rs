use super::numeric::{generate, replace};
use super::*;

#[test]
fn utf8_manifest_digest_binds_original_bytes() {
    let raw = fixture("synthetic-manifest", "json");
    let output = generate(&raw, "zeta");
    assert_eq!(output.status.code(), Some(0));
    let plan: Value = serde_json::from_slice(&output.stdout).expect("plan");
    assert_eq!(plan["manifest_sha256"], hex::encode(Sha256::digest(&raw)));
}

#[test]
fn plan_keeps_consumed_ascii_escape_format() {
    let raw = replace(
        &fixture("synthetic-manifest", "json"),
        b"\"notes\": \"synthetic parity only\"",
        br#""notes": "x\u007fy""#,
    );
    let output = generate(&raw, "zeta");
    assert_eq!(output.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&output.stdout).contains(r#""notes": "x\u007fy""#));
    assert!(!output.stdout.contains(&0x7f));
}

#[test]
fn help_succeeds_without_loading_manifest() {
    let output = run(&["--help"]);
    assert_eq!(output.status.code(), Some(0));
    assert!(output.stderr.is_empty());
    assert!(!output.stdout.is_empty());
}
