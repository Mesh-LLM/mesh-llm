use super::numeric::{generate, replace};
use super::*;

#[test]
fn object_tokens_cannot_supply_manifest_integers() {
    let raw = replace(
        &fixture("synthetic-manifest", "json"),
        b"\"mtp_layers\": 0",
        br#""mtp_layers": {"$serde_json::private::Number":"1"}"#,
    );
    let output = generate(&raw, "zeta");
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
}
