use super::numeric::{generate, replace};
use super::*;

#[test]
fn malformed_unicode_is_rejected_at_the_json_boundary() {
    let raw = replace(
        &fixture("synthetic-manifest", "json"),
        b"\"notes\": \"synthetic parity only\"",
        br#""notes": "\ud800""#,
    );
    let output = generate(&raw, "zeta");
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
}
