#[path = "migration_patch_bytes/cases.rs"]
mod cases;
#[path = "../src/automation/rewriter_patch.rs"]
mod rewriter_patch;

use cases::{Case, LF_PATCH, SUBJECT};
use rewriter_patch::{PatchError, encode_mail_patch};

#[test]
fn mail_patch_matches_consumed_golden() {
    let encoded = encode_mail_patch(SUBJECT, Case::Lf.diff()).unwrap();
    assert_eq!(encoded.bytes, LF_PATCH);
}

#[test]
fn diff_payload_preserves_crlf_and_utf8_bytes() {
    for case in [Case::UnicodeCrlf, Case::ChangedHunk] {
        let encoded = encode_mail_patch(SUBJECT, case.diff()).unwrap();
        assert_eq!(encoded.diff_sha256, case.digest().unwrap());
        assert!(
            encoded
                .bytes
                .windows(case.diff().len())
                .any(|window| window == case.diff())
        );
    }
}

#[test]
fn empty_diff_is_rejected() {
    assert!(matches!(
        encode_mail_patch(SUBJECT, Case::Empty.diff()),
        Err(PatchError::EmptyDiff)
    ));
}

#[test]
fn malformed_utf8_is_rejected() {
    assert!(matches!(
        encode_mail_patch(SUBJECT, Case::Malformed.diff()),
        Err(PatchError::InvalidUtf8)
    ));
}
