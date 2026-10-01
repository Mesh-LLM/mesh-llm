use super::support::{DENSE, context, family};
use crate::canary_receipts::{Error, ErrorKind, validate_results};

fn check(bytes: &[u8]) -> Result<(), Error> {
    let context = context();
    let family = family("dense");
    validate_results(bytes, &family, context.model(&family).unwrap())
}

#[test]
fn migration_canary_receipts_accepts_literal_separator_1c() {
    let given = include_bytes!("fixtures/separator-1c.jsonl");
    let when = check(given);
    assert_eq!(when.unwrap_err().kind, ErrorKind::Json);
}

#[test]
fn migration_canary_receipts_accepts_literal_separator_1d() {
    let given = include_bytes!("fixtures/separator-1d.jsonl");
    let when = check(given);
    assert_eq!(when.unwrap_err().kind, ErrorKind::Json);
}

#[test]
fn migration_canary_receipts_accepts_literal_separator_1e() {
    let given = include_bytes!("fixtures/separator-1e.jsonl");
    let when = check(given);
    assert_eq!(when.unwrap_err().kind, ErrorKind::Json);
}

#[test]
fn migration_canary_receipts_accepts_literal_separator_1f() {
    let given = include_bytes!("fixtures/separator-1f.jsonl");
    let when = check(given);
    assert_eq!(when.unwrap_err().kind, ErrorKind::Json);
}

#[test]
fn migration_canary_receipts_rejects_raw_controls_inside_objects() {
    for separator in [0x1c, 0x1d, 0x1e, 0x1f] {
        let mut given = DENSE.to_vec();
        given.insert(1, separator);
        let when = check(&given);
        assert_eq!(when.unwrap_err().kind, ErrorKind::Json, "{separator:02x}");
    }
}
