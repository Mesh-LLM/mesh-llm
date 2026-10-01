use super::support::{Stage, TestResult, valid};

#[test]
fn malformed_unicode_is_rejected_even_in_ignored_fields() -> TestResult {
    let stage = Stage::new()?;
    let raw = valid().replacen('{', r#"{"ignored":"\ud800","#, 1);
    let actual = stage.input(raw.as_bytes())?;
    assert_eq!(actual.code, 1);
    assert!(actual.stdout.is_empty());
    Ok(())
}

#[test]
fn valid_supplementary_unicode_is_accepted() -> TestResult {
    let stage = Stage::new()?;
    let raw = valid().replacen('{', r#"{"ignored":"\ud83d\ude00","#, 1);
    let actual = stage.input(raw.as_bytes())?;
    assert_eq!(actual.code, 0);
    Ok(())
}
