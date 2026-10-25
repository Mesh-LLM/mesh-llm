use super::support::CaseFixture;
use std::error::Error;

#[test]
fn complete_source_bound_package_is_green() -> Result<(), Box<dyn Error>> {
    let given = CaseFixture::complete()?;
    given.reverify_package()?;
    let input_hashes = given.input_hashes()?;
    let report = given.run_rust()?;
    assert_eq!(given.input_hashes()?, input_hashes);
    assert!(report.green);
    assert_eq!(report.passed_count, 2);
    assert!(report.outputs.is_some());
    assert!(!report.summary.is_empty());
    Ok(())
}

#[test]
fn newest_failure_withholds_green_without_fallback() -> Result<(), Box<dyn Error>> {
    let given = CaseFixture::newest_failure()?;
    given.reverify_package()?;
    let report = given.run_rust()?;
    assert!(!report.green);
    assert_eq!(report.passed_count, 1);
    assert_eq!(report.dense_attempt.as_deref(), Some("3"));
    assert_eq!(report.outputs, None);
    Ok(())
}
