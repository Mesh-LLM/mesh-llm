use super::export_cases;
use super::export_process;
use super::support::TestResult;

#[test]
fn export_writes_json_and_appends_environment_fields() -> TestResult {
    let case = export_cases::roster()
        .into_iter()
        .find(|case| case.id == "P03")
        .ok_or("missing complete export case")?;
    let actual = export_process::observe(&case, None)?;
    assert_eq!(actual.status, 0);
    let json = actual
        .files
        .get("params.json")
        .and_then(Option::as_ref)
        .ok_or("missing JSON")?;
    let document: serde_json::Value = serde_json::from_slice(json)?;
    assert!(document.is_object());
    let environment = actual
        .files
        .get("github.env")
        .and_then(Option::as_ref)
        .ok_or("missing environment")?;
    assert_eq!(std::str::from_utf8(environment)?.lines().count(), 12);
    Ok(())
}
