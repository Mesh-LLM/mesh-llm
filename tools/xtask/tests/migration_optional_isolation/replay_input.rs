use super::support::*;

#[test]
fn invalid_json_is_rejected_before_policy() -> TestResult {
    let stage = Stage::new()?;
    for raw in [
        b"{\"replay\":null,\"ignored\":!}".as_slice(),
        b"{} trailing",
        b"\xff",
        b"{\"ignored\":NaN}",
    ] {
        let actual = stage.input(raw)?;
        assert_eq!(actual.code, 1);
        assert!(actual.stdout.is_empty());
        assert!(!actual.stderr.is_empty());
    }
    Ok(())
}

#[test]
fn excessive_nesting_is_rejected_without_crashing() -> TestResult {
    let stage = Stage::new()?;
    let deep = format!("{}0{}", "[".repeat(500), "]".repeat(500));
    let actual = stage.input(deep.as_bytes())?;
    assert_eq!(actual.code, 1);
    assert!(actual.stdout.is_empty());
    Ok(())
}

#[test]
fn valid_policy_accepts_ignored_json_extensions() -> TestResult {
    let stage = Stage::new()?;
    let raw = valid().replacen('{', r#"{"ignored":[null,1,{"field":"value"}],"#, 1);
    assert_outcome(
        "ignored extension",
        &stage.input(raw.as_bytes())?,
        (0, SHELL, ""),
    );
    Ok(())
}

#[test]
fn command_isolation_preserves_unrelated_file() -> TestResult {
    let stage = Stage::new()?;
    let sentinel = stage.cwd().join("sentinel");
    std::fs::write(&sentinel, b"unchanged")?;
    assert_outcome(
        "relative input",
        &stage.input(valid().as_bytes())?,
        (0, SHELL, ""),
    );
    assert_eq!(std::fs::read(sentinel)?, b"unchanged");
    Ok(())
}
