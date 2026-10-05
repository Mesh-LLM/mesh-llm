//! Compile-once admission and complete quantization handoff contracts.
mod fixture;
use fixture::{fixture, run};
use std::fs;

const BUILD: &str = "Build and locate the SafeTensors smoke test";
const EXERCISE: &str = "Exercise every current direct-load quantization";

#[test]
fn safetensors_smoke_selects_the_actual_workspace_owner_and_compiles_once() {
    for adapter in [false, true] {
        let fixture = fixture(adapter, "none");
        let output = run(&fixture, BUILD, adapter, "none", "[]", "");
        assert!(output.status.success(), "{output:?}");
        let calls = fs::read_to_string(fixture.path().join("cargo-calls")).unwrap();
        assert_eq!(calls.lines().count(), 2);
        assert_eq!(
            calls
                .lines()
                .filter(|line| line.starts_with("test "))
                .count(),
            1
        );
        let outputs = fixture.outputs();
        assert_eq!(
            outputs["test_name"],
            if adapter {
                fixture::ADAPTER
            } else {
                fixture::HOST
            }
        );
        assert!(String::from_utf8_lossy(&output.stderr).contains("finite compiler diagnostic"));
    }
}

#[test]
fn safetensors_smoke_refuses_compiler_or_test_admission_failures_before_output() {
    for failure in [
        "compile",
        "wrong-target",
        "not-test",
        "missing-binary",
        "listing",
        "absent-test",
    ] {
        let fixture = fixture(false, failure);
        let output = run(&fixture, BUILD, false, failure, "[]", "");
        assert!(!output.status.success(), "failure={failure}: {output:?}");
        assert!(!fixture.path().join("outputs").exists());
        assert!(!fixture.path().join("measured-calls").exists());
    }
}

#[test]
fn safetensors_smoke_reuses_the_selected_binary_for_every_quantization() {
    let fixture = fixture(true, "none");
    let output = run(
        &fixture,
        EXERCISE,
        true,
        "none",
        r#"["F16","Q8_0","Q4_K_M"]"#,
        "",
    );
    assert!(output.status.success(), "{output:?}");
    assert_eq!(
        fs::read_to_string(fixture.path().join("measured-calls")).unwrap(),
        "F16\nQ8_0\nQ4_K_M\n"
    );
    assert!(!fixture.path().join("cargo-calls").exists());
}

#[test]
fn safetensors_smoke_stops_at_the_first_failed_quantization() {
    let fixture = fixture(false, "none");
    let output = run(
        &fixture,
        EXERCISE,
        false,
        "none",
        r#"["F16","Q8_0","Q4_K_M"]"#,
        "Q8_0",
    );
    assert_eq!(output.status.code(), Some(45));
    assert_eq!(
        fs::read_to_string(fixture.path().join("measured-calls")).unwrap(),
        "F16\nQ8_0\n"
    );
}

#[test]
fn safetensors_smoke_refuses_invalid_or_empty_quantization_inputs_before_execution() {
    for quantizations in [
        "broken-json",
        "[]",
        "{}",
        r#"["F16",null]"#,
        r#"[""]"#,
        r#"["F16\nQ8_0"]"#,
    ] {
        let fixture = fixture(false, "none");
        let output = run(&fixture, EXERCISE, false, "none", quantizations, "");
        assert!(
            !output.status.success(),
            "input={quantizations}: {output:?}"
        );
        assert!(!fixture.path().join("measured-calls").exists());
    }
}
