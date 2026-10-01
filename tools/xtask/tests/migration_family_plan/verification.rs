use super::numeric::replace;
use super::*;

#[test]
fn supplied_plan_rejects_numeric_type_substitution() {
    for (needle, replacement) in [
        (
            b"\"schema_version\": 1".as_slice(),
            b"\"schema_version\": true".as_slice(),
        ),
        (
            b"\"selected_family_count\": 2",
            b"\"selected_family_count\": 2.0",
        ),
        (b"\"mtp_layers\": 0", b"\"mtp_layers\": false"),
    ] {
        let path = temp_path("substitution.json");
        fs::write(
            &path,
            replace(&fixture("real-reversed", "stdout"), needle, replacement),
        )
        .expect("plan");
        let output = run(&["--verify-plan", path.to_str().expect("path")]);
        assert_eq!(output.status.code(), Some(2));
        assert!(output.stdout.is_empty());
        fs::remove_file(path).expect("cleanup");
    }
}

#[test]
fn malformed_supplied_plan_is_a_runtime_failure() {
    let path = temp_path("invalid-plan.json");
    fs::write(&path, b"{").expect("plan");
    let output = run(&["--verify-plan", path.to_str().expect("path")]);
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    fs::remove_file(path).expect("cleanup");
}

#[test]
fn missing_supplied_plan_is_a_runtime_failure() {
    let path = temp_path("absent-plan.json");
    let output = run(&["--verify-plan", path.to_str().expect("path")]);
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
}

#[test]
fn output_write_failure_exits_one_when_parent_is_a_file() {
    let parent = temp_path("output-parent");
    fs::write(&parent, b"not a directory").expect("blocked parent");
    let path = parent.join("plan.json");
    let output = run(&[
        "--families",
        "llama",
        "--output",
        path.to_str().expect("path"),
    ]);
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    fs::remove_file(parent).expect("cleanup");
}
