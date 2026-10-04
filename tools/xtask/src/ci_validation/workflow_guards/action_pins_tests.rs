use super::*;

fn action(reference: &str, provenance: Option<&str>) -> ActionReference {
    ActionReference {
        reference: reference.into(),
        provenance: provenance.map(str::to_owned),
        line: 1,
    }
}

#[test]
fn immutable_external_reference_requires_full_sha_and_real_comment() {
    let reference = format!("owner/action@{}", "a".repeat(40));
    assert!(validate("fixture.yml", &action(&reference, Some("v1"))).is_ok());
    assert!(validate("fixture.yml", &action(&reference, None)).is_err());
    for invalid in [
        "owner/action@main",
        "owner/action@v1",
        "owner/action@abc",
        "owner/action",
        "@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    ] {
        assert!(validate("fixture.yml", &action(invalid, Some("v1"))).is_err());
    }
}

#[test]
fn protected_main_reference_is_bound_to_its_matching_entrypoint() {
    for lane in ["quality", "website", "linux", "macos", "windows"] {
        let reference = format!("Mesh-LLM/mesh-llm/.github/workflows/ci-{lane}-lane.yml@main");
        assert!(validate(&format!("pr_{lane}.yml"), &action(&reference, None)).is_ok());
        assert!(validate("other.yml", &action(&reference, Some("comment"))).is_err());
    }
    let reference = "Mesh-LLM/mesh-llm/.github/workflows/ci-pr-canary-lane.yml@main";
    assert!(validate("pr_ci_canary.yml", &action(reference, None)).is_ok());
    assert!(validate("pr_linux.yml", &action(reference, None)).is_err());
}

#[test]
fn pre_checkout_reference_keeps_exact_pin_and_bounded_callers() {
    for caller in PRE_CHECKOUT_CALLERS {
        assert!(validate(caller, &action(PRE_CHECKOUT, None)).is_ok());
    }
    assert!(validate("unapproved.yml", &action(PRE_CHECKOUT, Some("comment"))).is_err());
    assert!(
        validate(
            "ci-quality-slice.yml",
            &action(&PRE_CHECKOUT.replace("ed07043b", "main"), None)
        )
        .is_err()
    );
}

#[test]
fn local_actions_remain_local_and_shell_data_cannot_supply_provenance() {
    assert!(validate("fixture.yml", &action("./.github/actions/prepare", None)).is_ok());
    let reference = format!("owner/action@{}", "a".repeat(40));
    let source = format!(
        "jobs:\n  build:\n    steps:\n      - uses: {reference}\n      - run: |\n          uses: {reference} # v1\n"
    );
    let (_, references) = workflow_yaml::parse_with_actions(&source).unwrap();
    assert_eq!(references.len(), 1);
    assert!(validate("fixture.yml", &references[0]).is_err());
}

#[test]
fn checked_in_workflows_and_composite_actions_keep_pin_provenance() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    check(&root).unwrap();
}

#[test]
fn unsupported_job_and_step_structures_cannot_hide_external_references() {
    for source in [
        "jobs: {build: {uses: owner/action@main}}\n",
        "jobs:\n  build:\n    steps: [{uses: owner/action@main}]\n",
        "runs:\n  using: composite\n  steps: [{uses: owner/action@main}]\n",
        "jobs:\n  build:\n    steps: *aliased_steps\n",
    ] {
        let (document, _) = workflow_yaml::parse_with_actions(source).unwrap();
        assert!(execution_structure(&document).is_err(), "{source}");
    }
}

#[test]
fn reusable_jobs_and_javascript_actions_do_not_require_a_steps_sequence() {
    for source in [
        "jobs:\n  call:\n    uses: ./workflow.yml\n",
        "runs:\n  using: node24\n  main: index.js\n",
    ] {
        let (document, _) = workflow_yaml::parse_with_actions(source).unwrap();
        assert!(execution_structure(&document).is_ok());
    }
}
