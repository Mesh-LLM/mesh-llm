use super::*;
use std::path::PathBuf;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .into()
}
fn actual() -> Node {
    workflow_yaml::parse(&fs::read_to_string(root().join(ACTION)).unwrap()).unwrap()
}
fn field<'a>(node: &'a mut Node, name: &str) -> &'a mut Node {
    let Node::Map(entries) = node else {
        panic!("expected action mapping");
    };
    &mut entries
        .iter_mut()
        .find(|(key, _)| key == name)
        .expect("owning field")
        .1
}
fn steps(action: &mut Node) -> &mut Vec<Node> {
    let Node::Seq(steps) = field(field(action, "runs"), "steps") else {
        panic!("expected action steps");
    };
    steps
}
fn mutate_run(action: &mut Node, run: &str) {
    let step = steps(action)
        .iter_mut()
        .find(|step| step.get("id").and_then(Node::text) == Some("derive"))
        .unwrap();
    *field(step, "run") = Node::Scalar(run.into());
}
fn admit(action: &Node) -> DynResult<()> {
    check_document(action, &root().join(DERIVE))
}

#[test]
fn actual_compute_changes_action_and_sibling_script_fit_startup_budget() {
    check(&root()).unwrap();
}
#[test]
fn actual_run_budget_rejects_21000_and_21001_but_accepts_20999() {
    for (size, accepted) in [(20999, true), (21000, false), (21001, false)] {
        let mut action = actual();
        mutate_run(&mut action, &"x".repeat(size));
        assert_eq!(admit(&action).is_ok(), accepted, "size={size}");
    }
}
#[test]
fn expanded_sha_and_longest_supported_event_determine_budget() {
    let expression = "${{ inputs.event_name }}${{ inputs.base_sha }}${{inputs.head_sha}}";
    let substitution = LONGEST_EVENT.len() + SHA_LENGTH * 2;
    for (size, accepted) in [(20999, true), (21000, false), (21001, false)] {
        let mut action = actual();
        mutate_run(
            &mut action,
            &format!("{}{expression}", "x".repeat(size - substitution)),
        );
        assert_eq!(admit(&action).is_ok(), accepted, "expanded={size}");
    }
    // Count source characters, including non-ASCII text, and every occurrence.
    assert_eq!(
        expanded_length("é${{ inputs.head_sha }}${{ inputs.head_sha }}").unwrap(),
        81
    );
    assert_eq!(expanded_length("${{inputs.event_name}}").unwrap(), 17);
}
#[test]
fn unknown_nested_or_unterminated_run_expressions_fail_closed() {
    for expression in [
        "${{ github.event_name }}",
        "${{ inputs.unknown }}",
        "${{ inputs.head_sha || '' }}",
        "${{ inputs.head_sha",
        "${{ ${{inputs.head_sha}} }}",
    ] {
        let mut action = actual();
        mutate_run(&mut action, expression);
        assert!(admit(&action).is_err(), "accepted {expression}");
    }
}
#[test]
fn actual_action_requires_derive_step_and_real_sibling_file() {
    let mut action = actual();
    steps(&mut action).retain(|step| step.get("id").and_then(Node::text) != Some("derive"));
    assert!(
        admit(&action)
            .unwrap_err()
            .to_string()
            .contains("derive step")
    );
    let directory = tempfile::tempdir().unwrap();
    let missing = directory.path().join("derive-outputs.sh");
    assert!(
        check_document(&actual(), &missing)
            .unwrap_err()
            .to_string()
            .contains("sibling")
    );
    fs::create_dir(&missing).unwrap();
    assert!(check_document(&actual(), &missing).is_err());
    fs::remove_dir(&missing).unwrap();
    fs::write(&missing, "source-owned derive fixture").unwrap();
    check_document(&actual(), &missing).unwrap();
}
#[test]
fn non_scalar_run_and_non_sequence_steps_are_rejected() {
    let mut action = actual();
    let derive = steps(&mut action)
        .iter_mut()
        .find(|step| step.get("id").and_then(Node::text) == Some("derive"))
        .unwrap();
    *field(derive, "run") = Node::Map(Vec::new());
    assert!(admit(&action).is_err());
    *field(field(&mut action, "runs"), "steps") = Node::Scalar("invalid steps".into());
    assert!(admit(&action).is_err());
}
