use super::*;

#[test]
fn action_reference_evidence_binds_actual_job_and_step_locations() {
    let source = "jobs:\n  build:\n    uses: owner/repo/workflow.yml@abc # v1\n  smoke:\n    steps:\n      - uses: 'owner/action@def' # v2\n";
    let (document, references) = parse_with_actions(source).unwrap();
    assert_eq!(document, parse(source).unwrap());
    assert_eq!(references.len(), 2);
    assert_eq!(references[0].reference, "owner/repo/workflow.yml@abc");
    assert_eq!(references[0].provenance.as_deref(), Some("v1"));
    assert_eq!(references[0].line, 3);
    assert_eq!(references[1].reference, "owner/action@def");
    assert_eq!(references[1].provenance.as_deref(), Some("v2"));
    assert_eq!(references[1].line, 6);
}

#[test]
fn composite_actions_preserve_reference_comments_and_quoted_hashes() {
    let source = "runs:\n  using: composite\n  steps:\n    - name: fixture\n      uses: \"owner/action@abc#part\" # version\n";
    let (_, references) = parse_with_actions(source).unwrap();
    assert_eq!(references.len(), 1);
    assert_eq!(references[0].reference, "owner/action@abc#part");
    assert_eq!(references[0].provenance.as_deref(), Some("version"));
    assert_eq!(references[0].line, 5);
}

#[test]
fn shell_text_and_unrelated_mapping_keys_cannot_supply_action_evidence() {
    let source = "inputs:\n  uses: owner/fake@abc # wrong\njobs:\n  build:\n    steps:\n      - uses: owner/action@abc\n        with:\n          uses: owner/action@abc # fake version\n      - run: |\n          uses: owner/action@abc # fake version\n";
    let (_, references) = parse_with_actions(source).unwrap();
    assert_eq!(references.len(), 1);
    assert_eq!(references[0].provenance, None);
    assert_eq!(references[0].line, 6);
}

#[test]
fn block_scalar_reference_has_no_comment_provenance() {
    let source =
        "jobs:\n  build:\n    steps:\n      - uses: |\n          owner/action@abc # data\n";
    let (_, references) = parse_with_actions(source).unwrap();
    assert_eq!(references.len(), 1);
    assert_eq!(references[0].provenance, None);
    assert!(references[0].reference.contains("# data"));
}

#[test]
fn duplicate_structural_keys_are_refused_before_evidence_is_accepted() {
    let source = "jobs:\n  build:\n    steps:\n      - uses: owner/action@abc # version\n        uses: owner/action@def # another\n";
    assert!(parse_with_actions(source).is_err());
}
