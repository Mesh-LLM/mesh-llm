use super::*;
use crate::ci_validation::lane_results::workflow_yaml;
fn document(source: &str) -> Node {
    workflow_yaml::parse(source).unwrap()
}
#[test]
fn native_entry_effective_checks_none_follows_job_override_and_workflow_inheritance() {
    for (declaration, expected) in [
        ("  contents: read\n", true),
        ("  checks: none\n", true),
        ("  checks: read\n", false),
        ("  checks: write\n", false),
    ] {
        let source = format!(
            "permissions: write-all\njobs:\n  plan:\n    permissions:\n{}",
            declaration.replace("  ", "      ")
        );
        let tree = document(&source);
        let job = tree.get("jobs").unwrap().get("plan").unwrap();
        assert_eq!(effective_none(&tree, job, "checks").unwrap(), expected);
    }
    for (workflow, expected) in [
        ("{}", true),
        ("{checks: none}", true),
        ("{contents: read}", true),
        ("read-all", false),
        ("write-all", false),
    ] {
        let tree = document(&format!("permissions: {workflow}\njobs:\n  plan: {{}}\n"));
        assert_eq!(
            effective_none(
                &tree,
                tree.get("jobs").unwrap().get("plan").unwrap(),
                "checks"
            )
            .unwrap(),
            expected
        );
    }
    let tree = document("jobs:\n  plan: {}\n");
    assert!(
        effective_none(
            &tree,
            tree.get("jobs").unwrap().get("plan").unwrap(),
            "checks"
        )
        .is_err()
    );
}
#[test]
fn native_entry_effective_checks_none_refuses_ambiguous_permissions() {
    for source in [
        "permissions: {checks: write, checks: none}\n",
        "permissions: {checks: none, checks: write}\n",
        "permissions: invalid\n",
    ] {
        let tree = document(source);
        assert!(effective_none(&tree, &Node::Map(vec![]), "checks").is_err());
    }
    let declaration = Node::Map(vec![
        ("checks".into(), Node::Scalar("write".into())),
        ("checks".into(), Node::Scalar("none".into())),
    ]);
    assert!(parse(&declaration).is_err());
}

#[test]
fn native_entry_effective_checks_none_normalizes_simple_quoted_keys() {
    for (permissions, expected) in [
        ("{'checks': write}", false),
        (r#"{"checks": write}"#, false),
        ("{'checks': none}", true),
        (r#"{"checks": none}"#, true),
        ("{'contents': read}", true),
        (r#"{"contents": read}"#, true),
    ] {
        let tree = document(&format!("permissions: {permissions}\n"));
        assert_eq!(
            effective_none(&tree, &Node::Map(vec![]), "checks").unwrap(),
            expected
        );
    }
    for name in ["'checks'", "\"checks\""] {
        let tree = Node::Map(vec![(
            "permissions".into(),
            Node::Map(vec![(name.into(), Node::Scalar("write".into()))]),
        )]);
        assert!(!effective_none(&tree, &Node::Map(vec![]), "checks").unwrap());
    }
}
#[test]
fn native_entry_effective_checks_none_refuses_quoted_duplicates_and_complex_keys() {
    for permissions in [
        "{checks: none, 'checks': write}",
        r#"{"checks": none, checks: write}"#,
        "{'checks': none, 'checks': write}",
        r#"{'checks': none, "checks": write}"#,
    ] {
        let tree = document(&format!("permissions: {permissions}\n"));
        let error = effective_none(&tree, &Node::Map(vec![]), "checks").unwrap_err();
        assert!(error.contains("duplicate permission checks"), "{error}");
    }
    for name in [
        "' checks '",
        r#""\u0063hecks""#,
        "'chec''ks'",
        "[checks]",
        "'checks",
        "checks'",
        "",
    ] {
        let declaration = Node::Map(vec![(name.into(), Node::Scalar("write".into()))]);
        assert!(parse(&declaration).is_err(), "accepted {name:?}");
        let declaration = Node::Scalar(format!("{{{name}: write}}"));
        assert!(parse(&declaration).is_err(), "accepted inline {name:?}");
    }
    let declaration = Node::Map(vec![
        ("checks".into(), Node::Scalar("none".into())),
        ("'checks'".into(), Node::Scalar("write".into())),
    ]);
    assert!(parse(&declaration).is_err());
}
