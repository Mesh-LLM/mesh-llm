use super::*;

#[test]
fn migration_ci_graph_yaml_reads_the_workflow_subset() {
    let source = "on:\n  workflow_call:\n    inputs: &i\n      a:\n        type: string\n  \
        workflow_dispatch:\n    inputs: *i\njobs:\n  x:\n    needs: [a, 'b']\n    \
        permissions: {}\n    steps:\n      - uses: actions/checkout@abc # v5\n        \
        with:\n          ref: main\n      - run: |\n          echo 1\n\n          echo 2\n    \
        if: ${{ !cancelled() }}\n  y:\n    needs:\n      - x\n";
    let tree = parse(source).expect("valid subset");
    let on = tree.get("on").expect("on");
    assert_eq!(
        on.get("workflow_dispatch").and_then(|d| d.get("inputs")),
        Some(&Node::Scalar("*i".into()))
    );
    assert!(
        on.get("workflow_call")
            .and_then(|c| c.get("inputs"))
            .and_then(|i| i.get("a"))
            .is_some()
    );
    let x = tree.get("jobs").and_then(|jobs| jobs.get("x")).expect("x");
    assert_eq!(x.get("needs").map(Node::list), Some(vec!["a", "b"]));
    assert_eq!(x.get("permissions"), Some(&Node::Map(Vec::new())));
    assert_eq!(
        x.get("if").and_then(Node::text),
        Some("${{ !cancelled() }}")
    );
    let Some(Node::Seq(steps)) = x.get("steps") else {
        panic!("steps")
    };
    assert_eq!(
        steps[0].get("uses").and_then(Node::text),
        Some("actions/checkout@abc")
    );
    assert_eq!(
        steps[0]
            .get("with")
            .and_then(|w| w.get("ref"))
            .and_then(Node::text),
        Some("main")
    );
    assert_eq!(
        steps[1].get("run").and_then(Node::text),
        Some("echo 1\n\necho 2")
    );
    let y = tree.get("jobs").and_then(|jobs| jobs.get("y")).expect("y");
    assert_eq!(y.get("needs").map(Node::list), Some(vec!["x"]));
}
