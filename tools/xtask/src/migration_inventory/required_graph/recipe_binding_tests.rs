use super::super::required_graph::report;
use super::super::required_graph_tests::source;
use crate::command::DynResult;
use crate::migration_inventory::{ledger, required_closure, scan, shards};
use std::collections::BTreeSet;
use std::fs;

#[test]
fn graph_reaches_finite_just_argument_child() -> DynResult<()> {
    // Given an actual Just-style star parameter passed by a selected recipe.
    let root = crate::command::unique_temp_dir("graph-just-finite");
    source(
        &root,
        "Justfile",
        "default: ci-validate\nci-validate:\n    just with-lld python3 scripts/child.py\nwith-lld *COMMAND:\n    #!/usr/bin/env bash\n    exec {{ COMMAND }}\n",
    )?;
    source(&root, "scripts/child.py", "pass\n")?;
    let paths = vec!["Justfile".to_owned(), "scripts/child.py".to_owned()];
    // When selected recipes are expanded, then the argument binds to the launched child.
    let graph = report(&root, &paths, &[], &BTreeSet::new(), &["Justfile"])?;
    assert!(
        graph.edges.iter().any(|edge| edge.parent == "just:with-lld"
            && edge.child.as_deref() == Some("scripts/child.py")),
        "{graph:?}"
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn graph_does_not_traverse_environment_selected_just_child() -> DynResult<()> {
    // Given an environment-selected recipe name with a default that happens to exist.
    let root = crate::command::unique_temp_dir("graph-just-env");
    source(
        &root,
        "Justfile",
        "child := env('SELECTED_RECIPE', 'safe')\ndefault:\n    just {{ child }}\nsafe:\n    python3 scripts/child.py\n",
    )?;
    source(&root, "scripts/child.py", "pass\n")?;
    let paths = vec!["Justfile".to_owned(), "scripts/child.py".to_owned()];
    // When the graph is built, then the default is not promoted to a bounded child.
    let graph = report(&root, &paths, &[], &BTreeSet::new(), &["Justfile"])?;
    assert!(graph.edges.iter().any(|edge| {
        edge.status == "unknown_selection"
            && edge
                .unresolved_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("child"))
    }));
    assert!(!graph.edges.iter().any(|edge| edge.parent == "just:safe"));
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn graph_keeps_dynamic_just_argument_and_escaped_data_unresolved() -> DynResult<()> {
    // Given a caller-supplied command and an escaped string that names a Python file.
    let root = crate::command::unique_temp_dir("graph-just-escaped");
    source(
        &root,
        "Justfile",
        "default:\n    just with-lld $SELECTED_COMMAND\n    echo 'python3 scripts/data.py'\nwith-lld *COMMAND:\n    exec {{ COMMAND }}\n",
    )?;
    source(&root, "scripts/data.py", "pass\n")?;
    let paths = vec!["Justfile".to_owned(), "scripts/data.py".to_owned()];
    // When only literal arguments can bind, then neither data nor a guessed default launches.
    let graph = report(&root, &paths, &[], &BTreeSet::new(), &["Justfile"])?;
    assert!(graph.edges.iter().any(|edge| {
        edge.status == "unknown_selection"
            && edge
                .unresolved_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("dynamic Just arguments"))
    }));
    assert!(
        !graph.edges.iter().any(|edge| edge.parent == "just:with-lld"
            || edge.child.as_deref() == Some("scripts/data.py")
                && edge.parent.starts_with("just:"))
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn graph_keeps_platform_selected_recipe_conditional() -> DynResult<()> {
    // Given a platform-qualified recipe in Just's parsed source.
    let root = crate::command::unique_temp_dir("graph-just-platform");
    source(
        &root,
        "Justfile",
        "default: build\n[macos]\nbuild:\n    python3 scripts/child.py\n",
    )?;
    source(&root, "scripts/child.py", "pass\n")?;
    let paths = vec!["Justfile".to_owned(), "scripts/child.py".to_owned()];
    // When the graph spans source platforms, then the recipe's call remains conditional.
    let graph = report(&root, &paths, &[], &BTreeSet::new(), &["Justfile"])?;
    assert!(graph.edges.iter().any(|edge| edge.parent == "just:build"
        && edge.child.as_deref() == Some("scripts/child.py")
        && edge.status == "optional_branch"));
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn graph_does_not_call_unbound_star_argument() -> DynResult<()> {
    // Given a star parameter with no arguments on the only selected path.
    let root = crate::command::unique_temp_dir("graph-just-unbound-star");
    source(
        &root,
        "Justfile",
        "default:\n    just with-lld\nwith-lld *COMMAND:\n    exec {{ COMMAND }}\n",
    )?;
    let paths = vec!["Justfile".to_owned()];
    // When rendered, then exec without a command stays unknown rather than selected.
    let graph = report(&root, &paths, &[], &BTreeSet::new(), &["Justfile"])?;
    assert!(graph.edges.iter().any(|edge| edge.parent == "just:with-lld"
        && edge.status == "unknown_selection"
        && edge.unresolved_reason.as_deref() == Some("required executable argument is unbound")));
    assert!(
        !graph
            .edges
            .iter()
            .any(|edge| edge.parent == "just:with-lld" && edge.status == "selected")
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn graph_expands_checked_in_just_bindings_without_claiming_census() -> DynResult<()> {
    // Given the checked-in Justfile and the same required roots as the graph CLI.
    let root = crate::repo_consistency::repo_root()?;
    let paths = ledger::tracked_paths(&root)?;
    let observed = scan::scan_paths(&root, &paths)?;
    let validated = shards::check_shards(&root, &observed)?;
    let roots = required_closure::required_roots(&root, &paths)?;
    let refs = roots.iter().map(String::as_str).collect::<Vec<_>>();
    // When the graph expands the actual dump, then the finite wrapper is reached but a platform-selected build remains conditional.
    let graph = report(&root, &paths, &observed, &validated, &refs)?;
    assert!(graph.edges.iter().any(|edge| edge.parent == "just:with-lld"
        && edge.source_block == "exec cargo run -p xtask -- repo-consistency ci-crate-lists"
        && edge.status == "optional_branch"));
    assert!(graph.edges.iter().any(|edge| edge.parent == "just:build"
        && edge.child.as_deref() == Some("scripts/manage-build-cache.py")
        && edge.status == "optional_branch"));
    // Then each gated edge carries a platform_conditional record, so no class-c edge remains.
    assert!(graph.edges.iter().any(|edge| edge.parent == "just:build"
        && edge.disposition == super::boundaries::EdgeDisposition::PlatformConditional));
    assert_eq!(graph.unresolved, 0);
    Ok(())
}
