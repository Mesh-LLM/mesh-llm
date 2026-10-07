//! The shipped Python language component is separate from retained external SDK clients.
use crate::command::DynResult;
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;

const COMPONENT: [&str; 8] = [
    "mesh/sdk/python/hatch_build.py",
    "mesh/sdk/python/src/meshllm/__init__.py",
    "mesh/sdk/python/src/meshllm/_binding.py",
    "mesh/sdk/python/src/meshllm/_generated/__init__.py",
    "mesh/sdk/python/src/meshllm/_generated/mesh_ffi.py",
    "mesh/sdk/python/src/meshllm/client.py",
    "mesh/sdk/python/src/meshllm/types.py",
    "mesh/sdk/python/tests/test_client.py",
];

fn component_execution(parent: &str, child: Option<&str>, block: &str) -> bool {
    // Release version propagation lists this manifest as data, as recorded by
    // the complete graph's source-backed SDK boundary. It executes no component.
    let component_parent =
        parent.starts_with("mesh/sdk/python/") || parent.starts_with("sdk/python/");
    if !component_parent
        && matches!(
            block,
            "\"mesh/sdk/python/pyproject.toml\"" | "\"sdk/python/pyproject.toml\""
        )
    {
        return false;
    }
    component_parent
        || child.is_some_and(|child| COMPONENT.contains(&child) || child.starts_with("sdk/python/"))
        || block.contains("sdk/python")
        || block.contains("import meshllm")
        || block.contains("from meshllm")
        || ((block.contains("pip ") || block.contains("uv pip ")) && block.contains("mesh-llm"))
}

fn assert_isolated(
    graph: &Value,
    observed: &[super::super::scan::Candidate],
    selected: &[(String, String)],
) -> DynResult<()> {
    if graph["complete_census"].as_bool() != Some(true) {
        return Err("Python SDK component isolation requires a complete required census".into());
    }
    for (recipe, command) in selected {
        if component_execution(recipe, None, command) {
            return Err(format!(
                "required recipe executes Python SDK component: {recipe}: {command}"
            )
            .into());
        }
    }
    let edges = graph["edges"].as_array().ok_or("missing graph edges")?;
    let mut reached = graph["roots"]
        .as_array()
        .ok_or("missing graph roots")?
        .iter()
        .filter_map(Value::as_str)
        .collect::<BTreeSet<_>>();
    for edge in edges {
        let parent = edge["parent"].as_str().ok_or("missing edge parent")?;
        reached.insert(parent);
        if edge["status"].as_str() != Some("reference_only")
            && component_execution(
                parent,
                edge["child"].as_str(),
                edge["source_block"].as_str().ok_or("missing edge source")?,
            )
        {
            return Err(format!("required graph reaches Python SDK component: {edge}").into());
        }
    }
    // The graph intentionally does not tokenize sdk/ as a generic script prefix.
    // Inspect its reached parents' executable scanner rows too, including inline installs/imports.
    for row in observed {
        if row.executable
            && row.path != "Justfile"
            && !row.path.ends_with(".just")
            && reached.contains(row.path.as_str())
            && component_execution(&row.path, None, &row.source_block)
        {
            return Err(
                format!("required source executes Python SDK component: {}", row.id).into(),
            );
        }
    }
    Ok(())
}

#[test]
fn python_sdk_component_stays_outside_actual_required_graph_and_closure() -> DynResult<()> {
    let root = crate::repo_consistency::repo_root()?;
    let paths = super::super::ledger::tracked_paths(&root)?;
    let actual = paths
        .iter()
        .filter(|path| path.starts_with("mesh/sdk/python/") && path.ends_with(".py"))
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    assert_eq!(actual, COMPONENT.into_iter().collect());
    let observed = super::super::scan::scan_paths(&root, &paths)?;
    let owned = super::super::shards::check_shards(&root, &observed)?;
    let roots = super::super::required_closure::required_roots(&root, &paths)?;
    let roots = roots.iter().map(String::as_str).collect::<Vec<_>>();
    super::super::required_closure::check_required_closure(
        &root, &paths, &observed, &owned, &roots,
    )?;
    let graph = super::super::required_graph::report(&root, &paths, &observed, &owned, &roots)?;
    assert!(graph.complete_census, "{graph:?}");
    assert_isolated(
        &serde_json::to_value(&graph)?,
        &observed,
        &graph.selected_recipe_commands,
    )?;
    let data = fs::read_to_string(root.join("mesh/scripts/check-sdk-contract.sh"))?;
    assert!(data.contains("PYTHON_SDK=\"$ROOT/mesh/sdk/python/src/meshllm/client.py\""));
    assert!(data.contains("if ! grep -Fq \"$pattern\" \"$file\""));
    let docs = fs::read_to_string(root.join("mesh/sdk/python/README.md"))?;
    assert!(docs.contains("python3 -I mesh/sdk/python/tests/test_client.py"));
    let tests = fs::read_to_string(root.join(COMPONENT[7]))?;
    assert!(tests.contains("sys.path.insert(0,"));
    assert!(tests.contains("unittest.main()"));
    let project: toml::Value = toml::from_str(&fs::read_to_string(
        root.join("mesh/sdk/python/pyproject.toml"),
    )?)?;
    assert_eq!(
        project["build-system"]["build-backend"].as_str(),
        Some("hatchling.build")
    );
    Ok(())
}

#[test]
fn python_sdk_component_guard_rejects_actual_inline_install_import_and_test_calls() -> DynResult<()>
{
    let private = tempfile::tempdir()?;
    let paths = vec!["Justfile".to_owned()];
    for command in [
        "python3 -m pip install -e sdk/python",
        "python3 -c 'import meshllm'",
        "python3 -I mesh/sdk/python/tests/test_client.py",
        "python3 -I sdk/python/tests/test_client.py",
        "python3 -m pip install -e mesh/sdk/python",
    ] {
        fs::write(
            private.path().join("Justfile"),
            format!("default:\n    {command}\n"),
        )?;
        let observed = super::super::scan::scan_paths(private.path(), &paths)?;
        assert!(observed.iter().any(|row| row.executable));
        let graph = super::super::required_graph::report(
            private.path(),
            &paths,
            &observed,
            &BTreeSet::new(),
            &["Justfile"],
        )?;
        assert!(graph.complete_census, "{graph:?}");
        let error = assert_isolated(
            &serde_json::to_value(&graph)?,
            &observed,
            &graph.selected_recipe_commands,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("required recipe executes Python SDK component"),
            "{error}"
        );
    }
    fs::create_dir(private.path().join("scripts"))?;
    fs::write(
        private.path().join("scripts/run.sh"),
        "PYTHON_SDK=\"$ROOT/mesh/sdk/python/src/meshllm/client.py\"\n",
    )?;
    let paths = vec!["scripts/run.sh".to_owned()];
    let observed = super::super::scan::scan_paths(private.path(), &paths)?;
    let graph = super::super::required_graph::report(
        private.path(),
        &paths,
        &observed,
        &BTreeSet::new(),
        &["scripts/run.sh"],
    )?;
    assert!(graph.complete_census, "{graph:?}");
    assert_isolated(
        &serde_json::to_value(&graph)?,
        &observed,
        &graph.selected_recipe_commands,
    )?;
    Ok(())
}

#[test]
fn python_sdk_component_guard_keeps_optional_recipe_and_refuses_required_or_incomplete()
-> DynResult<()> {
    let private = tempfile::tempdir()?;
    let paths = vec!["Justfile".to_owned()];
    let native = "default: ci-validate\nci-validate:\n    printf 'native'\noptional-sdk:\n    python3 -m pip install -e sdk/python\n";
    fs::write(private.path().join("Justfile"), native)?;
    let observed = super::super::scan::scan_paths(private.path(), &paths)?;
    assert!(
        observed
            .iter()
            .any(|row| row.executable && row.source_block.contains("sdk/python"))
    );
    let graph = super::super::required_graph::report(
        private.path(),
        &paths,
        &observed,
        &BTreeSet::new(),
        &["Justfile"],
    )?;
    assert!(graph.complete_census, "{graph:?}");
    assert!(
        !graph
            .selected_recipe_commands
            .iter()
            .any(|(recipe, _)| recipe == "just:optional-sdk")
    );
    assert_isolated(
        &serde_json::to_value(&graph)?,
        &observed,
        &graph.selected_recipe_commands,
    )?;
    let mut incomplete = serde_json::to_value(&graph)?;
    incomplete["complete_census"] = Value::Bool(false);
    assert!(assert_isolated(&incomplete, &observed, &graph.selected_recipe_commands).is_err());
    for required in [
        "    python3 -m pip install -e sdk/python",
        "    printf 'native'; python3 -m pip install -e sdk/python",
        "    echo native; python3 -m pip install -e sdk/python",
    ] {
        fs::write(
            private.path().join("Justfile"),
            native.replace("    printf 'native'", required),
        )?;
        let observed = super::super::scan::scan_paths(private.path(), &paths)?;
        let graph = super::super::required_graph::report(
            private.path(),
            &paths,
            &observed,
            &BTreeSet::new(),
            &["Justfile"],
        )?;
        assert!(graph.complete_census, "{graph:?}");
        assert!(
            graph
                .selected_recipe_commands
                .iter()
                .any(|(recipe, command)| recipe == "just:ci-validate"
                    && command == required.trim())
        );
        assert!(
            assert_isolated(
                &serde_json::to_value(&graph)?,
                &observed,
                &graph.selected_recipe_commands
            )
            .unwrap_err()
            .to_string()
            .contains("required recipe")
        );
    }
    Ok(())
}
