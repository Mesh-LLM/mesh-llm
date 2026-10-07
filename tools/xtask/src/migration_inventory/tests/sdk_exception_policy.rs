//! Status admission only; installed dependencies and SDK execution remain unqualified.
use super::super::exception_policy::check_exceptions;
use super::*;

const CLIENTS: [&str; 4] = [
    "scripts/ci-openai-python-smoke.py",
    "scripts/ci-langchain-openai-smoke.py",
    "scripts/ci-litellm-smoke.py",
    "scripts/ci-openai-embeddings-smoke.py",
];

fn entry(path: &str, status: &str) -> ExceptionEntry {
    serde_json::from_value(serde_json::json!({
        "path": path, "status": status,
        "callers": ["source-owned existing caller"],
        "local_dependency_files": ["source-owned existing lock"],
        "cadence": "retained L8 cadence", "purpose": "real SDK behavior",
        "isolation_test": "source evidence; real execution qualification pending"
    }))
    .unwrap()
}

fn recorded(path: &str, paths: &mut Vec<String>, ledgers: &mut MigrationLedgers) {
    paths.push(path.into());
    ledgers.inventory.files.push(PythonFile {
        path: path.into(),
        classification: "candidate".into(),
        replacement_owner: "isolated client".into(),
        deletion_condition: "qualification".into(),
    });
}

fn with_ledgers(
    check: impl FnOnce(Vec<String>, MigrationLedgers) -> DynResult<()>,
) -> DynResult<()> {
    let root = crate::command::unique_temp_dir("sdk-exception-status");
    let result = fixture(&root).and_then(|(paths, ledgers)| check(paths, ledgers));
    cleanup(root)?;
    result
}

#[test]
fn sdk_policy_keeps_all_four_conditional_independent_of_advisory_filename() -> DynResult<()> {
    with_ledgers(|mut paths, mut ledgers| {
        for client in CLIENTS {
            recorded(client, &mut paths, &mut ledgers);
            ledgers
                .exceptions
                .exceptions
                .push(entry(client, "conditional_unqualified"));
        }
        check_exceptions(&paths, &ledgers)?;
        paths.push(".github/workflows/python-sdk-compatibility.yml".into());
        check_exceptions(&paths, &ledgers)?;
        assert!(
            ledgers
                .exceptions
                .exceptions
                .iter()
                .all(|e| e.status == "conditional_unqualified")
        );
        Ok(())
    })
}

#[test]
fn sdk_policy_refuses_qualified_even_with_recorded_source_and_advisory_filename() -> DynResult<()> {
    with_ledgers(|mut paths, mut ledgers| {
        paths.push(".github/workflows/python-sdk-compatibility.yml".into());
        for client in CLIENTS {
            recorded(client, &mut paths, &mut ledgers);
            ledgers.exceptions.exceptions = vec![entry(client, "qualified")];
            let error = check_exceptions(&paths, &ledgers).unwrap_err().to_string();
            assert!(
                error.contains(client) && error.contains("(qualified)"),
                "{error}"
            );
        }
        Ok(())
    })
}

#[test]
fn sdk_policy_preserves_closed_reader_approval_and_metadata_refusals() -> DynResult<()> {
    with_ledgers(|mut paths, mut ledgers| {
        for reader in [
            "mesh/evals/agentic-trajectory-manifest.py",
            "scripts/generate-bench-corpus.py",
        ] {
            ledgers.exceptions.exceptions = vec![entry(reader, "maintainer_retained")];
            assert!(check_exceptions(&paths, &ledgers).is_err());
            recorded(reader, &mut paths, &mut ledgers);
            check_exceptions(&paths, &ledgers)?;
            ledgers.exceptions.exceptions[0].status = "conditional_unqualified".into();
            assert!(check_exceptions(&paths, &ledgers).is_err());
            ledgers.exceptions.exceptions[0].status = "qualified".into();
            assert!(check_exceptions(&paths, &ledgers).is_err());
        }
        for client in CLIENTS {
            ledgers.exceptions.exceptions = vec![entry(client, "maintainer_retained")];
            assert!(check_exceptions(&paths, &ledgers).is_err());
            let candidate = entry(client, "conditional_unqualified");
            ledgers.exceptions.exceptions = vec![candidate];
            check_exceptions(&paths, &ledgers)?;
            ledgers.exceptions.exceptions[0].isolation_test = Some(String::new());
            assert!(
                check_exceptions(&paths, &ledgers)
                    .unwrap_err()
                    .to_string()
                    .contains("incomplete")
            );
            ledgers.exceptions.exceptions = vec![
                entry(client, "conditional_unqualified"),
                entry(client, "conditional_unqualified"),
            ];
            assert!(
                check_exceptions(&paths, &ledgers)
                    .unwrap_err()
                    .to_string()
                    .contains("duplicate")
            );
        }
        recorded("scripts/not-approved.py", &mut paths, &mut ledgers);
        ledgers.exceptions.exceptions =
            vec![entry("scripts/not-approved.py", "maintainer_retained")];
        assert!(
            check_exceptions(&paths, &ledgers)
                .unwrap_err()
                .to_string()
                .contains("fabricated")
        );
        Ok(())
    })
}

#[test]
fn granite_reference_policy_is_exact_optional_status_and_recorded_source() -> DynResult<()> {
    with_ledgers(|mut paths, mut ledgers| {
        let path = "skippy/evals/skippy-granite-tensor-equivalence.py";
        let mut candidate = entry(path, "isolated_model_reference");
        candidate.local_dependency_files = Some(vec![
            "evals/granite-reference/pyproject.toml".into(),
            "evals/granite-reference/uv.lock".into(),
        ]);
        ledgers.exceptions.exceptions = vec![candidate];
        assert!(check_exceptions(&paths, &ledgers).is_err());
        recorded(path, &mut paths, &mut ledgers);
        check_exceptions(&paths, &ledgers)?;
        for status in [
            "maintainer_retained",
            "conditional_unqualified",
            "qualified",
        ] {
            ledgers.exceptions.exceptions[0].status = status.into();
            assert!(check_exceptions(&paths, &ledgers).is_err());
        }
        ledgers.exceptions.exceptions[0].status = "isolated_model_reference".into();
        ledgers.exceptions.exceptions[0].local_dependency_files =
            Some(vec!["ci/requirements-ci-python.txt".into()]);
        assert!(check_exceptions(&paths, &ledgers).is_err());
        ledgers.exceptions.exceptions[0].path = "evals/other-model-reference.py".into();
        assert!(check_exceptions(&paths, &ledgers).is_err());
        Ok(())
    })
}

#[test]
fn granite_reference_project_has_separate_finite_python_and_complete_hashed_lock() -> DynResult<()>
{
    let root = crate::repo_consistency::repo_root()?;
    let project: toml::Value = toml::from_str(&fs::read_to_string(
        root.join("evals/granite-reference/pyproject.toml"),
    )?)?;
    assert_eq!(
        project["project"]["requires-python"].as_str(),
        Some(">=3.12,<3.13")
    );
    let expected = BTreeSet::from(["gguf", "numpy", "safetensors", "torch"]);
    let declared = project["project"]["dependencies"]
        .as_array()
        .ok_or("missing reference dependencies")?
        .iter()
        .map(|value| value.as_str().ok_or("invalid dependency"))
        .collect::<Result<BTreeSet<_>, _>>()?;
    assert_eq!(declared, expected);
    let lock: toml::Value = toml::from_str(&fs::read_to_string(
        root.join("evals/granite-reference/uv.lock"),
    )?)?;
    assert_eq!(lock["requires-python"].as_str(), Some("==3.12.*"));
    let packages = lock["package"].as_array().ok_or("missing package lock")?;
    for name in expected {
        let package = packages
            .iter()
            .find(|p| p["name"].as_str() == Some(name))
            .ok_or("reference import not locked")?;
        assert!(
            !package["version"]
                .as_str()
                .ok_or("missing version")?
                .is_empty()
        );
        assert_eq!(
            package["source"]["registry"].as_str(),
            Some("https://pypi.org/simple")
        );
        let wheels = package["wheels"]
            .as_array()
            .ok_or("missing reference wheels")?;
        assert!(
            !wheels.is_empty(),
            "direct dependency {name} has no locked artifact"
        );
        assert!(
            wheels
                .iter()
                .all(|wheel| wheel["hash"].as_str().is_some_and(|hash| hash
                    .strip_prefix("sha256:")
                    .is_some_and(
                        |hex| hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit())
                    )))
        );
    }
    let docs = fs::read_to_string(root.join("skippy/docs/COMPETITIVE_BENCHMARK.md"))?;
    assert!(docs.contains(
        "evals/granite-reference/.venv/bin/python -I skippy/evals/skippy-granite-tensor-equivalence.py"
    ));
    assert!(docs.contains("uv sync --locked --no-python-downloads --project evals/granite-reference --python python3.12"));
    Ok(())
}

#[test]
fn granite_reference_is_absent_from_actual_required_and_default_execution_graph() -> DynResult<()> {
    let root = crate::repo_consistency::repo_root()?;
    let paths = super::super::ledger::tracked_paths(&root)?;
    let observed = super::super::scan::scan_paths(&root, &paths)?;
    let owned = super::super::shards::check_shards(&root, &observed)?;
    let roots = super::super::required_closure::required_roots(&root, &paths)?;
    let roots = roots.iter().map(String::as_str).collect::<Vec<_>>();
    let graph = super::super::required_graph::report(&root, &paths, &observed, &owned, &roots)?;
    for edge in &graph.edges {
        assert_ne!(
            edge.child.as_deref(),
            Some("skippy/evals/skippy-granite-tensor-equivalence.py")
        );
        assert!(
            !edge.source_block.contains("evals/granite-reference"),
            "{edge:?}"
        );
        assert!(
            !edge
                .source_block
                .contains("skippy-granite-tensor-equivalence"),
            "{edge:?}"
        );
    }
    Ok(())
}

#[test]
fn research16_is_absent_from_actual_required_and_default_execution_graph() -> DynResult<()> {
    let root = crate::repo_consistency::repo_root()?;
    let declaration: serde_json::Value = serde_json::from_slice(&std::fs::read(
        root.join("evals/research-python/BOUNDARIES.json"),
    )?)?;
    let rows = declaration["rows"]
        .as_array()
        .ok_or("research roster absent")?;
    assert_eq!(rows.len(), 16);
    let forbidden = rows
        .iter()
        .map(|row| row["path"].as_str().expect("exact research path"))
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(forbidden.len(), 16);
    let expected = [
        "skippy/crates/skippy-cache/src/cachegen/fixtures/generate_lmcache_compat.py",
        "skippy/crates/skippy-quantize/scripts/compare-reference-quantization.py",
        "skippy/evals/latency-benchmarking/latency-proxy.py",
        "skippy/evals/latency-benchmarking/measure.py",
        "mesh/evals/moa-openrouter/analyze_ablation.py",
        "mesh/evals/moa-openrouter/lite_agent.py",
        "mesh/evals/moa-openrouter/make_fixture.py",
        "mesh/evals/moa-openrouter/orclient.py",
        "mesh/evals/moa-openrouter/probe_tools.py",
        "mesh/evals/moa-openrouter/record.py",
        "mesh/evals/moa-openrouter/record_agentic.py",
        "mesh/evals/scenarios/debug-session/buggy.py",
        "mesh/evals/scenarios/edit-file/server.py",
        "mesh/evals/scenarios/refactor/config.py",
        "mesh/evals/test_injection_framing.py",
        "mesh/evals/virtual_llm_eval.py",
    ]
    .into_iter()
    .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(forbidden, expected);
    let project: toml::Value = toml::from_str(&std::fs::read_to_string(
        root.join("evals/research-python/pyproject.toml"),
    )?)?;
    assert_eq!(
        project["project"]["requires-python"].as_str(),
        Some(">=3.11,<3.15")
    );
    assert!(
        project["project"]["dependencies"]
            .as_array()
            .unwrap()
            .is_empty()
    );

    let reference: toml::Value = toml::from_str(&std::fs::read_to_string(
        root.join("evals/quantizer-reference/pyproject.toml"),
    )?)?;
    assert_eq!(
        reference["project"]["requires-python"].as_str(),
        Some(">=3.12,<3.13")
    );
    let direct = reference["project"]["dependencies"].as_array().unwrap();
    assert_eq!(
        direct
            .iter()
            .map(|item| item.as_str().unwrap())
            .collect::<Vec<_>>(),
        ["numpy~=2.2.6", "gguf>=0.1.0"]
    );
    let paths = super::super::ledger::tracked_paths(&root)?;
    let observed = super::super::scan::scan_paths(&root, &paths)?;
    let owned = super::super::shards::check_shards(&root, &observed)?;
    let roots = super::super::required_closure::required_roots(&root, &paths)?;
    let roots = roots.iter().map(String::as_str).collect::<Vec<_>>();
    let graph = super::super::required_graph::report(&root, &paths, &observed, &owned, &roots)?;
    assert!(graph.complete_census, "{graph:?}");
    for edge in &graph.edges {
        assert!(
            !edge
                .child
                .as_deref()
                .is_some_and(|child| forbidden.contains(child)),
            "{edge:?}"
        );
        for path in &forbidden {
            assert!(!edge.source_block.contains(path), "{edge:?}");
        }
        assert!(
            !edge.source_block.contains("evals/research-python")
                && !edge.source_block.contains("evals/quantizer-reference"),
            "{edge:?}"
        );
    }
    Ok(())
}

#[test]
fn research16_policy_refuses_unrecorded_wrong_class_and_undeclared_dependencies() -> DynResult<()> {
    with_ledgers(|mut paths, mut ledgers| {
        for path in super::super::exception_policy::research::PATHS {
            ledgers.exceptions.exceptions = vec![research_entry(path)];
            assert!(check_exceptions(&paths, &ledgers).is_err());
            recorded(path, &mut paths, &mut ledgers);
            check_exceptions(&paths, &ledgers)?;
            ledgers.exceptions.exceptions[0].status =
                if super::super::exception_policy::research::is_upstream(path) {
                    "isolated_research_evaluation"
                } else {
                    "isolated_upstream_reference"
                }
                .into();
            assert!(check_exceptions(&paths, &ledgers).is_err());
            ledgers.exceptions.exceptions[0].status = "qualified".into();
            assert!(check_exceptions(&paths, &ledgers).is_err());
            ledgers.exceptions.exceptions[0] = research_entry(path);
            ledgers.exceptions.exceptions[0].local_dependency_files =
                Some(vec!["ci/requirements-ci-python.txt".into()]);
            assert!(check_exceptions(&paths, &ledgers).is_err());
            ledgers.exceptions.exceptions = vec![research_entry(path), research_entry(path)];
            assert!(check_exceptions(&paths, &ledgers).is_err());
        }
        ledgers.exceptions.exceptions =
            vec![entry("evals/unreviewed.py", "isolated_research_evaluation")];
        assert!(check_exceptions(&paths, &ledgers).is_err());
        Ok(())
    })
}

fn research_entry(path: &str) -> ExceptionEntry {
    let mut candidate = entry(
        path,
        if super::super::exception_policy::research::is_upstream(path) {
            "isolated_upstream_reference"
        } else {
            "isolated_research_evaluation"
        },
    );
    candidate.cadence = Some("optional explicit operator research/upstream regeneration only; never PR/main required setup".into());
    candidate.local_dependency_files = Some(vec![
        super::super::exception_policy::research::project(path).into(),
        "evals/research-python/BOUNDARIES.json".into(),
    ]);
    candidate
}
