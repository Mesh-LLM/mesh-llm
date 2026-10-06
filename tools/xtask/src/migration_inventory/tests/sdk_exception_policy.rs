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
            "evals/agentic-trajectory-manifest.py",
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
