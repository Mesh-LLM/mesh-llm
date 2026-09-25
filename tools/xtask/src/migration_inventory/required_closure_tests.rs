use super::required_closure::check_required_closure;
use super::scan;
use crate::command::DynResult;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

fn source(root: &Path, path: &str, text: &str) -> DynResult<()> {
    let file = root.join(path);
    fs::create_dir_all(file.parent().ok_or("missing parent")?)?;
    fs::write(file, text)?;
    Ok(())
}

fn check(root: &Path, files: &[&str], roots: &[&str]) -> DynResult<()> {
    let paths = files
        .iter()
        .map(|path| (*path).to_owned())
        .collect::<Vec<_>>();
    let observed = scan::scan_paths(root, &paths)?;
    check_required_closure(root, &paths, &observed, &BTreeSet::new(), roots)
}

#[test]
fn required_closure_allows_source_owned_unittest_but_rejects_new_inline() -> DynResult<()> {
    // Given the checked-in Just recipe and its independently checked shard ownership.
    let repo = crate::repo_consistency::repo_root()?;
    let paths = super::ledger::tracked_paths(&repo)?;
    let observed = scan::scan_paths(&repo, &paths)?;
    let owned = super::shards::check_shards(&repo, &observed)?;
    let baseline = observed
        .iter()
        .find(|row| {
            row.path == "just/ci.just" && row.source_block.starts_with("python3 -m unittest")
        })
        .ok_or("missing checked-in unittest candidate")?;
    assert_eq!(baseline.id, "just/ci.just#candidate:807df0c5a71213d8:1");
    // When the required recipe is checked, then its owned Python call is transitional.
    check_required_closure(&repo, &paths, &observed, &owned, &["just/ci.just"])?;

    // Given the same owned source ID but a newly inserted required inline call.
    let root = crate::command::unique_temp_dir("closure-transitional");
    source(
        &root,
        "just/ci.just",
        "ci-validate:\n    python3 -m unittest discover -s scripts/tests -p 'test_*.py'\n    python3 -c 'print(1)'\n",
    )?;
    let paths = vec!["just/ci.just".to_owned()];
    let observed = scan::scan_paths(&root, &paths)?;
    // When checked with only the verified baseline ID, the new call fails.
    let error = check_required_closure(
        &root,
        &paths,
        &observed,
        &BTreeSet::from([baseline.id.clone()]),
        &["just/ci.just"],
    )
    .unwrap_err();
    assert!(error.to_string().contains("inline Python"), "{error}");
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn required_closure_rejects_new_inline_when_required() -> DynResult<()> {
    // Given a required Just recipe with an inline Python execution.
    let root = crate::command::unique_temp_dir("closure-inline");
    source(
        &root,
        "just/ci.just",
        "ci-validate:\n    python3 -c 'print(1)'\n",
    )?;
    // When checking its closure, then the interpreter is a required dependency.
    let error = check(&root, &["just/ci.just"], &["just/ci.just"]).unwrap_err();
    assert!(error.to_string().contains("just/ci.just"), "{error}");
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn required_closure_rejects_new_dynamic_import_when_required() -> DynResult<()> {
    // Given a required shell caller of a Python module with a computed import.
    let root = crate::command::unique_temp_dir("closure-import");
    source(
        &root,
        "just/ci.just",
        "ci-validate:\n    python3 scripts/loader.py\n",
    )?;
    source(
        &root,
        "scripts/loader.py",
        "importlib.util.spec_from_file_location('x', path)\n",
    )?;
    let error = check(
        &root,
        &["just/ci.just", "scripts/loader.py"],
        &["just/ci.just"],
    )
    .unwrap_err();
    assert!(error.to_string().contains("scripts/loader.py"), "{error}");
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn required_closure_rejects_stale_source() -> DynResult<()> {
    // Given a required caller of a script missing from the checked-in source set.
    let root = crate::command::unique_temp_dir("closure-stale");
    source(
        &root,
        "just/ci.just",
        "ci-validate:\n    scripts/missing.sh\n",
    )?;
    let error = check(&root, &["just/ci.just"], &["just/ci.just"]).unwrap_err();
    assert!(error.to_string().contains("scripts/missing.sh"), "{error}");
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn required_closure_rejects_new_unowned_required_child() -> DynResult<()> {
    // Given a required shell script that starts a previously unowned Python child.
    let root = crate::command::unique_temp_dir("closure-child");
    source(
        &root,
        "just/ci.just",
        "ci-validate:\n    scripts/runner.sh\n",
    )?;
    source(&root, "scripts/runner.sh", "python3 scripts/new.py\n")?;
    source(&root, "scripts/new.py", "pass\n")?;
    let error = check(
        &root,
        &["just/ci.just", "scripts/runner.sh", "scripts/new.py"],
        &["just/ci.just"],
    )
    .unwrap_err();
    assert!(error.to_string().contains("scripts/new.py"), "{error}");
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn required_closure_does_not_execute_owned_data_reference() -> DynResult<()> {
    // Given an owned data reference to a Python child, not an owned launch.
    let root = crate::command::unique_temp_dir("closure-data-child");
    source(
        &root,
        "just/ci.just",
        "ci-validate:\n    echo scripts/tests/new.py\n",
    )?;
    source(&root, "scripts/tests/new.py", "pass\n")?;
    let paths = vec!["just/ci.just".to_owned(), "scripts/tests/new.py".to_owned()];
    let mut observed = scan::scan_paths(&root, &paths)?;
    let candidate = observed
        .iter_mut()
        .find(|row| row.path == "just/ci.just")
        .ok_or("missing data candidate")?;
    candidate.executable = false;
    let owned = BTreeSet::from([candidate.id.clone()]);
    // When following required references, then data is not an execution edge.
    check_required_closure(&root, &paths, &observed, &owned, &["just/ci.just"])?;
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn required_closure_rejects_fabricated_optional_classification() -> DynResult<()> {
    // Given an eval imported by required CI, its directory cannot make it optional.
    let root = crate::command::unique_temp_dir("closure-eval");
    source(
        &root,
        "just/ci.just",
        "ci-validate:\n    python3 evals/benchmark.py\n",
    )?;
    source(&root, "evals/benchmark.py", "pass\n")?;
    let error = check(
        &root,
        &["just/ci.just", "evals/benchmark.py"],
        &["just/ci.just"],
    )
    .unwrap_err();
    assert!(error.to_string().contains("evals/benchmark.py"), "{error}");
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn required_closure_allows_operator_selected_optional_destination() -> DynResult<()> {
    // Given an unrelated operator-invoked eval with a caller-selected executable.
    let root = crate::command::unique_temp_dir("closure-optional");
    source(
        &root,
        "just/ci.just",
        "ci-validate:\n    cargo test -p xtask\n",
    )?;
    source(&root, "evals/manual.py", "subprocess.run(args.binary)\n")?;
    check(
        &root,
        &["just/ci.just", "evals/manual.py"],
        &["just/ci.just"],
    )?;
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn required_closure_rejects_python_shebang_in_required_child() -> DynResult<()> {
    // Given a required shell command whose child starts Python through its shebang.
    let root = crate::command::unique_temp_dir("closure-shebang");
    source(
        &root,
        "just/ci.just",
        "ci-validate:\n    scripts/runner.sh\n",
    )?;
    source(&root, "scripts/runner.sh", "#!/usr/bin/env python3\npass\n")?;
    // When checking closure, then the shebang cannot bypass interpreter scanning.
    let error = check(
        &root,
        &["just/ci.just", "scripts/runner.sh"],
        &["just/ci.just"],
    )
    .unwrap_err();
    assert!(error.to_string().contains("scripts/runner.sh"), "{error}");
    fs::remove_dir_all(root)?;
    Ok(())
}
