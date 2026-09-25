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

fn check(root: &Path, files: &[&str], entry: &str) -> DynResult<()> {
    let paths = files
        .iter()
        .map(|path| (*path).to_owned())
        .collect::<Vec<_>>();
    let observed = scan::scan_paths(root, &paths)?;
    check_required_closure(root, &paths, &observed, &BTreeSet::new(), &[entry])
}

#[test]
fn required_workflow_reaches_local_reusable_and_its_action() -> DynResult<()> {
    // Given a required entry calling a local reusable workflow, then a local action.
    let root = crate::command::unique_temp_dir("closure-reusable");
    source(
        &root,
        ".github/workflows/main_linux.yml",
        "jobs:\n  lane:\n    uses: ./.github/workflows/ci-linux-lane.yml\n",
    )?;
    source(
        &root,
        ".github/workflows/ci-linux-lane.yml",
        "jobs:\n  product:\n    uses: ./.github/workflows/ci-linux-product-slice.yml\n",
    )?;
    source(
        &root,
        ".github/workflows/ci-linux-product-slice.yml",
        "steps:\n  - uses: ./.github/actions/compose-product-input\n",
    )?;
    source(
        &root,
        ".github/actions/compose-product-input/action.yml",
        "run: scripts/ci-compose-product-input.sh\n",
    )?;
    source(
        &root,
        "scripts/ci-compose-product-input.sh",
        "python3 scripts/compose-product-bundle.py\n",
    )?;
    source(&root, "scripts/compose-product-bundle.py", "pass\n")?;
    let paths = [
        ".github/workflows/main_linux.yml",
        ".github/workflows/ci-linux-lane.yml",
        ".github/workflows/ci-linux-product-slice.yml",
        ".github/actions/compose-product-input/action.yml",
        "scripts/ci-compose-product-input.sh",
        "scripts/compose-product-bundle.py",
    ];
    // When checking the entry, then the unowned Python call in the reached script fails.
    let error = check(&root, &paths, paths[0]).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("scripts/compose-product-bundle.py"),
        "{error}"
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn required_workflow_does_not_follow_remote_ref_as_local() -> DynResult<()> {
    // Given a PR entry pinned to a protected workflow on main.
    let root = crate::command::unique_temp_dir("closure-remote-reusable");
    let entry = ".github/workflows/pr_linux.yml";
    source(
        &root,
        entry,
        "jobs:\n  lane:\n    uses: Mesh-LLM/mesh-llm/.github/workflows/ci-linux-lane.yml@main\n",
    )?;
    // When checking the local tree, then remote revision bytes are not inferred.
    let result = check(&root, &[entry], entry);
    fs::remove_dir_all(root)?;
    result
}

#[test]
fn required_workflow_rejects_missing_local_reusable() -> DynResult<()> {
    // Given a required main entry whose local reusable workflow is absent.
    let root = crate::command::unique_temp_dir("closure-missing-reusable");
    let entry = ".github/workflows/main_linux.yml";
    source(
        &root,
        entry,
        "jobs:\n  lane:\n    uses: ./.github/workflows/ci-linux-lane.yml\n",
    )?;
    // When checking its closure, then the missing required child is reported.
    let error = check(&root, &[entry], entry).unwrap_err();
    assert!(error.to_string().contains("ci-linux-lane.yml"), "{error}");
    fs::remove_dir_all(root)?;
    Ok(())
}
