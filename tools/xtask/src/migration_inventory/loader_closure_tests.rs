use super::ledger::MigrationLedgers;
use super::loader_closure::{check_family_canary_loaders, check_runner_image_planner_loader};
use crate::command::DynResult;
use std::fs;

#[test]
fn runner_image_planner_loader_requires_a_record() -> DynResult<()> {
    // Given the checked-in loader and a ledger with no ownership record.
    let root = crate::repo_consistency::repo_root()?;
    let mut ledger = MigrationLedgers::load(&root)?.invocations;
    ledger.runner_image_planner_loader = None;
    // When inventory checks the loader, then omission must fail.
    let error = check_runner_image_planner_loader(&root, &ledger).unwrap_err();
    assert!(
        error.to_string().contains("missing invocation contract"),
        "{error}"
    );
    Ok(())
}

#[test]
fn runner_image_planner_loader_rejects_target_drift_and_unknown_import() -> DynResult<()> {
    // Given a copy of the exact loader and planner with the recorded digest.
    let repo = crate::repo_consistency::repo_root()?;
    let root = crate::command::unique_temp_dir("runner-planner-loader");
    fs::create_dir_all(root.join("scripts"))?;
    let loader = fs::read_to_string(repo.join("scripts/runner-image-identity.py"))?;
    let planner = fs::read(repo.join("scripts/plan-ci.py"))?;
    fs::write(root.join("scripts/runner-image-identity.py"), &loader)?;
    fs::write(root.join("scripts/plan-ci.py"), &planner)?;
    let ledger = MigrationLedgers::load(&repo)?.invocations;
    check_runner_image_planner_loader(&root, &ledger)?;

    // When target bytes or the selected import path change, then ownership is stale.
    fs::write(root.join("scripts/plan-ci.py"), b"# changed planner\n")?;
    let error = check_runner_image_planner_loader(&root, &ledger).unwrap_err();
    assert!(
        error.to_string().contains("changed planner target bytes"),
        "{error}"
    );
    fs::write(root.join("scripts/plan-ci.py"), &planner)?;
    fs::write(
        root.join("scripts/runner-image-identity.py"),
        loader.replace(
            "root / \"scripts/plan-ci.py\"",
            "root / \"scripts/other.py\"",
        ),
    )?;
    let error = check_runner_image_planner_loader(&root, &ledger).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("changed or unowned dynamic import"),
        "{error}"
    );

    // When a further unresolved file-location import is added, then it cannot be approved implicitly.
    fs::write(
        root.join("scripts/runner-image-identity.py"),
        format!(
            "{loader}\nextra = importlib.util.spec_from_file_location('extra', runtime_path)\n"
        ),
    )?;
    let error = check_runner_image_planner_loader(&root, &ledger).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("changed or unowned dynamic import"),
        "{error}"
    );
    let mut ledger = ledger;
    ledger
        .runner_image_planner_loader
        .as_mut()
        .ok_or("missing fixture")?
        .target_sha256 = "0".repeat(64);
    fs::write(root.join("scripts/runner-image-identity.py"), loader)?;
    let error = check_runner_image_planner_loader(&root, &ledger).unwrap_err();
    assert!(
        error.to_string().contains("changed planner target bytes"),
        "{error}"
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn family_canary_loaders_require_both_records() -> DynResult<()> {
    // Given the checked-in family loader and an omitted ownership record.
    let root = crate::repo_consistency::repo_root()?;
    let mut ledger = MigrationLedgers::load(&root)?.invocations;
    ledger.family_canary_loaders.clear();
    // When checking the loader, then missing contracts must fail.
    let error = check_family_canary_loaders(&root, &ledger).unwrap_err();
    assert!(
        error.to_string().contains("missing family loader"),
        "{error}"
    );
    let mut ledger = MigrationLedgers::load(&root)?.invocations;
    ledger.family_canary_loaders.pop();
    assert!(check_family_canary_loaders(&root, &ledger).is_err());
    Ok(())
}

#[test]
fn family_canary_loaders_reject_target_bytes_path_and_new_import() -> DynResult<()> {
    // Given copied source with the recorded relative targets.
    let repo = crate::repo_consistency::repo_root()?;
    let root = crate::command::unique_temp_dir("family-canary-loader");
    fs::create_dir_all(root.join("scripts/lib"))?;
    let loader = fs::read_to_string(repo.join("scripts/llama-canary-family-evidence.py"))?;
    let memory = fs::read(repo.join("scripts/lib/canary_family_memory.py"))?;
    let planner = fs::read(repo.join("scripts/plan-family-battery.py"))?;
    fs::write(
        root.join("scripts/llama-canary-family-evidence.py"),
        &loader,
    )?;
    fs::write(root.join("scripts/lib/canary_family_memory.py"), &memory)?;
    fs::write(root.join("scripts/plan-family-battery.py"), &planner)?;
    let mut wrong_record = MigrationLedgers::load(&repo)?.invocations;
    wrong_record.family_canary_loaders[1].target = "scripts/other-planner.py".into();
    assert!(check_family_canary_loaders(&root, &wrong_record).is_err());
    wrong_record.family_canary_loaders[1].target = "scripts/plan-family-battery.py".into();
    wrong_record.family_canary_loaders[1].target_sha256 = "0".repeat(64);
    assert!(check_family_canary_loaders(&root, &wrong_record).is_err());
    let ledger = MigrationLedgers::load(&repo)?.invocations;
    check_family_canary_loaders(&root, &ledger)?;

    // When either target changes, then the corresponding source identity fails.
    fs::write(
        root.join("scripts/lib/canary_family_memory.py"),
        b"# changed\n",
    )?;
    assert!(check_family_canary_loaders(&root, &ledger).is_err());
    fs::write(root.join("scripts/lib/canary_family_memory.py"), &memory)?;
    fs::write(root.join("scripts/plan-family-battery.py"), b"# changed\n")?;
    assert!(check_family_canary_loaders(&root, &ledger).is_err());
    fs::write(root.join("scripts/plan-family-battery.py"), &planner)?;

    // When a fixed target path or an extra computed import appears, neither is reviewed.
    fs::write(
        root.join("scripts/llama-canary-family-evidence.py"),
        loader.replace("lib/canary_family_memory.py", "lib/other_memory.py"),
    )?;
    assert!(check_family_canary_loaders(&root, &ledger).is_err());
    fs::write(
        root.join("scripts/llama-canary-family-evidence.py"),
        format!(
            "{loader}\nextra = importlib.util.spec_from_file_location('extra', runtime_path)\n"
        ),
    )?;
    assert!(check_family_canary_loaders(&root, &ledger).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}
