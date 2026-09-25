use super::just_recipes::check_recipe_children;
use super::required_closure::check_required_closure;
use super::scan;
use crate::command::DynResult;
use std::collections::BTreeSet;
use std::fs;

#[test]
fn required_closure_rejects_python_child_in_imported_recipe_variable() -> DynResult<()> {
    // Given a root Justfile importing CI and a recipe selected by the required CI recipe.
    let root = crate::command::unique_temp_dir("just-recipe-variable");
    fs::create_dir_all(root.join("just"))?;
    fs::create_dir_all(root.join("scripts"))?;
    fs::write(
        root.join("Justfile"),
        "interpreter := \"python3\"\nimport 'just/ci.just'\ndefault: ci-validate\n",
    )?;
    fs::write(
        root.join("just/ci.just"),
        "ci-validate:\n    just child-check\n\ntest-all:\n    cargo test -p xtask\n\nchild-check:\n    {{ interpreter }} scripts/new.py\n",
    )?;
    fs::write(root.join("scripts/new.py"), "pass\n")?;
    let paths = vec![
        "Justfile".to_owned(),
        "just/ci.just".to_owned(),
        "scripts/new.py".to_owned(),
    ];
    let observed = scan::scan_paths(&root, &paths)?;
    // When only previously observed candidate IDs are approved, the expanded child must fail.
    let owned = observed
        .iter()
        .map(|row| row.id.clone())
        .collect::<BTreeSet<_>>();
    let result = check_required_closure(
        &root,
        &paths,
        &observed,
        &owned,
        &["Justfile", "just/ci.just"],
    );
    fs::remove_dir_all(root)?;
    let error = result.unwrap_err();
    assert!(error.to_string().contains("interpreter"), "{error}");
    Ok(())
}

#[test]
fn required_closure_rejects_unresolved_interpreter_variable() -> DynResult<()> {
    // Given a required recipe with an executable selected at runtime.
    let root = crate::command::unique_temp_dir("just-unresolved-interpreter");
    fs::create_dir_all(root.join("just"))?;
    fs::write(
        root.join("Justfile"),
        "interpreter := env('SELECTED_INTERPRETER', 'python3')\nimport 'just/ci.just'\ndefault: ci-validate\n",
    )?;
    fs::write(
        root.join("just/ci.just"),
        "ci-validate:\n    {{ interpreter }} scripts/child.py\ntest-all:\n    cargo test -p xtask\n",
    )?;
    fs::create_dir_all(root.join("scripts"))?;
    fs::write(root.join("scripts/child.py"), "pass\n")?;
    let paths = vec![
        "Justfile".to_owned(),
        "just/ci.just".to_owned(),
        "scripts/child.py".to_owned(),
    ];
    let observed = scan::scan_paths(&root, &paths)?;
    let owned = observed
        .iter()
        .map(|row| row.id.clone())
        .collect::<BTreeSet<_>>();
    // When the Just expression has no finite binding, the guard must not invent a target.
    let result = check_required_closure(
        &root,
        &paths,
        &observed,
        &owned,
        &["Justfile", "just/ci.just"],
    );
    fs::remove_dir_all(root)?;
    let error = result.unwrap_err();
    assert!(
        error
            .to_string()
            .contains("unresolved variable target interpreter"),
        "{error}"
    );
    Ok(())
}

#[test]
fn required_closure_rejects_python_child_selected_by_recipe_variable() -> DynResult<()> {
    // Given a required recipe selecting a child through a constant Just assignment.
    let root = crate::command::unique_temp_dir("just-recipe-child-variable");
    fs::create_dir_all(&root)?;
    fs::write(
        root.join("Justfile"),
        "child := \"python-runner\"\ninterpreter := \"python3\"\nci-validate:\n    just {{ child }}\n\npython-runner:\n    {{ interpreter }} -c 'pass'\n",
    )?;
    // When the required recipe is traversed, its selected child's interpreter is reached.
    let result = check_recipe_children(&root, &BTreeSet::from(["ci-validate".to_owned()]));
    fs::remove_dir_all(root)?;
    // Then the unowned interpreter is rejected.
    let error = result.unwrap_err();
    assert!(error.to_string().contains("interpreter"), "{error}");
    Ok(())
}

#[test]
fn required_closure_accepts_constant_recipe_target_with_safe_child() -> DynResult<()> {
    // Given a required recipe selecting a safe child by constant assignment.
    let root = crate::command::unique_temp_dir("just-safe-recipe-target");
    fs::create_dir_all(&root)?;
    fs::write(
        root.join("Justfile"),
        "child := \"child-check\"\nci-validate:\n    just {{ child }}\n\nchild-check:\n    cargo test -p xtask\n",
    )?;
    // When traversed, the resolved child is permitted.
    let result = check_recipe_children(&root, &BTreeSet::from(["ci-validate".to_owned()]));
    fs::remove_dir_all(root)?;
    // Then no interpreter is found.
    result
}

#[test]
fn required_closure_rejects_runtime_selected_recipe_target() -> DynResult<()> {
    // Given a required recipe whose target depends on the process environment.
    let root = crate::command::unique_temp_dir("just-runtime-recipe-target");
    fs::create_dir_all(&root)?;
    fs::write(
        root.join("Justfile"),
        "child := env('SELECTED_RECIPE', 'child-check')\nci-validate:\n    just {{ child }}\n\nchild-check:\n    cargo test -p xtask\n",
    )?;
    // When the required recipe is traversed, the target must not be guessed.
    let result = check_recipe_children(&root, &BTreeSet::from(["ci-validate".to_owned()]));
    fs::remove_dir_all(root)?;
    // Then the unresolved recipe target fails closed.
    let error = result.unwrap_err();
    assert!(
        error
            .to_string()
            .contains("unresolved variable target child"),
        "{error}"
    );
    Ok(())
}
