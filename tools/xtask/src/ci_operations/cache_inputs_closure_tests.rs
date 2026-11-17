use super::*;

fn fixture() -> DynResult<tempfile::TempDir> {
    let sandbox = tempfile::tempdir()?;
    std::fs::write(
        sandbox.path().join("Cargo.toml"),
        "[workspace]\nmembers=[]\n[workspace.dependencies]\nleaf={path='tools/leaf'}\n",
    )?;
    for package in XTASK.iter().copied().chain(["tools/leaf"]) {
        let directory = sandbox.path().join(package);
        std::fs::create_dir_all(&directory)?;
        std::fs::write(directory.join("Cargo.toml"), "[package]\nname='fixture'\n")?;
    }
    Ok(sandbox)
}

#[test]
fn default_features_require_ownership_of_optional_target_dependencies() -> DynResult<()> {
    for declaration in ["path='../leaf'", "workspace=true"] {
        for kind in ["dependencies", "build-dependencies"] {
            let sandbox = fixture()?;
            std::fs::write(
                sandbox.path().join("tools/ci-controller/Cargo.toml"),
                format!(
                    "[package]\nname='controller'\n[features]\ndefault=['enabled']\nenabled=['dep:leaf']\n[target.'cfg(unix)'.{kind}]\nleaf={{{declaration},optional=true}}\n"
                ),
            )?;
            let error = check(sandbox.path(), Inventory::HostedController)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("optional") && error.contains("leaf"),
                "{error}"
            );
        }
    }
    Ok(())
}

#[test]
fn requested_features_cannot_forward_an_unowned_nested_local_closure() -> DynResult<()> {
    for forwarding in ["controller/nested", "controller?/nested"] {
        for table in ["dependencies", "target.'cfg(unix)'.dependencies"] {
            let sandbox = fixture()?;
            std::fs::write(
                sandbox.path().join("tools/xtask/Cargo.toml"),
                "[package]\nname='xtask'\n[dependencies]\nbridge={package='authority-automation',path='../authority-automation',features=['enabled']}\n",
            )?;
            std::fs::write(
                sandbox.path().join("tools/authority-automation/Cargo.toml"),
                format!(
                    "[package]\nname='authority-automation'\n[features]\nenabled=['{forwarding}']\n[{table}]\ncontroller={{package='ci-controller',path='../ci-controller'}}\n"
                ),
            )?;
            std::fs::write(
                sandbox.path().join("tools/ci-controller/Cargo.toml"),
                "[package]\nname='ci-controller'\n[features]\nnested=['dep:leaf']\n[dependencies]\nleaf={workspace=true,optional=true}\n",
            )?;
            let error = check(sandbox.path(), Inventory::HostedXtask)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("forwards local") && error.contains(forwarding),
                "{error}"
            );
        }
    }
    Ok(())
}

#[test]
fn requested_features_require_ownership_of_workspace_optional_dependencies() -> DynResult<()> {
    let sandbox = fixture()?;
    std::fs::write(
        sandbox.path().join("tools/xtask/Cargo.toml"),
        "[package]\nname='xtask'\n[dependencies]\ncontroller={path='../ci-controller',features=['nested']}\n",
    )?;
    std::fs::write(
        sandbox.path().join("tools/ci-controller/Cargo.toml"),
        "[package]\nname='ci-controller'\n[features]\nnested=['dep:leaf']\n[target.'cfg(unix)'.dependencies]\nleaf={workspace=true,optional=true}\n",
    )?;
    let error = check(sandbox.path(), Inventory::HostedXtask)
        .unwrap_err()
        .to_string();
    assert!(error.contains("optional local dependency leaf"), "{error}");
    Ok(())
}

#[test]
fn inactive_local_forwarding_and_registry_forwarding_remain_accepted() -> DynResult<()> {
    let sandbox = fixture()?;
    std::fs::write(
        sandbox.path().join("tools/ci-controller/Cargo.toml"),
        "[package]\nname='controller'\n[features]\ndefault=['registry/extra']\nunused=['leaf/nested']\n[dependencies]\nregistry='1'\nleaf={workspace=true,optional=true}\n",
    )?;
    check(sandbox.path(), Inventory::HostedController)
}
