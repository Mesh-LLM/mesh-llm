use super::*;
use sha2::{Digest, Sha256};

fn root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn digest(root: &Path, inventory: Inventory) -> DynResult<String> {
    let mut files = BTreeSet::new();
    for pattern in inventory.inputs() {
        // glob's terminal ** matches directories; Actions also hashes their files.
        let pattern = if pattern.ends_with("/**") {
            format!("{pattern}/*")
        } else {
            pattern
        };
        for path in glob::glob(root.join(pattern).to_str().ok_or("non-UTF8 fixture")?)? {
            let path = path?;
            if path.is_file() {
                files.insert(path);
            }
        }
    }
    let mut hash = Sha256::new();
    for file in files {
        hash.update(std::fs::read(file)?);
    }
    Ok(hex::encode(hash.finalize()))
}

#[test]
fn checked_in_cache_expressions_match_the_owned_inventory() -> DynResult<()> {
    check(&root())
}

#[test]
fn compiled_inputs_and_embedded_resources_change_identity() -> DynResult<()> {
    let sandbox = tempfile::tempdir()?;
    for (inventory, path) in [
        (Inventory::HostedXtask, "tools/xtask/build.rs"),
        (
            Inventory::HostedXtask,
            "tools/xtask/src/automation/catalog.json",
        ),
        (Inventory::HostedXtask, "ci/python-research-source.json"),
        (Inventory::HostedXtask, "scripts/ci-compat-smoke.sh"),
        (Inventory::HostedXtask, "scripts/linux-glibc-floor.txt"),
        (Inventory::HostedController, "scripts/cargo-linker"),
        (
            Inventory::HostedController,
            "tools/ci-controller/Cargo.toml",
        ),
        (
            Inventory::LinuxRuntime,
            "skippy/crates/skippy-hf-hub/src/client.rs",
        ),
        (
            Inventory::LinuxRuntime,
            "skippy/crates/skippy-ffi/build/native_link.rs",
        ),
        (
            Inventory::LinuxRuntime,
            "skippy/crates/skippy-protocol/proto/stage.proto",
        ),
        (
            Inventory::LinuxRuntime,
            "skippy/llama_cpp/patches/generated/series",
        ),
        (
            Inventory::LinuxRuntime,
            "skippy/llama_cpp/whisper-upstream.txt",
        ),
        (
            Inventory::LinuxRuntime,
            "tools/xtask/src/native_policy/runtime_manifest_writer.rs",
        ),
        (
            Inventory::LinuxRuntime,
            ".github/workflows/ci-linux-runtime-slice.yml",
        ),
        (
            Inventory::WindowsAbi,
            "tools/xtask/src/native_policy/runtime_manifest_backend.rs",
        ),
    ] {
        let file = sandbox.path().join(path);
        std::fs::create_dir_all(file.parent().ok_or("fixture parent")?)?;
        std::fs::write(&file, "before")?;
        let before = digest(sandbox.path(), inventory)?;
        std::fs::write(&file, "after")?;
        assert_ne!(before, digest(sandbox.path(), inventory)?, "{path}");
    }
    Ok(())
}

#[test]
fn uncompiled_tests_readmes_and_unrelated_products_preserve_identity() -> DynResult<()> {
    let sandbox = tempfile::tempdir()?;
    for (inventory, path) in [
        (Inventory::HostedXtask, "tools/xtask/tests/integration.rs"),
        (Inventory::HostedXtask, "tools/trajectory-reader/README.md"),
        (Inventory::HostedController, "tools/xtask/src/main.rs"),
        (
            Inventory::LinuxRuntime,
            "skippy/crates/skippy-cli/src/main.rs",
        ),
        (Inventory::LinuxRuntime, "skippy/docs/ARCHITECTURE.md"),
        (
            Inventory::WindowsAbi,
            "skippy/crates/skippy-package-builder/src/main.rs",
        ),
    ] {
        let before = digest(sandbox.path(), inventory)?;
        let file = sandbox.path().join(path);
        std::fs::create_dir_all(file.parent().ok_or("fixture parent")?)?;
        std::fs::write(file, "uncompiled edit")?;
        assert_eq!(before, digest(sandbox.path(), inventory)?, "{path}");
    }
    Ok(())
}

#[test]
fn new_local_normal_and_build_dependencies_require_inventory_ownership() -> DynResult<()> {
    let sandbox = tempfile::tempdir()?;
    std::fs::write(
        sandbox.path().join("Cargo.toml"),
        "[workspace]\nmembers=[]\n",
    )?;
    let controller = sandbox.path().join("tools/ci-controller");
    std::fs::create_dir_all(&controller)?;
    std::fs::create_dir_all(sandbox.path().join("tools/new-tool"))?;
    for (kind, accepted) in [
        ("dependencies", false),
        ("build-dependencies", false),
        ("dev-dependencies", true),
    ] {
        std::fs::write(
            controller.join("Cargo.toml"),
            format!("[package]\nname='ci-controller'\n[{kind}]\nnew-tool={{path='../new-tool'}}\n"),
        )?;
        assert_eq!(
            closure::check(sandbox.path(), Inventory::HostedController).is_ok(),
            accepted,
            "{kind}"
        );
    }
    Ok(())
}

#[test]
fn newly_added_build_script_requires_inventory_ownership() -> DynResult<()> {
    let sandbox = tempfile::tempdir()?;
    std::fs::write(
        sandbox.path().join("Cargo.toml"),
        "[workspace]\nmembers=[]\n",
    )?;
    let controller = sandbox.path().join("tools/ci-controller");
    std::fs::create_dir_all(&controller)?;
    std::fs::write(
        controller.join("Cargo.toml"),
        "[package]\nname='ci-controller'\n",
    )?;
    closure::check(sandbox.path(), Inventory::HostedController)?;
    std::fs::write(controller.join("build.rs"), "fn main() {}")?;
    assert!(
        closure::check(sandbox.path(), Inventory::HostedController)
            .unwrap_err()
            .to_string()
            .contains("omits build script")
    );
    Ok(())
}
