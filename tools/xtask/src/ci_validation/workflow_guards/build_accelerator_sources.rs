//! Portable policy checks for source-owned Windows adapters and bootstrap defaults.
//! These checks do not claim that a Windows compiler or PowerShell was executed.
use super::{field, workflow_yaml};
use std::{fs, path::Path};
fn source(path: &str) -> String {
    fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(path),
    )
    .unwrap()
}
fn commands(source: &str) -> String {
    source
        .lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n")
}
#[test]
fn build_accelerator_sources_release_requires_cache_and_action_creates_and_exports_short_paths() {
    let release = workflow_yaml::parse(&source(".github/workflows/release.yml")).unwrap();
    let jobs = release.get("jobs").unwrap();
    for name in ["windows_host_input", "build_native_runtime_windows_gpu"] {
        let environment = jobs.get(name).unwrap().get("env").unwrap();
        assert_eq!(
            field(environment, "MESH_LLM_REQUIRE_SCCACHE"),
            Some("1"),
            "{name}"
        );
    }
    assert!(!source(".github/workflows/release.yml").contains("use_sccache"));
    let action = workflow_yaml::parse(&source(
        ".github/actions/setup-windows-short-paths/action.yml",
    ))
    .unwrap();
    let super::Node::Seq(steps) = action.get("runs").unwrap().get("steps").unwrap() else {
        panic!("short-path action must have steps")
    };
    let run = commands(field(&steps[0], "run").unwrap());
    for assignment in [
        "SCCACHE_DIR = \"C:\\s\"",
        "TEMP = \"C:\\t\"",
        "TMP = \"C:\\t\"",
    ] {
        assert!(run.contains(assignment), "{assignment}");
    }
    assert!(
        run.contains("New-Item -ItemType Directory -Force -Path $paths.SCCACHE_DIR, $paths.TEMP")
    );
    assert!(run.contains("$paths.GetEnumerator()"));
    assert!(run.contains("Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append"));
}
#[test]
fn build_accelerator_sources_object_path_limit_is_shared_by_windows_native_backends() {
    let bash = commands(&source("skippy/scripts/build-llama.sh"));
    let platform = bash
        .split("case \"$(uname -s)\" in")
        .skip(1)
        .map(|body| body.split("esac").next().unwrap())
        .find(|body| body.contains("CMAKE_ARGS+=(-DCMAKE_OBJECT_PATH_MAX="))
        .unwrap();
    assert!(platform.contains("MINGW*|MSYS*|CYGWIN*"));
    assert!(platform.contains("CMAKE_ARGS+=(-DCMAKE_OBJECT_PATH_MAX=180)"));
    let windows = commands(&source("mesh/scripts/build-windows.ps1"));
    let shared = windows
        .split("$cmakeArgs = @(")
        .nth(1)
        .unwrap()
        .split("switch ($backendName)")
        .next()
        .unwrap();
    assert!(shared.contains("\"-DCMAKE_OBJECT_PATH_MAX=180\""));
    let backends = windows.rsplit("switch ($backendName)").next().unwrap();
    assert!(
        !backends.contains("-DCMAKE_OBJECT_PATH_MAX="),
        "backend branches must inherit the shared limit"
    );
}
#[test]
fn build_accelerator_sources_windows_linker_selects_rust_or_llvm_and_preserves_exit_and_arguments()
{
    let windows = source("scripts/cargo-linker.cmd");
    for command in [
        "rustc --print sysroot",
        "for %%T in (x86_64-pc-windows-msvc aarch64-pc-windows-msvc)",
        "for %%L in (rust-lld.exe lld-link.exe)",
        "set \"MESH_LLD_FLAVOR=-flavor link\"",
        "if \"%~1\"==\"--mesh-probe\"",
        "setlocal DisableDelayedExpansion",
        "\"%MESH_LLD%\" %MESH_LLD_FLAVOR% %*",
        "exit /b %ERRORLEVEL%",
    ] {
        assert!(windows.contains(command), "{command}");
    }
    assert!(
        windows.find("setlocal DisableDelayedExpansion").unwrap()
            < windows.find("\"%MESH_LLD%\" %MESH_LLD_FLAVOR% %*").unwrap()
    );
}
#[test]
fn build_accelerator_sources_windows_bootstrap_keeps_version_override_cache_restore_and_platform_facades()
 {
    let windows = commands(&source("scripts/bootstrap-build-tools.ps1"));
    for command in [
        "$env:MESH_LLM_SCCACHE_VERSION",
        "\"0.16.0\"",
        "& rustup component add llvm-tools-preview",
        "& cargo install sccache --version $sccacheVersion --locked --force",
        "$env:RUSTC_WRAPPER = \"\"",
        "finally {",
        "$env:RUSTC_WRAPPER = $previousRustcWrapper",
        "Remove-Item Env:RUSTC_WRAPPER",
        "cargo-linker.cmd\") --mesh-probe",
    ] {
        assert!(windows.contains(command), "{command}");
    }
    let recipes = source("just/build.just");
    assert!(recipes.contains("[unix]\nbootstrap-build-tools:\n    @scripts/bootstrap-build-tools"));
    assert!(recipes.contains("[windows]\nbootstrap-build-tools:\n    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/bootstrap-build-tools.ps1"));
}
