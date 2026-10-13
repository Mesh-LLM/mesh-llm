//! Actual installed Just recipes with inert compiler/build observers.
use super::*;
use std::{io::Read as _, os::unix::fs::PermissionsExt as _};

fn executable(path: &Path, source: &str) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, source).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}

fn installed(name: &str) -> PathBuf {
    std::env::split_paths(&std::env::var_os("PATH").unwrap())
        .map(|directory| directory.join(name))
        .find(|path| path.is_file())
        .unwrap()
        .canonicalize()
        .unwrap()
}

fn execute(root: &Path, command: &Path, arguments: &[&str]) -> (bool, Vec<u8>) {
    let search = format!(
        "{}:{}",
        root.join("bin").display(),
        std::env::var("PATH").unwrap()
    );
    let report = process::supervise_raw(
        &ProcessSpec {
            executable: command.into(),
            cwd: root.into(),
            arguments: arguments
                .iter()
                .map(|value| Value::Public((*value).into()))
                .collect(),
            environment: BTreeMap::from([
                ("PATH".into(), Value::Public(search.into())),
                ("HOME".into(), Value::Public(root.into())),
                ("TMPDIR".into(), Value::Public(root.into())),
                ("CI".into(), Value::Public("true".into())),
            ]),
        },
        &Limits {
            execution: Duration::from_secs(10),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(2),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: NonZeroUsize::new(65536),
            stderr: NonZeroUsize::new(65536),
        },
    )
    .unwrap();
    assert!(
        report.process.cleanup.complete && report.process.failure.is_none(),
        "{report:?}"
    );
    assert!(
        !report.process.stdout.truncated && !report.process.stderr.truncated,
        "{report:?}"
    );
    (
        report.process.status.unwrap().success(),
        report.stdout.unwrap().as_bytes().to_vec(),
    )
}

fn recipe(name: &str) -> String {
    String::from_utf8(just(&repo(), &["--show", name]).unwrap()).unwrap()
}

fn packaged(root: &Path) -> BTreeMap<String, Vec<u8>> {
    let file = fs::File::open(root.join("dist/openai-exchange-observer.tar.gz")).unwrap();
    let mut archive = tar::Archive::new(flate2::read::GzDecoder::new(file));
    archive
        .entries()
        .unwrap()
        .filter_map(|entry| {
            let mut entry = entry.unwrap();
            if !entry.header().entry_type().is_file() {
                return None;
            }
            let path = entry.path().unwrap().to_string_lossy().into_owned();
            let mut bytes = Vec::new();
            entry.read_to_end(&mut bytes).unwrap();
            Some((path, bytes))
        })
        .collect()
}

#[test]
fn just_layout_exemplar_packages_selected_cargo_artifact_and_refuses_missing_output() {
    for target in [
        "configured-target",
        "env-target/debug",
        "env-build-target/aarch64-apple-darwin/debug",
    ] {
        let state = tempfile::tempdir().unwrap();
        let root = state.path();
        let artifact = root.join(target).join("examples/openai-exchange-observer");
        let bytes = "#!/bin/sh\n[ \"$*\" = --manifest ] || exit 90\nprintf '{\"fresh\":true}\\n'\n";
        executable(&artifact, bytes);
        executable(
            &root.join("target/debug/examples/openai-exchange-observer"),
            "#!/bin/sh\nexit 91\n",
        );
        executable(
            &root.join("bin/just"),
            "#!/bin/sh\n[ \"$*\" = 'build-openai-exchange-exemplar json' ] || exit 92\ncat build.json\n",
        );
        fs::write(
            root.join("Justfile"),
            recipe("package-openai-exchange-exemplar"),
        )
        .unwrap();
        let output = serde_json::json!({"reason":"compiler-artifact","target":{"name":"openai-exchange-observer"},"executable":artifact});
        fs::write(
            root.join("build.json"),
            format!("{output}\n{{\"reason\":\"build-finished\",\"success\":true}}\n"),
        )
        .unwrap();
        assert!(
            execute(
                root,
                &installed("just"),
                &["--justfile", "Justfile", "package-openai-exchange-exemplar"]
            )
            .0
        );
        let archive = packaged(root);
        assert_eq!(
            archive["openai-exchange-observer/openai-exchange-observer"],
            bytes.as_bytes()
        );
        let manifest: serde_json::Value =
            serde_json::from_slice(&archive["openai-exchange-observer/plugin-manifest.json"])
                .unwrap();
        assert_eq!(manifest, serde_json::json!({"fresh":true}));
        let archive = root.join("dist/openai-exchange-observer.tar.gz");
        fs::remove_file(&archive).unwrap();
        fs::write(
            root.join("build.json"),
            "{\"reason\":\"build-finished\",\"success\":true}\n",
        )
        .unwrap();
        assert!(
            !execute(
                root,
                &installed("just"),
                &["--justfile", "Justfile", "package-openai-exchange-exemplar"]
            )
            .0
        );
        assert!(!archive.exists());
        fs::write(root.join("build.json"), output.to_string()).unwrap();
        fs::remove_file(&artifact).unwrap();
        assert!(
            !execute(
                root,
                &installed("just"),
                &["--justfile", "Justfile", "package-openai-exchange-exemplar"]
            )
            .0
        );
        assert!(!archive.exists());
    }
}

#[test]
fn just_layout_quality_windows_skips_only_unix_conformance_then_runs_portable_checks() {
    let source = recipe("test-all");
    let stage = source
        .split_once("echo \"=== 6/11 Plugin author exemplar ===\"")
        .unwrap()
        .1;
    let guard = stage
        .split_once("case \"$(uname -s)\" in")
        .unwrap()
        .1
        .split_once("esac")
        .unwrap()
        .0;
    let portable = stage
        .lines()
        .find(|line| line.trim_start().starts_with("just with-lld cargo run "))
        .unwrap()
        .trim();
    assert!(portable.contains("mesh/docs/plugins/exemplars/web-ui/Cargo.toml"));
    assert!(
        source
            .lines()
            .any(|line| line.trim() == "scripts/test-portable.sh")
    );
    let body = format!(
        "set -e\ncase \"$(uname -s)\" in{guard}esac\n{portable}\nscripts/test-portable.sh\n"
    );
    for platform in [
        "Linux",
        "Darwin",
        "MINGW64_NT-10.0",
        "MSYS_NT-10.0",
        "CYGWIN_NT-10.0",
    ] {
        let state = tempfile::tempdir().unwrap();
        let root = state.path();
        fs::create_dir(root.join("target")).unwrap();
        executable(
            &root.join("bin/uname"),
            &format!("#!/bin/sh\nprintf '%s\\n' '{platform}'\n"),
        );
        executable(
            &root.join("bin/just"),
            "#!/bin/sh\nprintf '%s\\n' \"$*\" >> calls\nprintf '{}\\n'\n",
        );
        executable(
            &root.join("scripts/test-portable.sh"),
            "#!/bin/sh\nprintf 'portable\\n' >> calls\n",
        );
        assert!(execute(root, &installed("bash"), &["-c", &body]).0);
        let calls = fs::read_to_string(root.join("calls")).unwrap();
        assert!(calls.contains("with-lld cargo run --quiet --manifest-path mesh/docs/plugins/exemplars/web-ui/Cargo.toml -- --print-package-manifest"));
        assert!(calls.ends_with("portable\n"));
        assert_eq!(
            calls.contains("test-openai-exchange-conformance"),
            matches!(platform, "Linux" | "Darwin")
        );
    }
}

#[test]
fn just_layout_short_products_preserve_independent_mesh_and_composition_order() {
    let skippy = recipe("skippy");
    let mesh = recipe("mesh");
    assert!(
        skippy.contains("skippy/scripts/build-development-product.sh --backend \"{{ backend }}\"")
    );
    assert!(mesh.contains("scripts/build-host.sh --profile \"{{ profile }}\""));
    assert!(!mesh.contains("skippy/scripts/build-development-product.sh"));
    let mesh_source = fs::read_to_string(repo().join("just/mesh.just")).unwrap();
    assert!(mesh_source.contains("-HostOnly"));
    let state = tempfile::tempdir().unwrap();
    let root = state.path();
    fs::create_dir_all(root.join("mesh/scripts")).unwrap();
    fs::create_dir(root.join("scripts")).unwrap();
    let owner = root.join("mesh/scripts/build-development-product.sh");
    fs::copy(
        repo().join("mesh/scripts/build-development-product.sh"),
        &owner,
    )
    .unwrap();
    executable(
        &root.join("bin/just"),
        "#!/bin/sh\nprintf '%s|%s\\n' \"$*\" \"${MESH_LLM_BUILD_PROFILE:-}\" >> calls\n",
    );
    assert!(
        execute(
            root,
            &installed("bash"),
            &[
                owner.to_str().unwrap(),
                "--backend",
                "cuda",
                "--cuda-arch",
                "75;90",
                "--rocm-arch",
                "gfx1100",
                "--profile",
                "dev"
            ]
        )
        .0
    );
    assert_eq!(
        fs::read_to_string(root.join("calls")).unwrap(),
        "skippy cuda 75;90 gfx1100|\nmesh dev|dev\n"
    );
}
