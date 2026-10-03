//! Actual installed Just parses the maintained import graph and renders selected recipes.
//! Execution uses copied recipes and inert host/runtime packagers; no compiler/native build.
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use std::{
    collections::BTreeMap,
    fs,
    num::NonZeroUsize,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::Duration,
};
fn tool() -> PathBuf {
    std::env::split_paths(&std::env::var_os("PATH").expect("native Just requires PATH"))
        .map(|p| p.join("just"))
        .find(|p| p.is_file() && fs::metadata(p).unwrap().permissions().mode() & 0o111 != 0)
        .expect("native Just is required")
        .canonicalize()
        .unwrap()
}
fn invoke(cwd: &Path, args: &[&str], values: &[(&str, &str)]) -> String {
    let mut environment = BTreeMap::from([
        (
            std::ffi::OsString::from("PATH"),
            Value::Public(std::env::var_os("PATH").unwrap()),
        ),
        (
            std::ffi::OsString::from("HOME"),
            Value::Public(cwd.as_os_str().to_owned()),
        ),
    ]);
    for (key, value) in values {
        environment.insert((*key).into(), Value::Public((*value).into()));
    }
    let result = process::supervise_raw(
        &ProcessSpec {
            executable: tool(),
            cwd: cwd.to_owned(),
            environment,
            arguments: args.iter().map(|s| Value::Public((*s).into())).collect(),
        },
        &Limits {
            execution: Duration::from_secs(20),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(2),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: NonZeroUsize::new(4 * 1024 * 1024),
            stderr: NonZeroUsize::new(65536),
        },
    )
    .unwrap();
    assert!(result.process.failure.is_none(), "{:?}", result.process);
    assert!(result.process.cleanup.complete);
    let stderr = result.stderr.unwrap();
    assert_eq!(
        result.process.status.unwrap().code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(stderr.as_bytes())
    );
    String::from_utf8(result.stdout.unwrap().as_bytes().to_vec()).unwrap()
}
fn recipe(name: &str) -> String {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    invoke(&repo, &["--show", name], &[])
}
fn executable(path: &Path, text: &str) {
    fs::write(path, text).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}
struct Fixture {
    _temp: tempfile::TempDir,
    root: PathBuf,
}
impl Fixture {
    fn new(name: &str) -> Self {
        let temp = tempfile::tempdir().unwrap();
        let root = temp
            .path()
            .canonicalize()
            .unwrap()
            .join("native recipe fixture");
        fs::create_dir_all(root.join("scripts")).unwrap();
        fs::create_dir_all(root.join("bin")).unwrap();
        // The selected recipe body is emitted by native Just, never flattened/reparsed in Rust.
        let dependencies = match name {
            "release-build-cuda" | "release-build-aarch64-cuda" => {
                "release-host-build:\n    @true\n\n"
            }
            "bundle" => "release-build:\n    @true\n\n",
            _ => "",
        };
        fs::write(
            root.join("Justfile"),
            format!(
                "mesh_bin := 'fixture-host'\n{dependencies}{}\n",
                recipe(name)
            ),
        )
        .unwrap();
        executable(&root.join("bin/uname"), "#!/bin/sh\nprintf 'Linux\\n'\n");
        executable(
            &root.join("scripts/detect-cuda-toolkit-version.sh"),
            "#!/bin/sh\nprintf '%s\\n' \"${MESH_CUDA_VERSION:-11}\"\n",
        );
        executable(
            &root.join("scripts/package-native-runtime.sh"),
            r#"#!/bin/bash
set -euo pipefail
printf '%s\n' "${LLAMA_STAGE_CUDA_ARCHITECTURES:-}" "${MESH_LLM_CUDA_TOOLKIT_MAJOR:-}" > "$FIXTURE_ROOT/environment"
printf '%s\0' "$@" > "$FIXTURE_ROOT/arguments"
"#,
        );
        Self { _temp: temp, root }
    }
    fn run(
        &self,
        name: &str,
        args: &[&str],
        version: Option<&str>,
        major: Option<&str>,
    ) -> (Vec<String>, Vec<String>) {
        let path = format!(
            "{}:{}",
            self.root.join("bin").display(),
            std::env::var("PATH").unwrap()
        );
        let mut env = vec![
            ("PATH", path.as_str()),
            ("FIXTURE_ROOT", self.root.to_str().unwrap()),
        ];
        if let Some(v) = version {
            env.push(("MESH_CUDA_VERSION", v));
        }
        if let Some(v) = major {
            env.push(("MESH_LLM_CUDA_TOOLKIT_MAJOR", v));
        }
        let mut command = vec!["--justfile", "Justfile", name];
        command.extend_from_slice(args);
        invoke(&self.root, &command, &env);
        let environment = fs::read_to_string(self.root.join("environment"))
            .unwrap()
            .lines()
            .map(str::to_owned)
            .collect();
        let arguments = fs::read(self.root.join("arguments"))
            .unwrap()
            .split(|b| *b == 0)
            .filter(|s| !s.is_empty())
            .map(|s| String::from_utf8(s.to_vec()).unwrap())
            .collect();
        (environment, arguments)
    }
}
#[test]
fn native_release_recipes_x86_cuda_boundary_detection_and_override_reach_packager() {
    let f = Fixture::new("release-build-cuda");
    for version in ["12", "12.0", "12.7", "12.8", "12.9", "13", "13.3"] {
        let (env, args) = f.run("release-build-cuda", &[], Some(version), None);
        let blackwell = matches!(version, "12.8" | "12.9" | "13" | "13.3");
        assert_eq!(
            env[0],
            if blackwell {
                "75;80;86;87;89;90;100;103;120;121"
            } else {
                "61;75;80;86;87;89;90"
            }
        );
        assert_eq!(env[1], version.split('.').next().unwrap());
        assert_eq!(
            args,
            [
                "--build",
                "--backend",
                "cuda",
                "--target",
                "x86_64-unknown-linux-gnu"
            ]
        );
    }
    let (env, _) = f.run("release-build-cuda", &[], None, None);
    assert_eq!(env, ["61;75;80;86;87;89;90", "11"]);
    let (env, _) = f.run("release-build-cuda", &[], Some("13.3"), Some("12"));
    assert_eq!(env[1], "12");
}
#[test]
fn native_release_recipes_aarch_cuda_boundary_detection_and_target_are_preserved() {
    let f = Fixture::new("release-build-aarch64-cuda");
    for version in ["12", "12.4", "12.8", "13", "13.1.2", "14"] {
        let (env, args) = f.run("release-build-aarch64-cuda", &[], Some(version), None);
        assert_eq!(
            env[0],
            if version.starts_with("12") {
                "61;75;80;86;87;89;90"
            } else {
                "75;80;86;87;89;90;110"
            }
        );
        assert_eq!(env[1], version.split('.').next().unwrap());
        assert_eq!(
            args,
            [
                "--build",
                "--backend",
                "cuda",
                "--target",
                "aarch64-unknown-linux-gnu"
            ]
        );
    }
    let (env, _) = f.run("release-build-aarch64-cuda", &[], None, None);
    assert_eq!(env, ["61;75;80;86;87;89;90", "11"]);
    let (env, _) = f.run(
        "release-build-aarch64-cuda",
        &[],
        Some("13.1.2"),
        Some("12"),
    );
    assert_eq!(env[1], "12");
}
#[test]
fn native_release_recipes_runtime_arguments_survive_empty_defaults_under_nounset() {
    let f = Fixture::new("release-runtime-build");
    for (parameters, expected) in [
        (vec![], vec!["--build", "--backend", "cpu"]),
        (vec!["cuda"], vec!["--build", "--backend", "cuda"]),
        (
            vec!["cuda", "aarch64-unknown-linux-gnu"],
            vec![
                "--build",
                "--backend",
                "cuda",
                "--target",
                "aarch64-unknown-linux-gnu",
            ],
        ),
    ] {
        assert_eq!(
            f.run("release-runtime-build", &parameters, None, None).1,
            expected
        );
    }
    #[cfg(target_os = "linux")]
    {
        let f = Fixture::new("build-runtime");
        assert_eq!(
            f.run("build-runtime", &[], None, None).1,
            ["--build", "--backend", "cpu"]
        );
        assert_eq!(
            f.run("build-runtime", &["cuda"], None, None).1,
            ["--build", "--backend", "cuda"]
        );
    }
}
#[test]
fn native_release_recipes_bundle_copies_product_and_checksum_not_host_binary() {
    let f = Fixture::new("bundle");
    executable(
        &f.root.join("bin/fixture-host"),
        "#!/bin/sh\nprintf 'mesh-llm 1.2.3\\n'\n",
    );
    executable(
        &f.root.join("scripts/package-release.sh"),
        r#"#!/bin/bash
set -euo pipefail
[[ "$1" == v1.2.3 ]]
printf 'product bytes\n' > "$2/mesh-llm-linux-x86_64.tar.gz"
printf 'product checksum\n' > "$2/mesh-llm-linux-x86_64.tar.gz.sha256"
printf 'version-specific bytes\n' > "$2/mesh-llm-v1.2.3-linux-x86_64.tar.gz"
"#,
    );
    let path = format!(
        "{}:{}",
        f.root.join("bin").display(),
        std::env::var("PATH").unwrap()
    );
    invoke(
        &f.root,
        &[
            "--justfile",
            "Justfile",
            "bundle",
            "output with spaces/product.tar.gz",
        ],
        &[("PATH", &path)],
    );
    assert_eq!(
        fs::read(f.root.join("output with spaces/product.tar.gz")).unwrap(),
        b"product bytes\n"
    );
    assert_eq!(
        fs::read(f.root.join("output with spaces/product.tar.gz.sha256")).unwrap(),
        b"product checksum\n"
    );
}
#[test]
fn native_release_recipes_windows_select_dynamic_hosts_and_cuda12_pascal_default() {
    for (name, backend) in [
        ("release-build-windows", "cpu"),
        ("release-build-cuda-windows", "cuda"),
        ("release-build-rocm-windows", "rocm"),
        ("release-build-vulkan-windows", "vulkan"),
    ] {
        let f = Fixture::new(name);
        executable(
            &f.root.join("bin/powershell"),
            r#"#!/bin/bash
set -euo pipefail
printf '%s\0' "$@" > "$FIXTURE_ROOT/windows.arguments"
"#,
        );
        let path = format!(
            "{}:{}",
            f.root.join("bin").display(),
            std::env::var("PATH").unwrap()
        );
        invoke(
            &f.root,
            &["--justfile", "Justfile", name],
            &[("PATH", &path), ("FIXTURE_ROOT", f.root.to_str().unwrap())],
        );
        let bytes = fs::read(f.root.join("windows.arguments")).unwrap();
        let args = bytes
            .split(|b| *b == 0)
            .filter(|s| !s.is_empty())
            .map(|s| std::str::from_utf8(s).unwrap())
            .collect::<Vec<_>>();
        assert!(args.contains(&"-DynamicHost"));
        assert!(!args.contains(&"-AbiOnly"));
        assert!(args.windows(2).any(|pair| pair == ["-Backend", backend]));
        assert!(
            args.windows(2)
                .any(|pair| pair == ["-File", "scripts/build-windows.ps1"])
        );
        assert!(
            args.windows(2)
                .any(|pair| pair == ["-BuildProfile", "release"])
        );
        if backend == "cuda" {
            assert!(
                args.windows(2)
                    .any(|pair| pair == ["-CudaArch", "61;75;80;86;87;89;90"])
            );
        }
    }
}
