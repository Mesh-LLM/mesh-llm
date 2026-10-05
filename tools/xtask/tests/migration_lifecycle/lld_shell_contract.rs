//! Copied real lld library/callers; fake tool boundaries, no compilation or install.
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
fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}
fn program(name: &str) -> PathBuf {
    std::env::split_paths(&std::env::var_os("PATH").unwrap())
        .map(|p| p.join(name))
        .find(|p| p.is_file() && fs::metadata(p).unwrap().permissions().mode() & 0o111 != 0)
        .unwrap_or_else(|| panic!("required fixture component:{name}"))
        .canonicalize()
        .unwrap()
}
fn executable(p: &Path, s: &str) {
    fs::create_dir_all(p.parent().unwrap()).unwrap();
    fs::write(p, s).unwrap();
    fs::set_permissions(p, fs::Permissions::from_mode(0o755)).unwrap();
}
struct Fixture {
    _temp: tempfile::TempDir,
    root: PathBuf,
}
struct Output {
    code: i32,
    stdout: String,
    stderr: String,
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let root = temp
            .path()
            .canonicalize()
            .unwrap()
            .join("linker fixture with spaces");
        fs::create_dir_all(root.join("bin")).unwrap();
        fs::create_dir_all(root.join("tmp")).unwrap();
        for name in [
            "bash", "mktemp", "rm", "cat", "dirname", "mkdir", "shasum", "awk", "mv", "cp", "chmod",
        ] {
            std::os::unix::fs::symlink(program(name), root.join("bin").join(name)).unwrap();
        }
        for path in [
            "scripts/lib/lld.sh",
            "scripts/cargo-linker",
            "scripts/cargo-linker-linux-aarch64",
            "scripts/cargo-linker-linux-x86_64",
            "scripts/build-host.sh",
            "scripts/lib/macos-deployment-target.sh",
            "scripts/lib/macos-deployment-target.txt",
        ] {
            let p = root.join(path);
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::copy(repo().join(path), &p).unwrap();
        }
        executable(
            &root.join("bin/uname"),
            "#!/bin/bash\ncase \"$1\" in -s) printf '%s\\n' \"$FAKE_OS\";;-m) printf '%s\\n' \"${FAKE_ARCH:-x86_64}\";;*)exit 95;;esac\n",
        );
        for name in [
            "cc",
            "target-cc",
            "aarch64-linux-gnu-gcc",
            "x86_64-linux-gnu-gcc",
        ] {
            executable(
                &root.join("bin").join(name),
                r#"#!/bin/bash
set -euo pipefail
if [[ "$1" == --version ]]; then echo 'finite compiler 1';exit 0;fi
printf '%s\0' "$0" "$@" >> "$FIXTURE_ROOT/compiler.arguments"
printf '%s\n' "${RUSTFLAGS:-}" "${RUSTC_WRAPPER:-}" "${COMPILER_ENV_SENTINEL:-}" > "$FIXTURE_ROOT/compiler.environment"
if [[ "$*" == *probe.c* ]]; then
 printf 'probe\n' >> "$FIXTURE_ROOT/compiler.events"
 if [[ "${CC_STATUS:-0}" != 0 ]];then echo 'ld64.lld: error: could not load TAPI file at /fixture-sdk/libSystem.tbd' >&2;exit "$CC_STATUS";fi
else printf 'link\n' >> "$FIXTURE_ROOT/compiler.events";fi
"#,
            );
        }
        executable(
            &root.join("bin/ld.lld"),
            "#!/bin/bash\necho 'finite linker 1'\n",
        );
        executable(
            &root.join("bin/ld64.lld"),
            "#!/bin/bash\necho 'finite linker 1'\n",
        );
        Self { _temp: temp, root }
    }
    fn run(&self, executable: PathBuf, args: &[&str], extra: &[(&str, &str)]) -> Output {
        let mut env: BTreeMap<_, _> = [
            ("PATH", self.root.join("bin").display().to_string()),
            ("HOME", self.root.display().to_string()),
            ("FIXTURE_ROOT", self.root.display().to_string()),
            ("TMPDIR", self.root.join("tmp").display().to_string()),
            ("FAKE_OS", "Linux".into()),
            ("MESH_LLM_MOLD", "finite-absent-mold".into()),
            (
                "MESH_LLM_LINKER_PROBE_CACHE_DIR",
                self.root.join("cache").display().to_string(),
            ),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), Value::Public(v.into())))
        .collect();
        for (k, v) in extra {
            env.insert((*k).into(), Value::Public((*v).into()));
        }
        let r = process::supervise_raw(
            &ProcessSpec {
                executable,
                cwd: self.root.clone(),
                environment: env,
                arguments: args.iter().map(|a| Value::Public((*a).into())).collect(),
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
        assert!(r.process.failure.is_none(), "{:?}", r.process);
        assert!(r.process.cleanup.complete);
        assert!(
            fs::read_dir(self.root.join("tmp"))
                .unwrap()
                .next()
                .is_none(),
            "probe directory leaked"
        );
        Output {
            code: r.process.status.unwrap().code().unwrap(),
            stdout: String::from_utf8(r.stdout.unwrap().as_bytes().to_vec()).unwrap(),
            stderr: String::from_utf8(r.stderr.unwrap().as_bytes().to_vec()).unwrap(),
        }
    }
    fn shell(&self, text: &str, extra: &[(&str, &str)]) -> Output {
        self.run(program("bash"), &["-c", text], extra)
    }
    fn lib(&self, text: &str, extra: &[(&str, &str)]) -> Output {
        self.shell(
            &format!("set -euo pipefail\nsource scripts/lib/lld.sh\n{text}"),
            extra,
        )
    }
    fn text(&self, path: &str) -> String {
        fs::read_to_string(self.root.join(path)).unwrap()
    }
    fn arguments(&self) -> Vec<String> {
        fs::read(self.root.join("compiler.arguments"))
            .unwrap()
            .split(|b| *b == 0)
            .filter(|x| !x.is_empty())
            .map(|s| String::from_utf8(s.to_vec()).unwrap())
            .collect()
    }
}
#[test]
fn lld_shell_contract_probe_uses_exact_linker_selected_compiler_and_diagnostics() {
    for (status, usable) in [("0", "USABLE"), ("17", "UNUSABLE")] {
        let f = Fixture::new();
        let r = f.lib(
            "lld_links lld && echo USABLE || echo UNUSABLE",
            &[("CC_STATUS", status)],
        );
        assert_eq!(r.code, 0);
        assert_eq!(r.stdout.trim(), usable);
        assert!(f.arguments().contains(&"-fuse-ld=lld".into()));
        assert_eq!(f.text("compiler.events"), "probe\n");
    }
    let f = Fixture::new();
    let r=f.lib("MESH_LLM_CC=target-cc lld_links '/selected linker/ld64.lld' -target 'target with spaces' || printf '%s' \"$LLD_PROBE_OUTPUT\"",&[("CC_STATUS","17"),("COMPILER_ENV_SENTINEL","retained value")]);
    assert_eq!(r.code, 0);
    assert!(r.stdout.contains("could not load TAPI file"));
    let args = f.arguments();
    assert!(args[0].ends_with("/target-cc"));
    assert_eq!(
        &args[1..4],
        [
            "-target",
            "target with spaces",
            "-fuse-ld=/selected linker/ld64.lld"
        ]
    );
    assert!(f.text("compiler.environment").contains("retained value"));
    let f = Fixture::new();
    let r = f.lib(
        "MESH_LLM_CC=missing-cc lld_links lld || printf '%s' \"$LLD_PROBE_OUTPUT\"",
        &[],
    );
    assert_eq!(r.code, 0);
    assert!(r.stdout.contains("no C compiler driver (missing-cc)"));
    assert!(!f.root.join("compiler.events").exists());
}
#[test]
fn lld_shell_contract_resolve_success_failure_missing_and_unsupported_are_soft() {
    let f = Fixture::new();
    let r = f.lib(
        "selected=\"$(resolve_usable_lld)\"; [[ \"$selected\" == \"$(find_lld)\" ]] && echo SAME",
        &[],
    );
    assert_eq!(r.code, 0);
    assert_eq!(r.stdout.trim(), "SAME");
    assert_eq!(r.stderr, "");
    let f = Fixture::new();
    let r = f.lib(
        "printf '[%s]' \"$(resolve_usable_lld)\"; resolve_usable_lld >/dev/null; echo SURVIVED",
        &[("CC_STATUS", "17")],
    );
    assert_eq!(r.code, 0);
    assert_eq!(r.stdout.trim(), "[]SURVIVED");
    for expected in [
        "cannot link against the active SDK",
        "platform default linker",
        "could not load TAPI file",
    ] {
        assert!(r.stderr.contains(expected));
    }
    let f = Fixture::new();
    fs::remove_file(f.root.join("bin/ld.lld")).unwrap();
    let r = f.lib("printf '[%s]' \"$(resolve_usable_lld)\"", &[]);
    assert_eq!(r.code, 0);
    assert_eq!(r.stdout, "[]");
    assert_eq!(r.stderr, "");
    assert!(!f.root.join("compiler.events").exists());
    let r = f.lib(
        "printf '[%s]' \"$(find_lld)\"",
        &[("FAKE_OS", "Unsupported")],
    );
    assert_eq!(r.code, 0);
    assert_eq!(r.stdout, "[]");
    let f = Fixture::new();
    let r = f.lib("find_lld", &[("FAKE_OS", "Darwin")]);
    assert_eq!(r.code, 0);
    assert_eq!(
        r.stdout.trim(),
        f.root.join("bin/ld64.lld").to_str().unwrap()
    );
}
#[test]
fn lld_shell_contract_actual_driver_keeps_target_arguments_and_falls_back_after_failed_probe() {
    for status in ["0", "17"] {
        let f = Fixture::new();
        let r = f.run(
            program("bash"),
            &[
                "scripts/cargo-linker",
                "-target",
                "x86_64-unknown-linux-gnu",
                "-isysroot",
                "sdk with spaces",
                "object with spaces.o",
                "-fuse-ld=unprobed",
            ],
            &[
                ("CC_STATUS", status),
                ("COMPILER_ENV_SENTINEL", "unchanged"),
                ("RUSTFLAGS", "caller flags"),
                ("RUSTC_WRAPPER", "caller cache"),
            ],
        );
        assert_eq!(r.code, 0, "{}", r.stderr);
        let args = f.arguments();
        assert!(!args.contains(&"-fuse-ld=unprobed".into()));
        assert_eq!(f.text("compiler.events"), "probe\nlink\n");
        assert!(
            args.windows(2)
                .any(|p| p == ["-isysroot", "sdk with spaces"])
        );
        assert!(args.contains(&"object with spaces.o".into()));
        assert_eq!(
            args.iter().filter(|a| a.starts_with("-fuse-ld=")).count(),
            if status == "0" { 2 } else { 1 }
        );
        assert_eq!(
            f.text("compiler.environment"),
            "caller flags\ncaller cache\nunchanged\n"
        );
    }
    let f = Fixture::new();
    let r = f.run(
        program("bash"),
        &["scripts/cargo-linker-linux-aarch64", "object.o"],
        &[],
    );
    assert_eq!(r.code, 0, "{}", r.stderr);
    assert!(f.arguments()[0].ends_with("/aarch64-linux-gnu-gcc"));
    let f = Fixture::new();
    let r = f.run(
        program("bash"),
        &["scripts/cargo-linker", "object.o"],
        &[("MESH_LLM_LINK_TARGET", "unsupported-target")],
    );
    assert_eq!(r.code, 1);
    assert!(r.stderr.contains("unsupported host/target linker pair"));
    assert!(!f.root.join("compiler.events").exists());
}
#[test]
fn lld_shell_contract_host_keeps_repository_and_operator_cargo_defaults() {
    let f = Fixture::new();
    executable(
        &f.root.join("bin/cargo"),
        "#!/bin/bash\nprintf '%s\\n' \"${RUSTFLAGS:-}\" \"${RUSTC_WRAPPER:-}\" > \"$FIXTURE_ROOT/host.environment\"\nprintf '%s\\0' \"$@\" > \"$FIXTURE_ROOT/host.arguments\"\n",
    );
    let r = f.run(
        program("bash"),
        &["scripts/build-host.sh", "--profile", "release"],
        &[
            ("MESH_LLM_SKIP_UI", "1"),
            ("MESH_LLM_BUILD_VERSION", "1.2.3"),
            ("RUSTFLAGS", "operator flags"),
            ("RUSTC_WRAPPER", "operator cache"),
        ],
    );
    assert_eq!(r.code, 0, "{}", r.stderr);
    assert_eq!(
        f.text("host.environment"),
        "operator flags\noperator cache\n"
    );
    assert!(!f.root.join("compiler.events").exists());
}
#[test]
fn lld_shell_contract_cargo_configuration_owns_wrapper_and_target_drivers() {
    let cfg: toml::Value =
        toml::from_str(&fs::read_to_string(repo().join(".cargo/config.toml")).unwrap()).unwrap();
    assert_eq!(cfg["build"]["rustc-wrapper"].as_str(), Some("sccache"));
    for (target, driver) in [
        ("aarch64-apple-darwin", "scripts/cargo-linker"),
        ("x86_64-apple-darwin", "scripts/cargo-linker"),
        (
            "aarch64-unknown-linux-gnu",
            "scripts/cargo-linker-linux-aarch64",
        ),
        (
            "x86_64-unknown-linux-gnu",
            "scripts/cargo-linker-linux-x86_64",
        ),
        ("aarch64-pc-windows-msvc", "scripts/cargo-linker.cmd"),
        ("x86_64-pc-windows-msvc", "scripts/cargo-linker.cmd"),
    ] {
        assert_eq!(cfg["target"][target]["linker"].as_str(), Some(driver));
    }
    fn unprobed(v: &toml::Value) -> bool {
        match v {
            toml::Value::String(s) => s.contains("fuse-ld"),
            toml::Value::Array(a) => a.iter().any(unprobed),
            toml::Value::Table(t) => t.values().any(unprobed),
            _ => false,
        }
    }
    assert!(!unprobed(&cfg), "Cargo must not bypass driver probing");
}
#[test]
fn lld_shell_contract_native_with_lld_probes_before_command_and_preserves_operator_environment() {
    for mode in ["success", "failed-probe", "missing-cache"] {
        let f = Fixture::new();
        let shown = f.run(
            program("just"),
            &[
                "--justfile",
                repo().join("Justfile").to_str().unwrap(),
                "--show",
                "with-lld",
            ],
            &[],
        );
        assert_eq!(shown.code, 0, "{}", shown.stderr);
        fs::write(f.root.join("Justfile"), shown.stdout).unwrap();
        executable(&f.root.join("bin/sccache"), "#!/bin/bash\nexit 0\n");
        executable(
            &f.root.join("scripts/cargo-linker"),
            "#!/bin/bash\n[[ \"$#\" == 1 && \"$1\" == --mesh-probe ]] || exit 98\nprintf 'probe\\n' >> \"$FIXTURE_ROOT/caller.events\"\nexit \"${PROBE_STATUS:-0}\"\n",
        );
        executable(
            &f.root.join("bin/fixture-command"),
            "#!/bin/bash\nprintf 'command\\n' >> \"$FIXTURE_ROOT/caller.events\"\nprintf '%s\\n' \"$#\" \"$1\" \"${RUSTFLAGS:-}\" \"${RUSTC_WRAPPER:-}\" > \"$FIXTURE_ROOT/caller.values\"\n",
        );
        if mode == "missing-cache" {
            fs::remove_file(f.root.join("bin/sccache")).unwrap();
        }
        let r = f.run(
            program("just"),
            &[
                "--justfile",
                "Justfile",
                "with-lld",
                "fixture-command",
                "'spaced value'",
            ],
            &[
                (
                    "PROBE_STATUS",
                    if mode == "failed-probe" { "9" } else { "0" },
                ),
                ("RUSTFLAGS", "operator flags"),
                ("RUSTC_WRAPPER", "operator cache"),
            ],
        );
        match mode {
            "success" => {
                assert_eq!(r.code, 0, "{}", r.stderr);
                assert_eq!(f.text("caller.events"), "probe\ncommand\n");
                assert_eq!(
                    f.text("caller.values"),
                    "1\nspaced value\noperator flags\noperator cache\n"
                );
            }
            "failed-probe" => {
                assert_ne!(r.code, 0);
                assert_eq!(f.text("caller.events"), "probe\n");
                assert!(!f.root.join("caller.values").exists());
            }
            _ => {
                assert_ne!(r.code, 0);
                assert!(r.stderr.contains("sccache is required"));
                assert!(!f.root.join("caller.events").exists());
            }
        }
    }
}
use super::workflow_yaml;
#[test]
fn lld_shell_contract_macos_action_installs_tools_and_probes_with_disk_cache_fallback() {
    let f = Fixture::new();
    let source =
        fs::read_to_string(repo().join(".github/actions/setup-macos-lld/action.yml")).unwrap();
    let action = workflow_yaml::parse(&source).unwrap();
    let workflow_yaml::Node::Seq(steps) = action.get("runs").unwrap().get("steps").unwrap() else {
        panic!("action steps")
    };
    assert_eq!(steps.len(), 2);
    let scalars = steps
        .iter()
        .map(|step| {
            assert_eq!(step.get("shell").unwrap().text(), Some("bash"));
            step.get("run").unwrap().text().unwrap()
        })
        .collect::<Vec<_>>();
    fs::create_dir_all(f.root.join("brew/bin")).unwrap();
    executable(&f.root.join("brew/bin/ld64.lld"), "#!/bin/bash\nexit 0\n");
    executable(
        &f.root.join("sccache-template"),
        "#!/bin/bash\nprintf '%s\\n' \"$*\" >> \"$FIXTURE_ROOT/cache.calls\"\n[[ \"$1\" == --version ]] && echo 'finite sccache'\nexit 0\n",
    );
    executable(
        &f.root.join("bin/brew"),
        r#"#!/bin/bash
set -euo pipefail
printf '%s\n' "$*" >> "$FIXTURE_ROOT/brew.calls"
if [[ "$1" == --prefix && "$2" == lld ]];then printf '%s\n' "$FIXTURE_ROOT/brew";exit 0;fi
[[ "$1" == install ]] || exit 97
if [[ "$2" == sccache ]];then cp "$FIXTURE_ROOT/sccache-template" "$FIXTURE_ROOT/bin/sccache";chmod +x "$FIXTURE_ROOT/bin/sccache";fi
"#,
    );
    executable(
        &f.root.join("scripts/cargo-linker"),
        r#"#!/bin/bash
set -euo pipefail
[[ "$#" == 1 && "$1" == --mesh-probe ]] || exit 96
[[ "$PATH" == "$FIXTURE_ROOT/brew/bin:"* ]] || exit 95
printf '%s\n' "$SCCACHE_GHA_ENABLED" "$SCCACHE_MULTILEVEL_CHAIN" "$SCCACHE_DIR" > "$FIXTURE_ROOT/action.probe"
"#,
    );
    let output = f.root.join("github.env");
    let github_path = f.root.join("github.path");
    let runner = f.root.join("runner temp");
    let r = f.shell(
        &scalars.join("\n"),
        &[
            ("SCCACHE_GHA_ENABLED", "true"),
            ("GITHUB_ENV", output.to_str().unwrap()),
            ("GITHUB_PATH", github_path.to_str().unwrap()),
            ("RUNNER_TEMP", runner.to_str().unwrap()),
        ],
    );
    assert_eq!(r.code, 0, "{}", r.stderr);
    assert_eq!(
        f.text("brew.calls"),
        "install lld\ninstall sccache\n--prefix lld\n"
    );
    assert_eq!(
        f.text("cache.calls"),
        "--version\n--stop-server\n--start-server\n"
    );
    assert_eq!(
        f.text("action.probe"),
        format!(
            "false\ndisk\n{}\n",
            runner.join("mesh-llm-sccache").display()
        )
    );
    let env = f.text("github.env");
    assert!(env.contains("SCCACHE_GHA_ENABLED=false\n"));
    assert!(env.contains("SCCACHE_MULTILEVEL_CHAIN=disk\n"));
    assert_eq!(
        f.text("github.path"),
        format!("{}\n", f.root.join("brew/bin").display())
    );
}

#[path = "lld_shell_contract/cargo_driver.rs"]
mod cargo_driver;
