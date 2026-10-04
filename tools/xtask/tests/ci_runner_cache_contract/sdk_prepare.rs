//! Actual SDK preparation step with finite package/verifier/crate boundaries.
use super::{
    support::{Fixture, action, root, step},
    workflow_yaml::Node,
};
use serde_json::json;
use std::{
    collections::BTreeMap,
    fs,
    os::unix::fs::PermissionsExt,
    path::Path,
    process::{Command, Output},
};

fn script(path: &Path, body: &str) {
    fs::write(path, format!("#!/bin/bash\nset -euo pipefail\n{body}\n")).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}
struct NativeFixture {
    runner: Fixture,
}
impl NativeFixture {
    fn new() -> Self {
        let runner = Fixture(
            tempfile::Builder::new()
                .prefix("SDK producer with spaces ")
                .tempdir()
                .unwrap(),
        );
        for dir in ["bin", "scripts/lib", "home", "tmp"] {
            fs::create_dir_all(runner.path().join(dir)).unwrap();
        }
        fs::copy(
            root().join("scripts/lib/automation.sh"),
            runner.path().join("scripts/lib/automation.sh"),
        )
        .unwrap();
        runner.executable(
            "uname",
            r#"case "$*" in -m) printf 'x86_64\n';; -s) printf 'Linux\n';; *) exit 97;; esac"#,
        );
        for name in ["cargo", "just", "cmake", "rustc", "cc"] {
            runner.executable(name, "printf forbidden >> \"$EVENTS\"; exit 98");
        }
        script(
            &runner.path().join("scripts/package-native-sdk.sh"),
            PACKAGE,
        );
        script(
            &runner.path().join("scripts/verify-native-sdk-package.sh"),
            VERIFY,
        );
        script(
            &runner.path().join("scripts/package-native-sdk-crate.sh"),
            CRATE,
        );
        Self { runner }
    }
    fn invoke(&self, mode: &str, extra: &[(&str, &str)]) -> Output {
        let mut values = BTreeMap::from([
            ("INPUT_BACKEND", "cpu"),
            ("INPUT_TARGET", "x86_64-unknown-linux-gnu"),
            ("INPUT_PROFILE", "debug"),
            ("INPUT_OUTPUT_DIR", "out"),
            ("INPUT_INCLUDE_RUNTIME_CRATE", "false"),
            ("INPUT_REQUIRE_PREBUILT_STATIC_ABI", "false"),
        ]);
        values.extend(extra.iter().copied());
        let mut manifest = json!({"target_triple":values["INPUT_TARGET"],"backend":values["INPUT_BACKEND"],"cargo_profile":values["INPUT_PROFILE"]});
        if mode == "identity" {
            manifest["target_triple"] = "unexpected-target".into();
        }
        fs::write(
            self.runner.path().join("input-manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let mut command = Command::new("/bin/bash");
        command
            .env_clear()
            .current_dir(self.runner.path())
            .env(
                "PATH",
                format!("{}:/usr/bin:/bin", self.runner.path().join("bin").display()),
            )
            .env("HOME", self.runner.path().join("home"))
            .env("TMPDIR", self.runner.path().join("tmp"))
            .env(
                "GITHUB_WORKSPACE",
                self.runner.path().canonicalize().unwrap(),
            )
            .env("GITHUB_OUTPUT", self.runner.path().join("outputs"))
            .env("EVENTS", self.runner.path().join("events"))
            .env("MANIFEST", self.runner.path().join("input-manifest.json"))
            .env("MODE", mode)
            .env("MESH_LLM_AUTOMATION_BIN", env!("CARGO_BIN_EXE_xtask"));
        for (key, value) in values {
            command.env(key, value);
        }
        command.args(["-c", &step(&action("prepare-native-sdk-input"), "run")]);
        self.runner.run(command)
    }
    fn events(&self) -> String {
        fs::read_to_string(self.runner.path().join("events")).unwrap_or_default()
    }
}
const PACKAGE: &str = r#"
[[ "$1" == --build && "$2" == --backend && "$3" == "$INPUT_BACKEND" && "$4" == --target && "$5" == "$INPUT_TARGET" && "$6" == --profile && "$7" == "$INPUT_PROFILE" && "$8" == --out && "$9" == "$INPUT_OUTPUT_DIR" ]] || exit 97
if [[ "$INPUT_REQUIRE_PREBUILT_STATIC_ABI" == true ]]; then [[ "$#" == 10 && "${10}" == --require-prebuilt-llama ]] || exit 97; else [[ "$#" == 9 ]] || exit 97; fi
printf 'package\n' >> "$EVENTS"
[[ "$MODE" != package ]] || exit 73
mkdir -p "$INPUT_OUTPUT_DIR/meshllm-native-fixture"
cp "$MANIFEST" "$INPUT_OUTPUT_DIR/meshllm-native-fixture/manifest.json"
if [[ "$MODE" != missing-archive ]]; then printf 'immutable archive\n' > "$INPUT_OUTPUT_DIR/sdk.tar.gz"; fi
if [[ "$MODE" == duplicate-archive ]]; then printf 'second archive\n' > "$INPUT_OUTPUT_DIR/second.tar.gz"; fi
sidecar=sdk.tar.gz.sha256
if [[ "$MODE" == wrong-sidecar ]]; then sidecar=other.tar.gz.sha256; fi
printf 'immutable checksum\n' > "$INPUT_OUTPUT_DIR/$sidecar"
"#;
const VERIFY: &str = r#"
[[ "$#" == 1 ]] || exit 97
case "$1" in
  "$INPUT_OUTPUT_DIR/sdk.tar.gz")
    [[ -f "$1" ]] || exit 97; printf 'verify-archive\n' >> "$EVENTS"; [[ "$MODE" != verify-archive ]] || exit 74;;
  "$INPUT_OUTPUT_DIR/meshllm-native-fixture")
    [[ -f "$1/manifest.json" ]] || exit 97; printf 'verify-directory\n' >> "$EVENTS"; [[ "$MODE" != verify-directory ]] || exit 75;;
  *) exit 97;;
esac
"#;
const CRATE: &str = r#"
[[ "$#" == 3 && "$1" == --out && "$2" == "$INPUT_OUTPUT_DIR-crates" && "$3" == "$INPUT_OUTPUT_DIR/meshllm-native-fixture" ]] || exit 97
printf 'crate\n' >> "$EVENTS"
mkdir -p "$2/fixture/target/package"
if [[ "$MODE" != missing-crate ]]; then printf 'immutable runtime crate\n' > "$2/fixture/target/package/runtime.crate"; fi
if [[ "$MODE" == duplicate-crate ]]; then printf 'second crate\n' > "$2/fixture/target/package/second.crate"; fi
"#;
fn bindings(document: &Node) -> bool {
    let Some(Node::Seq(steps)) = document.get("runs").and_then(|n| n.get("steps")) else {
        return false;
    };
    let [prepare] = steps.as_slice() else {
        return false;
    };
    if prepare.get("id").and_then(Node::text) != Some("prepare")
        || prepare.get("shell").and_then(Node::text) != Some("bash")
    {
        return false;
    }
    let Some(env) = prepare.get("env") else {
        return false;
    };
    let inputs = [
        ("INPUT_BACKEND", "backend"),
        ("INPUT_TARGET", "target"),
        ("INPUT_PROFILE", "profile"),
        ("INPUT_OUTPUT_DIR", "output_dir"),
        ("INPUT_INCLUDE_RUNTIME_CRATE", "include_runtime_crate"),
        (
            "INPUT_REQUIRE_PREBUILT_STATIC_ABI",
            "require_prebuilt_static_abi",
        ),
    ];
    if !inputs.iter().all(|(key, input)| {
        env.get(key).and_then(Node::text) == Some(format!("${{{{ inputs.{input} }}}}").as_str())
    }) {
        return false;
    }
    [
        "archive_path",
        "checksum_path",
        "artifact_dir",
        "upload_path",
    ]
    .iter()
    .all(|key| {
        document
            .get("outputs")
            .and_then(|n| n.get(key))
            .and_then(|n| n.get("value"))
            .and_then(Node::text)
            == Some(format!("${{{{ steps.prepare.outputs.{key} }}}}").as_str())
    })
}
#[test]
fn actual_native_sdk_action_binds_declared_inputs_and_outputs_to_its_prepare_step() {
    use super::workflow_yaml;
    let source =
        fs::read_to_string(root().join(".github/actions/prepare-native-sdk-input/action.yml"))
            .unwrap();
    assert!(bindings(&workflow_yaml::parse(&source).unwrap()));
    for (valid, invalid) in [
        (
            "INPUT_TARGET: ${{ inputs.target }}",
            "INPUT_TARGET: unexpected-target",
        ),
        (
            "steps.prepare.outputs.archive_path",
            "steps.unrelated.outputs.archive_path",
        ),
    ] {
        assert!(source.contains(valid));
        let changed = format!("{}\n# {valid}\n", source.replace(valid, invalid));
        assert!(!bindings(&workflow_yaml::parse(&changed).unwrap()));
    }
}
#[test]
fn actual_native_sdk_action_preserves_package_arguments_verification_order_and_flat_upload_bytes() {
    for include in [false, true] {
        let fixture = NativeFixture::new();
        let extra = if include {
            vec![
                ("INPUT_INCLUDE_RUNTIME_CRATE", "true"),
                ("INPUT_REQUIRE_PREBUILT_STATIC_ABI", "true"),
                ("INPUT_PROFILE", "release"),
            ]
        } else {
            vec![]
        };
        let result = fixture.invoke("valid", &extra);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(
            fixture.events(),
            if include {
                "package\nverify-archive\nverify-directory\ncrate\n"
            } else {
                "package\nverify-archive\nverify-directory\n"
            }
        );
        let mut names = fs::read_dir(fixture.runner.path().join("out-upload"))
            .unwrap()
            .map(|e| e.unwrap().file_name().to_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        names.sort();
        let expected = if include {
            vec!["runtime.crate", "sdk.tar.gz", "sdk.tar.gz.sha256"]
        } else {
            vec!["sdk.tar.gz", "sdk.tar.gz.sha256"]
        };
        assert_eq!(names, expected);
        for name in names {
            let source = if name == "runtime.crate" {
                fixture
                    .runner
                    .path()
                    .join("out-crates/fixture/target/package")
                    .join(&name)
            } else {
                fixture.runner.path().join("out").join(&name)
            };
            assert_eq!(
                fs::read(source).unwrap(),
                fs::read(fixture.runner.path().join("out-upload").join(name)).unwrap()
            );
        }
        let output = fixture.runner.outputs();
        let physical = fixture.runner.path().canonicalize().unwrap();
        for (key, suffix) in [
            ("archive_path", "out/sdk.tar.gz"),
            ("checksum_path", "out/sdk.tar.gz.sha256"),
            ("artifact_dir", "out/meshllm-native-fixture"),
            ("upload_path", "out-upload/*"),
        ] {
            assert_eq!(output[key], physical.join(suffix).display().to_string());
        }
    }
}
#[test]
fn actual_native_sdk_action_failure_stops_before_upload_or_output_publication() {
    for (mode, expected) in [
        ("package", "package\n"),
        ("verify-archive", "package\nverify-archive\n"),
        (
            "verify-directory",
            "package\nverify-archive\nverify-directory\n",
        ),
        ("identity", "package\nverify-archive\nverify-directory\n"),
    ] {
        let fixture = NativeFixture::new();
        let result = fixture.invoke(mode, &[]);
        assert!(!result.status.success());
        assert_eq!(fixture.events(), expected);
        assert!(!fixture.runner.path().join("out-upload").exists());
        assert!(!fixture.runner.path().join("outputs").exists());
        if mode == "identity" {
            assert!(String::from_utf8_lossy(&result.stderr).contains("target_triple mismatch"));
        }
    }
}
#[test]
fn actual_native_sdk_action_refuses_ambiguous_or_mismatched_package_outputs_before_verification() {
    for mode in ["missing-archive", "duplicate-archive", "wrong-sidecar"] {
        let fixture = NativeFixture::new();
        let result = fixture.invoke(mode, &[]);
        assert!(!result.status.success());
        assert_eq!(fixture.events(), "package\n");
        assert!(!fixture.runner.path().join("outputs").exists());
    }
}
#[test]
fn actual_native_sdk_action_refuses_invalid_inputs_and_preserves_existing_output_namespaces() {
    for input in [
        ("INPUT_BACKEND", "unknown"),
        ("INPUT_TARGET", "unsupported"),
        ("INPUT_TARGET", "aarch64-unknown-linux-gnu"),
        ("INPUT_PROFILE", "unknown"),
        ("INPUT_INCLUDE_RUNTIME_CRATE", "1"),
        ("INPUT_REQUIRE_PREBUILT_STATIC_ABI", "1"),
        ("INPUT_OUTPUT_DIR", "../outside"),
    ] {
        let fixture = NativeFixture::new();
        let result = fixture.invoke("valid", &[input]);
        assert!(!result.status.success());
        assert!(fixture.events().is_empty());
        assert!(!fixture.runner.path().join("outputs").exists());
    }
    for existing in ["out", "out-upload"] {
        let fixture = NativeFixture::new();
        fs::create_dir(fixture.runner.path().join(existing)).unwrap();
        fs::write(
            fixture.runner.path().join(existing).join("sentinel"),
            b"preserve",
        )
        .unwrap();
        let result = fixture.invoke("valid", &[]);
        assert!(!result.status.success());
        assert_eq!(
            fs::read(fixture.runner.path().join(existing).join("sentinel")).unwrap(),
            b"preserve"
        );
        assert!(!fixture.runner.path().join("outputs").exists());
    }
}
#[test]
fn actual_native_sdk_action_requires_exactly_one_runtime_crate_before_upload_publication() {
    for mode in ["missing-crate", "duplicate-crate"] {
        let fixture = NativeFixture::new();
        let result = fixture.invoke(mode, &[("INPUT_INCLUDE_RUNTIME_CRATE", "true")]);
        assert!(!result.status.success());
        assert_eq!(
            fixture.events(),
            "package\nverify-archive\nverify-directory\ncrate\n"
        );
        assert!(!fixture.runner.path().join("outputs").exists());
    }
}
