//! Execute the actual runtime preparation action with finite package observers.
use super::{
    support::{Fixture, action},
    workflow_yaml::Node,
};
use std::{fs, process::Command};
fn prepare() -> Node {
    let document = action("prepare-native-runtime-input");
    let Node::Seq(steps) = document.get("runs").unwrap().get("steps").unwrap() else {
        panic!("action steps")
    };
    let [install, step] = steps.as_slice() else {
        panic!("just install and runtime preparation steps required")
    };
    assert_eq!(
        install.get("uses").and_then(Node::text),
        Some("taiki-e/install-action@3d23c1bbdafe696dfccad2664945a04f47d03dc3")
    );
    assert_eq!(step.get("id").and_then(Node::text), Some("prepare"));
    assert_eq!(step.get("shell").and_then(Node::text), Some("bash"));
    for key in ["backend", "target", "output_dir", "build"] {
        let env_key = format!("INPUT_{}", key.to_uppercase());
        assert_eq!(
            step.get("env").unwrap().get(&env_key).and_then(Node::text),
            Some(format!("${{{{ inputs.{key} }}}}").as_str())
        );
    }
    step.clone()
}
fn fixture() -> Fixture {
    let fixture = Fixture::new();
    fs::create_dir_all(fixture.path().join("scripts")).unwrap();
    fixture.executable("package-observer",r#"expected=(--backend cpu --out 'runtime output')
[[ "$INPUT_BUILD" != true ]] || expected=(--build "${expected[@]}")
[[ -z "$INPUT_TARGET" ]] || expected+=(--target "$INPUT_TARGET")
[[ "$#" == "${#expected[@]}" ]] || exit 97
actual=("$@")
for ((index=0;index<$#;index++)); do [[ "${actual[index]}" == "${expected[index]}" ]] || exit 97; done
printf 'package\n' > events
[[ "$FAILURE" != package ]] || exit 97
mkdir -p 'runtime output/cpu-runtime'
printf '{}\n' > 'runtime output/cpu-runtime/manifest.json'
[[ "$FAILURE" != duplicate ]] || { mkdir -p 'runtime output/other-runtime'; printf '{}\n' > 'runtime output/other-runtime/manifest.json'; }
[[ "$FAILURE" == archive ]] || printf 'finite archive\n' > 'runtime output/cpu-runtime.tar.gz'
printf 'finite checksum\n' > 'runtime output/cpu-runtime.tar.gz.sha256'
"#);
    fixture.executable("verify-observer",r#"[[ "$#" == 2 && "$1" == 'runtime output/cpu-runtime' && "$2" == 'runtime output/cpu-runtime.tar.gz' ]] || exit 97
printf 'verify\n' >> events
[[ "$FAILURE" != verify ]] || exit 97
"#);
    for (from, to) in [
        ("package-observer", "package-native-runtime.sh"),
        ("verify-observer", "verify-native-runtime-package.sh"),
    ] {
        fs::copy(
            fixture.path().join("bin").join(from),
            fixture.path().join("scripts").join(to),
        )
        .unwrap();
    }
    for tool in ["cargo", "just", "cmake", "rustc"] {
        fixture.executable(tool, "printf 'forbidden build\\n' >> events; exit 97");
    }
    fixture
}
fn run(fixture: &Fixture, build: &str, target: &str, failure: &str) -> std::process::Output {
    let mut command = Command::new("/bin/bash");
    command.env_clear().current_dir(fixture.path());
    command.env(
        "PATH",
        format!("{}:/usr/bin:/bin", fixture.path().join("bin").display()),
    );
    command.env("GITHUB_OUTPUT", fixture.path().join("outputs"));
    for (key, value) in [
        ("INPUT_BACKEND", "cpu"),
        ("INPUT_OUTPUT_DIR", "runtime output"),
        ("INPUT_BUILD", build),
        ("INPUT_TARGET", target),
        ("FAILURE", failure),
    ] {
        command.env(key, value);
    }
    command.args(["-c", prepare().get("run").unwrap().text().unwrap()]);
    fixture.run(command)
}
#[test]
fn runtime_action_packages_only_the_runtime_and_publishes_verified_absolute_outputs() {
    for (build, target) in [
        ("true", ""),
        ("true", "aarch64-unknown-linux-gnu"),
        ("false", "x86_64-unknown-linux-gnu"),
    ] {
        let fixture = fixture();
        let result = run(&fixture, build, target, "none");
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(
            fs::read_to_string(fixture.path().join("events")).unwrap(),
            "package\nverify\n"
        );
        let output = fs::read_to_string(fixture.path().join("outputs")).unwrap();
        let base = fixture
            .path()
            .canonicalize()
            .unwrap()
            .join("runtime output/cpu-runtime");
        assert_eq!(
            output,
            format!(
                "runtime_dir={}\narchive_path={}.tar.gz\nchecksum_path={}.tar.gz.sha256\n",
                base.display(),
                base.display(),
                base.display()
            )
        );
    }
}
#[test]
fn runtime_action_rejects_failed_package_verification_and_ambiguous_or_missing_inputs_before_publication()
 {
    for failure in ["package", "verify", "duplicate", "archive"] {
        let fixture = fixture();
        let result = run(&fixture, "true", "", failure);
        assert!(
            !result.status.success(),
            "failure case {failure} accepted; stdout={}, stderr={}, events={:?}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr),
            fs::read_to_string(fixture.path().join("events"))
        );
        assert!(!fixture.path().join("outputs").exists());
        let expected = if failure == "verify" {
            "package\nverify\n"
        } else {
            "package\n"
        };
        assert_eq!(
            fs::read_to_string(fixture.path().join("events")).unwrap(),
            expected
        );
    }
}
