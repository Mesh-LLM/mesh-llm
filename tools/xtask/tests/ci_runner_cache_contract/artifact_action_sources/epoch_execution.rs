//! Execute the actual epoch resolver using bounded fake native tools.
use super::super::support::{self, Fixture};
use sha2::{Digest, Sha256};
use std::{fs, process::Command};
fn epoch_case(versions: &str, pinned: &str, cmake: &str) -> (Fixture, std::process::Output) {
    let fixture = Fixture::new();
    for command in ["sw_vers", "xcodebuild", "clang", "cmake", "ninja"] {
        fixture.executable(
            command,
            &format!(
                "printf '%s %s\\n' \"${{0##*/}}\" \"$*\" >> \"$EPOCH_FIXTURE_CALLS\"\nprintf '%s\\n' 'fixture-{command}-{}'",
                if command == "cmake" { cmake } else { "1" }
            ),
        );
    }
    let mut command = Command::new("bash");
    command
        .args([
            "-c",
            &support::step(&support::action("resolve-native-toolchain-epoch"), "run"),
        ])
        .env_clear()
        .env(
            "PATH",
            format!("{}:/usr/bin:/bin", fixture.path().join("bin").display()),
        )
        .env("GITHUB_OUTPUT", fixture.path().join("outputs"))
        .env("GITHUB_ENV", fixture.path().join("environment"))
        .env("EPOCH_FIXTURE_CALLS", fixture.path().join("tool-calls"))
        .env("INPUT_PINNED_EPOCH", pinned)
        .env("INPUT_INCLUDE_TOOL_VERSIONS", versions)
        .env("RUNNER_OS_VALUE", "macOS")
        .env("RUNNER_ARCH_VALUE", "ARM64");
    let output = fixture.run(command);
    (fixture, output)
}
#[test]
fn artifact_epoch_without_image_variables_fingerprints_tools_and_matches_build_environment() {
    let (first, output) = epoch_case("true", "", "1");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let first_output = fs::read_to_string(first.path().join("outputs")).unwrap();
    let epoch = first_output.trim().strip_prefix("epoch=").unwrap();
    let hash = epoch.strip_prefix("runner-macOS-ARM64-native-").unwrap();
    assert_eq!(hash.len(), 64);
    assert!(hash.bytes().all(|byte| byte.is_ascii_hexdigit()));
    let tool_bytes = b"fixture-sw_vers-1\nfixture-xcodebuild-1\nfixture-clang-1\nfixture-cmake-1\nfixture-ninja-1\n";
    assert_eq!(hash, hex::encode(Sha256::digest(tool_bytes)));
    assert_eq!(
        fs::read_to_string(first.path().join("tool-calls")).unwrap(),
        "sw_vers -productVersion\nxcodebuild -version\nclang --version\ncmake --version\nninja --version\n"
    );
    assert_eq!(
        fs::read_to_string(first.path().join("environment")).unwrap(),
        format!("MESH_LLM_LLAMA_TOOLCHAIN_EPOCH={epoch}\n")
    );
    let (second, output) = epoch_case("true", "", "2");
    assert!(output.status.success());
    assert_ne!(
        first_output,
        fs::read_to_string(second.path().join("outputs")).unwrap()
    );
    let (denied, output) = epoch_case("false", "", "1");
    assert!(!output.status.success());
    assert!(!denied.path().join("outputs").exists());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("ImageOS and ImageVersion are required")
    );
}
#[test]
fn artifact_epoch_pinned_identity_is_exact_and_conflicting_or_unsafe_modes_fail_closed() {
    let (fixture, output) = epoch_case("false", "pinned.exact-123", "1");
    assert!(output.status.success());
    assert!(!fixture.path().join("tool-calls").exists());
    assert_eq!(
        fs::read_to_string(fixture.path().join("outputs")).unwrap(),
        "epoch=pinned.exact-123\n"
    );
    for (versions, pinned) in [
        ("true", "pinned"),
        ("invalid", "pinned"),
        ("false", "unsafe/path"),
    ] {
        let (fixture, output) = epoch_case(versions, pinned, "1");
        assert!(!output.status.success());
        assert!(!fixture.path().join("outputs").exists());
        assert!(!fixture.path().join("environment").exists());
    }
}
