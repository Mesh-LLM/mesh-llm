use super::fixture::{Fixture, source};
use std::collections::BTreeSet;

fn cargo_features(fixture: &Fixture, release: bool) -> BTreeSet<String> {
    let arguments = fixture.log("cargo.args");
    let args: Vec<_> = arguments.lines().collect();
    assert_eq!(args.first(), Some(&"build"));
    assert!(args.contains(&"--locked"));
    assert!(args.contains(&"--no-default-features"));
    assert_eq!(args.contains(&"--release"), release);
    for flag in ["-p", "--bin"] {
        let index = args.iter().position(|arg| *arg == flag).unwrap();
        assert_eq!(args[index + 1], "mesh-llm");
    }
    let index = args.iter().position(|arg| *arg == "--features").unwrap();
    args[index + 1].split(',').map(str::to_owned).collect()
}

#[test]
fn neutral_release_host_features_are_backend_independent_of_legacy_wallet_selection() {
    for backend in ["cpu", "cuda", "rocm"] {
        for wallet in [false, true] {
            let fixture = Fixture::new();
            let result = fixture.invoke(
                "scripts/build-release.sh",
                &[],
                &[
                    ("LLAMA_STAGE_BACKEND", backend),
                    ("MESH_LLM_WALLET_LEXE", if wallet { "1" } else { "0" }),
                ],
            );
            assert!(result.process.success(), "{backend}: {result:?}");
            let expected: BTreeSet<_> = ["web-ui", "dynamic-native-runtime", "payments"]
                .into_iter()
                .map(str::to_owned)
                .collect();
            assert_eq!(cargo_features(&fixture, true), expected);
            assert_eq!(fixture.log("version"), "0.68.0\n");
            assert_eq!(fixture.log("events"), "ui\ncargo\n");
        }
    }
}

#[test]
fn release_entry_delegates_profile_to_canonical_host() {
    let fixture = Fixture::new();
    fixture.write(
        "scripts/build-host.sh",
        "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$BUILD_FIXTURE_ROOT/host.args\"\n",
    );
    let result = fixture.invoke("scripts/build-release.sh", &[], &[]);
    assert!(result.process.success(), "{result:?}");
    assert_eq!(fixture.log("host.args"), "--profile\nrelease\n");
    assert!(fixture.log("cargo.args").is_empty());
}

#[test]
fn unsupported_static_host_is_rejected_before_ui_or_cargo() {
    let fixture = Fixture::new();
    let result = fixture.invoke(
        "scripts/build-release.sh",
        &[],
        &[
            ("LLAMA_STAGE_BACKEND", "metal"),
            ("MESH_LLM_DYNAMIC_NATIVE_RUNTIME", "0"),
        ],
    );
    assert_eq!(result.process.status.unwrap().code(), Some(1));
    assert!(!result.process.success());
    assert!(fixture.log("events").is_empty());
    assert!(fixture.log("cargo.args").is_empty());
    assert!(fixture.log("pkgid.args").is_empty());
    assert!(
        String::from_utf8_lossy(result.stderr.as_ref().unwrap().as_bytes())
            .contains("MESH_LLM_DYNAMIC_NATIVE_RUNTIME=0 is unsupported")
    );
}

#[test]
fn actual_host_profiles_stamp_sha_only_for_nonrelease_builds() {
    for (profile, expected) in [
        ("debug", "0.68.0+gABC123\n"),
        ("dev", "0.68.0+gABC123\n"),
        ("release", "0.68.0\n"),
    ] {
        let fixture = Fixture::new();
        let result = fixture.invoke(
            "scripts/build-host.sh",
            &["--profile", profile],
            &[("MESH_LLM_SKIP_UI", "1")],
        );
        assert!(result.process.success(), "{profile}: {result:?}");
        assert_eq!(fixture.log("version"), expected);
        assert_eq!(fixture.log("events"), "cargo\n");
        assert!(cargo_features(&fixture, profile == "release").contains("dynamic-native-runtime"));
    }
}

fn neutral_source(source: &str) -> bool {
    let active = source
        .lines()
        .map(str::trim)
        .filter(|line| !line.starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n");
    !["gpu-bench-cuda", "gpu-bench-hip", "build-llama.sh"]
        .iter()
        .any(|name| active.contains(name))
}
#[test]
fn host_source_cannot_add_backend_features_or_native_preparation() {
    let current = source("mesh/scripts/build-host.sh");
    assert!(neutral_source(&current));
    for addition in [
        "host_features=\"gpu-bench-cuda\"",
        "host_features=\"gpu-bench-hip\"",
        "scripts/build-llama.sh",
    ] {
        assert!(!neutral_source(&format!("{current}\n{addition}\n")));
    }
}
