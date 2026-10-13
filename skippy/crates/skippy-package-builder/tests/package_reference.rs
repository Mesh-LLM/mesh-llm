use std::process::Command;
fn builder() -> &'static str {
    env!("CARGO_BIN_EXE_skippy-package-builder")
}
#[test]
fn actual_cli_preserves_revision_and_refuses_before_cache_preparation() {
    let scratch = tempfile::tempdir().unwrap();
    let cache = scratch.path().join("absent");
    for (input, expected) in [
        ("hf://org/repo", "org/repo\nmain\n"),
        ("hf://org/repo:topic/one", "org/repo\ntopic/one\n"),
        ("hf://org/repo@topic/one", "org/repo\ntopic/one\n"),
        ("hf://org/repo:release=v2", "org/repo\nrelease=v2\n"),
        ("hf://org/repo@release/étape", "org/repo\nrelease/étape\n"),
        ("hf://org/repo@release#v2", "org/repo\nrelease#v2\n"),
    ] {
        let out = Command::new(builder())
            .args(["parse-package-reference", input])
            .env("HF_HOME", &cache)
            .env("MESH_LLM_DATA_DIR", &cache)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(out.stdout, expected.as_bytes());
        assert!(!cache.exists());
    }
    for input in [
        "hf://org/repo:x@y",
        "hf://org/repo@../bad",
        "hf://org/repo@",
    ] {
        let out = Command::new(builder())
            .args(["parse-package-reference", input])
            .output()
            .unwrap();
        assert!(!out.status.success());
        assert!(out.stdout.is_empty());
    }
    assert!(
        Command::new(builder())
            .args(["parse-package-reference", "--help"])
            .status()
            .unwrap()
            .success()
    );
}
#[cfg(unix)]
fn function(source: &str, name: &str) -> String {
    let start = source.find(&format!("{name}() {{")).unwrap();
    let end = source[start..].find("\n}\n").unwrap() + start + 3;
    source[start..end].to_owned()
}
#[cfg(unix)]
const CALLER_CASES: [(&str, Option<&str>); 5] = [
    ("hf://org/repo:x@y", None),
    ("hf://org/repo:topic/one", Some("topic/one")),
    ("hf://org/repo:release=v2", Some("release=v2")),
    ("hf://org/repo@release/étape", Some("release/étape")),
    ("hf://org/repo@release#v2", Some("release#v2")),
];
#[cfg(unix)]
#[test]
fn actual_wan_functions_stop_acquisition_on_parse_failure() {
    use std::{os::unix::fs::PermissionsExt, path::Path};
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .unwrap();
    for host in [false, true] {
        let source = std::fs::read_to_string(root.join(if host {
            "skippy/evals/wan-lab/up.sh"
        } else {
            "skippy/evals/wan-lab/entrypoint.sh"
        }))
        .unwrap();
        let parser = function(&source, "parse_hf_package_ref");
        let caller = function(
            &source,
            if host {
                "ensure_hf_package"
            } else {
                "prepare_hf_layer_package"
            },
        );
        let parser = if host {
            parser
        } else {
            assert!(
                parser.contains("/usr/local/bin/skippy-package-builder parse-package-reference")
            );
            parser.replace("/usr/local/bin/skippy-package-builder", "\"$REAL_BUILDER\"")
        };
        for (reference, revision) in CALLER_CASES {
            let valid = revision.is_some();
            let scratch = tempfile::tempdir().unwrap();
            let marker = scratch.path().join("acquisition");
            let just = scratch.path().join("just");
            std::fs::write(&just, "#!/usr/bin/env bash\nset -euo pipefail\n[[ $1 == --justfile ]]\ncase $3 in\n  skippy-package-reference) [[ $# == 4 ]]; exec \"$REAL_BUILDER\" parse-package-reference \"$4\" ;;\n  skippy-layer-package-cache) exit 1 ;;\n  skippy-layer-package-fetch) printf '%s\\n' \"$*\" > \"$MARKER\"; printf '{\"commit\":\"0123abc\",\"snapshot_path\":\"%s\"}\\n' \"$SCRATCH\" ;;\n  *) exit 64 ;;\nesac\n").unwrap();
            std::fs::set_permissions(&just, std::fs::Permissions::from_mode(0o755)).unwrap();
            let invoke = if host {
                "MODEL_PACKAGE_REF=\"$REFERENCE\"; ensure_hf_package"
            } else {
                "prepare_hf_layer_package \"$REFERENCE\" 0 1"
            };
            let script = format!(
                "set -euo pipefail\n{parser}\n{caller}\nROOT=unused\n\
                log() {{ :; }}\nhf_home_dir() {{ printf '%s\\n' \"$SCRATCH\"; }}\n\
                hf() {{ printf '%s\\n' \"$*\" > \"$MARKER\"; }}\n\
                hf_snapshot_dir() {{ printf '%s\\n' \"$SCRATCH\"; }}\nverify_package_cache() {{ :; }}\n\
                prepare_hf_layer_package_from_host_cache() {{ printf '%s\\n' \"$*\" > \"$MARKER\"; }}\n{invoke}\n"
            );
            let bash = if Path::new("/opt/homebrew/bin/bash").exists() {
                "/opt/homebrew/bin/bash"
            } else {
                "bash"
            };
            let out = Command::new(bash)
                .args(["-c", &script])
                .env("REAL_BUILDER", builder())
                .env("SCRATCH", scratch.path())
                .env("MARKER", &marker)
                .env("HF_PACKAGE_SOURCE", "host-cache")
                .env(
                    "PATH",
                    format!(
                        "{}:{}",
                        scratch.path().display(),
                        std::env::var("PATH").unwrap()
                    ),
                )
                .env("REFERENCE", reference)
                .output()
                .unwrap();
            assert_eq!(
                out.status.success(),
                valid,
                "host={host}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            assert_eq!(marker.exists(), valid);
            if valid {
                let args = std::fs::read_to_string(marker).unwrap();
                assert!(args.contains("org/repo") && args.contains(revision.unwrap()));
            }
        }
    }
}
