//! Package flavor validates the selected compiler, never a declared label alone.
use super::Fixture;
use std::{fs, path::Path};

fn package_flavor_owner() -> String {
    let source = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/package-native-runtime.sh"),
    )
    .unwrap();
    let start = "backend_flavor() {";
    assert_eq!(source.matches(start).count(), 1);
    let region = source
        .split_once(start)
        .unwrap()
        .1
        .split_once("build_backend() {")
        .unwrap()
        .0;
    format!("{start}{region}\nbackend_flavor\n")
}

#[test]
fn package_flavor_preserves_compiler_evidence_and_refuses_false_cuda_declarations() {
    let body = package_flavor_owner();
    for (backend, version, declaration, major, expected) in [
        ("cuda", "13.1", "13.1.2", "", Some("cuda13")),
        ("cuda", "13.0", "13", "", Some("cuda13")),
        ("cuda-blackwell", "13.0", "", "", Some("cuda13-sm120")),
        ("cuda", "12.9", "", "12", Some("cuda12")),
        ("cuda", "13.0", "", "", Some("cuda13")),
        ("cuda", "13.0", "13.1.2", "", None),
        ("cuda", "13.0", "12.9.2", "", None),
        ("cuda", "13.0", "", "12", None),
        ("cuda", "13.0", "", "12.1", None),
        ("cuda", "13.0", "", "cuda12", None),
        ("cuda", "", "13", "13", None),
        ("cuda-blackwell", "", "13", "13", None),
    ] {
        let fixture = Fixture::new();
        let mut values = vec![
            ("BACKEND", backend),
            ("MESH_CUDA_VERSION", declaration),
            ("MESH_LLM_CUDA_TOOLKIT_MAJOR", major),
        ];
        if version.is_empty() {
            values.push(("NVCC", "unavailable-explicit-compiler"));
        } else {
            fixture.compiler("bin/nvcc", version);
        }
        let (ok, output, error) = fixture.run(&body, &values);
        match expected {
            Some(flavor) => {
                assert!(ok, "{backend} {version} {declaration} {major}: {error}");
                assert_eq!(output.trim(), flavor);
            }
            None => {
                assert!(
                    !ok,
                    "false declaration admitted: {backend} {version} {declaration} {major}"
                );
                assert!(output.is_empty(), "failure fabricated flavor {output}");
                assert!(
                    error.contains("does not match")
                        || error.contains("digits-only")
                        || error.contains("could not be detected"),
                    "{error}"
                );
            }
        }
        fixture._temporary.close().unwrap();
    }
}
