//! Windows release declarations only; no Windows/toolkit/product execution.
use super::{Node, document, input, job, named, steps, text};
fn action<'a>(node: &'a Node, uses: &str) -> &'a Node {
    let found = steps(node)
        .iter()
        .filter(|s| text(s, "uses") == Some(uses))
        .collect::<Vec<_>>();
    assert_eq!(found.len(), 1);
    found[0]
}
#[test]
fn graph_windows_release_host_publishes_verifier_without_runtime_composition() {
    let doc = document("release.yml");
    let host = job(&doc, "windows_host_input");
    assert_eq!(text(host, "runs-on"), Some("windows-2022"));
    let producer = action(host, "./.github/actions/prepare-windows-host-input");
    for (key, value) in [
        ("profile", "release"),
        ("skip_ui", "true"),
        ("commit", "${{ needs.metadata.outputs.source_sha }}"),
        (
            "attestation_signing_key_file",
            "${{ runner.temp }}/mesh-release-attestation-private-key.json",
        ),
        (
            "attestation_public_key_file",
            "${{ runner.temp }}/mesh-release-attestation-public-key.json",
        ),
    ] {
        assert_eq!(input(producer, key), Some(value));
    }
    let upload = steps(host)
        .iter()
        .filter(|s| input(s, "name") == Some("host-input-windows-x86_64"))
        .collect::<Vec<_>>();
    assert_eq!(upload.len(), 1);
    assert_eq!(input(upload[0], "path"), Some("host-input/*"));
    assert!(!steps(host).iter().any(|s| matches!(
        text(s, "uses"),
        Some(
            "./.github/actions/prepare-native-runtime-input"
                | "./.github/actions/compose-product-input"
        )
    )));
    let run = super::support::step(&super::support::action("prepare-windows-host-input"), "run");
    assert!(run.contains("release-attestation-verifier.exe"));
}
#[test]
fn graph_windows_release_composers_require_prebuilt_verifier_before_packaging() {
    let doc = document("release.yml");
    for (name, kind) in [
        ("compose_windows_cpu", "CPU"),
        ("compose_windows_gpu", "GPU"),
    ] {
        let j = job(&doc, name);
        let compose = action(j, "./.github/actions/compose-product-input");
        for (key, value) in [
            ("host_input_dir", "host-input"),
            ("runtime_input_dir", "runtime-input"),
            ("output_dir", "product-input"),
            ("version", "${{ needs.metadata.outputs.tag }}"),
            ("binary_name", "mesh-llm.exe"),
            ("readiness_smoke", "true"),
            (
                "attestation_verifier",
                "host-input/release-attestation-verifier.exe",
            ),
        ] {
            assert_eq!(input(compose, key), Some(value));
        }
        let package = named(j, &format!("Package verified Windows {kind} product"));
        let env = package.get("env").unwrap();
        for (key, value) in [
            (
                "MESH_LLM_PRECOMPOSED_PRODUCT_DIR",
                "${{ steps.compose.outputs.product_dir }}",
            ),
            ("MESH_RELEASE_HOST_PRESTAMPED", "1"),
            ("MESH_RELEASE_ATTESTATION_PREVERIFIED", "1"),
        ] {
            assert_eq!(text(env, key), Some(value));
        }
        assert!(
            steps(j)
                .iter()
                .position(|s| std::ptr::eq(s, compose))
                .unwrap()
                < steps(j)
                    .iter()
                    .position(|s| std::ptr::eq(s, package))
                    .unwrap()
        );
        for s in steps(j) {
            let uses = text(s, "uses").unwrap_or("");
            assert_ne!(uses, "./.github/actions/prepare-windows-host-input");
            assert_ne!(uses, "./.github/actions/prepare-native-runtime-input");
            assert_ne!(uses, "./.github/actions/configure-sccache-gha");
            assert!(!uses.starts_with("dtolnay/rust-toolchain@"));
            if let Some(run) = text(s, "run") {
                assert!(!run.contains("cargo build"));
                assert!(!run.contains("--skip-attestation"));
            }
        }
    }
}
#[test]
fn graph_windows_cuda_labels_are_validated_before_install_and_package() {
    let doc = document("release.yml");
    let j = job(&doc, "build_native_runtime_windows_gpu");
    let env = j.get("env").unwrap();
    for key in ["WINDOWS_CUDA_VERSION", "MESH_CUDA_VERSION"] {
        assert_eq!(
            text(env, key),
            Some("${{ matrix.cuda_version || vars.CUDA_VERSION || '12.9.2' }}")
        );
    }
    let validation = named(j, "Validate CUDA artifact contract");
    assert_eq!(
        text(validation, "if"),
        Some("${{ matrix.backend == 'cuda' }}")
    );
    let run = text(validation, "run").unwrap();
    assert_eq!(
        text(validation.get("env").unwrap(), "EXPECTED_CUDA_MAJOR"),
        Some("${{ matrix.cuda_major }}")
    );
    assert!(run.contains("$cudaMajor -ne $env:EXPECTED_CUDA_MAJOR"));
    assert!(run.contains("throw"));
    let install = named(j, "Install CUDA toolkit");
    assert_eq!(
        input(install, "cuda"),
        Some("${{ env.WINDOWS_CUDA_VERSION }}")
    );
    let package = action(j, "./.github/actions/prepare-native-runtime-input");
    assert_eq!(
        text(package.get("env").unwrap(), "MESH_LLM_CUDA_TOOLKIT_MAJOR"),
        Some("${{ matrix.cuda_major || '12' }}")
    );
    let position = |n: &Node| steps(j).iter().position(|s| std::ptr::eq(s, n)).unwrap();
    assert!(position(validation) < position(install) && position(install) < position(package));
    let Node::Seq(rows) = j
        .get("strategy")
        .unwrap()
        .get("matrix")
        .unwrap()
        .get("include")
        .unwrap()
    else {
        panic!("matrix")
    };
    let cuda = rows
        .iter()
        .filter(|r| text(r, "backend") == Some("cuda"))
        .collect::<Vec<_>>();
    assert_eq!(cuda.len(), 2);
    assert_eq!(
        text(cuda[0], "artifact_name"),
        Some("release-native-runtime-windows-x86_64-cuda12")
    );
    assert!(
        text(cuda[0], "cuda_architectures")
            .unwrap()
            .split(';')
            .any(|a| a == "61")
    );
    assert_eq!(text(cuda[1], "cuda_version"), Some("13.1.2"));
    assert_eq!(
        text(cuda[1], "artifact_name"),
        Some("release-native-runtime-windows-x86_64-cuda13")
    );
    assert!(
        text(cuda[1], "cuda_architectures")
            .unwrap()
            .split(';')
            .any(|a| a == "120")
    );
    assert_eq!(
        text(
            package.get("env").unwrap(),
            "LLAMA_STAGE_CUDA_ARCHITECTURES"
        ),
        Some("${{ matrix.cuda_architectures }}")
    );
}
