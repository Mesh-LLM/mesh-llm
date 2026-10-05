//! Complete CUDA packager joins observed toolkit policy, native closure and manifest.
use super::Fixture;
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

fn inputs(fixture: &Fixture, version: &str, major: &str) {
    fs::create_dir_all(fixture.path().join("cuda/lib64")).unwrap();
    fs::write(
        fixture.path().join("build/libllama.so"),
        b"\x7fELFinert-llama",
    )
    .unwrap();
    fs::write(
        fixture
            .path()
            .join(format!("cuda/lib64/libcudart.so.{major}")),
        b"\x7fELFinert-cudart",
    )
    .unwrap();
    fs::write(
        fixture.path().join("cuda-license"),
        b"owned inert CUDA redistribution license\n",
    )
    .unwrap();
    let compiler = if version.is_empty() {
        "if [[ ${1:-} == --version ]]; then exit 89; fi\nexit 90".to_owned()
    } else {
        format!(
            "if [[ ${{1:-}} == --version ]]; then printf 'Cuda compilation tools, release {version}, V{version}.0\\n'; exit 0; fi\nprintf '%s\\n' \"$@\" > \"$TEST_ROOT/compiler-arguments\"\nprevious=''\nfor argument in \"$@\"; do\n if [[ \"$previous\" == -o ]]; then output=\"$argument\"; fi\n previous=\"$argument\"\ndone\n: > \"$output\""
        )
    };
    fixture.tool("nvcc", &compiler);
    fixture.tool("model-tool", "exit 0");
    fixture.tool("cargo", "[[ ${1:-} == xtool ]] || exit 96\nprintf '%s\\n' \"$*\" >> \"$TEST_ROOT/native-calls\"\nshift\nexec \"$AUTOMATION\" \"$@\"");
    fixture.tool("readelf", "[[ $LC_ALL == C ]] || exit 95\ncase ${1:-} in\n -h) printf '  Class: ELF64\\n  Machine: Advanced Micro Devices X86-64\\n' ;;\n -d) case ${2##*/} in\n       libllama.so) printf ' (NEEDED) Shared library: [libcudart.so.%s]\\n' \"$TEST_CUDA_MAJOR\" ;;\n       libcudart.so.*) printf ' (SONAME) Library soname: [%s]\\n' \"${2##*/}\" ;;\n       *) exit 94 ;;\n     esac ;;\n -V) printf 'Version needs section GLIBC_2.17\\n' ;;\n *) exit 93 ;;\nesac");
}

fn package(
    fixture: &Fixture,
    version: &str,
    declaration: &str,
    explicit_major: &str,
) -> (bool, String) {
    let observed_major = version
        .split('.')
        .next()
        .filter(|v| !v.is_empty())
        .unwrap_or("13");
    inputs(fixture, version, observed_major);
    let script =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/package-native-runtime.sh");
    let values = [
        ("PACKAGE_SCRIPT", script.to_string_lossy().into_owned()),
        (
            "LLAMA_STAGE_BUILD_DIR",
            fixture.path().join("build").to_string_lossy().into_owned(),
        ),
        (
            "LLAMA_WORKDIR",
            fixture
                .path()
                .join("absent-owned-source")
                .to_string_lossy()
                .into_owned(),
        ),
        (
            "CUDACXX",
            fixture
                .path()
                .join("bin/nvcc")
                .to_string_lossy()
                .into_owned(),
        ),
        (
            "CUDAToolkit_ROOT",
            fixture.path().join("cuda").to_string_lossy().into_owned(),
        ),
        (
            "MESH_NATIVE_RUNTIME_MODEL_PACKAGE_TOOL",
            fixture
                .path()
                .join("bin/model-tool")
                .to_string_lossy()
                .into_owned(),
        ),
        (
            "MESH_LLM_CUDA_LICENSE_FILE",
            fixture
                .path()
                .join("cuda-license")
                .to_string_lossy()
                .into_owned(),
        ),
        ("MESH_CUDA_VERSION", declaration.into()),
        ("MESH_LLM_CUDA_TOOLKIT_MAJOR", explicit_major.into()),
        ("TEST_CUDA_MAJOR", observed_major.into()),
        ("LC_ALL", "POSIX".into()),
    ];
    let (ok, _, error) = fixture.run("\"$PACKAGE_SCRIPT\" --backend cuda --target x86_64-unknown-linux-gnu --out \"$TEST_ROOT/output\"", &values);
    (ok, error)
}

fn emitted(fixture: &Fixture, major: u64) {
    let artifact = format!("meshllm-native-runtime-linux-x86_64-cuda{major}");
    let output = fixture.path().join("output");
    let stage = output.join(&artifact);
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(stage.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["runtime"]["backend"]["kind"], "cuda");
    assert_eq!(
        manifest["runtime"]["backend"]["cuda"]["toolkit_major"],
        major
    );
    assert_eq!(manifest["build"]["primary_library"], "lib/libllama.so");
    let library = format!("lib/libcudart.so.{major}");
    assert_eq!(
        fs::read(stage.join(&library)).unwrap(),
        fs::read(
            fixture
                .path()
                .join(format!("cuda/lib64/libcudart.so.{major}"))
        )
        .unwrap()
    );
    assert_eq!(
        manifest["runtime"]["libraries"],
        serde_json::json!([library, "lib/libllama.so"])
    );
    let license = "licenses/NVIDIA-CUDA-LICENSE.txt";
    assert_eq!(
        fs::read(stage.join(license)).unwrap(),
        fs::read(fixture.path().join("cuda-license")).unwrap()
    );
    for relative in [library.as_str(), "lib/libllama.so", license] {
        let digest = hex::encode(Sha256::digest(fs::read(stage.join(relative)).unwrap()));
        assert_eq!(manifest["runtime"]["files"][relative], digest);
    }
    assert!(manifest["runtime"]["tools"]["tools/mesh-llm-gpu-benchmark"].is_string());
    let arguments = fs::read_to_string(fixture.path().join("compiler-arguments")).unwrap();
    let arguments = arguments.lines().collect::<Vec<_>>();
    let index = arguments.iter().position(|v| *v == "-cudart").unwrap();
    assert_eq!(arguments[index + 1], "shared");
    let calls = fs::read_to_string(fixture.path().join("native-calls")).unwrap();
    assert_eq!(
        calls
            .lines()
            .filter(|v| v.contains("linux-runtime-deps collect "))
            .count(),
        1
    );
    assert!(
        calls
            .lines()
            .any(|v| v.contains("linux-runtime-deps collect ")
                && v.contains(&format!("--cuda-major {major}")))
    );
    assert_eq!(
        calls
            .lines()
            .filter(|v| v.contains("linux-runtime-deps order "))
            .count(),
        1
    );
    assert_eq!(
        calls
            .lines()
            .filter(|v| v.contains("runtime-manifest-write "))
            .count(),
        1
    );
    assert!(output.join(format!("{artifact}.tar.gz")).is_file());
    assert!(output.join(format!("{artifact}.tar.gz.sha256")).is_file());
}

#[test]
fn package_cuda_full_script_joins_observed_evidence_native_closure_and_manifest_or_refuses_publication()
 {
    for (version, declaration, explicit_major, expected) in [
        ("13.0", "", "12", None),
        ("12.9", "", "12", Some(12)),
        ("13.0", "", "", Some(13)),
        ("13.0", "12.9.2", "", None),
        ("13.1", "13.1.2", "", Some(13)),
        ("13.0", "13", "", Some(13)),
        ("13.0", "13.1.2", "", None),
        ("", "", "", None),
    ] {
        let fixture = Fixture::new();
        let (ok, error) = package(&fixture, version, declaration, explicit_major);
        match expected {
            Some(major) => {
                assert!(ok, "{version} {declaration} {explicit_major}: {error}");
                emitted(&fixture, major);
            }
            None => {
                assert!(
                    !ok,
                    "false evidence admitted: {version} {declaration} {explicit_major}"
                );
                assert!(
                    error.contains("does not match") || error.contains("could not be detected"),
                    "{error}"
                );
                assert!(!fixture.path().join("compiler-arguments").exists());
                assert!(!fixture.path().join("native-calls").exists());
                assert!(
                    !fixture.path().join("output").exists(),
                    "refusal published staging, manifest or archive"
                );
            }
        }
        fixture.directory.close().unwrap();
    }
}
