use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;
#[cfg(unix)]
use std::path::PathBuf;
#[cfg(unix)]
use std::process::Command;

type TestResult = Result<(), Box<dyn std::error::Error>>;

#[cfg(unix)]
struct Scratch(PathBuf);
#[cfg(unix)]
impl Scratch {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let path = std::env::temp_dir().join(format!("l5-{}-{stamp}", std::process::id()));
        fs::create_dir(&path)?;
        Ok(Self(path))
    }
}
#[cfg(unix)]
impl Drop for Scratch {
    fn drop(&mut self) {
        let _cleanup = fs::remove_dir_all(&self.0);
    }
}

#[cfg(unix)]
fn executable(path: &Path, body: &[u8]) -> TestResult {
    use std::os::unix::fs::PermissionsExt;
    fs::write(path, body)?;
    fs::set_permissions(path, fs::Permissions::from_mode(0o755))?;
    Ok(())
}

#[test]
#[cfg(unix)]
fn copied_verifier_enforces_embedded_floor_without_source_or_cargo() -> TestResult {
    let scratch = Scratch::new()?;
    let root = &scratch.0;
    let verifier = root.join("automation");
    fs::copy(env!("CARGO_BIN_EXE_xtask"), &verifier)?;
    let tools = root.join("tools");
    fs::create_dir(&tools)?;
    executable(
        &tools.join("readelf"),
        br##"#!/bin/sh
case "$1" in
    -V) printf 'Version needs section\n Name: GLIBC_%s\n' "$L5_PROBE_FLOOR" ;;
    -d) printf 'Dynamic section\n' ;;
    *) exit 2 ;;
esac
"##,
    )?;
    let runtime = root.join("runtime");
    fs::create_dir_all(runtime.join("lib"))?;
    let library = b"\x7fELFfixture";
    fs::write(runtime.join("lib/runtime.so"), library)?;
    let digest = hex::encode(Sha256::digest(library));
    let manifest = serde_json::json!({
        "runtime": {
            "id":"runtime", "mesh_version":"1.0.0", "skippy_abi":"0.1.0",
            "platform":{"os":"linux","arch":"x86_64","target":"x86_64-unknown-linux-gnu"},
            "backend":{"kind":"cpu"}, "libraries":["lib/runtime.so"],
            "files":{"lib/runtime.so":digest}, "tools":{}
        },
        "build":{"primary_library":"lib/runtime.so","library_sha256":digest}
    });
    fs::write(
        runtime.join("manifest.json"),
        serde_json::to_vec(&manifest)?,
    )?;
    for (floor, accepted) in [("2.17", true), ("99.0", false)] {
        let output = Command::new(&verifier)
            .current_dir(root)
            .env("PATH", &tools)
            .env("L5_PROBE_FLOOR", floor)
            .args(["native", "verify-runtime-package"])
            .arg(&runtime)
            .output()?;
        assert_eq!(
            output.status.success(),
            accepted,
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        if !accepted {
            assert!(String::from_utf8_lossy(&output.stderr).contains("needs GLIBC_99.0"));
        }
    }
    Ok(())
}

#[test]
fn release_adapters_dispatch_rust_owners_without_interpreter_selection() -> TestResult {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for script in [
        "package-release.sh",
        "package-release.ps1",
        "verify-native-runtime-package.sh",
        "ci-compose-product-input.sh",
        "generate-native-runtime-release-manifest.sh",
        "publish-crates.sh",
        "rc-release-smoke.sh",
    ] {
        let source = fs::read_to_string(root.join("scripts").join(script))?;
        assert!(!source.contains("python3"), "{script}");
        assert!(!source.contains("Get-PythonCommand"), "{script}");
    }
    let workflow = fs::read_to_string(root.join(".github/workflows/release.yml"))?;
    assert!(!workflow.contains("setup-python"));
    assert!(!workflow.contains("scripts/validate-release-native-runtime-matrix.py"));
    Ok(())
}

#[test]
#[cfg(unix)]
fn release_manifest_adapter_runs_from_prepared_binary() -> TestResult {
    let scratch = Scratch::new()?;
    let runtime = scratch.0.join("runtime");
    fs::create_dir_all(runtime.join("lib"))?;
    let bytes = b"runtime";
    fs::write(runtime.join("lib/runtime.bin"), bytes)?;
    let digest = hex::encode(Sha256::digest(bytes));
    let manifest = serde_json::json!({
        "runtime":{
            "id":"runtime","mesh_version":"1.0.0","skippy_abi":"0.1.0",
            "platform":{"os":"macos","arch":"aarch64","target":"aarch64-apple-darwin"},
            "backend":{"kind":"metal"},"libraries":["lib/runtime.bin"],
            "files":{"lib/runtime.bin":digest},"tools":{}
        },
        "build":{"primary_library":"lib/runtime.bin","library_sha256":digest}
    });
    fs::write(
        runtime.join("manifest.json"),
        serde_json::to_vec(&manifest)?,
    )?;
    let archive = scratch.0.join("runtime.tar.gz");
    let status = Command::new("tar")
        .env("COPYFILE_DISABLE", "1")
        .current_dir(&scratch.0)
        .args(["-czf"])
        .arg(&archive)
        .arg("runtime")
        .status()?;
    assert!(status.success());
    let checksum = hex::encode(Sha256::digest(fs::read(&archive)?));
    fs::write(
        scratch.0.join("runtime.tar.gz.sha256"),
        format!("{checksum}  runtime.tar.gz\n"),
    )?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let destination = scratch.0.join("native-runtimes.json");
    let output = Command::new("bash")
        .arg(root.join("scripts/generate-native-runtime-release-manifest.sh"))
        .args(["--tag", "v1.0.0", "--out"])
        .arg(&destination)
        .arg(&archive)
        .env("MESH_LLM_AUTOMATION_BIN", env!("CARGO_BIN_EXE_xtask"))
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let release: serde_json::Value = serde_json::from_slice(&fs::read(destination)?)?;
    assert_eq!(release["artifacts"][0]["sha256"], checksum);
    assert_eq!(release["mesh_version"], "1.0.0");
    Ok(())
}

#[test]
#[cfg(unix)]
fn product_adapter_composes_immutable_inputs_and_rejects_host_checksum_drift() -> TestResult {
    let scratch = Scratch::new()?;
    let workspace = scratch.0.canonicalize()?;
    let host_input = workspace.join("host-input");
    let runtime = workspace.join("runtime-input/runtime");
    fs::create_dir(&host_input)?;
    fs::create_dir_all(runtime.join("lib"))?;
    let host = b"#!/bin/sh\nprintf 'mesh-llm 1.0.0\\n'\n";
    executable(&host_input.join("mesh-llm"), host)?;
    fs::write(host_input.join("host-imports.json"), b"{}")?;
    fs::write(
        host_input.join("mesh-llm.sha256"),
        format!("{}  mesh-llm\n", hex::encode(Sha256::digest(host))),
    )?;
    let library = b"runtime";
    fs::write(runtime.join("lib/runtime.bin"), library)?;
    let digest = hex::encode(Sha256::digest(library));
    let manifest = serde_json::json!({
        "runtime":{
            "id":"runtime","mesh_version":"1.0.0","skippy_abi":"0.1.0",
            "platform":{"os":"macos","arch":"aarch64","target":"aarch64-apple-darwin"},
            "backend":{"kind":"metal"},"libraries":["lib/runtime.bin"],
            "files":{"lib/runtime.bin":digest},"tools":{}
        },
        "build":{"backend":"metal","primary_library":"lib/runtime.bin","library_sha256":digest}
    });
    fs::write(
        runtime.join("manifest.json"),
        serde_json::to_vec(&manifest)?,
    )?;
    let script =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/ci-compose-product-input.sh");
    for accepted in [true, false] {
        if !accepted {
            fs::write(host_input.join("mesh-llm"), b"changed host")?;
        }
        let output = Command::new("bash")
            .arg(&script)
            .current_dir(&workspace)
            .env("MESH_LLM_AUTOMATION_BIN", env!("CARGO_BIN_EXE_xtask"))
            .env("GITHUB_WORKSPACE", &workspace)
            .env("GITHUB_OUTPUT", workspace.join("outputs"))
            .env("INPUT_HOST_INPUT_DIR", "host-input")
            .env("INPUT_RUNTIME_INPUT_DIR", "runtime-input")
            .env("INPUT_OUTPUT_DIR", "product")
            .env("INPUT_BACKEND", "metal")
            .env("INPUT_BINARY_NAME", "mesh-llm")
            .env("INPUT_READINESS_SMOKE", "false")
            .output()?;
        assert_eq!(
            output.status.success(),
            accepted,
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        if accepted {
            assert_eq!(fs::read(workspace.join("product/mesh-llm"))?, host);
            assert!(workspace.join("product.tar.gz").is_file());
            let product: serde_json::Value = serde_json::from_slice(&fs::read(
                workspace.join("product/product-manifest.json"),
            )?)?;
            assert_eq!(product["host"]["sha256"], hex::encode(Sha256::digest(host)));
        }
    }
    Ok(())
}
