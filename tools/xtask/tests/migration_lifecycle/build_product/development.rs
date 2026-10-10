use super::fixture::{Fixture, source};
use std::path::Path;

fn product_fixture() -> Fixture {
    let fixture = Fixture::new();
    fixture.write("scripts/build-host.sh", "#!/bin/sh\nset -eu\nprintf 'host\\n' >> \"$BUILD_FIXTURE_ROOT/events\"\nprintf '%s\\n' \"$@\" > \"$BUILD_FIXTURE_ROOT/host.args\"\nmkdir -p \"$BUILD_FIXTURE_ROOT/target/debug\"\n");
    fixture.write(
        "scripts/package-native-runtime.sh",
        r#"#!/bin/sh
set -eu
printf 'runtime\n' >> "$BUILD_FIXTURE_ROOT/events"
printf '%s\n' "$@" > "$BUILD_FIXTURE_ROOT/runtime.args"
printf '%s\n' "${LLAMA_STAGE_CUDA_ARCHITECTURES:-}" > "$BUILD_FIXTURE_ROOT/cuda.arch"
printf '%s\n' "${LLAMA_STAGE_AMDGPU_TARGETS:-}" > "$BUILD_FIXTURE_ROOT/rocm.arch"
previous=''
for argument do
  if [ "$previous" = --out ]; then out="$argument"; fi
  previous="$argument"
done
mkdir -p "$out"
"#,
    );
    fixture
}
fn option(fixture: &Fixture, name: &str) -> String {
    let log = fixture.log("runtime.args");
    let args: Vec<_> = log.lines().collect();
    let index = args.iter().position(|arg| *arg == name).unwrap();
    args[index + 1].into()
}
#[test]
fn actual_development_product_builds_skippy_runtime_cli_then_mesh_host() {
    let fixture = product_fixture();
    let result = fixture.invoke(
        "scripts/build-development-product.sh",
        &["--backend", "cpu", "--profile", "dev"],
        &[],
    );
    assert!(result.process.success(), "{result:?}");
    assert_eq!(fixture.log("events"), "runtime\ncli\nhost\n");
    assert_eq!(
        fixture.log("skippy-cargo.args"),
        "build\n--locked\n-p\nskippy-cli\n--features\ndynamic-native-runtime\n"
    );
    assert_eq!(fixture.log("host.args"), "--profile\ndev\n");
    assert!(
        fixture
            .log("runtime.args")
            .lines()
            .any(|arg| arg == "--build")
    );
    assert_eq!(option(&fixture, "--backend"), "cpu");
    let runtime = option(&fixture, "--out");
    assert_eq!(
        Path::new(&runtime).canonicalize().unwrap(),
        fixture
            .root()
            .join("target/debug/native-runtimes")
            .canonicalize()
            .unwrap()
    );
}
#[test]
fn actual_development_product_normalizes_documented_named_recipe_arch_arguments() {
    for alias in ["cuda_arch", "cuda-arch"] {
        let fixture = product_fixture();
        let value = format!("{alias}=89;90");
        let result = fixture.invoke(
            "scripts/build-development-product.sh",
            &["--backend", "backend=cuda", "--cuda-arch", &value],
            &[],
        );
        assert!(result.process.success(), "{alias}: {result:?}");
        assert_eq!(option(&fixture, "--backend"), "cuda");
        assert_eq!(fixture.log("cuda.arch"), "89;90\n");
        assert_eq!(fixture.log("events"), "runtime\ncli\nhost\n");
    }
    for alias in ["rocm_arch", "rocm-arch", "amd_arch", "amd-arch"] {
        let fixture = product_fixture();
        let value = format!("{alias}=gfx1100;gfx1101");
        let result = fixture.invoke(
            "scripts/build-development-product.sh",
            &["--backend", "backend=rocm", "--rocm-arch", &value],
            &[],
        );
        assert!(result.process.success(), "{alias}: {result:?}");
        assert_eq!(option(&fixture, "--backend"), "rocm");
        assert_eq!(fixture.log("rocm.arch"), "gfx1100;gfx1101\n");
    }
}

fn active(source: &str) -> String {
    source
        .lines()
        .map(str::trim)
        .filter(|line| !line.starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n")
}
fn linux_detection(source: &str) -> bool {
    let active = active(source);
    let Some((_, linux)) = active.split_once("Linux)") else {
        return false;
    };
    let Some((linux, _)) = linux.split_once(";;") else {
        return false;
    };
    let selected: Option<Vec<_>> = [
        "BACKEND=cuda",
        "BACKEND=rocm",
        "BACKEND=vulkan",
        "BACKEND=cpu",
    ]
    .iter()
    .map(|branch| linux.find(branch))
    .collect();
    selected.is_some_and(|positions| positions.windows(2).all(|pair| pair[0] < pair[1]))
        && linux.contains("if command -v nvidia-smi")
        && linux.contains("elif command -v rocm-smi")
        && linux.contains("elif command -v glslc")
        && linux.contains("vulkaninfo --summary")
        && linux.contains("pkg-config --exists vulkan")
}
#[test]
fn linux_source_retains_gpu_precedence_and_vulkan_runtime_admission() {
    let current = source("skippy/scripts/build-development-product.sh");
    assert!(linux_detection(&current));
    let swapped = current
        .replace("BACKEND=cuda", "BACKEND=temporary")
        .replace("BACKEND=rocm", "BACKEND=cuda")
        .replace("BACKEND=temporary", "BACKEND=rocm");
    assert!(!linux_detection(&swapped));
    for admission in ["vulkaninfo --summary", "pkg-config --exists vulkan"] {
        assert!(!linux_detection(&current.replace(admission, "true")));
    }
}

fn runtime_recipe(source: &str) -> bool {
    let source = active(source);
    let Some((_, recipe)) = source.split_once("release-runtime-build backend=\"\" target=\"\":")
    else {
        return false;
    };
    let body = recipe.split("\n\n").next().unwrap();
    body.contains("selected_backend=\"{{ backend }}\"")
        && body.contains("[[ -z \"$selected_backend\" ]]")
        && body.contains("if [[ \"$(uname -s)\" == Darwin ]]; then selected_backend=metal; else selected_backend=cpu; fi")
        && body.contains("--backend \"$selected_backend\"")
        && body.contains("--target \"{{ target }}\"")
        && !body.contains("$$selected_backend")
}
#[test]
fn native_release_runtime_recipe_selects_current_platform_default_without_shell_pid() {
    let current = source("just/release-build.just");
    assert!(runtime_recipe(&current));
    assert!(!runtime_recipe(
        &current.replace("selected_backend=cpu", "selected_backend=foreign")
    ));
    assert!(!runtime_recipe(
        &current.replace("$selected_backend", "$$selected_backend")
    ));
    assert!(!runtime_recipe(
        &current.replace("--target \"{{ target }}\"", "")
    ));
}

#[test]
fn actual_linux_entry_rejects_unknown_option_before_product_side_effects() {
    let fixture = Fixture::new();
    fixture.write(
        "scripts/build-development-product.sh",
        "#!/bin/sh\nprintf 'unexpected-product\\n' >> \"$BUILD_FIXTURE_ROOT/events\"\nexit 84\n",
    );
    let result = fixture.invoke("scripts/build-linux.sh", &["--cuda-archh", "90"], &[]);
    assert_eq!(result.process.status.unwrap().code(), Some(2));
    assert!(fixture.log("events").is_empty());
    assert!(fixture.log("cargo.args").is_empty());
    let stderr = String::from_utf8_lossy(result.stderr.as_ref().unwrap().as_bytes());
    assert!(stderr.contains("unknown option: --cuda-archh"));
    assert!(stderr.contains("usage: scripts/build-linux.sh"));
}
