use super::{Input, Phase};
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Clone, Serialize)]
pub(super) struct Step {
    pub label: String,
    pub arguments: Vec<String>,
    pub environment: BTreeMap<String, String>,
    pub identity: bool,
}

fn step(label: &str, arguments: &[&str]) -> Step {
    Step {
        label: label.into(),
        arguments: arguments.iter().map(|s| (*s).into()).collect(),
        environment: BTreeMap::new(),
        identity: false,
    }
}

fn native_compile(label: &str, focused: bool) -> Step {
    let mut compile = step(label, &["bash", "scripts/build-llama.sh", "--build-only"]);
    if focused {
        // The normal link closure/probes remain required; defer unrelated upstream executables.
        compile
            .arguments
            .extend(["--target".into(), "test-jinja".into()]);
    }
    if cfg!(target_os = "macos") {
        compile.arguments.extend([
            "-DCMAKE_OSX_ARCHITECTURES=arm64".into(),
            "-DGGML_METAL_EMBED_LIBRARY=ON".into(),
        ]);
    }
    compile.environment = BTreeMap::from([
        ("LLAMA_STAGE_UPSTREAM_TESTS".into(), "ON".into()),
        ("LLAMA_STAGE_BUILD_TESTS".into(), "ON".into()),
    ]);
    compile
}

pub(super) fn steps(input: &Input) -> Vec<Step> {
    let mut result = vec![step(
        "native-test-environment",
        &[
            "uv",
            "sync",
            "--locked",
            "--no-python-downloads",
            "--project",
            &input
                .controller
                .join("ci/llama-canary/jinja")
                .to_string_lossy(),
            "--python",
            &std::env::var("SKIPPY_WORKLOAD_SDK_PYTHON").unwrap_or_else(|_| "3.12".into()),
        ],
    )];
    if input.phase == Phase::Environment {
        return result;
    }
    result.extend([
        step(
            "native-prepare",
            &["bash", "scripts/prepare-llama.sh", "pinned"],
        ),
        native_compile("native-compile", true),
        step(
            "upstream-python-tests",
            &[
                "ctest",
                "--test-dir",
                &input.build.to_string_lossy(),
                "--output-on-failure",
                "--timeout",
                "900",
                "-L",
                "python",
                "--no-tests=error",
            ],
        ),
        step(
            "static-repair-binaries",
            &[
                "just",
                "with-lld",
                "cargo",
                "build",
                "--locked",
                "--no-default-features",
                "-p",
                "skippy-runtime",
                "-p",
                "skippy-cli",
                "-p",
                "skippy-package-builder",
                "-p",
                "skippy-correctness",
                "-p",
                "skippy-topology",
                "--bins",
            ],
        ),
    ]);
    result.push(Step {
        identity: true,
        ..step("static-runtime-identity", &[])
    });
    let mut smoke = step(
        "focused-split-state-smoke",
        &[
            "bash",
            &input
                .controller
                .join("skippy/scripts/skippy-ci-smoke.sh")
                .to_string_lossy(),
        ],
    );
    smoke.environment = BTreeMap::from([
        (
            "WORK_DIR".into(),
            input
                .state
                .join("focused-smoke")
                .to_string_lossy()
                .into_owned(),
        ),
        ("RUN_DENSE_CHAIN_SMOKE".into(), "1".into()),
        ("SKIPPY_SMOKE_FOCUSED_ONLY".into(), "1".into()),
        ("SKIPPY_SMOKE_SKIP_BUILD".into(), "1".into()),
        (
            "SMOKE_FLASH_ATTN".into(),
            if input.backend == "cpu" {
                "disabled"
            } else {
                "enabled"
            }
            .into(),
        ),
        (
            "SMOKE_N_GPU_LAYERS".into(),
            if input.backend == "cpu" { "0" } else { "999" }.into(),
        ),
    ]);
    result.push(smoke);
    if input.phase == Phase::Fast {
        return result;
    }
    result.extend([
        native_compile("complete-native-compile", false),
        step(
            "generated-family-and-native-suite",
            &[
                "bash",
                "scripts/check-skippy-generated-family-patch.sh",
                "--full-native-suite",
            ],
        ),
        step(
            "canonical-skippy-smoke",
            &[
                "bash",
                &input
                    .controller
                    .join("skippy/scripts/skippy-ci-smoke.sh")
                    .to_string_lossy(),
            ],
        ),
        step(
            "workload-oracles",
            &[
                "just",
                "skippy-workload-oracles-build",
                &format!("{}-workloads", input.build.display()),
            ],
        ),
    ]);
    let mut system = step(
        "system-one-smoke",
        &["bash", "scripts/skippy-system-one-smoke.sh"],
    );
    system.environment = BTreeMap::from([
        (
            "WORK_DIR".into(),
            input
                .root
                .join("target/skippy-system-one-smoke")
                .to_string_lossy()
                .into_owned(),
        ),
        ("SYSTEMONE_SMOKE_CADENCE".into(), "llama-bump".into()),
        (
            "SYSTEMONE_SMOKE_BUILD_BACKEND".into(),
            input.backend.clone(),
        ),
        (
            "SYSTEMONE_SMOKE_CERTIFIED_BACKENDS".into(),
            input.backend.clone(),
        ),
        ("SYSTEMONE_SMOKE_REQUIRE_QUALIFIED".into(), "1".into()),
    ]);
    let mut laya = step("laya-smoke", &["bash", "scripts/skippy-laya-smoke.sh"]);
    laya.environment = BTreeMap::from([
        (
            "WORK_DIR".into(),
            input
                .root
                .join("target/skippy-system-one-smoke")
                .to_string_lossy()
                .into_owned(),
        ),
        ("LAYA_SMOKE_CADENCE".into(), "llama-bump".into()),
        ("LAYA_SMOKE_DEVICE".into(), "CPU".into()),
    ]);
    result.extend([system, laya]);
    result
}
