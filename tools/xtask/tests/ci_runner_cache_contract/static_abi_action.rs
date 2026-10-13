//! Actual portable ABI action with inert archive bytes and native verification.
use super::{
    support::{Fixture, action},
    workflow_yaml::Node,
};
use std::{
    fs,
    process::{Command, Output},
};
const BUILD: &str = ".deps/llama.cpp/build-stage-abi-static";
const ARCHIVES: [&str; 8] = [
    "src/libllama.a",
    "common/libllama-common.a",
    "common/libllama-common-base.a",
    "ggml/src/libggml.a",
    "ggml/src/libggml-base.a",
    "tools/mtmd/libmtmd.a",
    "vendor/hash/libvendor-hash.a",
    "ggml/src/libggml-cpu.a",
];
const SHA: &str = "0123456789abcdef0123456789abcdef01234567";
fn fixture() -> Fixture {
    let fixture = Fixture::new();
    for relative in ARCHIVES {
        let path = fixture.path().join(BUILD).join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, format!("inert archive {relative}\n")).unwrap();
    }
    fs::write(
        fixture.path().join(".deps/llama.cpp/.mesh-llm-patched-sha"),
        format!("{SHA}\n"),
    )
    .unwrap();
    fs::write(fixture.path().join(BUILD).join(".mesh-llm-build-stamp"),format!("stamp-version=3\npatched-sha={SHA}\nbackend=cpu\nlink-mode=static\ntoolchain-epoch=finite-epoch\ncmake-arg=-DGGML_NATIVE=OFF\n")).unwrap();
    fs::write(fixture.path().join(BUILD).join("CMakeCache.txt"),format!("CMAKE_HOME_DIRECTORY:PATH={}\nGGML_OPENMP_ENABLED:BOOL=ON\nOpenMP_C_LIB_NAMES:STRING=gomp\n",fixture.path().display())).unwrap();
    fs::write(
        fixture.path().join(BUILD).join("not-a-link-input.o"),
        b"unrelated compiler output",
    )
    .unwrap();
    fs::create_dir_all(fixture.path().join("scripts")).unwrap();
    fixture.executable(
        "prepare-observer",
        "[[ \"$#\" == 1 && \"$1\" == pinned ]] || exit 97; printf 'prepare\\n' >> events",
    );
    fixture.executable(
        "build-observer",
        "[[ \"$#\" == 0 ]] || exit 97; printf 'build\\n' >> events",
    );
    for (from, to) in [
        ("prepare-observer", "prepare-llama.sh"),
        ("build-observer", "build-llama.sh"),
    ] {
        fs::copy(
            fixture.path().join("bin").join(from),
            fixture.path().join("scripts").join(to),
        )
        .unwrap();
    }
    fixture.executable(
        "uname",
        "[[ \"$#\" == 1 && \"$1\" == -m ]] || exit 97; printf '%s\\n' \"$RUNNER_ARCH\"",
    );
    for name in ["cargo", "just", "cmake", "rustc"] {
        fixture.executable(name, "printf 'forbidden compiler\\n' >> events; exit 97");
    }
    fixture
}
fn run(fixture: &Fixture, build: &str, target: &str, arch: &str) -> Output {
    let document = action("prepare-static-abi-input");
    let Node::Seq(steps) = document.get("runs").unwrap().get("steps").unwrap() else {
        panic!("action steps")
    };
    let [step] = steps.as_slice() else {
        panic!("one ABI action step required")
    };
    let mut command = Command::new("/bin/bash");
    command.env_clear().current_dir(fixture.path());
    command.env(
        "PATH",
        format!("{}:/usr/bin:/bin", fixture.path().join("bin").display()),
    );
    command
        .env("RUNNER_TEMP", fixture.path())
        .env("GITHUB_OUTPUT", fixture.path().join("outputs"));
    command.env("MESH_LLM_AUTOMATION_BIN", env!("CARGO_BIN_EXE_xtask"));
    for (key, value) in [
        ("INPUT_BACKEND", "cpu"),
        ("INPUT_TARGET", target),
        ("INPUT_BUILD", build),
        ("LLAMA_STAGE_BUILD_DIR", BUILD),
        ("MESH_LLM_LLAMA_TOOLCHAIN_EPOCH", "finite-epoch"),
        ("RUNNER_ARCH", arch),
    ] {
        command.env(key, value);
    }
    command.args(["-c", step.get("run").unwrap().text().unwrap()]);
    fixture.run(command)
}
fn unpack(fixture: &Fixture) {
    fs::create_dir(fixture.path().join("unpacked")).unwrap();
    let mut command = Command::new("/usr/bin/tar");
    command.current_dir(fixture.path()).args([
        "-xzf",
        "static-abi-artifact-output/mesh-llm-static-abi.tar.gz",
        "-C",
        "unpacked",
    ]);
    assert!(fixture.run(command).status.success());
}
#[test]
fn static_abi_action_archives_exact_link_closure_with_native_stamp_manifest_filter_and_checksum() {
    for (build, cpu_nested) in [("true", false), ("false", true)] {
        let fixture = fixture();
        if cpu_nested {
            let from = fixture.path().join(BUILD).join("ggml/src/libggml-cpu.a");
            let to = fixture
                .path()
                .join(BUILD)
                .join("ggml/src/ggml-cpu/libggml-cpu.a");
            fs::create_dir_all(to.parent().unwrap()).unwrap();
            fs::rename(from, to).unwrap();
        }
        let result = run(&fixture, build, "x86_64-unknown-linux-gnu", "x86_64");
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(
            fs::read_to_string(fixture.path().join("events")).unwrap(),
            if build == "true" {
                "prepare\nbuild\n"
            } else {
                "prepare\n"
            }
        );
        unpack(&fixture);
        let stage = fixture.path().join("unpacked/build-stage-abi-static");
        for relative in ARCHIVES {
            let selected = if cpu_nested && relative == "ggml/src/libggml-cpu.a" {
                "ggml/src/ggml-cpu/libggml-cpu.a"
            } else {
                relative
            };
            assert_eq!(
                fs::read(stage.join(selected)).unwrap(),
                format!("inert archive {relative}\n").as_bytes()
            );
        }
        assert!(!stage.join("not-a-link-input.o").exists());
        let cache = fs::read_to_string(stage.join("CMakeCache.txt")).unwrap();
        assert!(!cache.contains(fixture.path().to_str().unwrap()));
        assert!(cache.contains("GGML_OPENMP_ENABLED:BOOL=ON"));
        let manifest: serde_json::Value = serde_json::from_slice(
            &fs::read(stage.join(".mesh-llm-static-abi-input.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(manifest["target_triple"], "x86_64-unknown-linux-gnu");
        assert_eq!(manifest["toolchain_epoch"], "finite-epoch");
        assert_eq!(manifest["schema_version"], 3);
        let base = fixture
            .path()
            .canonicalize()
            .unwrap()
            .join("static-abi-artifact-output");
        assert_eq!(
            fs::read_to_string(fixture.path().join("outputs")).unwrap(),
            format!(
                "archive_path={}/mesh-llm-static-abi.tar.gz\nchecksum_path={}/mesh-llm-static-abi.tar.gz.sha256\nupload_path={}/*\n",
                base.display(),
                base.display(),
                base.display()
            )
        );
    }
}
#[test]
fn static_abi_action_refuses_every_missing_link_input_before_publication() {
    for relative in ARCHIVES {
        let fixture = fixture();
        fs::remove_file(fixture.path().join(BUILD).join(relative)).unwrap();
        let result = run(&fixture, "false", "x86_64-unknown-linux-gnu", "amd64");
        assert!(!result.status.success());
        assert!(!fixture.path().join("outputs").exists());
        assert!(!fixture.path().join("static-abi-artifact-output").exists());
    }
}
#[test]
fn static_abi_action_refuses_architecture_or_build_admission_and_preserves_occupied_upload() {
    for (build, target, arch) in [
        ("true", "aarch64-unknown-linux-gnu", "x86_64"),
        ("invalid", "x86_64-unknown-linux-gnu", "x86_64"),
        ("true", "unsupported", "x86_64"),
    ] {
        let fixture = fixture();
        let result = run(&fixture, build, target, arch);
        assert!(!result.status.success());
        assert!(!fixture.path().join("events").exists());
        assert!(!fixture.path().join("outputs").exists());
    }
    let fixture = fixture();
    fs::create_dir(fixture.path().join("static-abi-artifact-output")).unwrap();
    fs::write(
        fixture.path().join("static-abi-artifact-output/sentinel"),
        b"preserve",
    )
    .unwrap();
    assert!(
        !run(&fixture, "false", "x86_64-unknown-linux-gnu", "x86_64")
            .status
            .success()
    );
    assert_eq!(
        fs::read(fixture.path().join("static-abi-artifact-output/sentinel")).unwrap(),
        b"preserve"
    );
    assert!(!fixture.path().join("outputs").exists());
}
#[test]
fn static_abi_action_rejects_stamp_identity_and_producer_path_leakage_without_archive_publication()
{
    for case in ["stamp", "path"] {
        let fixture = fixture();
        if case == "stamp" {
            fs::write(fixture.path().join(BUILD).join(".mesh-llm-build-stamp"),format!("stamp-version=3\npatched-sha={SHA}\nbackend=cpu\nlink-mode=static\ntoolchain-epoch=wrong-epoch\ncmake-arg=-DGGML_NATIVE=OFF\n")).unwrap();
        } else {
            fs::write(
                fixture.path().join(BUILD).join(ARCHIVES[0]),
                fixture.path().canonicalize().unwrap().to_str().unwrap(),
            )
            .unwrap();
        }
        let result = run(&fixture, "false", "x86_64-unknown-linux-gnu", "x86_64");
        assert!(
            !result.status.success(),
            "case {case} accepted; stdout={}, stderr={}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(!fixture.path().join("outputs").exists());
        assert!(
            !fixture
                .path()
                .join("static-abi-artifact-output/mesh-llm-static-abi.tar.gz")
                .exists()
        );
    }
}
