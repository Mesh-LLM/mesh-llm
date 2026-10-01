#![cfg(unix)]

use sha2::{Digest, Sha256};
use std::path::Path;
use std::process::Command;

const EPOCH: &str = "test-runner-image-sha256-deadbeef";
const ARCHIVES: [&str; 8] = [
    "src/libllama.a",
    "common/libllama-common.a",
    "common/libllama-common-base.a",
    "ggml/src/libggml.a",
    "ggml/src/libggml-base.a",
    "ggml/src/ggml-cpu/libggml-cpu.a",
    "tools/mtmd/libmtmd.a",
    "vendor/hash/libvendor-hash.a",
];

fn fixture(root: &Path, mutation: &str) -> std::path::PathBuf {
    let source = root.join("source/build-stage-abi-static");
    for archive in ARCHIVES {
        if mutation == archive {
            continue;
        }
        let path = source.join(archive);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, archive).unwrap();
    }
    let linkage = if mutation == "linkage" {
        "dynamic"
    } else {
        "static"
    };
    let stamp = format!(
        "stamp-version=3\npatched-sha=abc\nbackend=cpu\nlink-mode={linkage}\ntoolchain-epoch={EPOCH}\ncmake-arg=-DGGML_NATIVE=OFF\n"
    );
    std::fs::write(source.join(".mesh-llm-build-stamp"), &stamp).unwrap();
    std::fs::write(
        source.join("CMakeCache.txt"),
        "GGML_OPENMP_ENABLED:BOOL=OFF\n",
    )
    .unwrap();
    let target = if cfg!(target_arch = "aarch64") {
        "aarch64-unknown-linux-gnu"
    } else {
        "x86_64-unknown-linux-gnu"
    };
    let manifest = serde_json::json!({
        "schema_version": 3, "contract": "mesh-llm-static-abi-v3",
        "target_triple": if mutation == "target" { "other-target" } else { target },
        "backend": "cpu", "build_directory": "build-stage-abi-static",
        "toolchain_epoch": if mutation == "epoch" { "other-epoch" } else { EPOCH },
        "build_stamp_sha256": hex::encode(Sha256::digest(stamp.as_bytes())),
    });
    std::fs::write(
        source.join(".mesh-llm-static-abi-input.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let download = root.join("download");
    std::fs::create_dir(&download).unwrap();
    let archive = download.join("mesh-llm-static-abi.tar.gz");
    let mut tar = Command::new("tar");
    tar.env("COPYFILE_DISABLE", "1");
    tar.arg("-C")
        .arg(root.join("source"))
        .arg("-czf")
        .arg(&archive)
        .arg("build-stage-abi-static");
    if mutation == "sibling" {
        std::fs::write(root.join("source/sibling"), "unexpected").unwrap();
        tar.arg("sibling");
    }
    assert!(tar.status().unwrap().success());
    let digest = hex::encode(Sha256::digest(std::fs::read(&archive).unwrap()));
    std::fs::write(
        download.join("mesh-llm-static-abi.tar.gz.sha256"),
        format!("{digest}  mesh-llm-static-abi.tar.gz\n"),
    )
    .unwrap();
    if mutation == "extra" {
        std::fs::write(download.join("extra"), "extra").unwrap();
    }
    download
}

#[test]
fn restore_checks_real_archive_identity_and_complete_link_closure() {
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let target = if cfg!(target_arch = "aarch64") {
        "aarch64-unknown-linux-gnu"
    } else {
        "x86_64-unknown-linux-gnu"
    };
    for mutation in [
        "",
        "target",
        "epoch",
        "linkage",
        "sibling",
        "extra",
        "common/libllama-common-base.a",
        "vendor/hash/libvendor-hash.a",
    ] {
        let temporary = tempfile::tempdir().unwrap();
        let download = fixture(temporary.path(), mutation);
        let destination = temporary.path().join("restored/build-stage-abi-static");
        let output = Command::new("bash")
            .arg(repository.join("scripts/restore-static-abi-input.sh"))
            .arg(download)
            .arg(&destination)
            .args([target, "cpu"])
            .env("MESH_LLM_AUTOMATION_BIN", env!("CARGO_BIN_EXE_xtask"))
            .env("MESH_LLM_LLAMA_TOOLCHAIN_EPOCH", EPOCH)
            .output()
            .unwrap();
        assert_eq!(
            output.status.success(),
            mutation.is_empty(),
            "{mutation}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        if mutation.is_empty() {
            assert_eq!(
                std::fs::read(destination.join("vendor/hash/libvendor-hash.a")).unwrap(),
                b"vendor/hash/libvendor-hash.a"
            );
        } else {
            assert!(!destination.exists());
        }
    }
}
