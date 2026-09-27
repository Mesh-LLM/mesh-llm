use crate::support::{Scratch, TestResult};
use flate2::read::{DeflateDecoder, GzDecoder};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::Path;
use std::process::{Command, Output};

fn command(
    root: &Path,
    verb: &str,
    source: &Path,
    target: &Path,
    kind: &str,
) -> std::io::Result<Output> {
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(root)
        .args(["product", verb])
        .args([source, target])
        .arg(kind)
        .output()
}

fn composed(source: &Path, host: &str) -> TestResult {
    let runtime = source.join("native-runtimes/rt");
    fs::create_dir_all(&runtime)?;
    fs::write(
        runtime.join("manifest.json"),
        b"{\"runtime\":{\"id\":\"rt\",\"mesh_version\":\"1.0\",\"backend\":{\"kind\":\"cpu\"}}}",
    )?;
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["product", "compose", "--bundle"])
        .arg(source)
        .arg("--host")
        .arg(source.join(host))
        .arg("--runtime")
        .arg(runtime)
        .args(["--version", "1.0", "--backend", "cpu"])
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

#[test]
fn migration_product_archive_matches_legacy_unpacked_contents_and_modes() -> TestResult {
    let scratch = Scratch::new()?;
    let source = scratch.path().join("mesh-bundle");
    fs::create_dir_all(&source)?;
    fs::write(source.join("mesh-llm"), b"host")?;
    composed(&source, "mesh-llm")?;
    let legacy = scratch.path().join("legacy.tar.gz");
    let modern = scratch.path().join("modern.tar.gz");
    let historical = Command::new("git")
        .current_dir(crate::support::repository_root())
        .args(["show", "9cf28138c"])
        .output()?;
    assert!(historical.status.success());
    let script = String::from_utf8(historical.stdout)?;
    let start = script
        .find("create_archive() {")
        .ok_or("missing legacy create_archive")?;
    let end = script[start..]
        .find("\nwrite_checksum_sidecar() {")
        .ok_or("missing legacy sidecar")?
        + start;
    let legacy_function = &script[start..end];
    let old = Command::new("bash")
        .arg("-c").arg(format!("set -euo pipefail\npython_bin() {{ command -v python3; }}\n{legacy_function}\ncreate_archive \"$1\" \"$2\" tar.gz"))
        .arg("bash").arg(&source).arg(&legacy).output()?;
    assert!(
        old.status.success(),
        "{}",
        String::from_utf8_lossy(&old.stderr)
    );
    let new = command(scratch.path(), "archive-write", &source, &modern, "tar.gz")?;
    assert!(
        new.status.success(),
        "{}",
        String::from_utf8_lossy(&new.stderr)
    );
    let inspect = |archive: &Path| -> Result<String, Box<dyn std::error::Error>> {
        let result = Command::new("tar").arg("-tvzf").arg(archive).output()?;
        assert!(result.status.success());
        Ok(String::from_utf8(result.stdout)?)
    };
    let old_listing = inspect(&legacy)?;
    let new_listing = inspect(&modern)?;
    let old_entries: Vec<_> = old_listing
        .lines()
        .map(|line| {
            (
                line.split_whitespace().next().unwrap_or(""),
                line.split_whitespace().last().unwrap_or(""),
            )
        })
        .collect();
    let new_entries: Vec<_> = new_listing
        .lines()
        .map(|line| {
            (
                line.split_whitespace().next().unwrap_or(""),
                line.split_whitespace().last().unwrap_or(""),
            )
        })
        .collect();
    assert_eq!(old_entries, new_entries);
    for entry in [
        "mesh-bundle/mesh-llm",
        "mesh-bundle/product-manifest.json",
        "mesh-bundle/native-runtimes/rt/manifest.json",
    ] {
        let extract = |archive: &Path| -> Result<Vec<u8>, Box<dyn std::error::Error>> {
            let output = Command::new("tar")
                .arg("-xOzf")
                .arg(archive)
                .arg(entry)
                .output()?;
            assert!(output.status.success());
            Ok(output.stdout)
        };
        assert_eq!(
            Sha256::digest(extract(&legacy)?),
            Sha256::digest(extract(&modern)?),
            "{entry}"
        );
    }
    let sidecar = fs::read_to_string(modern.with_file_name("modern.tar.gz.sha256"))?;
    assert_eq!(
        sidecar,
        format!(
            "{}  modern.tar.gz\n",
            hex::encode(Sha256::digest(fs::read(modern)?))
        )
    );
    Ok(())
}

#[test]
fn migration_product_cuda_license_survives_both_release_tar_names() -> TestResult {
    let scratch = Scratch::new()?;
    let source = scratch.path().join("mesh-bundle");
    let license = "native-runtimes/meshllm-native-runtime-linux-x86_64-cuda12/licenses/NVIDIA-CUDA-LICENSE.txt";
    let license_path = source.join(license);
    fs::create_dir_all(license_path.parent().ok_or("license has no parent")?)?;
    fs::write(&license_path, b"cuda license fixture")?;
    fs::write(source.join("mesh-llm"), b"host")?;
    let runtime = source.join("native-runtimes/meshllm-native-runtime-linux-x86_64-cuda12");
    fs::write(runtime.join("manifest.json"), b"{\"runtime\":{\"id\":\"meshllm-native-runtime-linux-x86_64-cuda12\",\"mesh_version\":\"1.0\",\"backend\":{\"kind\":\"cuda\"}}}")?;
    let compose = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["product", "compose", "--bundle"])
        .arg(&source)
        .arg("--host")
        .arg(source.join("mesh-llm"))
        .arg("--runtime")
        .arg(&runtime)
        .args(["--version", "1.0", "--backend", "cuda"])
        .output()?;
    assert!(
        compose.status.success(),
        "{}",
        String::from_utf8_lossy(&compose.stderr)
    );
    let member = format!("mesh-bundle/{license}");
    assert_eq!(member.len(), 103);

    for name in [
        "mesh-llm-v1.0-x86_64-unknown-linux-gnu-cuda.tar.gz",
        "mesh-llm-x86_64-unknown-linux-gnu-cuda.tar.gz",
    ] {
        let archive = scratch.path().join(name);
        let result = command(scratch.path(), "archive-write", &source, &archive, "tar.gz")?;
        assert!(
            result.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        let unpacked = Command::new("tar")
            .arg("-xOzf")
            .arg(&archive)
            .arg(&member)
            .output()?;
        assert!(
            unpacked.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&unpacked.stderr)
        );
        assert_eq!(unpacked.stdout, b"cuda license fixture", "{name}");
        let sidecar = fs::read_to_string(archive.with_file_name(format!("{name}.sha256")))?;
        assert_eq!(
            sidecar,
            format!(
                "{}  {name}\n",
                hex::encode(Sha256::digest(fs::read(&archive)?))
            )
        );
    }
    Ok(())
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(
        bytes[offset..offset + 2]
            .try_into()
            .expect("zip short field"),
    )
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("zip long field"),
    )
}

#[test]
fn migration_product_archive_plan_when_bundle_contains_nested_files() -> TestResult {
    let scratch = Scratch::new()?;
    let source = scratch.path().join("mesh-bundle");
    fs::create_dir_all(source.join("native-runtimes/rt/lib"))?;
    fs::write(source.join("mesh-llm"), b"host")?;
    fs::write(source.join("native-runtimes/rt/lib/runtime"), b"runtime")?;
    let target = scratch.path().join("result.tar.gz");

    let output = command(scratch.path(), "archive-plan", &source, &target, "tar.gz")?;

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let plan: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    let names: Vec<&str> = plan["entries"]
        .as_array()
        .expect("entries")
        .iter()
        .map(|item| item["name"].as_str().expect("name"))
        .collect();
    assert_eq!(
        names,
        [
            "mesh-bundle/",
            "mesh-bundle/mesh-llm",
            "mesh-bundle/native-runtimes/",
            "mesh-bundle/native-runtimes/rt/",
            "mesh-bundle/native-runtimes/rt/lib/",
            "mesh-bundle/native-runtimes/rt/lib/runtime"
        ]
    );
    assert!(!target.exists());
    Ok(())
}

#[test]
fn migration_product_tar_archive_when_bundle_contains_host_and_runtime() -> TestResult {
    let scratch = Scratch::new()?;
    let source = scratch.path().join("mesh-bundle");
    fs::create_dir_all(source.join("native-runtimes"))?;
    fs::write(source.join("mesh-llm"), b"original-host")?;
    fs::write(source.join("native-runtimes/runtime"), b"original-runtime")?;
    composed(&source, "mesh-llm")?;
    let target = scratch.path().join("output.tar.gz");

    let output = command(scratch.path(), "archive-write", &source, &target, "tar.gz")?;

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let mut tar = Vec::new();
    GzDecoder::new(fs::File::open(target)?).read_to_end(&mut tar)?;
    let mut names = Vec::new();
    let mut offset = 0;
    while tar[offset..offset + 512].iter().any(|byte| *byte != 0) {
        let header = &tar[offset..offset + 512];
        let name = std::str::from_utf8(&header[..100])?.trim_end_matches('\0');
        let size = u64::from_str_radix(
            std::str::from_utf8(&header[124..136])?
                .trim_end_matches('\0')
                .trim(),
            8,
        )?;
        let size = usize::try_from(size)?;
        names.push((
            name.to_owned(),
            tar[offset + 512..offset + 512 + size].to_vec(),
        ));
        offset += 512 + size.div_ceil(512) * 512;
    }
    assert!(names.contains(&("mesh-bundle/mesh-llm".into(), b"original-host".to_vec())));
    assert!(names.contains(&(
        "mesh-bundle/native-runtimes/runtime".into(),
        b"original-runtime".to_vec()
    )));
    Ok(())
}

#[test]
fn migration_product_zip_archive_when_bundle_contains_host() -> TestResult {
    let scratch = Scratch::new()?;
    let source = scratch.path().join("mesh-bundle");
    fs::create_dir_all(&source)?;
    fs::write(source.join("mesh-llm.exe"), b"portable-host")?;
    composed(&source, "mesh-llm.exe")?;
    let target = scratch.path().join("output.zip");

    let output = command(scratch.path(), "archive-write", &source, &target, "zip")?;

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let zip = fs::read(target)?;
    assert_eq!(read_u32(&zip, 0), 0x0403_4b50);
    let offset = 30 + usize::from(read_u16(&zip, 26));
    assert_eq!(&zip[30..offset], b"mesh-bundle/");
    let second = offset;
    assert_eq!(read_u32(&zip, second), 0x0403_4b50);
    let len = usize::from(read_u16(&zip, second + 26));
    assert_eq!(
        &zip[second + 30..second + 30 + len],
        b"mesh-bundle/mesh-llm.exe"
    );
    let compressed = read_u32(&zip, second + 18);
    let start = second + 30 + len;
    let mut body = Vec::new();
    DeflateDecoder::new(&zip[start..start + usize::try_from(compressed)?])
        .read_to_end(&mut body)?;
    assert_eq!(body, b"portable-host");
    assert!(
        zip.windows(4)
            .any(|part| part == 0x0605_4b50_u32.to_le_bytes())
    );
    Ok(())
}

#[test]
fn migration_product_archive_rejects_stale_host_and_missing_verifier() -> TestResult {
    let scratch = Scratch::new()?;
    let source = scratch.path().join("mesh-bundle");
    fs::create_dir_all(&source)?;
    fs::write(source.join("mesh-llm"), b"host")?;
    fs::create_dir_all(source.join("native-runtimes/rt/bin"))?;
    fs::write(source.join("native-runtimes/rt/bin/verifier"), b"verifier")?;
    composed(&source, "mesh-llm")?;
    let target = scratch.path().join("release.tar.gz");
    fs::write(source.join("mesh-llm"), b"changed")?;
    let stale = command(scratch.path(), "archive-write", &source, &target, "tar.gz")?;
    assert!(!stale.status.success());
    assert!(!target.exists());
    fs::write(source.join("mesh-llm"), b"host")?;
    fs::remove_file(source.join("native-runtimes/rt/bin/verifier"))?;
    let missing = command(scratch.path(), "archive-write", &source, &target, "tar.gz")?;
    assert!(!missing.status.success());
    assert!(!target.exists());
    Ok(())
}

#[test]
fn migration_product_archive_rejects_same_size_digest_drift() -> TestResult {
    let scratch = Scratch::new()?;
    let source = scratch.path().join("mesh-bundle");
    fs::create_dir_all(&source)?;
    fs::write(source.join("mesh-llm"), b"host")?;
    composed(&source, "mesh-llm")?;
    fs::write(source.join("mesh-llm"), b"HOST")?;

    for kind in ["tar.gz", "zip"] {
        let target = scratch.path().join(format!("drift.{kind}"));
        let result = command(scratch.path(), "archive-write", &source, &target, kind)?;
        assert!(!result.status.success(), "{kind}");
        assert!(!target.exists(), "{kind}");
        assert!(
            !target
                .with_file_name(format!("drift.{kind}.sha256"))
                .exists(),
            "{kind}"
        );
    }
    Ok(())
}

#[test]
fn migration_product_archive_rejects_self_inclusion_and_malformed_asset() -> TestResult {
    let scratch = Scratch::new()?;
    let source = scratch.path().join("mesh-bundle");
    fs::create_dir_all(&source)?;
    fs::write(source.join("mesh-llm"), b"host")?;
    composed(&source, "mesh-llm")?;

    let inside = source.join("release.tar.gz");
    let self_inclusion = command(scratch.path(), "archive-write", &source, &inside, "tar.gz")?;
    assert!(!self_inclusion.status.success());
    assert!(!inside.exists());

    let malformed = scratch.path().join("release.tar.gz");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(source.join("product-manifest.json"))?)?;
    manifest["host"]["path"] = "../outside".into();
    fs::write(
        source.join("product-manifest.json"),
        serde_json::to_vec(&manifest)?,
    )?;
    let rejected = command(
        scratch.path(),
        "archive-write",
        &source,
        &malformed,
        "tar.gz",
    )?;
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("unsafe product manifest path"));
    assert!(!malformed.exists());
    Ok(())
}

#[test]
fn migration_product_archive_rejects_missing_symlink_and_traversal() -> TestResult {
    let scratch = Scratch::new()?;
    let source = scratch.path().join("mesh-bundle");
    let target = scratch.path().join("archive.tar.gz");
    let missing = command(scratch.path(), "archive-write", &source, &target, "tar.gz")?;
    assert!(!missing.status.success());
    assert!(!target.exists());

    fs::create_dir_all(&source)?;
    std::os::unix::fs::symlink("missing", source.join("mesh-llm"))?;
    let linked = command(scratch.path(), "archive-write", &source, &target, "tar.gz")?;
    assert!(!linked.status.success());
    assert!(String::from_utf8_lossy(&linked.stderr).contains("symlink"));
    assert!(!target.exists());

    fs::remove_file(source.join("mesh-llm"))?;
    let escaped = command(
        scratch.path(),
        "archive-write",
        &source,
        Path::new("../escape.tar.gz"),
        "tar.gz",
    )?;
    assert!(!escaped.status.success());
    assert!(String::from_utf8_lossy(&escaped.stderr).contains("traverse"));
    Ok(())
}

#[test]
fn migration_product_portable_verifier_when_archive_is_moved_without_source() -> TestResult {
    let scratch = Scratch::new()?;
    let source = scratch.path().join("mesh-bundle");
    let runtime = source.join("native-runtimes/rt");
    fs::create_dir_all(&runtime)?;
    let host = source.join("mesh-llm");
    fs::write(&host, b"host bytes for attestation")?;
    let private = scratch.path().join("private.json");
    let public = scratch.path().join("public.json");
    let keypair = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args([
            "release-attestation",
            "generate-keypair",
            "--private-key-out",
        ])
        .arg(&private)
        .arg("--public-key-out")
        .arg(&public)
        .output()?;
    assert!(
        keypair.status.success(),
        "{}",
        String::from_utf8_lossy(&keypair.stderr)
    );
    let stamped = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["release-attestation", "stamp", "--binary"])
        .arg(&host)
        .arg("--signing-key-file")
        .arg(&private)
        .args(["--node-version", "1.0"])
        .output()?;
    assert!(
        stamped.status.success(),
        "{}",
        String::from_utf8_lossy(&stamped.stderr)
    );
    let producer_digest = Sha256::digest(fs::read(&host)?);
    composed(&source, "mesh-llm")?;
    let archive = scratch.path().join("portable.tar.gz");
    let written = command(scratch.path(), "archive-write", &source, &archive, "tar.gz")?;
    assert!(
        written.status.success(),
        "{}",
        String::from_utf8_lossy(&written.stderr)
    );
    assert_eq!(Sha256::digest(fs::read(&host)?), producer_digest);
    let moved = scratch.path().join("moved outside checkout");
    fs::create_dir(&moved)?;
    let verifier = moved.join("release-attestation-verifier");
    fs::copy(env!("CARGO_BIN_EXE_xtask"), &verifier)?;
    let moved_archive = moved.join("portable.tar.gz");
    fs::copy(&archive, &moved_archive)?;
    let sidecar = moved.join("portable.tar.gz.sha256");
    fs::copy(archive.with_file_name("portable.tar.gz.sha256"), &sidecar)?;
    let checksum = || -> std::io::Result<Output> {
        Command::new("shasum")
            .current_dir(&moved)
            .args(["-a", "256", "-c", "portable.tar.gz.sha256"])
            .output()
    };
    assert_eq!(checksum()?.status.code(), Some(0));
    let extraction = Command::new("tar")
        .args(["-xzf"])
        .arg(&moved_archive)
        .arg("-C")
        .arg(&moved)
        .output()?;
    assert!(
        extraction.status.success(),
        "{}",
        String::from_utf8_lossy(&extraction.stderr)
    );
    fs::write(&sidecar, format!("{}  portable.tar.gz\n", "0".repeat(64)))?;
    let bad_digest = checksum()?;
    assert_eq!(bad_digest.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&bad_digest.stdout).contains("FAILED"));
    let bundle = moved.join("mesh-bundle");
    let host = bundle.join("mesh-llm");
    let public_in_moved = moved.join("public.json");
    fs::copy(&public, &public_in_moved)?;
    fs::remove_dir_all(&source)?;
    let inspect = || -> std::io::Result<Output> {
        Command::new(&verifier)
            .current_dir(&moved)
            .env("PATH", "/usr/bin:/bin")
            .args(["release-attestation", "inspect", "--binary"])
            .arg(&host)
            .arg("--public-key-file")
            .arg(&public_in_moved)
            .arg("--json")
            .output()
    };
    let valid = inspect()?;
    assert_eq!(
        valid.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&valid.stderr)
    );
    let valid_json: serde_json::Value = serde_json::from_slice(&valid.stdout)?;
    assert_eq!(valid_json["status"], "valid");
    let malformed_key = moved.join("malformed public.json");
    fs::write(&malformed_key, b"{bad digest")?;
    let invalid_key = Command::new(&verifier)
        .current_dir(&moved)
        .env("PATH", "/usr/bin:/bin")
        .args(["release-attestation", "inspect", "--binary"])
        .arg(&host)
        .arg("--public-key-file")
        .arg(&malformed_key)
        .arg("--json")
        .output()?;
    assert_eq!(invalid_key.status.code(), Some(1));
    assert!(!invalid_key.stderr.is_empty());
    let mut altered = fs::read(&host)?;
    altered[0] ^= 1;
    fs::write(&host, altered)?;
    let malformed = inspect()?;
    assert_eq!(malformed.status.code(), Some(0));
    let invalid_json: serde_json::Value = serde_json::from_slice(&malformed.stdout)?;
    assert_eq!(invalid_json["status"], "invalid");
    fs::remove_file(&verifier)?;
    assert_eq!(
        inspect()
            .err()
            .ok_or("missing verifier was executable")?
            .kind(),
        std::io::ErrorKind::NotFound
    );
    Ok(())
}
