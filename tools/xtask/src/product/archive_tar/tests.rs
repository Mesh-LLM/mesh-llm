use super::*;
use std::fs;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn entry(root: &std::path::Path, name: &str, bytes: &[u8], mode: u32) -> Entry {
    let path = root.join(name);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, bytes).unwrap();
    Entry {
        name: name.into(),
        path,
        directory: false,
        size: bytes.len() as u64,
        mode,
        sha256: Some(hex::encode(Sha256::digest(bytes))),
    }
}

#[test]
fn aliases_round_trip_as_regular_files_with_one_payload() -> TestResult {
    let root = tempfile::tempdir()?;
    let entries = ["libcuda.so", "libcuda.so.0", "libcuda.so.0.29.0"].map(|name| {
        entry(
            root.path(),
            &format!("runtime/lib/{name}"),
            b"cuda-payload",
            0o755,
        )
    });
    let path = root.path().join("runtime.tar.gz");
    write(File::create(&path)?, &entries)?;
    let mut archive = tar::Archive::new(flate2::read::GzDecoder::new(File::open(&path)?));
    let mut payloads = 0;
    let mut links = 0;
    for member in archive.entries()? {
        let member = member?;
        if member.header().entry_type().is_hard_link() {
            links += 1;
            assert_eq!(member.header().size()?, 0);
            assert_eq!(
                member.link_name()?.unwrap(),
                std::path::Path::new(&entries[0].name)
            );
        } else {
            payloads += 1;
        }
    }
    assert_eq!((payloads, links), (1, 2));
    // This is the same extractor used by skippy-runtime-install.
    let output = root.path().join("extracted");
    tar::Archive::new(flate2::read::GzDecoder::new(File::open(path)?)).unpack(&output)?;
    for entry in entries {
        let path = output.join(entry.name);
        assert!(fs::symlink_metadata(&path)?.is_file());
        assert_eq!(fs::read(&path)?, b"cuda-payload");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(fs::metadata(path)?.permissions().mode() & 0o777, 0o755);
        }
    }
    Ok(())
}

#[test]
fn distinct_bytes_modes_directories_and_nonlibraries_keep_payloads() -> TestResult {
    let root = tempfile::tempdir()?;
    let entries = [
        entry(root.path(), "runtime/lib/libcuda.so", b"cuda", 0o755),
        entry(root.path(), "runtime/lib/libcuda.so.0", b"CUDA", 0o755),
        entry(root.path(), "runtime/lib/libmode.so", b"cuda", 0o644),
        entry(root.path(), "other/lib/libcuda.so", b"cuda", 0o755),
        entry(root.path(), "runtime/lib/README", b"cuda", 0o755),
    ];
    let path = root.path().join("runtime.tar.gz");
    write(File::create(&path)?, &entries)?;
    let mut archive = tar::Archive::new(flate2::read::GzDecoder::new(File::open(path)?));
    for member in archive.entries()? {
        assert!(member?.header().entry_type().is_file());
    }
    Ok(())
}

#[test]
fn alias_drift_is_checked_even_without_writing_its_payload() -> TestResult {
    let root = tempfile::tempdir()?;
    let entries = [
        entry(root.path(), "runtime/lib/libcuda.so", b"cuda", 0o755),
        entry(root.path(), "runtime/lib/libcuda.so.0", b"cuda", 0o755),
    ];
    fs::write(&entries[1].path, b"CUDA")?;
    let result = write(File::create(root.path().join("drift.tar.gz"))?, &entries);
    assert!(result.is_err_and(|error| error.contains("archive input changed")));
    Ok(())
}

#[test]
fn long_link_targets_fall_back_to_regular_payloads() -> TestResult {
    let root = tempfile::tempdir()?;
    let prefix = "r".repeat(100);
    let entries = ["libcuda.so", "libcuda.so.0"]
        .map(|name| entry(root.path(), &format!("{prefix}/lib/{name}"), b"cuda", 0o755));
    let path = root.path().join("long.tar.gz");
    write(File::create(&path)?, &entries)?;
    let mut archive = tar::Archive::new(flate2::read::GzDecoder::new(File::open(path)?));
    for member in archive.entries()? {
        assert!(member?.header().entry_type().is_file());
    }
    Ok(())
}
