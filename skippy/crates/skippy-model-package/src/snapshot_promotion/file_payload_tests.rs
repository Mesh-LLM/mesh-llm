use super::*;
use std::fs;

#[test]
fn file_rows_preserve_binary_bytes_across_chunks_and_escape_paths() {
    let directory = tempfile::tempdir().unwrap();
    let bytes = (0..65543)
        .map(|index| (index % 256) as u8)
        .collect::<Vec<_>>();
    let digest = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    let logical = "shared/quoted \"name\".gguf";
    let manifest = super::super::fixtures::manifest(logical, bytes.len() as u64, digest);
    let plan =
        super::super::policy::promote(&manifest, "automation/republish-file", &"a".repeat(40))
            .unwrap();
    let file = directory.path().join("artifact");
    fs::write(&file, &bytes).unwrap();
    let contents = BTreeMap::from([
        (logical.into(), CopyContent::File(file.clone())),
        ("model-package.json".into(), CopyContent::Inline(manifest)),
    ]);
    let payload = encode(&plan, &contents).unwrap();
    let row: serde_json::Value =
        serde_json::from_slice(payload.split(|byte| *byte == b'\n').nth(1).unwrap()).unwrap();
    assert_eq!(row["value"]["path"], logical);
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(row["value"]["content"].as_str().unwrap())
        .unwrap();
    assert_eq!(decoded, bytes);
    fs::write(file, b"changed source").unwrap();
    assert!(
        encode(&plan, &contents)
            .unwrap_err()
            .to_string()
            .contains("identity differs")
    );
}

fn zero_digest(size: u64) -> String {
    let mut digest = Sha256::new();
    let buffer = [0_u8; 64 * 1024];
    let mut remaining = size;
    while remaining > 0 {
        let count = remaining.min(buffer.len() as u64) as usize;
        digest.update(&buffer[..count]);
        remaining -= count as u64;
    }
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[test]
fn regular_files_above_old_individual_and_aggregate_limits_use_file_output() {
    for size in [33 * 1024 * 1024_u64, 65 * 1024 * 1024_u64] {
        let directory = tempfile::tempdir().unwrap();
        let digest = zero_digest(size);
        let original = super::super::fixtures::manifest("first.gguf", size, digest.clone());
        let mut manifest: skippy_package_format::PackageManifest =
            serde_json::from_slice(&original).unwrap();
        manifest
            .artifact_catalog
            .entries
            .push(skippy_package_format::Artifact {
                id: "second".into(),
                path: "second.gguf".into(),
                byte_size: size,
                sha256: digest,
            });
        manifest.package_id = manifest.computed_package_id().unwrap();
        manifest.validate_root().unwrap();
        let manifest = serde_json::to_vec(&manifest).unwrap();
        let plan =
            super::super::policy::promote(&manifest, "automation/republish-large", &"a".repeat(40))
                .unwrap();
        let mut contents = BTreeMap::new();
        for path in ["first.gguf", "second.gguf"] {
            let local = directory.path().join(path);
            File::create(&local).unwrap().set_len(size).unwrap();
            contents.insert(path.to_owned(), CopyContent::File(local));
        }
        let root = directory.path().join("model-package.json");
        fs::write(&root, manifest).unwrap();
        contents.insert("model-package.json".into(), CopyContent::File(root));
        let mut payload = tempfile::NamedTempFile::new().unwrap();
        write_payload(&plan, &contents, payload.as_file_mut()).unwrap();
        assert!(payload.as_file().metadata().unwrap().len() > 64 * 1024 * 1024);
        let mut input = payload.reopen().unwrap();
        let mut buffer = [0_u8; 64 * 1024];
        let mut lines = 0;
        loop {
            let read = input.read(&mut buffer).unwrap();
            if read == 0 {
                break;
            }
            lines += buffer[..read].iter().filter(|byte| **byte == b'\n').count();
        }
        assert_eq!(
            lines, 4,
            "header, two artifacts and manifest must be complete rows"
        );
    }
}
