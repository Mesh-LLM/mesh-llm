//! NDJSON copy payload matching the Hub's parent-bound commit API.
use super::policy::{ArtifactIdentity, PromotionPlan};
use anyhow::{Context, Result, bail};
#[cfg(test)]
use base64::Engine;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Write},
    path::PathBuf,
};

#[derive(Clone, Debug)]
pub enum CopyContent {
    Lfs {
        sha256: String,
        byte_size: u64,
    },
    File(PathBuf),
    #[cfg(test)]
    Inline(Vec<u8>),
}

/// Finish and validate every row before making the only main commit request.
/// Output can be a file, so regular artifacts impose no aggregate RAM limit.
pub fn write_payload(
    plan: &PromotionPlan,
    contents: &BTreeMap<String, CopyContent>,
    output: &mut impl Write,
) -> Result<()> {
    if contents.len() != plan.paths().len() {
        bail!("staged contents do not match the promotion catalog");
    }
    serde_json::to_writer(
        &mut *output,
        &json!({"key":"header","value":{
            "summary":format!("Atomically promote layer package from {}",plan.staging_revision()),
            "description":"", "parentCommit":plan.parent_commit(),
        }}),
    )?;
    output.write_all(b"\n")?;
    for path in plan.paths() {
        let content = contents.get(path).context("missing staged catalog file")?;
        let expected = plan
            .identity(path)
            .context("catalog entry lacks expected identity")?;
        match content {
            CopyContent::Lfs { sha256, byte_size } => {
                if *byte_size != expected.byte_size || *sha256 != expected.sha256 {
                    bail!("staged file identity differs from local package catalog");
                }
                serde_json::to_writer(
                    &mut *output,
                    &json!({"key":"lfsFile","value":{
                        "path":path,"algo":"sha256","oid":sha256,"size":byte_size,
                    }}),
                )?;
                output.write_all(b"\n")?;
            }
            CopyContent::File(file) => {
                write_regular_row(path, expected, File::open(file)?, output)?
            }
            #[cfg(test)]
            CopyContent::Inline(bytes) => {
                write_regular_row(path, expected, std::io::Cursor::new(bytes), output)?
            }
        }
    }
    output.flush()?;
    Ok(())
}

fn write_regular_row(
    path: &str,
    expected: &ArtifactIdentity,
    mut input: impl Read,
    output: &mut impl Write,
) -> Result<()> {
    output.write_all(b"{\"key\":\"file\",\"value\":{\"path\":")?;
    serde_json::to_writer(&mut *output, path)?;
    output.write_all(b",\"encoding\":\"base64\",\"content\":\"")?;
    let mut encoder =
        base64::write::EncoderWriter::new(&mut *output, &base64::engine::general_purpose::STANDARD);
    let mut digest = Sha256::new();
    let mut size = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = input.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        size = size
            .checked_add(u64::try_from(read)?)
            .context("staged file size overflow")?;
        if size > expected.byte_size {
            bail!("staged file identity differs from local package catalog");
        }
        digest.update(&buffer[..read]);
        encoder.write_all(&buffer[..read])?;
    }
    encoder.finish()?;
    drop(encoder);
    let hash: String = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    if size != expected.byte_size || hash != expected.sha256 {
        bail!("staged file identity differs from local package catalog");
    }
    output.write_all(b"\"}}\n")?;
    Ok(())
}

#[cfg(test)]
fn encode(plan: &PromotionPlan, contents: &BTreeMap<String, CopyContent>) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    write_payload(plan, contents, &mut bytes)?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> Vec<u8> {
        super::super::fixtures::manifest("shared/weights.gguf", 999, "b".repeat(64))
    }
    fn plan() -> PromotionPlan {
        super::super::policy::promote(&manifest(), "automation/republish-test", &"a".repeat(40))
            .unwrap()
    }
    fn contents() -> BTreeMap<String, CopyContent> {
        BTreeMap::from([
            (
                "shared/weights.gguf".into(),
                CopyContent::Lfs {
                    sha256: "b".repeat(64),
                    byte_size: 999,
                },
            ),
            ("model-package.json".into(), CopyContent::Inline(manifest())),
        ])
    }

    #[test]
    fn payload_binds_parent_and_all_lfs_and_regular_paths_in_one_commit() {
        let bytes = encode(&plan(), &contents()).unwrap();
        let rows = bytes
            .split(|b| *b == b'\n')
            .filter(|row| !row.is_empty())
            .map(|row| serde_json::from_slice::<serde_json::Value>(row).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0]["value"]["parentCommit"], "a".repeat(40));
        assert_eq!(rows[1]["key"], "lfsFile");
        assert_eq!(rows[1]["value"]["path"], "shared/weights.gguf");
        assert_eq!(rows[1]["value"]["oid"], "b".repeat(64));
        assert_eq!(rows[1]["value"]["size"], 999);
        assert_eq!(rows[2]["key"], "file");
        assert_eq!(rows[2]["value"]["path"], "model-package.json");
        let manifest = base64::engine::general_purpose::STANDARD
            .decode(rows[2]["value"]["content"].as_str().unwrap())
            .unwrap();
        assert_eq!(manifest, self::manifest());
    }

    #[test]
    fn missing_extra_and_invalid_lfs_entries_refuse_complete_payload() {
        let mut missing = contents();
        missing.remove("model-package.json");
        assert!(encode(&plan(), &missing).is_err());
        let mut extra = contents();
        extra.insert("unexpected".into(), CopyContent::Inline(vec![]));
        assert!(encode(&plan(), &extra).is_err());
        let mut invalid = contents();
        invalid.insert(
            "shared/weights.gguf".into(),
            CopyContent::Lfs {
                sha256: "not-a-digest".into(),
                byte_size: 999,
            },
        );
        assert!(encode(&plan(), &invalid).is_err());
    }
}

#[cfg(test)]
mod identity_tests {
    use super::*;
    #[test]
    fn changed_regular_bytes_and_lfs_size_refuse_before_payload_publication() {
        let manifest = super::super::fixtures::manifest("weights.gguf", 999, "b".repeat(64));
        let plan =
            super::super::policy::promote(&manifest, "automation/republish-test", &"a".repeat(40))
                .unwrap();
        for (size, bytes) in [(999, b"changed manifest".to_vec()), (998, manifest.clone())] {
            let content = BTreeMap::from([
                (
                    "weights.gguf".into(),
                    CopyContent::Lfs {
                        sha256: "b".repeat(64),
                        byte_size: size,
                    },
                ),
                ("model-package.json".into(), CopyContent::Inline(bytes)),
            ]);
            assert!(encode(&plan, &content).is_err());
        }
    }
}

#[cfg(test)]
#[path = "file_payload_tests.rs"]
mod file_tests;
