//! Exact file planning for a Hugging Face SafeTensors checkpoint.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Component, Path},
};

use anyhow::{Context, Result, ensure};
use serde::Deserialize;

use crate::ModelArtifactFile;

const OPTIONAL_SIDECARS: &[&str] = &[
    "tokenizer_config.json",
    "chat_template.jinja",
    "chat_template.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "generation_config.json",
    "mtp.safetensors",
];

#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: BTreeMap<String, String>,
}

/// The index is read before calling this function when the repository has one.
/// The primary weight remains first so callers can keep their existing model path.
pub fn checkpoint_files(
    primary: &str,
    siblings: &[ModelArtifactFile],
    index_bytes: Option<&[u8]>,
) -> Result<Vec<ModelArtifactFile>> {
    validate_relative_file(primary)?;
    let root = Path::new(primary).parent().unwrap_or_else(|| Path::new(""));
    let primary_relative = Path::new(primary)
        .strip_prefix(root)
        .unwrap_or_else(|_| Path::new(primary))
        .to_string_lossy();
    let listed = siblings
        .iter()
        .map(|file| (file.path.as_str(), file))
        .collect::<BTreeMap<_, _>>();
    let primary_file = listed
        .get(primary)
        .context("selected SafeTensors file is absent from repository listing")?;
    let mut selected = BTreeSet::from([primary.to_string()]);
    let index_name = root.join("model.safetensors.index.json");
    let index_name = index_name.to_string_lossy().into_owned();
    if listed.contains_key(index_name.as_str()) {
        let bytes = index_bytes.context("SafeTensors index must be downloaded before planning")?;
        let index: SafetensorsIndex =
            serde_json::from_slice(bytes).context("parse model.safetensors.index.json")?;
        ensure!(
            !index.weight_map.is_empty(),
            "SafeTensors index has no weights"
        );
        let shards = index.weight_map.into_values().collect::<BTreeSet<_>>();
        ensure!(
            primary == index_name || shards.contains(primary_relative.as_ref()),
            "selected SafeTensors file {primary} is not referenced by the checkpoint index"
        );
        for shard in shards {
            validate_relative_file(&shard)?;
            ensure!(
                Path::new(&shard)
                    .extension()
                    .is_some_and(|ext| ext == "safetensors"),
                "SafeTensors index references a non-weight file: {shard}"
            );
            let shard_path = root.join(&shard).to_string_lossy().into_owned();
            ensure!(
                listed.contains_key(shard_path.as_str()),
                "SafeTensors index references missing repository file {shard_path}"
            );
            selected.insert(shard_path);
        }
        selected.insert(index_name);
    } else if let Some((prefix, total)) = numbered_shard(primary) {
        for part in 1..=total {
            let shard = format!("{prefix}-{part:05}-of-{total:05}.safetensors");
            ensure!(
                listed.contains_key(shard.as_str()),
                "SafeTensors shard is missing from repository: {shard}"
            );
            selected.insert(shard);
        }
    }
    for required in ["config.json", "tokenizer.json"] {
        let name = root.join(required).to_string_lossy().into_owned();
        ensure!(
            listed.contains_key(name.as_str()),
            "SafeTensors checkpoint requires {name}"
        );
        selected.insert(name);
    }
    for sidecar in OPTIONAL_SIDECARS {
        let name = root.join(sidecar).to_string_lossy().into_owned();
        if listed.contains_key(name.as_str()) {
            selected.insert(name);
        }
    }
    let mut files = vec![(*primary_file).clone()];
    files.extend(
        selected
            .into_iter()
            .filter(|name| name != primary)
            .map(|name| listed[&name.as_str()].clone()),
    );
    Ok(files)
}

fn validate_relative_file(file: &str) -> Result<()> {
    ensure!(
        !file.is_empty()
            && Path::new(file)
                .components()
                .all(|part| matches!(part, Component::Normal(_))),
        "SafeTensors index shard path must remain within the checkpoint directory: {file:?}"
    );
    Ok(())
}

fn numbered_shard(primary: &str) -> Option<(String, u32)> {
    let stem = primary.strip_suffix(".safetensors")?;
    let (prefix_and_part, total) = stem.rsplit_once("-of-")?;
    let (prefix, part) = prefix_and_part.rsplit_once('-')?;
    if part.len() != 5
        || !part.bytes().all(|b| b.is_ascii_digit())
        || total.len() != 5
        || !total.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let part: u32 = part.parse().ok()?;
    let total: u32 = total.parse().ok()?;
    (part > 0 && part <= total).then(|| (prefix.to_string(), total))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files(names: &[&str]) -> Vec<ModelArtifactFile> {
        names
            .iter()
            .map(|name| ModelArtifactFile::new(*name))
            .collect()
    }

    #[test]
    fn indexed_checkpoint_selects_exact_shards_and_sidecars() {
        let listed = files(&[
            "model.safetensors-00001-of-00001.safetensors",
            "model.safetensors.index.json",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "other.safetensors",
        ]);
        let index = br#"{"weight_map":{"x":"model.safetensors-00001-of-00001.safetensors"}}"#;
        let planned = checkpoint_files(&listed[0].path, &listed, Some(index)).unwrap();
        assert_eq!(planned[0].path, listed[0].path);
        assert_eq!(planned.len(), 6);
        assert!(!planned.iter().any(|file| file.path == "other.safetensors"));
    }

    #[test]
    fn split_checkpoint_without_index_requires_every_shard() {
        let listed = files(&[
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "config.json",
            "tokenizer.json",
        ]);
        assert_eq!(
            checkpoint_files(&listed[0].path, &listed, None)
                .unwrap()
                .len(),
            4
        );
        let error = checkpoint_files(&listed[0].path, &listed[..1], None).unwrap_err();
        assert!(error.to_string().contains("shard is missing"));
        let from_second = checkpoint_files(&listed[1].path, &listed, None).unwrap();
        assert_eq!(from_second[0].path, listed[1].path);
        assert_eq!(from_second.len(), 4);
        let error = checkpoint_files(&listed[1].path, &listed[1..], None).unwrap_err();
        assert!(error.to_string().contains("shard is missing"));
    }

    #[test]
    fn index_rejects_traversal_and_missing_shards() {
        let listed = files(&[
            "model-00001-of-00002.safetensors",
            "model.safetensors.index.json",
            "config.json",
            "tokenizer.json",
        ]);
        let traversal = br#"{"weight_map":{"x":"model-00001-of-00002.safetensors","y":"../outside.safetensors"}}"#;
        assert!(
            checkpoint_files(&listed[0].path, &listed, Some(traversal))
                .unwrap_err()
                .to_string()
                .contains("must remain within")
        );
        let missing = br#"{"weight_map":{"x":"model-00001-of-00002.safetensors","y":"model-00002-of-00002.safetensors"}}"#;
        assert!(
            checkpoint_files(&listed[0].path, &listed, Some(missing))
                .unwrap_err()
                .to_string()
                .contains("missing repository file")
        );
    }
}
