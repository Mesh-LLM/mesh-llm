use super::super::regular_publication;
use super::Input;
use anyhow::{Result, anyhow, bail};
use std::{
    collections::BTreeSet,
    io::{Read, Seek},
    time::Instant,
};
pub(super) fn validate(input: &mut Input, until: Instant) -> Result<()> {
    if input.shards.is_empty() || input.shards.len() > 128 {
        bail!("model ordered shard roster bound refused");
    }
    regular_publication::Plan {
        repo: input.repo.clone(),
        parent_commit: input.parent_commit.clone(),
        paths: vec!["receipt.json".into()],
    }
    .validate()?;
    if !input.sidecars.is_empty() {
        regular_publication::Plan {
            repo: input.repo.clone(),
            parent_commit: input.parent_commit.clone(),
            paths: input
                .sidecars
                .iter()
                .map(|f| f.path_in_repo.clone())
                .collect(),
        }
        .validate()?;
    }
    let count = input.shards.len();
    let mut seen = BTreeSet::new();
    let mut prefix = None;
    for (index, shard) in input.shards.iter_mut().enumerate() {
        let path = &shard.path_in_repo;
        if path.len() > 256
            || !path.ends_with(".gguf")
            || !seen.insert(path.clone())
            || path.split('/').any(|p| {
                p.is_empty()
                    || matches!(p, "." | "..")
                    || !p
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
            })
        {
            bail!("model shard repository path refused");
        }
        if let Some(info) = skippy_model_ref::split_gguf_shard_info(path) {
            let part = info.part.parse::<usize>()?;
            let total = info.total.parse::<usize>()?;
            if part != index + 1
                || total != count
                || prefix
                    .as_ref()
                    .is_some_and(|prior: &String| prior != info.prefix)
            {
                bail!("model complete ordered sibling roster mismatch");
            }
            prefix = Some(info.prefix.to_owned());
        } else if count != 1 {
            bail!("model multi-shard names require complete split roster");
        }
        shard.object.verify(until)?;
        let mut magic = [0; 4];
        shard
            .object
            .file
            .read_exact(&mut magic)
            .map_err(|_| anyhow!("model shard GGUF magic absent"))?;
        if magic != *b"GGUF" {
            bail!("model shard GGUF magic invalid");
        }
        shard.object.file.rewind()?;
    }
    for sidecar in &input.sidecars {
        if !seen.insert(sidecar.path_in_repo.clone()) {
            bail!("model sidecar/shard path collision");
        }
    }
    regular_publication::contract::check(until)
}
