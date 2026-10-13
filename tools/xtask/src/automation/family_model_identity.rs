//! Path-derived certification identity; never resolves or loads a model.
use crate::{command::DynResult, repository::check_report::CheckReport};
use serde_json::{Map, Value};
use std::path::{Component, Path};

fn first_shard_stem(stem: &str) -> &str {
    if let Some((prefix, count)) = stem.rsplit_once("-00001-of-")
        && count.len() == 5
        && count.bytes().all(|byte| byte.is_ascii_digit())
    {
        prefix
    } else {
        stem
    }
}

fn selector(stem: &str) -> Option<&str> {
    stem.char_indices().find_map(|(index, _)| {
        let suffix = &stem[index..];
        let upper = suffix.to_ascii_uppercase();
        let quant_name = upper.strip_prefix("UD-").unwrap_or(&upper);
        let tail = quant_name
            .strip_prefix("IQ")
            .or_else(|| quant_name.strip_prefix('Q'));
        let quant = tail.is_some_and(|tail| {
            tail.as_bytes().first().is_some_and(u8::is_ascii_digit)
                && tail
                    .bytes()
                    .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || byte == b'_')
        });
        (quant || matches!(quant_name, "BF16" | "F16" | "F32")).then_some(suffix)
    })
}

fn identity(model_id: &str, model_path: &str) -> Value {
    let mut identity = Map::new();
    identity.insert("model_id".into(), model_id.into());
    let parts = Path::new(model_path)
        .components()
        .filter_map(|part| match part {
            Component::Normal(value) => value.to_str(),
            Component::ParentDir => Some(".."),
            _ => None,
        })
        .collect::<Vec<_>>();
    for (index, part) in parts.iter().enumerate() {
        let Some(repo) = part.strip_prefix("models--") else {
            continue;
        };
        if index + 3 >= parts.len() || parts[index + 1] != "snapshots" {
            continue;
        }
        let repo = repo.replace("--", "/");
        let revision = parts[index + 2];
        let file = parts[index + 3..].join("/");
        let basename = parts.last().copied().unwrap_or_default();
        let stem = first_shard_stem(basename.strip_suffix(".gguf").unwrap_or(basename));
        for (key, value) in [
            ("source_repo", repo.clone()),
            ("source_revision", revision.to_owned()),
            ("source_file", file.clone()),
            ("canonical_ref", format!("{repo}@{revision}/{file}")),
            ("distribution_id", stem.to_owned()),
        ] {
            identity.insert(key.into(), value.into());
        }
        if basename.ends_with(".gguf")
            && let Some(quant) = selector(stem)
        {
            identity.insert("selector".into(), quant.into());
        }
        break;
    }
    Value::Object(identity)
}

fn snapshot_revision(path: &Path) -> DynResult<&str> {
    let mut revision = None;
    let mut parts = path.components();
    while let Some(part) = parts.next() {
        if part.as_os_str() != "snapshots" {
            continue;
        }
        let Some(Component::Normal(candidate)) = parts.next() else {
            continue;
        };
        let Some(candidate) = candidate.to_str() else {
            continue;
        };
        if !(40..=64).contains(&candidate.len())
            || !candidate
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            continue;
        }
        if revision.is_some_and(|previous| previous != candidate) {
            return Err("ambiguous immutable snapshot revision".into());
        }
        revision = Some(candidate);
    }
    revision.ok_or_else(|| "path has no immutable snapshot revision".into())
}

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    if let [option, path] = args
        && option == "--snapshot-revision"
    {
        return CheckReport::success(format!("{}\n", snapshot_revision(Path::new(path))?)).emit();
    }

    let [model_id, model_path] = args else {
        return Err("usage: automation family-model-identity MODEL_ID MODEL_PATH".into());
    };
    CheckReport::success(format!(
        "{}\n",
        serde_json::to_string(&identity(model_id, model_path))?
    ))
    .emit()
}
