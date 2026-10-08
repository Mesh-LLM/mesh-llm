use crate::command::DynResult;
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
pub(super) struct Model {
    pub family: String,
    pub class: Class,
    pub repo: String,
    pub revision: String,
    pub file: String,
    pub sha256: String,
    pub native_context_tokens: u64,
}

#[derive(Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub(super) enum Class {
    Dense,
    Moe,
    HybridRecurrent,
}

pub(super) struct VerifiedModel {
    pub file: PathBuf,
    pub reference: String,
    pub sha256: String,
    pub recurrent: bool,
}

pub(super) fn verify(
    models: &[Model],
    family: &str,
    file: &Path,
    minimum_context: u64,
) -> DynResult<VerifiedModel> {
    let mut matches = models.iter().filter(|model| model.family == family);
    let model = matches
        .next()
        .ok_or("family is absent from replay matrix")?;
    if matches.next().is_some() {
        return Err("family occurs more than once in replay matrix".into());
    }
    for text in [&model.repo, &model.file] {
        if text.is_empty() || text.chars().any(char::is_control) {
            return Err("model pin text must be nonempty and contain no control characters".into());
        }
    }
    if model.revision.len() != 40
        || !model
            .revision
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
        || model.native_context_tokens < minimum_context
    {
        return Err(
            "model requires an immutable revision and sufficient declared native context".into(),
        );
    }
    let file = file.canonicalize()?;
    super::model_preflight::verify(&file, &model.sha256, minimum_context)?;
    Ok(VerifiedModel {
        file,
        reference: format!("{}@{}/{}", model.repo, model.revision, model.file),
        sha256: model.sha256.clone(),
        recurrent: model.class == Class::HybridRecurrent,
    })
}
