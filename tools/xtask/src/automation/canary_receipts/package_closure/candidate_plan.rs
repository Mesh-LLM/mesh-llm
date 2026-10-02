use super::{
    candidate_view, process,
    producer_receipt::{self, Context},
};
use crate::{
    automation::{canary_receipts::Digest, canary_source_plan},
    command::DynResult,
};
use serde::Deserialize;
use std::{fs, path::PathBuf};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub(super) context: Context,
    pub(super) root: PathBuf,
    pub(super) base: String,
    pub(super) candidate: String,
    pub(super) cache_root: PathBuf,
    pub(super) output: PathBuf,
}

pub(super) fn admit(input: &Input) -> DynResult<Digest> {
    input.context.validate()?;
    if !input.root.is_absolute() || !input.cache_root.is_absolute() || !input.cache_root.is_dir() {
        return Err("candidate source and selected cache must be absolute existing paths".into());
    }
    let root = input.root.canonicalize()?;
    candidate_view::producer(&root, &input.base, &input.candidate)?;
    if (!input.context.selected_source.is_empty()
        && (input.candidate != input.context.selected_source || input.base != input.candidate))
        || (input.context.selected_source.is_empty()
            && input.base != input.context.controller_revision)
    {
        return Err("candidate plan differs from frozen controller/selected source".into());
    }
    let output = producer_receipt::new_output(
        &input.output,
        &[&root, &input.context.controller_root.canonicalize()?],
    )?;
    let scratch = candidate_view::materialize(
        &root,
        &input.candidate,
        output.parent().ok_or("plan output has no parent")?,
    )?;
    let request = serde_json::json!({"controller_root":input.context.controller_root,"source_root":scratch.path.join("source"),"controller_revision":input.context.controller_revision,
        "selected_revision":input.candidate,"manifest":"ci/llama-canary/family-certified.json","output":output,
        "cache":{"mode":"gguf_metadata","root":input.cache_root}});
    canary_source_plan::admit_document(&serde_json::to_vec(&request)?, &process::cancellation())?;
    process::check()?;
    candidate_view::producer(&root, &input.base, &input.candidate)?;
    input.context.validate()?;
    // Preserve the admitted receipt bytes; pack will compare them to the dependency output.
    let receipt = fs::read(output.join("source-plan.json"))?;
    scratch.cleanup()?;
    Ok(Digest::of_bytes(&receipt))
}
