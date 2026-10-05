//! Stable family prefixes and interleaved tasks for the synthetic A/B workload.
use super::{Prompt, PromptManifest, options, publish};
use crate::command::DynResult;
use std::fmt::Write;

const MAX_REQUESTS: u64 = 10_000;
const MAX_MANIFEST_BYTES: u64 = 256 * 1024 * 1024;

fn admit(families: u64, requests_per_family: u64, blocks: u64) -> DynResult<usize> {
    if [families, requests_per_family, blocks].contains(&0) {
        return Err("synthetic workload sizes must be positive".into());
    }
    let requests = families
        .checked_mul(requests_per_family)
        .filter(|count| *count <= MAX_REQUESTS)
        .ok_or("synthetic workload exceeds the request phase limit")?;
    // Include a conservative line allowance and room for the header/task. This
    // bounds allocations before constructing repeated long-context prompts.
    let bytes = blocks
        .checked_mul(128)
        .and_then(|bytes| bytes.checked_add(512))
        .and_then(|bytes| bytes.checked_mul(requests))
        .filter(|bytes| *bytes <= MAX_MANIFEST_BYTES)
        .ok_or("synthetic prompt manifest exceeds the 256 MiB allocation budget")?;
    let _ = usize::try_from(bytes)?;
    Ok(usize::try_from(requests)?)
}

fn stable_prefix(family: &str, blocks: u64) -> DynResult<String> {
    let mut prefix =
        format!("You are working in repository family {family}. Follow its fixed rules.\n");
    for index in 0..blocks {
        if index != 0 {
            prefix.push('\n');
        }
        write!(
            prefix,
            "{family}-context-{index:04}: src/{family}/module_{}.rs owns invariant {index}; preserve it exactly.",
            index % 37
        )?;
    }
    Ok(prefix)
}

pub(super) fn interleaved(
    families: u64,
    requests_per_family: u64,
    blocks: u64,
) -> DynResult<Vec<Prompt>> {
    let count = admit(families, requests_per_family, blocks)?;
    let prefixes = (0..families)
        .map(|index| stable_prefix(&format!("family-{index}"), blocks))
        .collect::<DynResult<Vec<_>>>()?;
    let mut prompts = Vec::new();
    prompts.try_reserve_exact(count)?;
    for request in 0..requests_per_family {
        for (family, prefix) in prefixes.iter().enumerate() {
            prompts.push(Prompt {
                family: format!("family-{family}"),
                prompt: format!(
                    "{prefix}\nUnique task {request}: inspect module_{}.rs and return its invariant only.",
                    request % 37
                ),
            });
        }
    }
    Ok(prompts)
}

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let opts = options(
        args,
        &[
            "--families",
            "--requests-per-family",
            "--prefix-blocks",
            "--output",
        ],
        &[
            "--families",
            "--requests-per-family",
            "--prefix-blocks",
            "--output",
        ],
    )?;
    let families = opts["--families"].parse()?;
    let requests = opts["--requests-per-family"].parse()?;
    let blocks = opts["--prefix-blocks"].parse()?;
    let document = PromptManifest {
        metadata: serde_json::json!({"generator":"stable-prefix-v1",
            "families":families,"requests_per_family":requests,"prefix_blocks":blocks})
        .as_object()
        .ok_or("synthetic metadata must be an object")?
        .clone(),
        prompts: interleaved(families, requests, blocks)?,
    };
    let mut bytes = serde_json::to_vec_pretty(&document)?;
    bytes.push(b'\n');
    publish(std::path::Path::new(opts["--output"]), &bytes)
}

#[cfg(test)]
#[path = "synthetic_prompts_tests.rs"]
mod tests;
