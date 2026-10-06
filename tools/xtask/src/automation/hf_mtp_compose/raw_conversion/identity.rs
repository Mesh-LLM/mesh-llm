//! Supervised byte observation, including complete immediate checkpoint roster.
use super::Template;
use crate::{
    automation::hf_certify::{
        admission::{self, Artifact},
        execution,
    },
    command::DynResult,
    process::Cancellation,
};
use serde::{Deserialize, Serialize};
use std::{path::Path, time::Instant};
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema_version: u32,
    template: Template,
    binary: Artifact,
    phase: String,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Receipt {
    pub schema_version: u32,
    pub request_sha256: String,
    pub source_sha256: String,
    pub binary: Artifact,
    pub output: Option<Artifact>,
}
fn sources(template: &Template) -> DynResult<String> {
    let root = template.checkpoint_directory.canonicalize()?;
    if root != template.checkpoint_directory || !std::fs::metadata(&root)?.is_dir() {
        return Err("checkpoint source must be canonical directory".into());
    }
    let expected = template
        .checkpoint_files
        .iter()
        .map(|a| a.path.clone())
        .collect::<std::collections::BTreeSet<_>>();
    let mut actual = std::collections::BTreeSet::new();
    for entry in std::fs::read_dir(&root)? {
        let entry = entry?;
        if actual.len() >= 1024 {
            return Err("checkpoint roster bound".into());
        }
        actual.insert(entry.path());
    }
    if actual != expected {
        return Err("checkpoint complete roster differs from pinned source files".into());
    }
    let mut observed = Vec::new();
    for pin in template
        .checkpoint_files
        .iter()
        .chain(std::iter::once(&template.tokenizer_profile))
    {
        let actual = admission::observe(&pin.path, false)?;
        if actual.sha256 != pin.sha256 {
            return Err("native conversion source byte pin mismatch".into());
        }
        // Preserve caller-visible source leaf as well as canonical observed target.
        observed.push((pin.path.clone(), actual));
    }
    Ok(admission::digest(&serde_json::to_vec(&observed)?))
}
pub(super) fn worker(args: &[String]) -> DynResult<()> {
    let [a, input, b, output] = args else {
        return Err("raw identity flags".into());
    };
    if a != "--input" || b != "--output" {
        return Err("raw identity flags".into());
    }
    let request: Request = serde_json::from_slice(&admission::read(Path::new(input), 1048576)?)?;
    if request.schema_version != 1
        || !["before", "after", "final"].contains(&request.phase.as_str())
    {
        return Err("raw identity schema/phase".into());
    }
    request.template.validate()?;
    let mut binary = request.binary.clone();
    admission::admit(&mut binary, false)?;
    let source_sha256 = sources(&request.template)?;
    let root = Path::new(output)
        .parent()
        .ok_or("raw identity output parent")?
        .canonicalize()?;
    let result = Receipt {
        schema_version: 1,
        request_sha256: admission::digest(&serde_json::to_vec(&request)?),
        source_sha256,
        binary,
        output: if request.phase != "before" {
            Some(admission::observe(&root.join("mtp.gguf"), true)?)
        } else {
            None
        },
    };
    admission::publish(Path::new(output), &result)
}
pub(super) fn observe(
    template: &Template,
    binary: &Artifact,
    root: &Path,
    phase: &str,
    deadline: Instant,
    cancel: &Cancellation,
) -> DynResult<Receipt> {
    let request = Request {
        schema_version: 1,
        template: template.clone(),
        binary: binary.clone(),
        phase: phase.into(),
    };
    let input = root.join(format!("{phase}-raw-input.json"));
    let output = root.join(format!("{phase}-raw-identity.json"));
    admission::publish(&input, &request)?;
    let args = vec![
        "automation".into(),
        "hf-mtp-compose".into(),
        "raw-identity-worker".into(),
        "--input".into(),
        input.to_str().ok_or("raw identity Unicode")?.into(),
        "--output".into(),
        output.to_str().ok_or("raw identity Unicode")?.into(),
    ];
    let raw = execution::run_process(
        &std::env::current_exe()?,
        args,
        root,
        &format!("raw-identity-{phase}"),
        deadline,
        cancel,
    )?;
    if !execution::clean(&raw) {
        return Err("raw checkpoint identity child incomplete/nonzero/cleanup".into());
    }
    let receipt: Receipt = serde_json::from_slice(&admission::read(&output, 1048576)?)?;
    if receipt.schema_version != 1
        || receipt.request_sha256 != admission::digest(&serde_json::to_vec(&request)?)
        || receipt.source_sha256.len() != 64
        || (phase == "before") != receipt.output.is_none()
    {
        return Err("raw checkpoint identity correlation refused".into());
    }
    Ok(receipt)
}
