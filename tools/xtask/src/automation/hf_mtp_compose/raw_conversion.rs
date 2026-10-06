//! Pinned local Nemotron conversion and composition; no acquisition or upload.
#[path = "raw_conversion/identity.rs"]
mod identity;
use super::job_phase::{self, MtpSource};
use crate::{
    automation::hf_certify::{
        admission::{self, Artifact},
        execution as process_owner,
    },
    command::DynResult,
    process::Cancellation,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(in crate::automation) struct Template {
    pub checkpoint_directory: PathBuf,
    pub checkpoint_files: Vec<Artifact>,
    pub tokenizer_profile: Artifact,
    pub target_parts: Vec<Artifact>,
    pub target_basename: String,
    pub composite_basename: String,
    pub expected_parts: usize,
    pub mtp_block: u32,
    pub composite_repo: String,
}
pub(in crate::automation) struct Context<'a> {
    pub binary: &'a Artifact,
    pub mesh_revision: &'a str,
    pub deadline: Instant,
    pub cancellation: &'a Cancellation,
}
fn guard(deadline: Instant, cancel: &Cancellation) -> DynResult<()> {
    if cancel.is_cancelled() || Instant::now() >= deadline {
        return Err("native conversion phase cancelled/deadline".into());
    }
    Ok(())
}
fn text(path: &Path) -> DynResult<String> {
    Ok(path
        .to_str()
        .ok_or("native conversion Unicode path")?
        .into())
}
impl Template {
    fn composition(&self, artifact: Artifact) -> job_phase::Template {
        job_phase::Template {
            target_parts: self.target_parts.clone(),
            mtp: MtpSource::SuppliedConverted { artifact },
            target_basename: self.target_basename.clone(),
            composite_basename: self.composite_basename.clone(),
            expected_parts: self.expected_parts,
            mtp_block: self.mtp_block,
            composite_repo: self.composite_repo.clone(),
        }
    }
    pub(in crate::automation) fn validate(&self) -> DynResult<()> {
        if !self.checkpoint_directory.is_absolute()
            || self.checkpoint_files.is_empty()
            || self.checkpoint_files.len() > 1024
        {
            return Err("native conversion bounded checkpoint roster".into());
        }
        let mut names = std::collections::BTreeSet::new();
        for artifact in self
            .checkpoint_files
            .iter()
            .chain(std::iter::once(&self.tokenizer_profile))
        {
            if !artifact.path.is_absolute()
                || artifact.sha256.len() != 64
                || !artifact
                    .sha256
                    .bytes()
                    .all(|b| b.is_ascii_digit() || matches!(b, b'a'..=b'f'))
            {
                return Err("native conversion complete absolute byte pins".into());
            }
        }
        for artifact in &self.checkpoint_files {
            if artifact.path.parent() != Some(self.checkpoint_directory.as_path())
                || !names.insert(artifact.path.file_name().ok_or("checkpoint leaf")?)
            {
                return Err("checkpoint files must be unique immediate source leaves".into());
            }
        }
        if !names.contains(std::ffi::OsStr::new("config.json"))
            || !names.contains(std::ffi::OsStr::new("tokenizer.json"))
            || !self
                .checkpoint_files
                .iter()
                .any(|a| a.path.extension().is_some_and(|e| e == "safetensors"))
        {
            return Err("native conversion requires pinned config/tokenizer/SafeTensors".into());
        }
        self.composition(Artifact {
            path: self
                .tokenizer_profile
                .path
                .with_file_name("__native_converted_mtp__.gguf"),
            sha256: "0".repeat(64),
        })
        .validate()
    }
    fn convert_args(&self, root: &Path) -> DynResult<Vec<String>> {
        Ok(vec![
            "convert".into(),
            "--backend".into(),
            "native-rust".into(),
            "--mtp".into(),
            "--nemotron-mtp-tokenizer-profile".into(),
            text(&self.tokenizer_profile.path)?,
            "--target-prefix".into(),
            "".into(),
            "--output-basename".into(),
            "mtp".into(),
            "--output-type".into(),
            "bf16".into(),
            "--outfile".into(),
            text(&root.join("mtp.gguf"))?,
            "--manifest".into(),
            text(&root.join("convert-manifest.json"))?,
            "--expected-splits".into(),
            "1".into(),
            "--window-size".into(),
            "1".into(),
            "--split-max-size".into(),
            "0".into(),
            "--no-verify-on-complete".into(),
            "--json".into(),
            text(&self.checkpoint_directory)?,
        ])
    }
    fn verify_args(root: &Path) -> DynResult<Vec<String>> {
        Ok(vec![
            "verify-job".into(),
            "--manifest".into(),
            text(&root.join("convert-manifest.json"))?,
            "--json".into(),
        ])
    }
    pub(in crate::automation) fn execute(
        &self,
        root: &Path,
        context: &Context<'_>,
        evidence: &mut Value,
    ) -> DynResult<Value> {
        self.validate()?;
        guard(context.deadline, context.cancellation)?;
        let deadline = context
            .deadline
            .min(Instant::now() + Duration::from_secs(3600));
        let conversion = root.join("native-conversion");
        std::fs::create_dir(&conversion)?;
        let before = identity::observe(
            self,
            context.binary,
            &conversion,
            "before",
            deadline,
            context.cancellation,
        )?;
        evidence["native_conversion_sources_before"] = serde_json::to_value(&before)?;
        invoke(
            &before.binary,
            self.convert_args(&conversion)?,
            &conversion,
            "convert",
            deadline,
            context.cancellation,
            evidence,
        )?;
        let verification = invoke(
            &before.binary,
            Self::verify_args(&conversion)?,
            &conversion,
            "verify",
            deadline,
            context.cancellation,
            evidence,
        )?;
        correlate_verification(&conversion, &verification)?;
        evidence["native_conversion_verification"] = verification;
        let after = identity::observe(
            self,
            context.binary,
            &conversion,
            "after",
            deadline,
            context.cancellation,
        )?;
        evidence["native_conversion_sources_after"] = serde_json::to_value(&after)?;
        if after.source_sha256 != before.source_sha256 || after.binary != before.binary {
            return Err("native conversion input/binary drift".into());
        }
        let mtp = after
            .output
            .clone()
            .ok_or("native conversion missing observed GGUF")?;
        guard(deadline, context.cancellation)?;
        evidence["converted_mtp"] = serde_json::to_value(&mtp)?;
        let composed = self.composition(mtp).execute(
            root,
            &job_phase::Context {
                binary: &before.binary,
                mesh_revision: context.mesh_revision,
                deadline,
                cancellation: context.cancellation,
            },
            evidence,
        )?;
        evidence["source_unchanged"] = Value::Bool(false);
        let final_sources = identity::observe(
            self,
            context.binary,
            &conversion,
            "final",
            deadline,
            context.cancellation,
        )?;
        evidence["native_conversion_sources_final"] = serde_json::to_value(&final_sources)?;
        if final_sources.source_sha256 != before.source_sha256
            || final_sources.binary != before.binary
            || final_sources.output != after.output
        {
            return Err("native conversion source/output drift during composition".into());
        }
        guard(deadline, context.cancellation)?;
        evidence["source_unchanged"] = Value::Bool(true);
        Ok(composed)
    }
}
fn invoke(
    binary: &Artifact,
    args: Vec<String>,
    root: &Path,
    label: &str,
    deadline: Instant,
    cancel: &Cancellation,
    evidence: &mut Value,
) -> DynResult<Value> {
    let raw = process_owner::run_process(&binary.path, args, root, label, deadline, cancel)?;
    let observed = json!({"label":label,"outcome":format!("{:?}",raw.process.outcome),"exit_code":raw.process.status.as_ref().and_then(std::process::ExitStatus::code),"cleanup_complete":raw.process.cleanup.complete,"cleanup_forced":raw.process.cleanup.forced,"stdout_suppressed_lines":raw.process.stdout.suppressed_lines,"stderr_suppressed_lines":raw.process.stderr.suppressed_lines});
    admission::publish(&root.join(format!("{label}-process.json")), &observed)?;
    evidence[format!("native_conversion_{label}_process")] = observed;
    if !process_owner::clean(&raw) {
        return Err("native conversion child nonzero/incomplete/capture/cleanup refusal".into());
    }
    if label == "convert" {
        return Ok(Value::Null);
    }
    Ok(serde_json::from_slice(
        raw.stdout
            .as_ref()
            .ok_or("native verification stdout")?
            .as_bytes(),
    )?)
}
fn correlate_verification(root: &Path, value: &Value) -> DynResult<()> {
    // Native verify_job report is artifact completeness, not runtime model proof.
    let expected = [
        "root",
        "prefix",
        "basename",
        "expected_splits",
        "completed_count",
        "first_missing",
        "last_present",
        "first_shard",
        "last_shard",
        "complete",
    ];
    let object = value.as_object().ok_or("native verification object")?;
    if object.len() != expected.len()
        || expected.iter().any(|key| !object.contains_key(*key))
        || value["root"] != serde_json::to_value(root)?
        || value["prefix"] != ""
        || value["basename"] != "mtp"
        || value["expected_splits"] != 1
        || value["completed_count"] != 1
        || !value["first_missing"].is_null()
        || value["last_present"] != 1
        || value["first_shard"] != "mtp-00001-of-00001.gguf"
        || value["last_shard"] != "mtp-00001-of-00001.gguf"
        || value["complete"] != true
    {
        return Err("native conversion verification identity/completeness correlation".into());
    }
    Ok(())
}
pub(super) fn identity_worker(args: &[String]) -> DynResult<()> {
    identity::worker(args)
}
#[cfg(test)]
#[path = "raw_conversion/tests.rs"]
mod tests;

/// Local provided-tool frontdoor reuses parent terminal finalizer; Jobs binds
/// Context directly from ObservedBootstrap rather than these supplied fields.
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct LocalInput {
    schema_version: u32,
    conversion: Template,
    binary: Artifact,
    supplied_mesh_revision: String,
    timeout_secs: u64,
}
pub(super) fn run(args: &[String]) -> DynResult<()> {
    let [a, input, b, output] = args else {
        return Err("raw-checkpoint --input FILE --output-directory FRESH_DIRECTORY".into());
    };
    if a != "--input" || b != "--output-directory" {
        return Err("raw-checkpoint closed flags".into());
    }
    let input: LocalInput = serde_json::from_slice(&admission::read(Path::new(input), 1048576)?)?;
    if input.schema_version != 1
        || !(30..=3600).contains(&input.timeout_secs)
        || input.supplied_mesh_revision.len() != 40
        || !input
            .supplied_mesh_revision
            .bytes()
            .all(|b| b.is_ascii_hexdigit())
    {
        return Err("raw checkpoint local schema/budget/revision".into());
    }
    if !input.binary.path.is_absolute()
        || input.binary.sha256.len() != 64
        || !input
            .binary
            .sha256
            .bytes()
            .all(|b| b.is_ascii_digit() || matches!(b, b'a'..=b'f'))
    {
        return Err("raw checkpoint local tool byte pin".into());
    }
    input.conversion.validate()?;

    let requested = std::path::absolute(output)?;
    let parent = requested
        .parent()
        .ok_or("raw output parent")?
        .canonicalize()?;
    let root = parent.join(requested.file_name().ok_or("raw output leaf")?);
    std::fs::create_dir(&root)?;
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let cancel = interrupt.cancellation();
    let deadline = Instant::now() + Duration::from_secs(input.timeout_secs);
    let mut report = json!({"schema_version":1,"status":"FAILED","request_sha256":admission::digest(&serde_json::to_vec(&input)?),"source_unchanged":false,"error":null,"custody":"provided pinned local checkpoint/tokenizer/target/tool; no independent build, model equivalence or remote publication attestation"});
    let result = input.conversion.execute(
        &root,
        &Context {
            binary: &input.binary,
            mesh_revision: &input.supplied_mesh_revision,
            deadline,
            cancellation: &cancel,
        },
        &mut report,
    );
    let signal = interrupt.finish().map_err(|error| error.to_string().into());
    let outcome = super::finalize(&root, &mut report, result, signal, deadline, &cancel);
    println!("{}", root.join("report.json").display());
    outcome
}
