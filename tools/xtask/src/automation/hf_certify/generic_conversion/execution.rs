//! Observed native conversion window/job, verification and complete sidecar custody.
use super::{contract::Input, guard, identity};
use crate::{
    automation::hf_certify::{admission, execution as child},
    command::DynResult,
    process::Cancellation,
};
use serde_json::{Value, json};
use std::{path::Path, time::Instant};
fn text(path: &Path) -> DynResult<String> {
    Ok(path.to_str().ok_or("generic conversion UTF8 path")?.into())
}
fn copy_fresh_or_identical(source: &Path, target: &Path, limit: u64) -> DynResult<()> {
    let bytes = admission::read(source, limit)?;
    match std::fs::symlink_metadata(target) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            use std::io::Write;
            let mut file =
                tempfile::NamedTempFile::new_in(target.parent().ok_or("sidecar parent")?)?;
            file.write_all(&bytes)?;
            file.as_file().sync_all()?;
            file.persist_noclobber(target)?;
        }
        Ok(m) if m.is_file() && admission::read(target, limit)? == bytes => (),
        _ => return Err("generic sidecar existing identity differs".into()),
    }
    Ok(())
}
fn card(input: &Input) -> DynResult<()> {
    let target = input.artifact_directory().join("README.md");
    let bytes = input.card();
    match std::fs::symlink_metadata(&target) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            use std::io::Write;
            let mut file = tempfile::NamedTempFile::new_in(target.parent().ok_or("card parent")?)?;
            file.write_all(&bytes)?;
            file.as_file().sync_all()?;
            file.persist_noclobber(target)?;
        }
        Ok(m) if m.is_file() && admission::read(&target, 1048576)? == bytes => (),
        _ => return Err("generic beta card existing identity differs".into()),
    }
    Ok(())
}
pub(super) fn execute(
    input: &Input,
    root: &Path,
    until: Instant,
    cancel: &Cancellation,
    evidence: &mut Value,
) -> DynResult<()> {
    let before = identity::observe(input, false, root, "before", until, cancel)?;
    evidence["pre_identity"] = json!(before.files);
    if !input.upload_only {
        let binary = input.binary.as_ref().ok_or("generic binary absent")?;
        let result = child::run_process(
            &binary.path,
            input.convert_args(),
            root,
            "convert",
            until,
            cancel,
        )?;
        evidence["convert_process"] = super::super::publication::process_observation(&result);
        if !child::clean(&result) {
            return Err("generic convert-job child incomplete; native work retained".into());
        }
        guard(until, cancel)?;
        if input.dry_run {
            let after = identity::observe(input, false, root, "after-dry-run", until, cancel)?;
            if before.files != after.files {
                return Err("generic dry-run binary custody differs".into());
            }
            evidence["dry_run_completed"] = json!(true);
            return guard(until, cancel);
        }
        let manifest = input.work_directory.join("convert-manifest.json");
        let report = child::run_process(
            &binary.path,
            vec![
                "verify-job".into(),
                "--manifest".into(),
                text(&manifest)?,
                "--json".into(),
            ],
            root,
            "verify",
            until,
            cancel,
        )?;
        evidence["verify_process"] = super::super::publication::process_observation(&report);
        if !child::clean(&report) {
            return Err("generic verify-job child incomplete".into());
        }
        let verified: Value = serde_json::from_slice(
            report
                .stdout
                .as_ref()
                .ok_or("generic verify stdout absent")?
                .as_bytes(),
        )?;
        let actual = verified["expected_splits"]
            .as_u64()
            .filter(|n| *n >= input.expected_splits as u64 && *n <= 1024)
            .ok_or("generic verified effective split count refused")?;
        if verified["complete"] != true
            || verified["completed_count"].as_u64() != Some(actual)
            || verified["basename"] != input.output_basename
            || verified["prefix"] != input.target_prefix
        {
            return Err("generic actual native verification identity differs".into());
        }
        evidence["verification"] = verified;
        guard(until, cancel)?;
        copy_fresh_or_identical(
            &manifest,
            &input
                .artifact_directory()
                .join("skippy-convert-manifest.json"),
            8 * 1048576,
        )?;
        let status = input.work_directory.join("status.json");
        match std::fs::symlink_metadata(&status) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
            Ok(m) if m.is_file() => copy_fresh_or_identical(
                &status,
                &input
                    .artifact_directory()
                    .join("skippy-convert-status.json"),
                1048576,
            )?,
            _ => return Err("generic status sidecar type/read refused".into()),
        };
        card(input)?;
    }
    let after = identity::observe(input, true, root, "after", until, cancel)?;
    if !input.upload_only && after.files.get("__binary__") != before.files.get("__binary__") {
        return Err("generic native binary changed".into());
    }
    if !input.upload_only
        && after.effective_splits.map(|n| n as u64)
            != evidence["verification"]["expected_splits"].as_u64()
    {
        return Err("generic manifest and verified effective split counts differ".into());
    }
    evidence["effective_splits"] = json!(after.effective_splits);
    evidence["artifact_roster"] = json!(after.files);
    evidence["local_completed"] = json!(true);
    guard(until, cancel)
}
