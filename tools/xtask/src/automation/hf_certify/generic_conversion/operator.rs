//! Operator flags over observed G3 bootstrap and the complete generic conversion child.
#[path = "operator/options.rs"]
mod options;
use super::{contract::Input, guard};
use crate::{
    automation::{
        command_interrupt::Interrupt,
        hf_certify::{admission, bootstrap, execution, publication},
    },
    command::DynResult,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    bootstrap: Option<bootstrap::contract::Input>,
    conversion: Input,
}
fn prepare(manifest: &Manifest, output: &Path) -> DynResult<PathBuf> {
    if manifest.schema_version != 1 {
        return Err("generic operator schema".into());
    }
    manifest.conversion.validate_for_bootstrap()?;
    if !manifest.conversion.upload_only {
        let selected = manifest
            .bootstrap
            .as_ref()
            .ok_or("conversion requires observed G3 bootstrap input")?;
        selected.validate()?;
        if selected.mesh_commit != manifest.conversion.mesh_revision {
            return Err("operator mesh revision/bootstrap mismatch".into());
        }
    }
    let root = std::path::absolute(output)?;
    let root = root
        .parent()
        .ok_or("operator output parent")?
        .canonicalize()?
        .join(root.file_name().ok_or("operator output leaf")?);
    match std::fs::symlink_metadata(&root) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
        _ => return Err("operator requires fresh evidence output".into()),
    };
    let work = &manifest.conversion.work_directory;
    let fresh_work = match std::fs::symlink_metadata(work) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => true,
        Ok(m) if m.is_dir() => false,
        _ => return Err("operator work directory refused".into()),
    };
    let projected_work = if fresh_work {
        work.parent()
            .ok_or("operator work parent")?
            .canonicalize()?
            .join(work.file_name().ok_or("operator work leaf")?)
    } else {
        work.canonicalize()?
    };
    if projected_work != *work
        || root.starts_with(&projected_work)
        || projected_work.starts_with(&root)
    {
        return Err("operator canonical work/evidence ancestry refused".into());
    }
    if !manifest.conversion.upload_only {
        let source = manifest.conversion.source.canonicalize()?;
        if source != manifest.conversion.source
            || root.starts_with(&source)
            || source.starts_with(&root)
            || work.starts_with(&source)
            || source.starts_with(work)
        {
            return Err("operator source/work/evidence ancestry refused".into());
        }
    }
    if fresh_work {
        std::fs::create_dir(work)?;
    }
    super::workspace::admit(&manifest.conversion)?;
    std::fs::create_dir(&root)?;
    Ok(root)
}
fn execute(
    manifest: &Manifest,
    root: &Path,
    until: Instant,
    cancel: &crate::process::Cancellation,
    evidence: &mut Value,
) -> DynResult<()> {
    let mut conversion = manifest.conversion.clone();
    if !conversion.upload_only {
        let dir = root.join("bootstrap");
        std::fs::create_dir(&dir)?;
        let mut rows = Vec::new();
        let observed = bootstrap::execution::execute(
            manifest.bootstrap.as_ref().ok_or("bootstrap absent")?,
            &dir,
            until,
            cancel,
            &mut rows,
        );
        evidence["bootstrap_phases"] = json!(rows);
        let observed = observed?;
        if observed.mesh_commit != conversion.mesh_revision {
            return Err("observed bootstrap revision mismatch".into());
        }
        conversion.binary = Some(observed.binary.clone());
        evidence["observed_bootstrap"] = serde_json::to_value(observed)?;
    }
    guard(until, cancel)?;
    conversion.timeout_seconds = until
        .checked_duration_since(Instant::now())
        .and_then(|d| d.checked_sub(Duration::from_secs(3)))
        .map(|d| d.as_secs())
        .filter(|s| *s >= 5)
        .ok_or("operator remaining conversion/cleanup budget")?;
    conversion.validate()?;
    let path = root.join("conversion-input.json");
    admission::publish(&path, &conversion)?;
    let hash = admission::digest(&serde_json::to_vec(&conversion)?);
    evidence["conversion_request_sha256"] = json!(hash);
    let output = root.join("conversion");
    let report = execution::run_process(
        &std::env::current_exe()?.canonicalize()?,
        vec![
            "automation".into(),
            "hf-certify".into(),
            "generic-conversion".into(),
            "--input".into(),
            path.to_str().ok_or("input Unicode")?.into(),
            "--output-directory".into(),
            output.to_str().ok_or("output Unicode")?.into(),
        ],
        root,
        "conversion",
        until,
        cancel,
    )?;
    evidence["conversion_process"] = publication::process_observation(&report);
    let bytes = admission::read(&output.join("generic-conversion.json"), 8 * 1048576)?;
    let receipt: Value = serde_json::from_slice(&bytes)?;
    evidence["conversion_receipt"] = receipt.clone();
    let expected = if conversion.dry_run {
        "DRY_RUN_COMPLETED"
    } else if conversion.publish_confirmed {
        "PUBLISHED"
    } else {
        "LOCAL_ARTIFACT_READY"
    };
    if !execution::clean(&report)
        || receipt["request_sha256"] != hash
        || receipt["status"] != expected
        || !receipt["error"].is_null()
        || (conversion.publish_confirmed && receipt["publication_completed"] != true)
    {
        return Err("operator conversion child admission refused; partial receipt retained".into());
    }
    guard(until, cancel)
}
pub(super) fn run(args: &[String]) -> DynResult<()> {
    let parsed = options::parse(args)?;
    let mut manifest: Manifest = serde_json::from_slice(&admission::read(&parsed.input, 1048576)?)?;
    parsed.apply(&mut manifest.conversion)?;
    if parsed.xet {
        eprintln!(
            "native conversion uses basic/multipart upload; requested Xet performance mode is not qualified"
        );
    }
    let root = prepare(&manifest, &parsed.output)?;
    let interrupt = Interrupt::install()?;
    let cancel = interrupt.cancellation();
    let until = Instant::now() + Duration::from_secs(manifest.conversion.timeout_seconds);
    let mut evidence = json!({"schema_version":1,"status":"FAILED","request_sha256":admission::digest(&serde_json::to_vec(&manifest)?),"bootstrap_phases":[],"observed_bootstrap":null,"conversion_request_sha256":null,"conversion_process":null,"conversion_receipt":null,"error":null,"xet_high_performance_requested":parsed.xet,"transport":"native basic/multipart; no Xet performance equivalence","source_build_attribution":"observed G3 selected source/tool and fresh binary only; image/compiler/ABI qualification separate"});
    let result = execute(&manifest, &root, until, &cancel, &mut evidence);
    let finish: DynResult<()> = interrupt.finish().map_err(|e| e.to_string().into());
    let terminal = guard(until, &cancel);
    let complete = result.is_ok() && finish.is_ok() && terminal.is_ok();
    evidence["status"] = json!(if complete {
        "OPERATOR_COMPLETED"
    } else {
        "FAILED"
    });
    let errors = result
        .err()
        .into_iter()
        .chain(finish.err())
        .chain(terminal.err())
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join("; ");
    if !errors.is_empty() {
        evidence["error"] = json!(errors);
    }
    admission::publish(&root.join("generic-job.json"), &evidence)?;
    if complete {
        if manifest.conversion.publish_confirmed {
            println!(
                "published https://huggingface.co/{}",
                manifest.conversion.target_repo
            );
        }
        Ok(())
    } else {
        Err("generic operator incomplete; inspect partial phase receipts".into())
    }
}

/// Runs the existing operator under a caller-owned whole deadline and cancellation.
/// The caller owns Interrupt finalization and final delivery status. Observations survive refusal.
pub(in crate::automation::hf_certify) fn execute_inherited(
    bytes: &[u8],
    output: &Path,
    until: Instant,
    cancel: &crate::process::Cancellation,
) -> DynResult<(Value, DynResult<()>)> {
    if bytes.is_empty() || bytes.len() > 65536 {
        return Err("generic delivery manifest byte bound".into());
    }
    guard(until, cancel)?;
    let manifest: Manifest = serde_json::from_slice(bytes)?;
    let root = prepare(&manifest, output)?;
    let mut evidence = json!({"schema_version":1,"status":"FAILED","request_sha256":admission::digest(&serde_json::to_vec(&manifest)?),"bootstrap_phases":[],"observed_bootstrap":null,"conversion_request_sha256":null,"conversion_process":null,"conversion_receipt":null,"error":null,"xet_high_performance_requested":false,"transport":"native basic/multipart; no Xet performance equivalence","source_build_attribution":"observed G3 selected source/tool and fresh binary only; image/compiler/ABI qualification separate"});
    let result = execute(&manifest, &root, until, cancel, &mut evidence);
    let terminal = guard(until, cancel);
    let complete = result.is_ok() && terminal.is_ok();
    evidence["status"] = json!(if complete {
        "OPERATOR_COMPLETED"
    } else {
        "FAILED"
    });
    let errors = result
        .err()
        .into_iter()
        .chain(terminal.err())
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join("; ");
    if !errors.is_empty() {
        evidence["error"] = json!(errors);
    }
    admission::publish(&root.join("generic-job.json"), &evidence)?;
    let outcome = if complete {
        Ok(())
    } else {
        Err("generic delivered operator incomplete; partial observations retained".into())
    };
    Ok((evidence, outcome))
}
/// Closed worker transport delegates full native admission here rather than duplicating it.
pub(in crate::automation::hf_certify) fn delivery_manifest(
    value: &Value,
    timeout: u64,
    credential: Option<&Path>,
    executing: bool,
) -> DynResult<Vec<u8>> {
    let mut manifest: Manifest = serde_json::from_value(value.clone())?;
    if manifest.schema_version != 1 || manifest.conversion.credential_file.is_some() {
        return Err("generic Jobs operator schema/ambient credential file refused".into());
    }
    manifest.conversion.timeout_seconds = timeout;
    if manifest.conversion.publish_confirmed {
        manifest.conversion.credential_file = Some(if executing {
            credential
                .ok_or("generic Jobs private publication credential absent")?
                .into()
        } else {
            PathBuf::from("/work/native-job/private-publication-token")
        });
    }
    manifest.conversion.validate_for_bootstrap()?;
    if !manifest.conversion.upload_only {
        let bootstrap = manifest
            .bootstrap
            .as_ref()
            .ok_or("generic Jobs observed bootstrap absent")?;
        bootstrap.validate()?;
        if bootstrap.mesh_commit != manifest.conversion.mesh_revision {
            return Err("generic Jobs bootstrap/conversion revision mismatch".into());
        }
    }
    Ok(serde_json::to_vec(&manifest)?)
}
#[cfg(test)]
mod delivery_tests {
    use super::*;
    #[test]
    fn generic_inherited_delivery_precancel_and_expired_budget_create_no_output() {
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("uncreated");
        for cancelled in [false, true] {
            let cancel = crate::process::Cancellation::default();
            if cancelled {
                cancel.cancel();
            }
            let until = if cancelled {
                Instant::now() + Duration::from_secs(1)
            } else {
                Instant::now()
            };
            assert!(execute_inherited(b"{}", &output, until, &cancel).is_err());
            assert!(!output.exists());
        }
        temp.close().unwrap();
    }
}
