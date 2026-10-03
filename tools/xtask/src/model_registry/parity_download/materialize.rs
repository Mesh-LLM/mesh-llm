//! Bounded HF process ownership; only completely verified rows are admitted.
use super::{
    Request,
    selection::{Plan, Target},
};
use crate::{
    command::DynResult,
    command_interrupt::Interrupt,
    process::{self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value},
};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
fn arguments(target: &Target) -> Vec<OsString> {
    let mut args = vec!["download".into(), target.repo.clone().into()];
    if let Some(revision) = &target.revision {
        args.extend(target.includes.iter().map(OsString::from));
        args.extend(["--revision".into(), revision.into()]);
    } else {
        for include in &target.includes {
            args.extend(["--include".into(), include.into()]);
        }
    }
    args
}
fn environment() -> BTreeMap<OsString, Value> {
    std::env::vars_os()
        .map(|(key, value)| {
            let upper = key.to_string_lossy().to_ascii_uppercase();
            let secret = ["TOKEN", "PASSWORD", "SECRET", "AUTH"]
                .iter()
                .any(|part| upper.contains(part));
            (
                key,
                if secret {
                    Value::Secret(value)
                } else {
                    Value::Public(value)
                },
            )
        })
        .collect()
}
fn downloaded_root(target: &Target, stdout: &str) -> DynResult<PathBuf> {
    let resolved = stdout
        .lines()
        .rev()
        .map(|line| PathBuf::from(line.strip_prefix("path=").unwrap_or(line)))
        .find(|path| path.exists())
        .ok_or("HF output has no existing artifact path")?;
    if resolved.is_dir() {
        return Ok(resolved);
    }
    if !resolved.is_file() {
        return Err("HF output artifact must be file or directory".into());
    }
    let name = target
        .includes
        .iter()
        .filter(|name| resolved.ends_with(Path::new(name)))
        .max_by_key(|name| Path::new(name).components().count())
        .ok_or("HF output does not name a selected artifact file")?;
    let mut root = resolved;
    for _ in Path::new(name).components() {
        if !root.pop() {
            return Err("artifact path has no snapshot root".into());
        }
    }
    Ok(root)
}
fn guard(cancellation: &Cancellation, deadline: Instant) -> std::io::Result<()> {
    if cancellation.is_cancelled() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Interrupted,
            "parity downloads cancelled",
        ));
    }
    if Instant::now() >= deadline {
        return Err(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "parity verification deadline exceeded",
        ));
    }
    Ok(())
}
fn verify(
    target: &Target,
    stdout: &str,
    output: &mut String,
    cancellation: &Cancellation,
    deadline: Instant,
) -> DynResult<()> {
    guard(cancellation, deadline)?;
    let Some(manifest) = &target.verified else {
        return Ok(());
    };
    let root = downloaded_root(target, stdout)?;
    let artifact = crate::model_registry::manifest::resolve(
        manifest,
        &crate::model_registry::manifest::Selection {
            artifact_id: Some(&target.artifact_id),
            cadence: "manual",
        },
    )
    .map_err(|e| e.to_string())?;
    let mut verified = String::new();
    for file in &artifact.files {
        verified.push_str(
            &crate::model_registry::resolve::verify_with_guard(
                root.to_str().ok_or("artifact root must be UTF-8")?,
                file,
                &mut || guard(cancellation, deadline),
            )
            .map_err(|e| e.to_string())?,
        );
    }
    guard(cancellation, deadline)?;
    output.push_str(&verified);
    Ok(())
}
fn download(
    request: &Request,
    target: &Target,
    cancellation: &Cancellation,
    output: &mut String,
    verified: &mut String,
) -> DynResult<bool> {
    let deadline = Instant::now() + Duration::from_secs(request.timeout);
    let report = process::supervise(
        &ProcessSpec {
            executable: request.hf.clone(),
            cwd: std::env::current_dir()?,
            arguments: arguments(target).into_iter().map(Value::Public).collect(),
            environment: environment(),
        },
        &Limits {
            execution: Duration::from_secs(request.timeout),
            graceful_shutdown: Duration::from_secs(3),
            forced_shutdown: Duration::from_secs(2),
            retained_bytes_per_stream: 1024 * 1024,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        cancellation,
        OutputFiles::default(),
    )?;
    if !report.cleanup.complete || report.stdout.truncated || report.stderr.truncated {
        return Err("HF output or cleanup exceeded ownership bounds".into());
    }
    let stdout = std::str::from_utf8(&report.stdout.bytes_retained)?;
    output.push_str(&String::from_utf8_lossy(&report.stderr.bytes_retained));
    output.push_str(stdout);
    if !stdout.is_empty() && !stdout.ends_with('\n') {
        output.push('\n');
    }
    if report.outcome != process::Outcome::Exited {
        return Err(format!("HF download stopped: {:?}", report.outcome).into());
    }
    if !report.success() {
        output.push_str(&format!(
            "download skipped: {} (status {:?})\n",
            target.label,
            report.status.and_then(|status| status.code())
        ));
        return Ok(false);
    }
    verify(target, stdout, verified, cancellation, deadline)?;
    Ok(true)
}
pub(super) fn execute(request: &Request, plan: &Plan, output: &mut String) -> DynResult<()> {
    output.push_str(&format!("Download targets: {}\n", plan.targets.len()));
    for missing in &plan.missing {
        output.push_str(&format!("missing target: {missing}\n"));
    }
    if request.dry_run {
        for target in &plan.targets {
            output.push_str(&format!(
                "{}: {}\n",
                target.label,
                serde_json::to_string(
                    &arguments(target)
                        .iter()
                        .map(|a| a.to_string_lossy())
                        .collect::<Vec<_>>()
                )?
            ));
        }
        return Ok(());
    }
    let interrupt = Interrupt::install()?;
    let cancellation = interrupt.cancellation();
    let mut verified = String::new();
    let mut skipped = 0;
    let result: DynResult<()> = (|| {
        for target in &plan.targets {
            if cancellation.is_cancelled() {
                return Err("parity downloads cancelled".into());
            }
            output.push_str(&format!("# {}\n", target.label));
            if !download(request, target, &cancellation, output, &mut verified)? {
                skipped += 1;
            }
        }
        Ok(())
    })();
    let finish = interrupt.finish();
    result?;
    complete(output, &verified, plan.targets.len(), skipped, finish)
}

fn complete(
    output: &mut String,
    verified: &str,
    targets: usize,
    skipped: usize,
    admission: Result<(), crate::command_interrupt::Reason>,
) -> DynResult<()> {
    admission?;
    output.push_str(verified);
    output.push_str(&format!(
        "parity download complete: {targets} targets, {skipped} skipped\n"
    ));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn final_interruption_admission_preserves_diagnostics_without_immutable_claim_or_completion() {
        let queued = "verified immutable test artifact: model.gguf (9 bytes)\n";
        let mut output = "finite downloader diagnostic\n".to_owned();
        let refused = complete(
            &mut output,
            queued,
            1,
            0,
            Err(crate::command_interrupt::Reason::Interrupted),
        );
        assert!(refused.is_err());
        assert_eq!(output, "finite downloader diagnostic\n");
        complete(&mut output, queued, 1, 0, Ok(())).unwrap();
        assert!(output.contains(queued));
        assert!(output.contains("parity download complete:"));
    }

    #[test]
    fn parity_verification_guard_rejects_cancelled_and_expired_admission() {
        let cancellation = Cancellation::default();
        assert!(guard(&cancellation, Instant::now() + Duration::from_secs(1)).is_ok());
        assert_eq!(
            guard(&cancellation, Instant::now()).unwrap_err().kind(),
            std::io::ErrorKind::TimedOut
        );
        cancellation.cancel();
        assert_eq!(
            guard(&cancellation, Instant::now() + Duration::from_secs(1))
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::Interrupted
        );
    }
}
