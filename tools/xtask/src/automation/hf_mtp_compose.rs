//! Pinned local MTP conversion/composition; no acquisition or remote publication.
mod contract;
mod execution;
mod identity;
pub(in crate::automation) mod job_phase;
pub(in crate::automation) mod raw_conversion;
mod reports;
use crate::{
    automation::{command_interrupt::Interrupt, hf_certify::admission},
    command::DynResult,
};
use serde_json::json;
use std::{
    path::Path,
    time::{Duration, Instant},
};
pub(crate) fn run(args: &[String]) -> DynResult<()> {
    if let [verb, rest @ ..] = args
        && verb == "identity-worker"
    {
        return identity::worker(rest);
    }
    if let [verb, rest @ ..] = args {
        if verb == "raw-identity-worker" {
            return raw_conversion::identity_worker(rest);
        }
        if verb == "raw-checkpoint" {
            return raw_conversion::run(rest);
        }
    }
    if args == ["--help"] {
        println!(
            "automation hf-mtp-compose --input FILE --output-directory FRESH_DIRECTORY; already-converted pinned local GGUFs. Local native conversion: automation hf-mtp-compose raw-checkpoint --input FILE --output-directory FRESH_DIRECTORY; pinned checkpoint and explicit byte-bound tokenizer profile, no acquisition/upload"
        );
        return Ok(());
    }
    let [a, input, b, output] = args else {
        return Err("compose closed flags".into());
    };
    if a != "--input" || b != "--output-directory" {
        return Err("compose closed flags".into());
    }
    let bytes = admission::read(Path::new(input), 262144)?;
    let input: contract::Input = serde_json::from_slice(&bytes)?;
    input.validate()?;
    let requested = std::path::absolute(output)?;
    let parent = requested.parent().ok_or("compose parent")?.canonicalize()?;
    let root = parent.join(requested.file_name().ok_or("compose leaf")?);
    std::fs::create_dir(&root)?;
    let interrupt = Interrupt::install()?;
    let cancel = interrupt.cancellation();
    let deadline = Instant::now() + Duration::from_secs(input.timeout_secs);
    let mut report = json!({"schema_version":1,"status":"FAILED","request_sha256":admission::digest(&serde_json::to_vec(&input)?),"source_unchanged":false,"error":null,"custody":"pinned supplied local artifacts; no independent build, conversion, runtime or remote publication attestation"});
    let result = execution::execute(&input, &root, deadline, &cancel, &mut report);
    let signal: DynResult<()> = interrupt.finish().map_err(|e| e.to_string().into());
    let complete = finalize(&root, &mut report, result, signal, deadline, &cancel);
    println!("{}", root.join("report.json").display());
    complete
}
/// Make the terminal decision before publishing any eligible plan.
fn finalize(
    root: &Path,
    report: &mut serde_json::Value,
    result: DynResult<serde_json::Value>,
    signal: DynResult<()>,
    deadline: Instant,
    cancel: &crate::process::Cancellation,
) -> DynResult<()> {
    let plan = match (result, signal) {
        (Ok(mut plan), Ok(())) if !cancel.is_cancelled() && Instant::now() < deadline => {
            report["status"] = json!("PASS");
            plan["request_sha256"] = report["request_sha256"].clone();
            plan["required_receipt"] = json!({"path":"report.json","status":"PASS","request_sha256":report["request_sha256"],"consumer_rule":"require matching successful local receipt and all entry byte pins; plan alone is not publication authority"});
            Some(plan)
        }
        (result, signal) => {
            report["error"] = json!(format!(
                "{}{}",
                result
                    .err()
                    .map_or("late/cancelled compose".into(), |e| e.to_string()),
                signal.err().map_or(String::new(), |e| format!("; {e}"))
            ));
            None
        }
    };
    if let Some(plan) = plan {
        admission::publish(&root.join("publication-plan.json"), &plan)?;
        if let Err(error) = admission::publish(&root.join("report.json"), report) {
            // This freshly owned leaf must not survive failed receipt publication.
            std::fs::remove_file(root.join("publication-plan.json"))?;
            return Err(error);
        }
        Ok(())
    } else {
        admission::publish(&root.join("report.json"), report)?;
        Err("compose failed; owned partial evidence retained, publication forbidden".into())
    }
}
#[cfg(test)]
mod terminal_tests {
    use super::*;
    #[test]
    fn hf_compose_terminal_signal_cancel_deadline_and_receipt_failure_cannot_publish_eligible_plan()
    {
        for mode in ["signal", "cancel", "deadline", "receipt", "success"] {
            let root = tempfile::tempdir().unwrap();
            let cancel = crate::process::Cancellation::default();
            if mode == "cancel" {
                cancel.cancel();
            }
            if mode == "receipt" {
                std::fs::write(root.path().join("report.json"), b"existing").unwrap();
            }
            let mut report = json!({"status":"FAILED","request_sha256":"a".repeat(64)});
            let signal = if mode == "signal" {
                Err("injected finish failure".into())
            } else {
                Ok(())
            };
            let deadline =
                Instant::now() + Duration::from_secs(if mode == "deadline" { 0 } else { 60 });
            let result = finalize(
                root.path(),
                &mut report,
                Ok(json!({"status":"PLAN_ONLY_NOT_PUBLISHED"})),
                signal,
                deadline,
                &cancel,
            );
            assert_eq!(result.is_ok(), mode == "success");
            let plan = root.path().join("publication-plan.json");
            assert_eq!(plan.exists(), mode == "success");
            if mode == "success" {
                let plan: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(plan).unwrap()).unwrap();
                assert_eq!(plan["required_receipt"]["status"], "PASS");
                assert_eq!(plan["request_sha256"], report["request_sha256"]);
                let receipt: serde_json::Value = serde_json::from_slice(
                    &std::fs::read(root.path().join("report.json")).unwrap(),
                )
                .unwrap();
                assert_eq!(receipt["status"], "PASS");
            }
            root.close().unwrap();
        }
    }
}
