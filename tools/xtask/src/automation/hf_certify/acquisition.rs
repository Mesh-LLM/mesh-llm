//! G1 only: pinned projector acquisition feeding the existing supplied certification owner.
use super::{
    admission::{self, Artifact, Input},
    execution,
};
use crate::{automation::command_interrupt::Interrupt, command::DynResult, process::Cancellation};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    path::Path,
    time::{Duration, Instant},
};
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    pub(super) schema_version: u32,
    pub(super) certification: Input,
    pub(super) projector: Projector,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(super) enum Projector {
    Supplied {
        artifact: Artifact,
    },
    HfHttps {
        url: String,
        expected_sha256: String,
        max_bytes: u64,
    },
}
impl Request {
    pub(super) fn validate(&self) -> DynResult<()> {
        self.certification.validate()?;
        if self.schema_version != 1 {
            return Err("invalid acquisition schema".into());
        }
        match &self.projector {
            Projector::Supplied { artifact } if artifact == &self.certification.projector => Ok(()),
            Projector::Supplied { .. } => {
                Err("supplied projector differs from certification pin".into())
            }
            Projector::HfHttps {
                url,
                expected_sha256,
                max_bytes,
            } => {
                if url.len() > 16384
                    || expected_sha256 != &self.certification.projector.sha256
                    || !(4..=64 * 1024 * 1024 * 1024).contains(max_bytes)
                {
                    return Err("invalid projector acquisition bounds or pin".into());
                }
                crate::model_registry::validate_projector_origin(url)
            }
        }
    }
}
fn check(deadline: Instant, cancel: &Cancellation) -> DynResult<()> {
    if cancel.is_cancelled() {
        return Err("certify acquisition cancelled".into());
    }
    if Instant::now() >= deadline {
        return Err("certify acquisition total deadline".into());
    }
    Ok(())
}
pub(super) fn execute(
    request: &Request,
    root: &Path,
    deadline: Instant,
    cancel: &Cancellation,
    evidence: &mut Value,
) -> DynResult<()> {
    check(deadline, cancel)?;
    let mut input = request.certification.clone();
    match &request.projector {
        Projector::Supplied { artifact } => {
            input.projector = artifact.clone();
            evidence["acquisition"] =
                json!({"kind":"supplied","status":"PIN_DECLARED","sha256":artifact.sha256});
        }
        Projector::HfHttps {
            url,
            expected_sha256,
            max_bytes,
        } => {
            evidence["acquisition"] = json!({"kind":"hf-https","status":"FAILED","expected_sha256":expected_sha256,"max_bytes":max_bytes});
            let output = root.join("projector.gguf");
            let observed = crate::model_registry::acquire_pinned_projector(
                url,
                &output,
                expected_sha256,
                *max_bytes,
                deadline,
                cancel,
            )
            .inspect_err(|_error| {
                evidence["acquisition"]["error"] = json!("pinned projector transfer refused");
            })?;
            input.projector = Artifact {
                path: output,
                sha256: observed.clone(),
            };
            evidence["acquisition"]["status"] = json!("ACQUIRED");
            evidence["acquisition"]["observed_sha256"] = json!(observed);
        }
    }
    check(deadline, cancel)?;
    input.validate()?;
    let certification = root.join("certification");
    std::fs::create_dir(&certification)?;
    // No transport or certification parser is duplicated. Shared identity workers
    // observe all input pins before/after the actual native validation child.
    execution::execute(
        &input,
        &certification,
        deadline,
        cancel,
        &mut evidence["certification"],
    )
    .inspect_err(|_error| {
        evidence["certification"]["error"] = json!("supplied native certification refused");
    })?;
    check(deadline, cancel)?;
    evidence["certification"]["status"] = json!("PASS");
    if evidence["acquisition"]["kind"] == "supplied" {
        evidence["acquisition"]["status"] = json!("PIN_OBSERVED_BEFORE_AND_AFTER");
    }
    Ok(())
}
pub(super) fn run(args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        println!(
            "automation hf-certify acquire --input FILE --output-directory FRESH_DIRECTORY; G1 pinned projector only, no Jobs/build/upload"
        );
        return Ok(());
    }
    let [a, path, b, output] = args else {
        return Err("acquire requires --input/--output-directory".into());
    };
    if a != "--input" || b != "--output-directory" {
        return Err("acquire closed flags".into());
    }
    let request: Request = serde_json::from_slice(&admission::read(Path::new(path), 262144)?)?;
    request.validate()?;
    let requested = std::path::absolute(output)?;
    let parent = requested.parent().ok_or("output parent")?.canonicalize()?;
    let root = parent.join(requested.file_name().ok_or("output leaf")?);
    std::fs::create_dir(&root)?;
    let interrupt = Interrupt::install()?;
    let cancel = interrupt.cancellation();
    let deadline = Instant::now() + Duration::from_secs(request.certification.timeout_secs);
    let mut evidence = json!({"schema_version":1,"status":"FAILED","request_sha256":admission::digest(&serde_json::to_vec(&request)?),"acquisition":null,"certification":{"status":"FAILED","admitted":null,"native_report":null,"source_unchanged":false},"error":null,"scope":"G1 pinned projector acquisition and supplied native certification only"});
    let result = execute(&request, &root, deadline, &cancel, &mut evidence);
    let finish = interrupt.finish();
    let final_budget = check(deadline, &cancel);
    match (result, finish, final_budget) {
        (Ok(()), Ok(()), Ok(())) => evidence["status"] = json!("PASS"),
        (r, f, b) => {
            evidence["certification"]["status"] = json!("FAILED");
            evidence["error"] = json!(format!(
                "{}{}{}",
                r.err().map_or(String::new(), |e| e.to_string()),
                f.err().map_or(String::new(), |e| format!("; {e}")),
                b.err().map_or(String::new(), |e| format!("; {e}"))
            ));
        }
    }
    admission::publish(&root.join("acquisition-report.json"), &evidence)?;
    if evidence["status"] == "PASS" {
        Ok(())
    } else {
        Err("G1 acquisition/certification failed; owned evidence retained".into())
    }
}
