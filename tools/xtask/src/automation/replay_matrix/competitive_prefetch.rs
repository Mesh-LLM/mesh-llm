//! Pinned model-package helper isolation; no product dependency enters xtask.
#[path = "competitive_prefetch/trajectory.rs"]
mod trajectory;
use crate::{
    automation::{command_interrupt::Interrupt, hf_certify::execution, receipt_files},
    command::DynResult,
    process::Cancellation,
};
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use std::{
    io::Read as _,
    path::Path,
    time::{Duration, Instant},
};
fn check(deadline: Instant, cancel: &Cancellation) -> DynResult<()> {
    if cancel.is_cancelled() || Instant::now() >= deadline {
        return Err("prefetch deadline/cancellation refused".into());
    }
    Ok(())
}
fn pin(path: &Path, expected: &str, deadline: Instant, cancel: &Cancellation) -> DynResult<()> {
    if !path.is_absolute()
        || path.canonicalize()? != path
        || !std::fs::symlink_metadata(path)?.is_file()
    {
        return Err("canonical regular helper required".into());
    }
    let mut open = std::fs::OpenOptions::new();
    open.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        open.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let mut file = open.open(path)?;
    if !file.metadata()?.is_file() || file.metadata()?.len() > 1024 * 1024 * 1024 {
        return Err("helper identity bound".into());
    }
    let mut hash = Sha256::new();
    let mut chunk = [0; 65536];
    loop {
        check(deadline, cancel)?;
        let n = file.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        hash.update(&chunk[..n]);
    }
    if hex::encode(hash.finalize()) != expected {
        return Err("helper byte identity mismatch".into());
    }
    check(deadline, cancel)
}
pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    use crate::repository::check_args::Grammar;
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation replay-matrix competitive-inputs-prefetch --request ABS --helper ABS --helper-sha256 SHA --evidence-directory FRESH_ABS --timeout-seconds 5..86400 --reader ABS --reader-sha256 SHA",
        values: &[
            "--request",
            "--helper",
            "--helper-sha256",
            "--evidence-directory",
            "--timeout-seconds",
            "--reader",
            "--reader-sha256",
        ],
        flags: &["--help"],
    };
    let parsed = match GRAMMAR.parse(args) {
        Ok(p) => p,
        Err(e) => return e.emit(),
    };
    if parsed.flag("--help") {
        println!("{}", GRAMMAR.usage);
        return Ok(());
    }
    if !parsed.positionals.is_empty() {
        return Err("unexpected prefetch positional arguments".into());
    }
    let helper = Path::new(parsed.last("--helper").ok_or("helper required")?);
    let reader = parsed.last("--reader").map(Path::new);
    let reader_sha = parsed.last("--reader-sha256");
    let expected = parsed
        .last("--helper-sha256")
        .ok_or("helper pin required")?;
    let request = Path::new(parsed.last("--request").ok_or("request required")?);
    let root = Path::new(
        parsed
            .last("--evidence-directory")
            .ok_or("evidence required")?,
    );
    let timeout: u64 = parsed
        .last("--timeout-seconds")
        .ok_or("timeout required")?
        .parse()?;
    if !(5..=86400).contains(&timeout)
        || !root.is_absolute()
        || root.parent().ok_or("evidence parent")?.canonicalize()? != root.parent().unwrap()
    {
        return Err("prefetch budget/canonical evidence parent refused".into());
    }
    match std::fs::symlink_metadata(root) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
        _ => return Err("fresh evidence directory required".into()),
    };
    let bytes = receipt_files::bounded(request, 1024 * 1024)?;
    let mut input: Value = serde_json::from_slice(&bytes)?;
    let requested = input["timeout_seconds"]
        .as_u64()
        .filter(|v| (1..=86400).contains(v))
        .ok_or("helper timeout refused")?;
    let interrupt = Interrupt::install()?;
    let cancel = interrupt.cancellation();
    let deadline = Instant::now() + Duration::from_secs(timeout);
    pin(helper, expected, deadline, &cancel)?;
    let allowance = deadline
        .checked_duration_since(Instant::now())
        .and_then(|v| v.checked_sub(Duration::from_secs(3)))
        .filter(|v| v.as_secs() > 0)
        .ok_or("no helper execution allowance")?;
    input["timeout_seconds"] = json!(requested.min(allowance.as_secs()));
    let effective = serde_json::to_vec(&input)?;
    let hash = hex::encode(Sha256::digest(&effective));
    std::fs::create_dir(root)?;
    let path = root.join("request.json");
    receipt_files::fresh(&path, &effective)?;
    let output = input["output_directory"]
        .as_str()
        .ok_or("helper output path")?;
    let owner = trajectory::Context {
        root,
        deadline,
        cancel: &cancel,
        reader,
        reader_sha256: reader_sha,
    };
    let mut receipt = json!({"schema_version":1,"status":"FAILED","request_transport_sha256":hash,"helper_sha256":expected,"reader_sha256":reader_sha,"process_clean":false,"reader_preflight_clean":false,"trajectory_process_clean":false,"manifest_sha256":null,"helper_final":null,"helper_progress":null,"error":null});
    let observed = (|| -> DynResult<()> {
        owner.models(&input, &mut receipt)?;
        if input["skip_dataset"] != json!(true) {
            owner.preflight(&mut receipt)?;
        }
        let report = helper_process(
            helper,
            vec![
                "--input".into(),
                path.to_str().ok_or("Unicode input")?.into(),
            ],
            root,
            "acquisition",
            deadline,
            &cancel,
        )?;
        receipt["process_clean"] = json!(execution::clean(&report));
        for (file, key) in [
            ("progress.json", "helper_progress"),
            ("acquisition.json", "helper_final"),
        ] {
            let p = Path::new(output).join(file);
            match std::fs::symlink_metadata(&p) {
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
                Err(e) => return Err(e.into()),
                Ok(_) => {
                    let value: Value =
                        serde_json::from_slice(&receipt_files::bounded(&p, 8 * 1024 * 1024)?)?;
                    if value["request_transport_sha256"] != json!(hash) {
                        return Err("helper receipt transport correlation refused".into());
                    }
                    receipt[key] = value;
                }
            }
        }
        pin(helper, expected, deadline, &cancel)?;
        if !execution::clean(&report)
            || receipt["helper_final"]["status"] != json!("ACQUIRED_EXPORTED")
            || !receipt["helper_final"]["error"].is_null()
        {
            return Err("helper acquisition/export failed".into());
        }
        let verification = Verification {
            helper,
            input: &path,
            output: Path::new(output),
            root,
            hash: &hash,
            deadline,
            cancel: &cancel,
        };
        verify_phase(&verification, "before-manifest", &mut receipt)?;
        if input["skip_dataset"] != json!(true) {
            owner.materialize(&input, &mut receipt)?;
        }
        verify_phase(&verification, "after-manifest", &mut receipt)?;
        if receipt_files::bounded(request, 1024 * 1024)? != bytes {
            return Err("caller request changed".into());
        }
        check(deadline, &cancel)
    })();
    let finish: DynResult<()> = interrupt.finish().map_err(|e| e.to_string().into());
    let decision = observed.and(finish).and_then(|()| check(deadline, &cancel));
    receipt["status"] = json!(if decision.is_ok() {
        if ["skip_dataset", "skip_tokenizers", "skip_vllm_configs"]
            .iter()
            .any(|k| input[k] == json!(true))
        {
            "MATERIALIZED_SELECTED"
        } else {
            "MATERIALIZED"
        }
    } else {
        "FAILED"
    });
    receipt["error"] = decision
        .as_ref()
        .err()
        .map_or(Value::Null, |e| json!(e.to_string()));
    receipt_files::fresh(
        &root.join("prefetch.json"),
        &serde_json::to_vec_pretty(&receipt)?,
    )?;
    decision
}

struct Verification<'a> {
    helper: &'a Path,
    input: &'a Path,
    output: &'a Path,
    root: &'a Path,
    hash: &'a str,
    deadline: Instant,
    cancel: &'a Cancellation,
}
fn verify_phase(context: &Verification<'_>, phase: &str, evidence: &mut Value) -> DynResult<()> {
    let args = vec![
        "verify-acquired".into(),
        "--input".into(),
        context.input.to_str().ok_or("input Unicode")?.into(),
        "--phase".into(),
        phase.into(),
    ];
    let report = execution::run_process(
        context.helper,
        args,
        context.root,
        phase,
        context.deadline,
        context.cancel,
    )?;
    evidence[format!("{phase}-process-clean")] = json!(execution::clean(&report));
    if !execution::clean(&report) {
        return Err("acquired artifact custody process refused".into());
    }
    let receipt: Value = serde_json::from_slice(&receipt_files::bounded(
        &context.output.join(format!("custody-{phase}.json")),
        1024 * 1024,
    )?)?;
    if receipt["status"] != json!("CUSTODY_VERIFIED")
        || receipt["request_transport_sha256"] != json!(context.hash)
        || receipt["phase"] != json!(phase)
    {
        return Err("custody receipt correlation refused".into());
    }
    evidence[format!("{phase}-custody")] = receipt;
    check(context.deadline, context.cancel)
}

fn helper_process(
    binary: &Path,
    args: Vec<String>,
    root: &Path,
    label: &str,
    deadline: Instant,
    cancel: &Cancellation,
) -> DynResult<crate::process::RawProcessReport> {
    use crate::process::{
        self, Completion, Limits, OutputFiles, ProcessSpec, RawCaptureOptions, Readiness,
        Value as ProcessValue,
    };
    use std::{collections::BTreeMap, num::NonZeroUsize};
    check(deadline, cancel)?;
    let allowance = deadline
        .checked_duration_since(Instant::now())
        .and_then(|v| v.checked_sub(Duration::from_secs(3)))
        .filter(|v| !v.is_zero())
        .ok_or("helper cleanup reserve exhausted")?;
    let mut environment: BTreeMap<_, _> = ["PATH", "SYSTEMROOT", "WINDIR"]
        .into_iter()
        .filter_map(|k| std::env::var_os(k).map(|v| (k.into(), ProcessValue::Public(v))))
        .collect();
    environment.insert(
        "HF_HUB_DISABLE_IMPLICIT_TOKEN".into(),
        ProcessValue::Public("true".into()),
    );
    Ok(process::supervise_raw_with_files(
        &ProcessSpec {
            executable: binary.into(),
            arguments: args
                .into_iter()
                .map(|a| ProcessValue::Public(a.into()))
                .collect(),
            cwd: root.into(),
            environment,
        },
        &Limits {
            execution: allowance,
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 1048576,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        cancel,
        OutputFiles {
            stdout: Some(root.join(format!("{label}-stdout.log"))),
            stderr: Some(root.join(format!("{label}-stderr.log"))),
        },
        RawCaptureOptions {
            stdout: NonZeroUsize::new(1048576),
            stderr: NonZeroUsize::new(1048576),
        },
    )?)
}
