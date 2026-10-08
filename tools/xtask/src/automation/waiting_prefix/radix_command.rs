//! Alternating old/new cold/warm radix matrix over explicitly admitted local static arms.
use super::{
    adaptive_cell, adaptive_identity as io, options, radix_cell, radix_gate, radix_identity::Arm,
    radix_summary, radix_workload::Shape,
};
use crate::{
    command::DynResult,
    process::{self, Cancellation, Value as Arg},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::Path,
    time::{Duration, Instant},
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Case {
    pub key: String,
    pub family: String,
    pub old: Arm,
    pub new: Arm,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub schema_version: u64,
    pub cases: Vec<Case>,
    pub shape: Shape,
    pub request_timeout_secs: f64,
    pub batch_timeout_secs: u64,
    pub cell_timeout_secs: u64,
    pub timeout_secs: u64,
    pub require_gates: bool,
}
impl Input {
    pub fn validate(&self) -> DynResult<()> {
        self.shape.validate()?;
        if self.schema_version != 1
            || self.cases.is_empty()
            || self.cases.len() > 16
            || !(2..=10800).contains(&self.batch_timeout_secs)
            || self.cell_timeout_secs <= self.batch_timeout_secs
            || self.cell_timeout_secs > 86400
            || self.timeout_secs <= self.cell_timeout_secs + 6
            || self.timeout_secs > 86400
            || !self.request_timeout_secs.is_finite()
            || self.request_timeout_secs <= 0.0
            || self.request_timeout_secs > self.batch_timeout_secs as f64
        {
            return Err("radix matrix identity/workload/deadline invalid".into());
        }
        let mut keys = std::collections::BTreeSet::new();
        for case in &self.cases {
            case.old.validate()?;
            case.new.validate()?;
            if case.key.is_empty()
                || case.key.len() > 64
                || !case
                    .key
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
                || !keys.insert(&case.key)
                || case.family.trim().is_empty()
                || case.family.len() > 256
                || case.old.model_id != case.new.model_id
                || case.old.model_sha256 != case.new.model_sha256
                || case.old.ctx_size != case.new.ctx_size
                || case.old.layer_end != case.new.layer_end
                || case.old.payload != case.new.payload
                || case.old.native_build_sha256 != case.new.native_build_sha256
            {
                return Err("radix case model/profile OLD/NEW pairing differs".into());
            }
        }
        super::radix_workload::batches(&self.shape, true)?;
        Ok(())
    }
}
fn admission(arm: &Arm, directory: &Path, until: Instant, cancel: &Cancellation) -> DynResult<Arm> {
    std::fs::create_dir(directory)?;
    let bytes = serde_json::to_vec(arm)?;
    io::fresh(&directory.join("input.json"), &bytes)?;
    let report = process::supervise(
        &process::ProcessSpec {
            executable: std::env::current_exe()?,
            arguments: vec![
                Arg::Public("automation".into()),
                Arg::Public("waiting-prefix".into()),
                Arg::Public("radix-identity-worker".into()),
                Arg::Public("--input".into()),
                Arg::Public(directory.join("input.json").into_os_string()),
                Arg::Public("--output".into()),
                Arg::Public(directory.join("receipt.json").into_os_string()),
            ],
            cwd: directory.into(),
            environment: BTreeMap::new(),
        },
        &process::Limits {
            execution: adaptive_cell::remaining(until, Duration::from_millis(750))?,
            graceful_shutdown: Duration::from_millis(250),
            forced_shutdown: Duration::from_millis(250),
            retained_bytes_per_stream: 65536,
            readiness: process::Readiness::None,
            completion: process::Completion::Exit,
        },
        cancel,
        process::OutputFiles {
            stdout: Some(directory.join("stdout.log")),
            stderr: Some(directory.join("stderr.log")),
        },
    )?;
    if !report.success()
        || !adaptive_cell::clean(&report)
        || !report.stdout.line_capture_complete
        || !report.stderr.line_capture_complete
    {
        return Err("radix native identity worker failed/cleanup incomplete".into());
    }
    let receipt: Value =
        serde_json::from_slice(&io::bounded(&directory.join("receipt.json"), 128 * 1024)?)?;
    if receipt["schema_version"] != 1 || receipt["request_sha256"] != io::digest(&bytes) {
        return Err("radix identity receipt schema/correlation differs".into());
    }
    let admitted: Arm = serde_json::from_value(receipt["admitted"].clone())?;
    admitted.validate()?;
    Ok(admitted)
}
fn table(rows: &[Value]) -> String {
    let mut result="| version | cache | scenario | N | requests | hits | matched tokens | suffix prefill | TTFT p50 ms | cache lift ms |\n|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n".to_owned();
    for row in rows {
        let shown = |key: &str| {
            row[key]
                .as_f64()
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "—".into())
        };
        result.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            row["version"].as_str().unwrap_or("unknown"),
            row["cache"].as_str().unwrap_or("unknown"),
            row["scenario"].as_str().unwrap_or("unknown"),
            row["concurrency"],
            row["requests"],
            row["cache_hits"],
            shown("matched_prefix_tokens_median"),
            shown("suffix_prefill_tokens_median"),
            shown("ttft_ms_p50"),
            shown("cache_lift_ttft_ms")
        ));
    }
    result
}
fn chart(rows: &[Value]) -> String {
    let bars = rows
        .iter()
        .filter(|r| r["cache"] == "warm" && r["scenario"] == "divergent")
        .filter_map(|r| r["matched_prefix_tokens_median"].as_f64().map(|v| (r, v)))
        .collect::<Vec<_>>();
    let ceiling = bars.iter().map(|(_, v)| *v).fold(1.0, f64::max);
    let mut text="<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"900\" height=\"420\"><title>Divergent-prefix tokens reused; missing observations omitted</title>".to_owned();
    for (i, (r, v)) in bars.iter().enumerate() {
        let width = 760.0 / bars.len().max(1) as f64;
        let height = 280.0 * v / ceiling;
        let x = 70.0 + i as f64 * width;
        text.push_str(&format!("<rect x=\"{x:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{height:.1}\" fill=\"{}\"/><text x=\"{x:.1}\" y=\"370\">{} N{}</text>",350.0-height,width*0.65,if r["version"]=="old"{"#8a94a6"}else{"#2f80ed"},if r["version"]=="old"{"OLD"}else{"NEW"},r["concurrency"]));
    }
    text.push_str("</svg>\n");
    text
}
fn execute(input: &Input, directory: &Path, until: Instant, cancel: &Cancellation) -> Value {
    let mut cases = vec![];
    let mut error = None;
    for case in &input.cases {
        let case_dir = directory.join(&case.key);
        let result = (|| -> DynResult<Value> {
            std::fs::create_dir(&case_dir)?;
            let mut cells = vec![];
            let mut failure = None;
            'rounds: for round in 1..=input.shape.rounds {
                let order = if round % 2 == 1 {
                    ["old", "new"]
                } else {
                    ["new", "old"]
                };
                for version in order {
                    let declared = if version == "old" {
                        &case.old
                    } else {
                        &case.new
                    };
                    for warm in [false, true] {
                        if cancel.is_cancelled() {
                            failure = Some("radix matrix interrupted".into());
                            break 'rounds;
                        }
                        let stem = format!(
                            "round-{round}-{version}-{}",
                            if warm { "warm" } else { "cold" }
                        );
                        let admitted = match admission(
                            declared,
                            &case_dir.join(format!("{stem}-identity-before")),
                            until,
                            cancel,
                        ) {
                            Ok(arm) => arm,
                            Err(e) => {
                                failure = Some(e.to_string());
                                break 'rounds;
                            }
                        };
                        let mut cell = match radix_cell::execute(
                            &admitted,
                            &input.shape,
                            warm,
                            &case_dir.join(&stem),
                            &radix_cell::Budget {
                                batch_secs: input.batch_timeout_secs,
                                request_secs: input.request_timeout_secs,
                                cell_secs: input.cell_timeout_secs,
                                until,
                                cancel,
                            },
                        ) {
                            Ok(cell) => cell,
                            Err(e) => {
                                failure = Some(e.to_string());
                                break 'rounds;
                            }
                        };
                        cell["round"] = json!(round);
                        cell["version"] = json!(version);
                        let failed = !cell["error"].is_null();
                        cells.push(cell);
                        if failed {
                            failure =
                                Some("radix cell failed; earlier observations retained".into());
                            break 'rounds;
                        }
                        match admission(
                            declared,
                            &case_dir.join(format!("{stem}-identity-after")),
                            until,
                            cancel,
                        ) {
                            Ok(after) => {
                                if serde_json::to_value(&after)? != serde_json::to_value(&admitted)?
                                {
                                    failure = Some(
                                        "radix identity custody changed across serving cell".into(),
                                    );
                                    break 'rounds;
                                }
                            }
                            Err(e) => {
                                failure = Some(e.to_string());
                                break 'rounds;
                            }
                        }
                    }
                }
            }
            let rows = radix_summary::aggregate(&cells);
            let (parity, preservation) = radix_summary::comparisons(&rows);
            let mut gate = radix_gate::evaluate(&case.old.payload, &cells, &rows);
            if failure.is_some() {
                gate["passed"] = json!(false);
                gate["failures"]
                    .as_array_mut()
                    .unwrap()
                    .push(json!("radix matrix/cell incomplete"));
            }
            let result = json!({"case":{"key":case.key,"family":case.family,"model_id":case.old.model_id,"model_sha256":case.old.model_sha256,"layer_end":case.old.layer_end,"payload":case.old.payload},"cells":cells,"aggregate":rows,"output_parity":parity,"cache_output_preservation":preservation,"gate":gate,"error":failure});
            io::fresh(
                &case_dir.join("comparison.json"),
                &serde_json::to_vec_pretty(&result)?,
            )?;
            io::fresh(&case_dir.join("table.md"), table(&rows).as_bytes())?;
            io::fresh(
                &case_dir.join("divergent-prefix.svg"),
                chart(&rows).as_bytes(),
            )?;
            Ok(result)
        })();
        match result {
            Ok(case) => {
                let failed = !case["error"].is_null()
                    || (input.require_gates && case["gate"]["passed"] != true);
                cases.push(case);
                if failed {
                    error = Some("radix correctness/telemetry/lifecycle matrix gate failed");
                    break;
                }
            }
            Err(_) => {
                error = Some("radix case admission/publication failed");
                break;
            }
        }
    }
    json!({"schema_version":1,"kind":"radix-prefix-reuse/comparison","metadata":{"shape":input.shape,"require_gates":input.require_gates,"native_profile":"standalone-static-skippy-server","output_identity":"canonical content/reasoning/tool-calls SHA256; raw generated text is not persisted","telemetry_scope":"sole owned clients with exact asynchronous summary barriers and complete final EOF; no terminal telemetry drop attestation"},"cases":cases,"error":error})
}
pub(super) fn run(args: &[String]) -> DynResult<()> {
    let flags = options(
        args,
        &["--input", "--output-directory"],
        &["--input", "--output-directory"],
    )?;
    let input: Input =
        serde_json::from_slice(&io::bounded(Path::new(flags["--input"]), 1024 * 1024)?)?;
    input.validate()?;
    let directory = std::path::absolute(flags["--output-directory"])?;
    match std::fs::symlink_metadata(&directory) {
        Ok(meta) if meta.is_dir() => {
            if std::fs::read_dir(&directory)?.next().is_some() {
                return Err("radix output directory must be empty".into());
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        _ => return Err("radix output must be fresh/empty regular directory".into()),
    };
    std::fs::create_dir_all(&directory)?;
    let directory = directory.canonicalize()?;
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let cancellation = interrupt.cancellation();
    let deadline = Instant::now() + Duration::from_secs(input.timeout_secs);
    let mut value = execute(&input, &directory, deadline, &cancellation);
    let finished = interrupt.finish().map_err(Into::into);
    let terminal = super::radix_terminal::finalize(&mut value, finished, &cancellation, deadline);
    io::fresh(
        &directory.join("comparison.json"),
        &serde_json::to_vec_pretty(&value)?,
    )?;
    terminal
}
