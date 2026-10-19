//! Sweep ownership and partial-result publication, separate from child lifecycle.
use super::{Engine, Prepared, inputs, lifecycle, results, write};
use crate::command::DynResult;
use serde_json::{Value, json};

pub(super) fn run(input: &Prepared, engine: Engine) -> DynResult<()> {
    super::hardware::capture(input)?;
    let mut cells = Vec::new();
    let mut points = Vec::new();
    let mut notes = Vec::new();
    let engine = engine.label();
    inputs::recheck(input)?;
    for &concurrency in input.profile.concurrencies() {
        let mut cell: Prepared = serde_json::from_value(serde_json::to_value(input)?)?;
        cell.output = input.output.join(format!("{engine}-c{concurrency}"));
        std::fs::create_dir(&cell.output)?;
        let outcome = lifecycle::execute(&cell, engine, concurrency)
            .and_then(|()| results::read(&cell, engine, concurrency));
        match outcome {
            Ok(summary) if complete(&summary) => {
                points.push(results::point(&cell, concurrency)?);
                notes.push(format!("{engine} c{concurrency}: complete"));
                cells.push(summary);
            }
            Ok(mut summary) => {
                summary["status"] = json!("incomplete");
                notes.push(format!(
                    "{engine} c{concurrency}: incomplete; excluded from serving points"
                ));
                cells.push(summary);
            }
            Err(error) => {
                let mut status = failure_status(&cell.output);
                let context_support = super::context_support::evidence(&cell.output);
                if status == "failed" && context_support.is_some() {
                    status = "unsupported_context";
                }
                notes.push(format!("{engine} c{concurrency}: {status}; see cell logs"));
                cells.push(json!({"engine":engine,"concurrency":concurrency,
                    "status":status,"error":error.to_string(),
                    "context_support":context_support,
                    "memory":std::fs::read(cell.output.join("memory-summary.json")).ok()
                        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok()),
                    "memory_scope":"partial_attempt_not_comparable_to_completed_cells",
                    "capacity_policy":super::capacity::policy(engine),
                    "native_capacity":super::capacity::read(&cell.output)}));
            }
        }
        publish(input, engine, &cells, &points, &notes)?;
        // Never start another server if the previous process group was not reclaimed.
        let evidence = cell.output.join("lifecycle.json");
        let lifecycle: Value = serde_json::from_slice(&std::fs::read(evidence)?)?;
        if lifecycle["cleanup_complete"] != true {
            return Err(
                "cell cleanup incomplete; stopping sweep with prior results retained".into(),
            );
        }
    }
    finish_sweep(input.profile.concurrencies().len(), points.len())
}

fn finish_sweep(expected: usize, completed: usize) -> DynResult<()> {
    if completed != expected {
        return Err(format!(
            "AgentX sweep completed {completed}/{expected} cells; partial results retained"
        )
        .into());
    }
    Ok(())
}

fn complete(summary: &Value) -> bool {
    summary["phase_outcome"]["cancelled"] == 0 && summary["phase_outcome"]["errors"] == 0
}

fn failure_status(output: &std::path::Path) -> &'static str {
    let log = std::fs::read_to_string(output.join("server.stderr.log"))
        .unwrap_or_default()
        .to_ascii_lowercase();
    if [
        "out of memory",
        "cuda_error_out_of_memory",
        "cudamalloc failed",
    ]
    .iter()
    .any(|needle| log.contains(needle))
    {
        "out_of_memory"
    } else if std::fs::read_to_string(output.join("lifecycle.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        .is_some_and(|record| {
            matches!(
                record["outcome"].as_str(),
                Some("Deadline" | "ReadinessDeadline")
            )
        })
    {
        "timeout"
    } else {
        // A crash or readiness timeout is not evidence of a memory capacity limit.
        "failed"
    }
}

fn publish(
    input: &Prepared,
    engine: &str,
    cells: &[Value],
    points: &[Value],
    notes: &[String],
) -> DynResult<()> {
    write(
        &input.output.join("results.json"),
        &json!({"profile":input.profile,"model":input.model,"cells":cells}),
    )?;
    let Some(path) = std::env::var_os("JIANYANG_RESULT_PATH") else {
        return Ok(());
    };
    use sha2::Digest;
    let digest = if input.model.files.len() == 1 {
        input.model.files[0].sha256.clone()
    } else {
        hex::encode(sha2::Sha256::digest(serde_json::to_vec(
            &input.model.files,
        )?))
    };
    // A wholly capacity-limited sweep has no fabricated zero-throughput points.
    // Existing quality metrics let JY retain its explicit outcome counters.
    write(
        std::path::Path::new(&path),
        &json!({"points":points,"notes":notes,
        "quality":{"sweep":{"cells_finished":cells.len(),"cells_complete":points.len(),
            "cells_unsupported_context":cells.iter().filter(|cell| cell["status"] == "unsupported_context").count(),
            "cells_incomplete_or_failed":cells.len()-points.len()}},
        "provenance":{"stack_commit":std::env::var("JIANYANG_COMMIT")?,
            "stack_name":engine,"model_package_ref":input.model.registry_ref(),
            "model_package_sha256":digest}}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn failed_and_incomplete_sweeps_fail_the_job() {
        assert!(finish_sweep(3, 3).is_ok());
        for completed in 0..3 {
            let error = finish_sweep(3, completed).unwrap_err().to_string();
            assert!(error.contains(&format!("{completed}/3")));
            assert!(error.contains("partial results retained"));
        }
    }

    #[test]
    fn cancelled_requests_are_never_serving_points() {
        assert!(complete(
            &json!({"phase_outcome":{"cancelled":0,"errors":0}})
        ));
        for value in [
            json!({}),
            json!({"phase_outcome":{"cancelled":1,"errors":0}}),
            json!({"phase_outcome":{"cancelled":0,"errors":1}}),
        ] {
            assert!(!complete(&value));
        }
    }
    #[test]
    fn both_profiles_cover_every_requested_capacity() {
        for profile in [
            super::super::Profile::Smoke,
            super::super::Profile::Benchmark,
        ] {
            assert_eq!(profile.concurrencies(), &[1, 2, 4]);
        }
    }
    #[test]
    fn memory_failure_requires_explicit_evidence() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(failure_status(dir.path()), "failed");
        std::fs::write(
            dir.path().join("lifecycle.json"),
            r#"{"outcome":"Deadline"}"#,
        )
        .unwrap();
        assert_eq!(failure_status(dir.path()), "timeout");
        std::fs::write(
            dir.path().join("server.stderr.log"),
            "CUDA error: out of memory",
        )
        .unwrap();
        assert_eq!(failure_status(dir.path()), "out_of_memory");
    }
}
