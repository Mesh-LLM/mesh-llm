//! Readouts share normal batching and drain admitted work before cleanup.

use super::*;
use crate::frontend::iteration_scheduler::readout::ReadoutIteration;
use skippy_inference_api::CancellationToken;

/// Planning with a retained vocabulary reader does not acquire the runtime lock.
#[test]
#[ignore = "requires SKIPPY_DECISION_MODEL: a real causal GGUF"]
fn causal_decision_reader_plans_while_inference_context_is_locked() -> Result<()> {
    use skippy_runtime::decision::causal::DecisionPlan;
    use std::sync::mpsc;
    use std::time::Duration;

    let backend = backend()?;
    let reader = backend
        .iteration_scheduler
        .execute_runtime("decision-plan", |runtime| Ok(runtime.model.reader()))?;
    let request = serde_json::from_value(request())?;
    let decision = crate::frontend::system_one::decision_request(&request)?;
    let (send, receive) = mpsc::channel();
    let runtime = backend.runtime.lock().unwrap();
    let planning = std::thread::spawn(move || {
        let result = DecisionPlan::for_reader(&reader, &decision);
        let _ = send.send(result);
    });
    let completed = receive.recv_timeout(Duration::from_secs(10));
    // Release before joining even on failure, so a lock regression cannot hang.
    drop(runtime);
    planning.join().expect("decision planner panicked");
    let plan = completed.context("tokenization waited for the runtime lock")??;
    assert_eq!(plan.questions.len(), 2);
    assert_eq!(
        backend
            .iteration_scheduler
            .execute_runtime("decision-plan-probe", |runtime| {
                Ok(runtime.session_stats().active_sessions)
            })?,
        0
    );
    Ok(())
}

/// Scheduler waves batch readouts and drain admitted work before reporting cancellation.
#[test]
#[ignore = "requires SKIPPY_DECISION_MODEL: a real causal GGUF"]
fn causal_decisions_scheduler_batches_and_drains_cancelled_waves() -> Result<()> {
    let backend = backend()?;
    let (tokens, selected) = {
        let runtime = backend.runtime.lock().unwrap();
        (
            runtime.model.tokenize("Choose B or S. Answer:", true)?,
            runtime.model.tokenize("B", false)?,
        )
    };
    let scheduler = &backend.iteration_scheduler;
    let ids = [
        "decision-wave-a",
        "decision-wave-b",
        "decision-wave-c",
        "decision-wave-d",
    ];
    let make = |session| ReadoutIteration {
        session_id: session,
        position: 0,
        tokens: &tokens,
        selected_tokens: &selected,
        input: None,
        cancellation: None,
    };
    let outputs = scheduler.execute_readout_wave(ids.iter().map(|id| make(*id)).collect())?;
    assert_eq!(outputs.len(), ids.len());
    assert!(
        outputs.iter().any(|output| output.batch_size > 1),
        "independent readouts must use the existing native batch scheduler"
    );
    for output in outputs {
        assert_eq!(output.predicted, -1);
        assert_eq!(output.log_probabilities.len(), selected.len());
    }
    for id in ids {
        backend.runtime.lock().unwrap().drop_session_timed(id)?;
    }
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let mut cancelled = make("cancelled-readout");
    cancelled.cancellation = Some(cancellation);
    let error = scheduler
        .execute_readout_wave(vec![cancelled, make("admitted-readout")])
        .err()
        .context("cancelled wave must fail")?;
    assert!(error.to_string().contains("cancelled"));
    let mut runtime = backend.runtime.lock().unwrap();
    assert_eq!(runtime.session_token_count("cancelled-readout"), None);
    assert_eq!(
        runtime.session_token_count("admitted-readout"),
        Some(tokens.len() as u64),
        "successful queued work must finish before the failed wave returns"
    );
    runtime.drop_session_timed("admitted-readout")?;
    assert_eq!(runtime.session_stats().active_sessions, 0);
    Ok(())
}
