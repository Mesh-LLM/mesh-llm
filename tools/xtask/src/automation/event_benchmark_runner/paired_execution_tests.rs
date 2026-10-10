use super::*;
use serde_json::json;

fn inputs() -> (Vec<plan::Entry>, [plan::Side; 2]) {
    let sides = plan::sides(
        "current".into(),
        None,
        &[plan::Mode::Production, plan::Mode::EventDisabled],
    )
    .unwrap();
    let entries = plan::build(
        &plan::Spec {
            seed: 42,
            pairs_primary: 2,
            pairs_scenario: 1,
            scenarios: vec!["unicode/../日本語".into()],
        },
        &sides,
    )
    .unwrap();
    (entries, sides)
}
fn successful() -> Outcome {
    Outcome {
        launched: true,
        measurement: Some(Measurement {
            completion_tokens: Some(2),
            ttft_ms: Some(10.0),
            elapsed_ms: 100.0,
            decode_tok_s: Some(20.0),
            decode_only_tok_s: Some(2.0 / 0.09),
            malformed: false,
        }),
        ..Outcome::default()
    }
}

#[test]
fn every_pair_executes_back_to_back_with_identical_prompt_and_identity() {
    let (entries, sides) = inputs();
    let mut calls = Vec::new();
    let batch = run(&entries, &sides, |side, entry| {
        calls.push((side.side_id.clone(), entry.clone()));
        Ok(successful())
    })
    .unwrap();
    assert_eq!(calls.len(), entries.len() * 2);
    assert_eq!(batch.executed_order.len(), entries.len());
    for ((calls, entry), order) in calls
        .as_chunks::<2>()
        .0
        .iter()
        .zip(&entries)
        .zip(&batch.executed_order)
    {
        assert_eq!(calls[0].1, calls[1].1);
        assert_eq!(calls[0].0, entry.side_order_first);
        assert_eq!(order.order, [calls[0].0.clone(), calls[1].0.clone()]);
    }
    for rows in batch.trials {
        assert_eq!(rows.len(), entries.len());
        for (row, entry) in rows.iter().zip(&entries) {
            assert_eq!(row.prompt_sha256, entry.prompt_sha256());
            assert_eq!(row.status, "succeeded");
        }
    }
}

#[test]
fn each_side_keeps_its_own_last_trial_health_without_summing_prior_counts() {
    let (entries, sides) = inputs();
    let mut counts = [0_u64; 2];
    let batch = run(&entries, &sides, |side, _| {
        let index = usize::from(side.mode == plan::Mode::EventDisabled);
        counts[index] += 1;
        let mut outcome = successful();
        outcome.health = Observation {
            health: Some(
                serde_json::from_value(
                    json!({"dropped_progress":counts[index],"dropped_diagnostic":index}),
                )
                .unwrap(),
            ),
            ingress_p99_us: Some(10.0 + index as f64),
        };
        Ok(outcome)
    })
    .unwrap();
    for index in [0, 1] {
        let health = batch.final_health[index].health.as_ref().unwrap();
        assert_eq!(health["dropped_progress"], entries.len());
        assert_eq!(health["dropped_diagnostic"], index);
        assert_eq!(
            batch.final_health[index].ingress_p99_us,
            Some(10.0 + index as f64)
        );
    }
}

#[test]
fn absent_last_trial_health_replaces_previous_observations() {
    let (entries, sides) = inputs();
    let batch = run(&entries, &sides, |_, entry| {
        let mut outcome = successful();
        if entry.scenario == plan::PRIMARY {
            outcome.health = Observation {
                health: Some(serde_json::from_value(json!({"dropped_progress":0})).unwrap()),
                ingress_p99_us: Some(1.0),
            };
        }
        Ok(outcome)
    })
    .unwrap();
    assert_eq!(
        batch.final_health,
        [Observation::default(), Observation::default()]
    );
}

#[test]
fn spawn_refusal_preserves_partial_results_without_claiming_a_completed_pair() {
    let (entries, sides) = inputs();
    let mut calls = 0;
    let batch = run(&entries, &sides, |_, _| {
        calls += 1;
        if calls == 2 {
            Ok(Outcome {
                error: Some("spawn refused".into()),
                ..Outcome::default()
            })
        } else {
            Ok(successful())
        }
    })
    .unwrap();
    assert_eq!(calls, 2);
    assert_eq!(batch.trials.iter().map(Vec::len).sum::<usize>(), 2);
    assert!(batch.executed_order.is_empty());
    assert!(batch.interrupted.is_some());
    let row = batch
        .trials
        .iter()
        .flatten()
        .find(|row| row.status == "failed")
        .unwrap();
    assert_eq!(row.completion_tokens, None);
    assert_eq!(row.decode_tok_s, None);
    assert_eq!(row.error.as_deref(), Some("spawn refused"));
}

#[test]
fn launched_measurement_failure_still_records_actual_order_and_null_metrics() {
    let (entries, sides) = inputs();
    let batch = run(&entries, &sides, |_, _| {
        Ok(Outcome {
            launched: true,
            error: Some("request timeout".into()),
            ..Outcome::default()
        })
    })
    .unwrap();
    assert_eq!(batch.executed_order.len(), entries.len());
    assert!(batch.interrupted.is_none());
    for row in batch.trials.iter().flatten() {
        assert_eq!(row.status, "failed");
        assert_eq!(row.elapsed_ms, None);
        assert_eq!(row.ttft_ms, None);
    }
}

#[test]
fn cancellation_preserves_completed_pairs_and_invalid_plans_never_execute() {
    let (mut entries, sides) = inputs();
    let mut calls = 0;
    let batch = run(&entries, &sides, |_, _| {
        calls += 1;
        if calls == 3 {
            Err("interrupted".into())
        } else {
            Ok(successful())
        }
    })
    .unwrap();
    assert_eq!(batch.executed_order.len(), 1);
    assert_eq!(batch.interrupted.as_deref(), Some("interrupted"));
    entries[0].side_order_first = "unknown".into();
    assert!(
        run(&entries, &sides, |_, _| panic!(
            "invalid plan must not execute"
        ))
        .is_err()
    );
    entries[0].side_order_first = sides[0].side_id.clone();
    entries.push(entries[0].clone());
    assert!(
        run(&entries, &sides, |_, _| panic!(
            "duplicate plan must not execute"
        ))
        .is_err()
    );
}
