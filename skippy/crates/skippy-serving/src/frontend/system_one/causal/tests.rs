use super::*;
use skippy_runtime::decision::causal::QuestionPlan;

/// Construct a mixed-method plan for causal response-assembly tests.
fn fixture() -> (SystemOneRequest, DecisionPlan) {
    let request = serde_json::from_value(serde_json::json!({
        "model": "test", "state": "shared evidence", "questions": {
            "first": {"type": "choice", "criteria": {"A": "a", "B": "b"}},
            "second": {"type": "choice", "criteria": {"C": "c", "D": "d"}}
        }
    }))
    .unwrap();
    let questions = vec![
        QuestionPlan {
            key: "first".into(),
            prompt: vec![1, 2, 3],
            candidates: vec![vec![10], vec![20]],
            method: ReadoutMethod::SelectedTokenLogits,
        },
        QuestionPlan {
            key: "second".into(),
            prompt: vec![1, 2, 4],
            candidates: vec![vec![30, 31], vec![40]],
            method: ReadoutMethod::CandidateLogLikelihood,
        },
    ];
    (
        request,
        DecisionPlan {
            questions,
            shared_prefix_tokens: 2,
        },
    )
}

/// Construct one completed readout result with explicit token accounting.
fn result(scores: &[f64]) -> ReadoutResult {
    ReadoutResult {
        scores: scores.to_vec(),
        evaluated_tokens: 2,
        cached_tokens: 1,
    }
}

/// Jobs share only their own question prompt and preserve candidate boundaries.
#[test]
fn independent_jobs_preserve_candidates_and_share_only_their_question_prompt() {
    let (_, plan) = fixture();
    let jobs = decision_jobs(&plan);
    assert_eq!(jobs.len(), 3);
    assert_eq!(jobs[0].candidate, None);
    assert_eq!(jobs[0].method, Some(ReadoutMethod::SelectedTokenLogits));
    assert_eq!(jobs[1].method, Some(ReadoutMethod::CandidateLogLikelihood));
    assert_eq!(jobs[2].method, Some(ReadoutMethod::CandidateLogLikelihood));
    assert_eq!(jobs[0].candidates, vec![vec![10], vec![20]]);
    assert_eq!(jobs[1].candidate, Some(0));
    assert_eq!(jobs[2].candidate, Some(1));
    assert!(Arc::ptr_eq(&jobs[1].prompt, &jobs[2].prompt));
    assert_ne!(jobs[0].prompt, jobs[1].prompt);
}

/// Unordered completions preserve answer order and include prefix-preparation costs.
#[test]
fn out_of_order_completions_keep_answer_order_and_account_for_cache_preparation() {
    let (request, plan) = fixture();
    let jobs = decision_jobs(&plan);
    let results = vec![
        (jobs[2].clone(), result(&[-3.0])),
        (jobs[0].clone(), result(&[-1.0, -2.0])),
        (jobs[1].clone(), result(&[-0.5])),
    ];
    let preparation = DecisionReadoutUsage {
        evaluated_tokens: 4,
        cached_tokens: 2,
        cache_preparation_tokens: 4,
    };
    let response = decision_response(request, &plan, results, preparation).unwrap();
    assert_eq!(response.usage.input_tokens, 15);
    assert_eq!(response.usage.cached_tokens, 5);
    assert_eq!(response.usage.output_tokens, 0);
    assert_eq!(
        response.readouts["second"].candidate_log_probabilities,
        vec![-0.5, -3.0]
    );
    assert_eq!(
        response.readouts["second"].candidate_token_counts,
        vec![2, 1]
    );
    assert_eq!(
        response.readouts["second"].method,
        "candidate_log_likelihood"
    );
    assert_eq!(
        response.readout_usage.unwrap(),
        DecisionReadoutUsage {
            evaluated_tokens: 10,
            cached_tokens: 5,
            cache_preparation_tokens: 4
        }
    );
    let serde_json::Value::Object(answers) = serde_json::to_value(response.answers).unwrap() else {
        panic!("answers must be an object");
    };
    assert_eq!(answers["first"]["choice"], "A");
    assert_eq!(answers["second"]["choice"], "C");
}

/// Method validation prevents candidate cardinality from silently changing scoring.
#[test]
fn scoring_methods_validate_their_own_job_shapes() {
    let (_, plan) = fixture();
    for job in decision_jobs(&plan) {
        job.validate().unwrap();
    }
    let mut job = warmup_job(vec![1, 2]);
    job.validate().unwrap();
    job.candidates = vec![vec![3]];
    assert!(
        job.validate().is_err(),
        "preparation cannot score candidates"
    );
    job.method = Some(ReadoutMethod::SelectedTokenLogits);
    job.validate().unwrap(); // Even one selected token does not imply teacher forcing.
    job.candidates = vec![vec![3, 4]];
    assert!(
        job.validate().is_err(),
        "selected-token reads cannot truncate labels"
    );
    job.method = Some(ReadoutMethod::CandidateLogLikelihood);
    job.validate().unwrap();
    job.candidates.push(vec![5]);
    assert!(
        job.validate().is_err(),
        "teacher forcing requires one independent branch"
    );
    job.candidates = vec![vec![]];
    assert!(
        job.validate().is_err(),
        "empty labels must not reach indexing"
    );
}

/// Malformed or duplicate backend results fail response assembly.
#[test]
fn malformed_or_duplicate_backend_results_fail_closed() {
    for defect in 0..6 {
        let (request, plan) = fixture();
        let jobs = decision_jobs(&plan);
        let mut results = vec![
            (jobs[0].clone(), result(&[-1.0, -2.0])),
            (jobs[1].clone(), result(&[-1.0])),
            (jobs[2].clone(), result(&[-2.0])),
        ];
        match defect {
            0 => {
                results.pop();
            }
            1 => results[1].1.scores.clear(),
            2 => results[2].0.candidate = Some(0),
            3 => results[1].0.question = usize::MAX,
            4 => results[1].1.scores[0] = f64::NAN,
            _ => results[1].0.candidate = Some(usize::MAX),
        }
        assert!(
            decision_response(request, &plan, results, DecisionReadoutUsage::default()).is_err(),
            "defect {defect}"
        );
    }
}

/// Dropping frontend ownership cancels its admitted readout work.
#[test]
fn abandoned_frontend_cancels_admitted_readouts() {
    let cancellation = CancellationToken::new();
    {
        let _guard = CancelOnDrop(cancellation.clone());
    }
    assert!(cancellation.is_cancelled());
}

/// Context admission includes teacher-forced inputs but no generated answer token.
#[test]
fn context_admission_includes_teacher_forced_inputs_but_no_generated_token() {
    let (_, mut plan) = fixture();
    assert!(validate_context(&plan, 4).is_ok());
    assert!(validate_context(&plan, 3).is_err());
    plan.questions.pop();
    // The final prompt row supplies all single-token scores without decoding.
    assert!(validate_context(&plan, 3).is_ok());
    assert!(validate_context(&plan, 2).is_err());
}
