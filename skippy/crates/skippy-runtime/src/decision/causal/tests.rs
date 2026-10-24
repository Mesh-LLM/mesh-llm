use super::*;
use crate::decision::DecisionValue;

/// Construct a decision request with caller-ordered candidate labels.
fn request(labels: &[&str]) -> DecisionRequest {
    DecisionRequest {
        state: DecisionValue::String("shared evidence".into()),
        questions: vec![DecisionQuestion {
            key: "first".into(),
            instructions: Some(DecisionValue::String("choose".into())),
            kind: DecisionQuestionKind::Choice {
                options: labels
                    .iter()
                    .map(|label| {
                        (
                            label.to_string(),
                            DecisionValue::String(format!("description {label}")),
                        )
                    })
                    .collect(),
            },
        }],
        seed: [0; 32],
    }
}

/// Use character IDs as a controlled tokenizer for boundary-validation tests.
fn characters(text: &str) -> Result<Vec<i32>> {
    Ok(text.chars().map(|value| value as i32).collect())
}

/// Each question selects its own scoring method without consuming another answer.
#[test]
fn mixed_questions_select_their_own_methods_without_answer_leakage() {
    let mut request = request(&["A", "B"]);
    let mut second = request.questions[0].clone();
    second.key = "second".into();
    second.kind = DecisionQuestionKind::Choice {
        options: vec![
            ("long answer".into(), DecisionValue::Null),
            ("short".into(), DecisionValue::Null),
        ],
    };
    request.questions.push(second);
    let plan = DecisionPlan::tokenize(&request, characters).unwrap();
    assert_eq!(plan.questions[0].method, ReadoutMethod::SelectedTokenLogits);
    assert_eq!(
        plan.questions[1].method,
        ReadoutMethod::CandidateLogLikelihood
    );
    assert!(plan.shared_prefix_tokens > "Shared state:".len());
    assert_eq!(
        plan.questions[1].candidates[0],
        characters("long answer").unwrap()
    );
    assert!(
        !plan.questions[1]
            .prompt
            .windows(13)
            .any(|tokens| tokens == characters("description A").unwrap())
    );
}

/// Candidate token sequences retain the caller's order.
#[test]
fn candidate_order_is_request_order() {
    for labels in [["Z", "A", "M"], ["M", "Z", "A"]] {
        let plan = DecisionPlan::tokenize(&request(&labels), characters).unwrap();
        assert_eq!(
            plan.questions[0].candidates,
            labels
                .iter()
                .map(|label| characters(label).unwrap())
                .collect::<Vec<_>>()
        );
    }
}

/// Reject labels that merge across the prompt boundary or alias another label.
#[test]
fn rejects_context_boundary_merges_and_tokenizer_aliases() {
    let mut calls = 0;
    assert!(
        DecisionPlan::tokenize(&request(&["A", "B"]), |text| {
            calls += 1;
            let mut tokens = characters(text)?;
            if calls > 1 {
                tokens[0] = -1;
            }
            Ok(tokens)
        })
        .unwrap_err()
        .to_string()
        .contains("boundary")
    );
    assert!(
        DecisionPlan::tokenize(&request(&["A", "B"]), |text| characters(
            &text.replace('B', "A")
        ))
        .is_err()
    );
}

/// Reject requests whose question or label counts violate the bounded contract.
#[test]
fn rejects_ambiguous_and_unbounded_requests() {
    for labels in [&[""][..], &["A", "A"][..], &["A", "\nB"][..]] {
        assert!(DecisionPlan::tokenize(&request(labels), characters).is_err());
    }
    let mut request = request(&["A", "B"]);
    request.questions = vec![request.questions[0].clone(); MAX_QUESTIONS + 1];
    assert!(DecisionPlan::tokenize(&request, characters).is_err());
}

/// Relative candidate probabilities remain stable under score offsets and permutations.
#[test]
fn scores_are_stable_under_offset_and_permutation() {
    let scores = normalize_candidate_scores(&[-1000.0, -1001.0, -1002.0]).unwrap();
    assert_eq!(
        scores,
        normalize_candidate_scores(&[0.0, -1.0, -2.0]).unwrap()
    );
    let reversed = normalize_candidate_scores(&[-2.0, -1.0, 0.0]).unwrap();
    assert_eq!(scores, reversed.into_iter().rev().collect::<Vec<_>>());
    assert!((scores.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    assert!(normalize_candidate_scores(&[f64::NAN, 0.0]).is_err());
}
