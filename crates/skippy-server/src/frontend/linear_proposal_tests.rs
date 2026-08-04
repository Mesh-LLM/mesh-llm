
use super::*;

#[derive(Debug, PartialEq, Eq)]
struct RecordedQuery {
    request_id: u64,
    session_id: u64,
    prompt_token_count: usize,
    committed_token_count: usize,
    decode_step: usize,
    max_proposal_tokens: usize,
}

fn query_params(
    request_id: u64,
    session_id: u64,
    prompt_token_count: usize,
    decode_step: usize,
    committed_token_count: usize,
    remaining_new_tokens: usize,
    runtime_max_proposal_tokens: usize,
) -> LinearProposalQueryParams {
    LinearProposalQueryParams {
        request_id,
        session_id,
        prompt_token_count,
        decode_step,
        committed_token_count,
        remaining_new_tokens,
        runtime_max_proposal_tokens,
    }
}

#[derive(Default)]
struct FakeIngress {
    proposal: Mutex<Option<LinearProposal>>,
    delay: Mutex<Duration>,
    discard_fails: Mutex<bool>,
    report_fails: Mutex<bool>,
    queries: Mutex<Vec<RecordedQuery>>,
    receipts: Mutex<Vec<LinearProposalReceipt>>,
    discards: Mutex<Vec<(OpaqueProposalDecisionId, LinearProposalDiscardReason)>>,
}

impl LinearProposalIngress for FakeIngress {
    fn propose(&self, query: LinearProposalQuery) -> Result<Option<LinearProposal>> {
        self.queries.lock().unwrap().push(RecordedQuery {
            request_id: query.request_id,
            session_id: query.session_id,
            prompt_token_count: query.prompt_token_count,
            committed_token_count: query.committed_token_count,
            decode_step: query.decode_step,
            max_proposal_tokens: query.max_proposal_tokens,
        });
        thread::sleep(*self.delay.lock().unwrap());
        Ok(self.proposal.lock().unwrap().take())
    }

    fn report(&self, receipt: &LinearProposalReceipt) -> Result<()> {
        self.receipts.lock().unwrap().push(receipt.clone());
        if *self.report_fails.lock().unwrap() {
            bail!("synthetic report failure");
        }
        Ok(())
    }

    fn discard(
        &self,
        decision_id: &OpaqueProposalDecisionId,
        reason: LinearProposalDiscardReason,
    ) -> Result<()> {
        self.discards
            .lock()
            .unwrap()
            .push((decision_id.clone(), reason));
        if *self.discard_fails.lock().unwrap() {
            bail!("synthetic terminal discard failure");
        }
        Ok(())
    }
}

fn decision(proposal: &[i32], predictions: &[i32]) -> NativeMtpVerifyWindowDecision {
    classify_native_mtp_verify_window(proposal, predictions, 0, 64, |_| Ok(false)).unwrap()
}

#[test]
fn opaque_decision_ids_are_nonempty_and_bounded() {
    assert!(OpaqueProposalDecisionId::new(Vec::new()).is_err());
    assert!(OpaqueProposalDecisionId::new(vec![1; 64]).is_ok());
    assert!(OpaqueProposalDecisionId::new(vec![1; 65]).is_err());
}

#[test]
fn ingress_config_requires_positive_bounds() {
    let source = Arc::new(FakeIngress::default());
    assert!(LinearProposalIngressConfig::new(source.clone(), Duration::ZERO, 8).is_err());
    assert!(LinearProposalIngressConfig::new(source.clone(), Duration::from_millis(1), 0).is_err());
    assert!(
        LinearProposalIngressConfig::new(
            source.clone(),
            Duration::from_millis(1),
            MAX_LINEAR_PROPOSAL_TOKENS + 1,
        )
        .is_err()
    );
    assert!(LinearProposalIngressConfig::new(source, Duration::from_millis(1), 8).is_ok());
}

#[test]
fn native_classifier_is_the_only_acceptance_authority() {
    let full = decision(&[11, 12, 13], &[11, 12, 13, 14]);
    assert_eq!(full.accepted_proposal_tokens, 3);
    assert_eq!(full.commit_count, 4);
    assert!(!full.rejected);

    for accepted in [0, 1, 2] {
        let proposal = [11, 12, 13];
        let mut predictions = [11, 12, 13, 14];
        predictions[accepted] = 99;
        let mismatch = decision(&proposal, &predictions);
        assert_eq!(mismatch.accepted_proposal_tokens, accepted);
        assert_eq!(mismatch.commit_count, accepted + 1);
        assert!(mismatch.rejected);
    }
}

#[test]
fn disposition_distinguishes_full_mismatch_and_early_stop() {
    let full = decision(&[11, 12], &[11, 12, 13]);
    assert_eq!(
        linear_proposal_disposition(full, 2, 3, false),
        LinearProposalDisposition::FullAccept
    );

    let mismatch = decision(&[11, 12], &[11, 99, 13]);
    assert_eq!(
        linear_proposal_disposition(mismatch, 2, 2, false),
        LinearProposalDisposition::FirstMismatch
    );

    assert_eq!(
        linear_proposal_disposition(full, 2, 1, true),
        LinearProposalDisposition::Stopped
    );
    assert_eq!(
        linear_proposal_disposition(full, 2, 1, false),
        LinearProposalDisposition::Stopped
    );
}

#[test]
fn callback_error_is_returned_only_after_repair_runs() {
    let repair_ran = Cell::new(false);
    let result = finish_linear_proposal_after_repair(
        Some(OpenAiError::backend("synthetic callback failure")),
        || {
            repair_ran.set(true);
            Ok(())
        },
    );

    assert!(repair_ran.get());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("synthetic callback failure")
    );
}

#[test]
fn execution_error_discards_exactly_once_without_masking_primary_error() {
    let source = Arc::new(FakeIngress::default());
    let config =
        LinearProposalIngressConfig::new(source.clone(), Duration::from_secs(1), 4).unwrap();
    let id = OpaqueProposalDecisionId::new(vec![91]).unwrap();

    let result = execute_linear_proposal_with_terminal_discard(&config, &id, || {
        Err::<(), _>(OpenAiError::backend("synthetic execution failure"))
    });

    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("synthetic execution failure")
    );
    assert_eq!(
        source.discards.lock().unwrap().as_slice(),
        &[(id.clone(), LinearProposalDiscardReason::ExecutionFailed)]
    );

    *source.discard_fails.lock().unwrap() = true;
    let result = execute_linear_proposal_with_terminal_discard(&config, &id, || {
        Err::<(), _>(OpenAiError::backend("primary error survives"))
    });
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("primary error survives")
    );
    assert_eq!(
        source.discards.lock().unwrap().as_slice(),
        &[
            (id.clone(), LinearProposalDiscardReason::ExecutionFailed),
            (id, LinearProposalDiscardReason::ExecutionFailed),
        ]
    );
}

#[test]
fn report_failure_is_observed_without_becoming_an_execution_error() {
    let source = Arc::new(FakeIngress::default());
    *source.report_fails.lock().unwrap() = true;
    let config =
        LinearProposalIngressConfig::new(source.clone(), Duration::from_secs(1), 4).unwrap();
    let receipt = LinearProposalReceipt {
        decision_id: OpaqueProposalDecisionId::new(vec![90]).unwrap(),
        disposition: LinearProposalDisposition::FullAccept,
        proposal_token_count: 1,
        verification_rows: 2,
        accepted_proposal_tokens: 1,
        committed_tokens: vec![11, 12].into_boxed_slice(),
        verification_row_predictions: vec![11, 12].into_boxed_slice(),
        canonical_prediction_count: 2,
        correction_or_boundary_token: Some(12),
        base_position: 10,
        position_after_verification: 12,
        canonical_position: 12,
        trimmed_rows: 0,
        proposal_elapsed_us: 1,
        verification_elapsed_us: 2,
        repair_elapsed_us: 0,
        total_elapsed_us: 3,
        runtime_lock_wait_us: 0,
        runtime_lock_hold_us: 2,
        runtime_lock_acquires: 1,
    };

    let error = report_linear_proposal_receipt(&config, &receipt)
        .expect("report failure should be available for logging");

    assert!(error.to_string().contains("synthetic report failure"));
    assert_eq!(source.receipts.lock().unwrap().as_slice(), &[receipt]);
    assert!(source.discards.lock().unwrap().is_empty());
}

#[test]
fn greedy_admission_rejects_stochastic_sampling_but_accepts_valid_grammar_metadata() {
    let disabled = SamplingConfig::default();
    let temperature_zero = SamplingConfig {
        enabled: true,
        temperature: 0.0,
        top_p: 0.95,
        top_k: 40,
        min_p: 0.05,
        ..SamplingConfig::default()
    };
    let stochastic = SamplingConfig {
        enabled: true,
        temperature: 0.8,
        ..SamplingConfig::default()
    };
    let biased_greedy = SamplingConfig {
        enabled: true,
        temperature: 0.0,
        logit_bias: vec![skippy_runtime::LogitBias {
            token_id: 7,
            bias: 1.0,
        }],
        ..SamplingConfig::default()
    };

    assert!(greedy_linear_proposal_admitted(&disabled, None));
    assert!(greedy_linear_proposal_admitted(&disabled, Some("{}")));
    assert!(greedy_linear_proposal_admitted(
        &disabled,
        Some(r#"{"grammar":""}"#)
    ));
    assert!(greedy_linear_proposal_admitted(&temperature_zero, None));
    assert!(!greedy_linear_proposal_admitted(&stochastic, None));
    assert!(!greedy_linear_proposal_admitted(&biased_greedy, None));
    assert!(greedy_linear_proposal_admitted(
        &disabled,
        Some(r#"{"grammar":"root ::= value"}"#)
    ));
    assert!(!greedy_linear_proposal_admitted(&disabled, Some("{")));
}

#[test]
fn query_passes_bounded_committed_position_and_accepts_a_bounded_proposal() {
    let source = Arc::new(FakeIngress::default());
    let id = OpaqueProposalDecisionId::new(vec![1, 2, 3]).unwrap();
    *source.proposal.lock().unwrap() = Some(LinearProposal::new(id.clone(), vec![31, 32, 33]));
    let config =
        LinearProposalIngressConfig::new(source.clone(), Duration::from_secs(1), 4).unwrap();

    let LinearProposalQueryOutcome::Ready(queried) =
        query_linear_proposal(&config, query_params(7, 8, 2, 1, 3, 5, 4)).unwrap()
    else {
        panic!("bounded proposal should be ready");
    };

    assert_eq!(queried.proposal.decision_id, id);
    assert_eq!(queried.proposal.token_ids.as_ref(), &[31, 32, 33]);
    assert_eq!(
        source.queries.lock().unwrap().as_slice(),
        &[RecordedQuery {
            request_id: 7,
            session_id: 8,
            prompt_token_count: 2,
            decode_step: 1,
            committed_token_count: 3,
            max_proposal_tokens: 4,
        }]
    );
    assert!(source.discards.lock().unwrap().is_empty());
}

#[test]
fn query_rejects_an_inconsistent_prompt_decode_boundary_before_ingress() {
    let source = Arc::new(FakeIngress::default());
    let config =
        LinearProposalIngressConfig::new(source.clone(), Duration::from_secs(1), 4).unwrap();

    for (prompt_token_count, decode_step) in [(0, 3), (4, 0), (2, 0), (1, 3)] {
        assert!(
            query_linear_proposal(
                &config,
                query_params(7, 8, prompt_token_count, decode_step, 3, 5, 4),
            )
            .is_err()
        );
    }
    assert!(source.queries.lock().unwrap().is_empty());
}

#[test]
fn query_discards_invalid_and_late_proposals_without_verification() {
    let invalid_source = Arc::new(FakeIngress::default());
    let invalid_id = OpaqueProposalDecisionId::new(vec![4]).unwrap();
    *invalid_source.proposal.lock().unwrap() =
        Some(LinearProposal::new(invalid_id.clone(), Vec::new()));
    let invalid_config =
        LinearProposalIngressConfig::new(invalid_source.clone(), Duration::from_secs(1), 4)
            .unwrap();
    assert!(matches!(
        query_linear_proposal(&invalid_config, query_params(1, 2, 1, 0, 1, 5, 4)).unwrap(),
        LinearProposalQueryOutcome::NoProposal
    ));
    assert_eq!(
        invalid_source.discards.lock().unwrap().as_slice(),
        &[(invalid_id, LinearProposalDiscardReason::InvalidTokenCount)]
    );

    let late_source = Arc::new(FakeIngress::default());
    let late_id = OpaqueProposalDecisionId::new(vec![5]).unwrap();
    *late_source.proposal.lock().unwrap() = Some(LinearProposal::new(late_id.clone(), vec![41]));
    *late_source.delay.lock().unwrap() = Duration::from_millis(5);
    let late_config =
        LinearProposalIngressConfig::new(late_source.clone(), Duration::from_millis(1), 4).unwrap();
    let LinearProposalQueryOutcome::DeadlineExceeded {
        proposal_elapsed_us,
    } = query_linear_proposal(&late_config, query_params(1, 2, 1, 0, 1, 5, 4)).unwrap()
    else {
        panic!("late proposal should produce deadline telemetry");
    };
    assert!(proposal_elapsed_us >= 1_000);
    assert_eq!(
        late_source.discards.lock().unwrap().as_slice(),
        &[(late_id, LinearProposalDiscardReason::DeadlineExceeded)]
    );

    let invalid_token_source = Arc::new(FakeIngress::default());
    let invalid_token_id = OpaqueProposalDecisionId::new(vec![6]).unwrap();
    *invalid_token_source.proposal.lock().unwrap() =
        Some(LinearProposal::new(invalid_token_id.clone(), vec![41, -1]));
    let invalid_token_config =
        LinearProposalIngressConfig::new(invalid_token_source.clone(), Duration::from_secs(1), 4)
            .unwrap();
    assert!(matches!(
        query_linear_proposal(&invalid_token_config, query_params(1, 2, 1, 0, 1, 5, 4),).unwrap(),
        LinearProposalQueryOutcome::NoProposal
    ));
    assert_eq!(
        invalid_token_source.discards.lock().unwrap().as_slice(),
        &[(
            invalid_token_id,
            LinearProposalDiscardReason::InvalidTokenId
        )]
    );
}

#[test]
fn fake_source_preserves_exact_receipt_and_discard_identity() {
    let source = FakeIngress::default();
    let id = OpaqueProposalDecisionId::new(vec![7, 8, 9]).unwrap();
    let receipt = LinearProposalReceipt {
        decision_id: id.clone(),
        disposition: LinearProposalDisposition::FirstMismatch,
        proposal_token_count: 4,
        verification_rows: 5,
        accepted_proposal_tokens: 1,
        committed_tokens: vec![11, 42].into_boxed_slice(),
        verification_row_predictions: vec![11, 42, 43, 44, 45].into_boxed_slice(),
        canonical_prediction_count: 2,
        correction_or_boundary_token: Some(42),
        base_position: 100,
        position_after_verification: 105,
        canonical_position: 102,
        trimmed_rows: 3,
        proposal_elapsed_us: 5,
        verification_elapsed_us: 10,
        repair_elapsed_us: 2,
        total_elapsed_us: 17,
        runtime_lock_wait_us: 1,
        runtime_lock_hold_us: 9,
        runtime_lock_acquires: 2,
    };
    source.report(&receipt).unwrap();
    source
        .discard(&id, LinearProposalDiscardReason::DeadlineExceeded)
        .unwrap();

    assert_eq!(source.receipts.lock().unwrap().as_slice(), &[receipt]);
    assert_eq!(
        source.discards.lock().unwrap().as_slice(),
        &[(id, LinearProposalDiscardReason::DeadlineExceeded)]
    );
}

#[test]
fn query_caps_proposals_to_the_runtime_batch_window() {
    let source = Arc::new(FakeIngress::default());
    let config =
        LinearProposalIngressConfig::new(source.clone(), Duration::from_secs(1), 32).unwrap();

    assert!(matches!(
        query_linear_proposal(&config, query_params(1, 2, 1, 0, 1, 64, 7)).unwrap(),
        LinearProposalQueryOutcome::NoProposal
    ));
    assert_eq!(source.queries.lock().unwrap()[0].max_proposal_tokens, 7);
}

#[test]
fn receipt_telemetry_excludes_source_ids_tokens_and_error_text() {
    let secret = "private-decision-/Users/nick/prompt.txt";
    let receipt = LinearProposalReceipt {
        decision_id: OpaqueProposalDecisionId::new(secret.as_bytes().to_vec()).unwrap(),
        disposition: LinearProposalDisposition::FullAccept,
        proposal_token_count: 1,
        verification_rows: 2,
        accepted_proposal_tokens: 1,
        committed_tokens: vec![12_345, 67_890].into_boxed_slice(),
        verification_row_predictions: vec![12_345, 67_890].into_boxed_slice(),
        canonical_prediction_count: 2,
        correction_or_boundary_token: Some(67_890),
        base_position: 3,
        position_after_verification: 5,
        canonical_position: 5,
        trimmed_rows: 0,
        proposal_elapsed_us: 1,
        verification_elapsed_us: 2,
        repair_elapsed_us: 3,
        total_elapsed_us: 6,
        runtime_lock_wait_us: 1,
        runtime_lock_hold_us: 2,
        runtime_lock_acquires: 1,
    };
    let mut attrs = BTreeMap::new();

    receipt.insert_telemetry_attrs(&mut attrs);
    let encoded = serde_json::to_string(&attrs).unwrap();

    assert!(!encoded.contains(secret));
    assert!(!encoded.contains("12345"));
    assert!(!encoded.contains("67890"));
    assert!(!attrs.keys().any(|key| key.contains("decision_id")));
    assert!(!attrs.keys().any(|key| key.contains("error")));
}
