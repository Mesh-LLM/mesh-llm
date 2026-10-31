use super::super::tests::activation_desc;
use super::*;

/// Serial decode accepts an explicit decode phase with a matching live position.
#[test]
fn serial_decode_accepts_explicit_phase_and_matching_position() {
    let mut session = StageSession {
        raw: ptr::null_mut(),
        token_count: 4,
        terminal_stage: true,
        batched_activation_exports: true,
        dflash: None,
        dflash_split: None,
    };
    let request = IterationBatchRequest {
        session: &mut session,
        token_ids: &[7],
        positions: &[],
        sampling: None,
        input: None,
        sample_last: true,
        phase: IterationBatchPhase::Decode,
    };

    validate_serial_decode_request(&request).unwrap();
}

/// Recurrent downstream decode accepts matching framed-origin metadata.
#[test]
fn serial_decode_accepts_framed_origin_for_recurrent_downstream_stage() {
    for positions in [&[][..], &[0][..]] {
        let mut session = StageSession {
            raw: ptr::null_mut(),
            token_count: 0,
            terminal_stage: true,
            batched_activation_exports: true,
            dflash: None,
            dflash_split: None,
        };
        let frame = ActivationFrame {
            desc: activation_desc(1),
            payload: vec![1],
        };
        let request = IterationBatchRequest {
            session: &mut session,
            token_ids: &[7],
            positions,
            sampling: None,
            input: Some(&frame),
            sample_last: true,
            phase: IterationBatchPhase::Decode,
        };

        validate_serial_decode_request(&request).unwrap();
    }
}

/// Serial decode rejects invalid token counts, phases and positions.
#[test]
fn serial_decode_rejects_invalid_decode_shapes() {
    for (session_tokens, token_ids, positions, sample_last) in [
        (0, &[7][..], &[][..], true),
        (4, &[7, 8][..], &[][..], true),
        (4, &[7][..], &[3][..], true),
        (4, &[7][..], &[][..], false),
    ] {
        let mut session = StageSession {
            raw: ptr::null_mut(),
            token_count: session_tokens,
            terminal_stage: true,
            batched_activation_exports: true,
            dflash: None,
            dflash_split: None,
        };
        let request = IterationBatchRequest {
            session: &mut session,
            token_ids,
            positions,
            sampling: None,
            input: None,
            sample_last,
            phase: IterationBatchPhase::Decode,
        };

        assert!(validate_serial_decode_request(&request).is_err());
    }
}

/// A single-token prefill tail retains prefill semantics.
#[test]
fn one_token_prefill_tail_is_explicitly_not_decode() {
    let mut session = StageSession {
        raw: ptr::null_mut(),
        token_count: 4,
        terminal_stage: true,
        batched_activation_exports: true,
        dflash: None,
        dflash_split: None,
    };
    let request = IterationBatchRequest {
        session: &mut session,
        token_ids: &[7],
        positions: &[4],
        sampling: None,
        input: None,
        sample_last: false,
        phase: IterationBatchPhase::Prefill,
    };
    assert_eq!(request.phase, IterationBatchPhase::Prefill);
}

/// Intermediate stages do not fabricate sampled-token outputs.
#[test]
fn serial_iteration_omits_samples_for_intermediate_stages() {
    let mut session = StageSession {
        raw: ptr::null_mut(),
        token_count: 4,
        terminal_stage: false,
        batched_activation_exports: true,
        dflash: None,
        dflash_split: None,
    };
    let request = IterationBatchRequest {
        session: &mut session,
        token_ids: &[7],
        positions: &[],
        sampling: None,
        input: None,
        sample_last: true,
        phase: IterationBatchPhase::Decode,
    };

    assert!(!iteration_request_should_emit_sample(&request));
}

/// Sparse samples retain their explicit indexes within the original request batch.
#[test]
fn sparse_iteration_samples_preserve_explicit_request_indexes() -> anyhow::Result<()> {
    let samples = collect_iteration_samples(3, 2, &[0, 2, 0], &[41, 99, 0])?;

    assert_eq!(samples[0].request_index, 0);
    assert_eq!(samples[0].predicted_token, 41);
    assert_eq!(samples[1].request_index, 2);
    assert_eq!(samples[1].predicted_token, 99);
    Ok(())
}

/// Duplicate or out-of-range native sample indexes fail validation.
#[test]
fn iteration_samples_reject_duplicate_or_out_of_range_indexes() {
    assert!(collect_iteration_samples(3, 2, &[1, 1], &[41, 99]).is_err());
    assert!(collect_iteration_samples(3, 1, &[3], &[41]).is_err());
    assert!(collect_iteration_samples(3, 4, &[0, 1, 2, 0], &[1, 2, 3, 4]).is_err());
}

/// Open a small native session for iteration-batch regression fixtures.
fn batch_session(terminal_stage: bool, batched_activation_exports: bool) -> StageSession {
    StageSession {
        raw: ptr::null_mut(),
        token_count: 0,
        terminal_stage,
        batched_activation_exports,
        dflash: None,
        dflash_split: None,
    }
}

/// Construct one decode request with explicit sampling and phase metadata.
fn decode_request<'a>(
    session: &'a mut StageSession,
    input: Option<&'a ActivationFrame>,
) -> IterationBatchRequest<'a> {
    IterationBatchRequest {
        session,
        token_ids: &[7],
        positions: &[],
        sampling: None,
        input,
        sample_last: true,
        phase: IterationBatchPhase::Decode,
    }
}

/// Native invalid-argument errors propagate without retrying through serial fallback.
#[test]
fn native_invalid_argument_is_returned_without_serial_fallback() {
    crate::test_support::require_native_runtime();
    let mut first = batch_session(false, true);
    let mut second = batch_session(false, true);
    let error = StageSession::iteration_batch_sampled(&mut [
        decode_request(&mut first, None),
        decode_request(&mut second, None),
    ])
    .err()
    .expect("null native sessions must be rejected");

    assert!(error.to_string().contains("InvalidArgument"), "{error:#}");
    assert!(error.downcast_ref::<PartialBatchExecution>().is_none());
    assert_eq!(first.token_count, 0);
    assert_eq!(second.token_count, 0);
}

/// Exporting stages lacking batch-export support execute one request at a time.
#[test]
fn exporting_stages_that_cannot_batch_exports_run_one_at_a_time() {
    let mut dense = batch_session(false, true);
    assert!(
        !iteration_request_requires_one_at_a_time(&decode_request(&mut dense, None)),
        "an exporting attention stage with a unified split still batches"
    );

    let mut recurrent = batch_session(false, false);
    assert!(
        iteration_request_requires_one_at_a_time(&decode_request(&mut recurrent, None)),
        "recurrent and hybrid memory splits an all-output batch by sequence"
    );
}

/// Terminal stages can batch independently of memory-split export restrictions.
#[test]
fn terminal_stages_batch_regardless_of_memory_splits() {
    let mut terminal = batch_session(true, false);

    assert!(
        !iteration_request_requires_one_at_a_time(&decode_request(&mut terminal, None)),
        "a terminal stage samples through llama's output-row translation, never exports"
    );
}

/// Optional activation parts force serial request execution.
#[test]
fn optional_activation_parts_always_run_one_at_a_time() {
    let mut dense = batch_session(false, true);
    let mut desc = activation_desc(1);
    desc.parts[0].flags = skippy_ffi::ACTIVATION_PART_OPTIONAL;
    let frame = ActivationFrame {
        desc,
        payload: vec![1],
    };

    assert!(iteration_request_requires_one_at_a_time(&decode_request(
        &mut dense,
        Some(&frame)
    )));
}

/// Partial native execution reports exactly which requests completed.
#[test]
fn partial_batch_execution_reports_the_requests_that_already_ran() {
    let error = anyhow::Error::new(PartialBatchExecution::new(2, anyhow!("native failed")));
    let partial = error
        .downcast_ref::<PartialBatchExecution>()
        .expect("partial batch keeps its type through anyhow");

    assert_eq!(partial.executed(), 2);
    assert_eq!(partial.error().to_string(), "native failed");
    assert!(partial.to_string().contains("executed 2 request(s)"));
}

/// A decode run through the scheduler forwards the DFlash target-layer rows a
/// split stage captured, as a stage message does; a frame without them makes
/// the final stage's draft drop its context.
#[test]
fn iteration_frames_carry_captured_dflash_rows() {
    let mut rows = crate::DFlashFeatureRows::new(1, 1);
    rows.insert(6, vec![0.5]).unwrap();
    let mut capturing = batch_session(false, true);
    capturing.dflash_split = Some(Box::new(crate::dflash::DFlashSplitSession::with_captured(
        rows.clone(),
    )));
    let mut plain = batch_session(false, true);
    let mut requests = vec![
        decode_request(&mut capturing, None),
        decode_request(&mut plain, None),
    ];
    let mut frames = vec![
        ActivationFrame {
            desc: activation_desc(4),
            payload: vec![0; 4],
        },
        ActivationFrame {
            desc: activation_desc(4),
            payload: vec![0; 4],
        },
    ];

    forward_dflash_features(&mut requests, &mut frames);

    let forwarded = crate::DFlashFeatureRows::from_frame(&frames[0])
        .unwrap()
        .expect("the capturing stage's frame carries its rows");
    assert_eq!(forwarded.layer_ids().collect::<Vec<_>>(), vec![6]);
    let value = forwarded.rows_of(6, 0, 1).unwrap()[0];
    assert!((value - 0.5).abs() < 0.5 / 128.0, "{value}");
    assert_eq!(
        crate::DFlashFeatureRows::from_frame(&frames[1]).unwrap(),
        None
    );
}

/// A batched decode on a split final stage hands its draft the feature rows
/// the input frame carries, as a stage message does. Without them the draft
/// finds the decode's rows missing and drops its context.
#[test]
fn iteration_batches_hand_frame_features_to_the_draft() {
    let mut rows = crate::DFlashFeatureRows::new(1, 1);
    rows.insert(6, vec![0.5]).unwrap();
    let mut frame = ActivationFrame {
        desc: activation_desc(4),
        payload: vec![0; 4],
    };
    rows.append_to_frame_as(&mut frame, crate::DFlashFeatureEncoding::Bf16)
        .unwrap();
    let mut drafting = batch_session(true, true);
    drafting.dflash = Some(crate::dflash::DFlashSessionLink::detached(0));
    let mut capturing = batch_session(false, true);
    let mut requests = vec![
        decode_request(&mut drafting, Some(&frame)),
        decode_request(&mut capturing, Some(&frame)),
    ];

    receive_dflash_features(&mut requests);

    let remote = drafting
        .dflash_split
        .as_ref()
        .and_then(|split| split.remote.as_ref());
    assert_eq!(remote, Some(&(rows, 0)));
    // A stage without a draft keeps no rows; they ride its frame on.
    assert!(capturing.dflash_split.is_none());
}
