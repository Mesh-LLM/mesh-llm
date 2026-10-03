#[cfg(test)]
mod speculative_tests {
    use super::*;

    #[test]
    fn thinking_override_respects_no_think_and_budget_zero() {
        assert_eq!(normalized_prompt_thinking(false, None), None);
        assert_eq!(normalized_prompt_thinking(true, None), Some(false));
        assert_eq!(normalized_prompt_thinking(false, Some(0)), Some(false));
        assert_eq!(normalized_prompt_thinking(false, Some(128)), Some(true));
    }

    fn normalized_prompt_thinking(no_think: bool, budget: Option<usize>) -> Option<bool> {
        let reasoning = prompt_openai_reasoning_config(no_think, budget).unwrap();
        normalize_reasoning_template_options(reasoning.as_ref(), None, &BTreeMap::new())
            .unwrap()
            .enable_thinking
    }

    #[test]
    fn verify_inputs_align_with_draft_proposals() {
        assert_eq!(verify_inputs_for_proposals(10, &[]), Vec::<i32>::new());
        assert_eq!(verify_inputs_for_proposals(10, &[11]), vec![10]);
        assert_eq!(
            verify_inputs_for_proposals(10, &[11, 12, 13]),
            vec![10, 11, 12]
        );
    }

    #[test]
    fn classify_verify_window_full_accept() {
        let decision =
            classify_verify_window(&[10, 11, 12], &[10, 11, 12], 0, 16, |_| Ok(false)).unwrap();
        assert_eq!(
            decision,
            VerifyWindowDecision {
                kind: VerifyWindowDecisionKind::FullAccept,
                accepted_before_reject: 3,
                commit_count: 3,
            }
        );
        assert!(!decision.rejected());
    }

    #[test]
    fn classify_verify_window_tail_reject_keeps_state() {
        let decision =
            classify_verify_window(&[10, 11, 12], &[10, 11, 42], 0, 16, |_| Ok(false)).unwrap();
        assert_eq!(
            decision,
            VerifyWindowDecision {
                kind: VerifyWindowDecisionKind::TailReject,
                accepted_before_reject: 2,
                commit_count: 3,
            }
        );
        assert!(decision.rejected());
        assert!(decision.tail_reject());
    }

    #[test]
    fn classify_verify_window_early_reject_commits_correction() {
        let decision =
            classify_verify_window(&[10, 11, 12, 13], &[10, 42, 77, 88], 0, 16, |_| Ok(false))
                .unwrap();
        assert_eq!(
            decision,
            VerifyWindowDecision {
                kind: VerifyWindowDecisionKind::EarlyReject,
                accepted_before_reject: 1,
                commit_count: 2,
            }
        );
        assert!(decision.rejected());
        assert!(!decision.tail_reject());
    }

    #[test]
    fn classify_verify_window_accepted_eog_stops_without_growing_window() {
        let decision =
            classify_verify_window(&[10, 99, 12], &[10, 99, 12], 0, 16, |token| Ok(token == 99))
                .unwrap();
        assert_eq!(
            decision,
            VerifyWindowDecision {
                kind: VerifyWindowDecisionKind::AcceptedStop,
                accepted_before_reject: 2,
                commit_count: 2,
            }
        );
        assert!(!decision.rejected());
    }

    #[test]
    fn classify_verify_window_early_reject_at_limit_stops() {
        let decision =
            classify_verify_window(&[10, 11, 12], &[10, 42, 77], 2, 4, |_| Ok(false)).unwrap();
        assert_eq!(
            decision,
            VerifyWindowDecision {
                kind: VerifyWindowDecisionKind::EarlyRejectStop,
                accepted_before_reject: 1,
                commit_count: 2,
            }
        );
        assert!(decision.rejected());
        assert!(!decision.tail_reject());
    }

    #[test]
    fn classify_verify_window_accepts_a_sampler_guarded_short_reply() {
        // With a stateful sampler the runtime stops at the first mismatched
        // row and returns a short reply; every returned token is
        // authoritative, so it classifies as an early reject at its final row.
        let decision =
            classify_verify_window(&[10, 11, 12, 13], &[10, 42], 0, 16, |_| Ok(false)).unwrap();
        assert_eq!(
            decision,
            VerifyWindowDecision {
                kind: VerifyWindowDecisionKind::EarlyReject,
                accepted_before_reject: 1,
                commit_count: 2,
            }
        );

        let err = classify_verify_window(&[10, 11, 12], &[], 0, 16, |_| Ok(false)).unwrap_err();
        assert!(
            err.to_string().contains("returned no tokens"),
            "{err:#}"
        );
    }

    #[test]
    fn observe_verify_decision_grows_on_full_accept_only() {
        let mut stats = SpeculativeStats::default();
        let mut adaptive_window = 4;
        stats.observe_verify_decision(
            VerifyWindowDecision {
                kind: VerifyWindowDecisionKind::FullAccept,
                accepted_before_reject: 4,
                commit_count: 4,
            },
            &mut adaptive_window,
            true,
            8,
        );

        assert_eq!(adaptive_window, 5);
        assert_eq!(stats.full_accept_windows, 1);
        assert_eq!(stats.adaptive_window_grows, 1);
        assert_eq!(stats.accepted_tokens, 4);
    }

    #[test]
    fn observe_verify_decision_stop_outcomes_do_not_move_adaptive_window() {
        let mut stats = SpeculativeStats::default();
        let mut adaptive_window = 4;
        stats.observe_verify_decision(
            VerifyWindowDecision {
                kind: VerifyWindowDecisionKind::AcceptedStop,
                accepted_before_reject: 2,
                commit_count: 2,
            },
            &mut adaptive_window,
            true,
            8,
        );
        stats.observe_verify_decision(
            VerifyWindowDecision {
                kind: VerifyWindowDecisionKind::EarlyRejectStop,
                accepted_before_reject: 1,
                commit_count: 2,
            },
            &mut adaptive_window,
            true,
            8,
        );

        assert_eq!(adaptive_window, 4);
        assert_eq!(stats.accepted_stop_windows, 1);
        assert_eq!(stats.early_reject_stop_windows, 1);
        assert_eq!(stats.adaptive_window_grows, 0);
        assert_eq!(stats.adaptive_window_shrinks, 0);
    }

    #[test]
    fn observe_verify_decision_early_reject_shrinks() {
        let mut stats = SpeculativeStats::default();
        let mut adaptive_window = 6;
        stats.observe_verify_decision(
            VerifyWindowDecision {
                kind: VerifyWindowDecisionKind::EarlyReject,
                accepted_before_reject: 1,
                commit_count: 2,
            },
            &mut adaptive_window,
            true,
            8,
        );

        assert_eq!(adaptive_window, 5);
        assert_eq!(stats.early_reject_windows, 1);
        assert_eq!(stats.adaptive_window_shrinks, 1);
        assert_eq!(stats.rejected_windows, 1);
        assert_eq!(stats.first_reject_position_sum, 2);
    }

    #[test]
    fn stable_wire_ids_are_deterministic_and_namespaced() {
        let prompt_index = 7usize.to_le_bytes();
        let session = stable_wire_id(&[b"session-a"]);
        let request = stable_wire_id(&[b"session-a", &prompt_index]);
        assert_ne!(session, 0);
        assert_eq!(session, stable_wire_id(&[b"session-a"]));
        assert_ne!(session, request);
    }
}

fn connect_ready(addr: &str, timeout_secs: u64) -> Result<TcpStream> {
    let deadline = Instant::now() + Duration::from_secs(timeout_secs.max(1));
    let mut last_error = None;
    while Instant::now() < deadline {
        match TcpStream::connect(addr) {
            Ok(mut stream) => {
                stream.set_nodelay(true).ok();
                stream
                    .set_read_timeout(Some(Duration::from_millis(500)))
                    .ok();
                match recv_ready_until_deadline(&mut stream, deadline) {
                    Ok(()) => {
                        stream.set_read_timeout(None).ok();
                        return Ok(stream);
                    }
                    Err(error) => {
                        last_error = Some(anyhow!(error).context("ready handshake failed"))
                    }
                }
            }
            Err(error) => last_error = Some(anyhow!(error).context("connect failed")),
        }
        thread::sleep(Duration::from_millis(500));
    }
    Err(last_error.unwrap_or_else(|| anyhow!("timed out")))
}

fn recv_ready_until_deadline(stream: &mut TcpStream, deadline: Instant) -> io::Result<()> {
    let mut bytes = [0_u8; 4];
    let mut offset = 0usize;
    while offset < bytes.len() {
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "timed out waiting for ready handshake",
            ));
        }
        match stream.read(&mut bytes[offset..]) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "ready handshake stream closed",
                ));
            }
            Ok(read) => offset += read,
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::Interrupted
                        | io::ErrorKind::WouldBlock
                        | io::ErrorKind::TimedOut
                ) =>
            {
                thread::sleep(Duration::from_millis(20));
            }
            Err(error) => return Err(error),
        }
    }
    let magic = i32::from_le_bytes(bytes);
    if magic != READY_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "stage ready magic mismatch",
        ));
    }
    Ok(())
}
