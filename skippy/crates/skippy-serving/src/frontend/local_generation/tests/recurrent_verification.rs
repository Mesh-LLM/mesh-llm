use super::*;

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS"]
fn recurrent_verification_recovers_rejected_suffix_without_rolling_snapshots() -> Result<()> {
    let (backend, _) = recurrent_test_backend("verify", "verify", 128, (32, 32), 8, 128)?;
    let runtime = backend.runtime.lock().unwrap();
    let prompt = (1..=32).collect::<Vec<_>>();
    let proposal = [33, 34];

    // Exercise both complete rejection and replay of an accepted prefix. The
    // native context has no rolling device snapshots; verify must checkpoint
    // before advancing the recurrent state and trim must recover that state.
    for accepted in 0..proposal.len() {
        let mut control = runtime.model.create_session()?;
        control.prefill_chunk(&prompt)?;
        if accepted > 0 {
            control.prefill_chunk(&proposal[..accepted])?;
        }
        let mut expected = Vec::new();
        let mut token = 35;
        for _ in 0..4 {
            token = control.decode_step(token)?;
            expected.push(token);
        }
        drop(control);

        let mut verified = runtime.model.create_session()?;
        verified.prefill_chunk(&prompt)?;
        assert_eq!(verified.verify_tokens(&proposal)?.len(), proposal.len());
        verified.trim_session((prompt.len() + accepted) as u64)?;
        let mut actual = Vec::new();
        let mut token = 35;
        for _ in 0..4 {
            token = verified.decode_step(token)?;
            actual.push(token);
        }
        assert_eq!(actual, expected, "accepted prefix length {accepted}");
    }
    Ok(())
}
