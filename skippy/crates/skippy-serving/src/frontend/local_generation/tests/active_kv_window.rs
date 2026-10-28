//! Exercise attention views that start after a parked session's physical cells.
use super::*;
use native_decode_shapes::iterate;
use skippy_runtime::IterationBatchPhase;

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL, _LAYERS and optional _GPU_LAYERS"]
fn parked_prefix_preserves_active_decode_tokens() -> Result<()> {
    for lanes in [1, 2, 4] {
        let reference = decode_with_parked_prefix(lanes, false)?;
        let actual = decode_with_parked_prefix(lanes, true)?;
        assert_eq!(
            actual, reference,
            "parked physical prefix changed active output at c={lanes}"
        );
    }
    Ok(())
}

fn decode_with_parked_prefix(lanes: u32, parked: bool) -> Result<Vec<Vec<i32>>> {
    let (backend, _) = recurrent_test_backend_with_lanes(
        "active-kv-window",
        "active-kv-window",
        1024,
        (128, 128, lanes + 1),
        8,
        1024 * (lanes + 1) as usize,
        None,
    )?;
    let mut runtime = backend.runtime.lock().unwrap();
    if parked {
        let prompt: Vec<_> = (1..=512).collect();
        for chunk in prompt.chunks(32) {
            iterate(
                &mut runtime,
                &["parked-prefix"],
                &[chunk],
                IterationBatchPhase::Prefill,
            )?;
        }
    }
    let names: Vec<_> = (0..lanes).map(|lane| format!("active-{lane}")).collect();
    let sessions: Vec<_> = names.iter().map(String::as_str).collect();
    let prompts: Vec<Vec<i32>> = (0..lanes)
        .map(|lane| (1..=128).map(|token| token + 257 * lane as i32).collect())
        .collect();
    let mut next = Vec::new();
    for offset in (0..128).step_by(16) {
        let chunks: Vec<_> = prompts
            .iter()
            .map(|prompt| &prompt[offset..offset + 16])
            .collect();
        next = iterate(
            &mut runtime,
            &sessions,
            &chunks,
            IterationBatchPhase::Prefill,
        )?;
    }
    let mut output: Vec<_> = next.iter().map(|&token| vec![token]).collect();
    for _ in 1..8 {
        let inputs: Vec<_> = next.iter().map(std::slice::from_ref).collect();
        next = iterate(
            &mut runtime,
            &sessions,
            &inputs,
            IterationBatchPhase::Decode,
        )?;
        for (tokens, &token) in output.iter_mut().zip(&next) {
            tokens.push(token);
        }
    }
    Ok(output)
}
