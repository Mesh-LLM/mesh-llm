//! Model-backed coverage of startup after the binary transport fills the pool.
use super::*;

fn fixture(lanes: u32) -> Result<Arc<Mutex<RuntimeState>>> {
    let path = std::env::var_os("SKIPPY_WARMUP_STAGE_CONFIG")
        .context("set SKIPPY_WARMUP_STAGE_CONFIG to a full causal-model stage config")?;
    let mut config: StageConfig = serde_json::from_slice(&std::fs::read(path)?)?;
    config.lane_count = lanes;
    config.cache_idle_slots = None;
    crate::runtime_state::load_runtime(&config)?.context("fixture must load a native model")
}

#[test]
#[ignore = "requires SKIPPY_WARMUP_STAGE_CONFIG and a local causal GGUF"]
fn full_idle_pool_warms_repeatedly_and_all_lanes_still_decode() -> Result<()> {
    for lanes in [1, 4] {
        let runtime = fixture(lanes)?;
        let mut runtime = runtime.lock().unwrap();
        runtime.prewarm_idle_sessions(lanes as usize)?;
        let token = runtime.model.tokenize("Hello", true)?[0];
        for _ in 0..3 {
            assert!(runtime.warmup_generation_graph()?);
            assert_eq!(runtime.idle_sessions.len(), lanes as usize);
            assert!(
                runtime
                    .idle_sessions
                    .iter()
                    .all(|lane| lane.session.token_count() == 0)
            );
            assert!(runtime.free_lane_indices.is_empty());
        }
        let mut predictions = Vec::new();
        for index in 0..lanes {
            predictions.push(
                runtime
                    .session(&format!("request-{index}"))?
                    .decode_step(token)?,
            );
        }
        assert!(predictions.iter().all(|token| *token == predictions[0]));
        assert_eq!(runtime.session_stats().active_sessions, lanes as usize);
        for index in 0..lanes {
            runtime.drop_session_timed(&format!("request-{index}"))?;
        }
        assert_eq!(runtime.idle_sessions.len(), lanes as usize);
    }
    Ok(())
}

#[test]
#[ignore = "requires SKIPPY_WARMUP_STAGE_CONFIG and a local causal GGUF"]
fn temporary_warmup_releases_capacity_with_idle_pool_disabled() -> Result<()> {
    let runtime = fixture(1)?;
    let mut runtime = runtime.lock().unwrap();
    runtime.max_idle_sessions = Some(0);
    for _ in 0..3 {
        assert!(runtime.warmup_generation_graph()?);
        assert!(runtime.idle_sessions.is_empty());
        assert_eq!(runtime.next_lane_index, 1);
        assert_eq!(runtime.free_lane_indices, [0]);
    }
    let token = runtime.model.tokenize("Hello", true)?[0];
    runtime.session("request")?.decode_step(token)?;
    runtime.drop_session_timed("request")?;
    assert_eq!(runtime.free_lane_indices, [0]);
    Ok(())
}

#[test]
#[ignore = "requires SKIPPY_WARMUP_STAGE_CONFIG and a local causal GGUF"]
fn failed_warmup_discards_dirty_session_and_reclaims_native_lane() -> Result<()> {
    let runtime = fixture(1)?;
    let mut runtime = runtime.lock().unwrap();
    runtime.prewarm_idle_sessions(1)?;
    let mut lane = runtime.idle_sessions.pop().unwrap();
    let token = runtime.model.tokenize("Hello", true)?[0];
    lane.session.decode_step(token)?;
    assert_eq!(lane.session.token_count(), 1);
    // Exercise failure cleanup with a real, dirty native session. Both decode
    // and reset errors reach this branch; neither may park the session again.
    let error = runtime
        .finish_generation_warmup(lane, Err(anyhow::anyhow!("warmup reset failed")), true)
        .unwrap_err();
    assert_eq!(error.to_string(), "warmup reset failed");
    assert!(runtime.idle_sessions.is_empty());
    assert_eq!(runtime.free_lane_indices, [0]);
    runtime.prewarm_idle_sessions(1)?;
    assert_eq!(runtime.idle_sessions[0].session.token_count(), 0);
    assert!(runtime.warmup_generation_graph()?);
    assert!(runtime.free_lane_indices.is_empty());
    Ok(())
}
