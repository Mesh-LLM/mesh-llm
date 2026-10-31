//! Context limits admitted by the loaded native workload.

use anyhow::{Context, Result};
use skippy_runtime::{ModelWorkload, StageModel};

use super::RuntimeState;

#[derive(Clone, Copy)]
pub(super) struct ContextCapacity {
    per_lane_tokens: u32,
    pool_tokens: u32,
}

impl ContextCapacity {
    /// Mirror `skippy_context_capacity`: native ctx_size is per execution lane.
    /// This runs only after model open has admitted the corresponding config.
    pub(super) fn for_loaded_model(
        configured_tokens: u32,
        effective_lanes: u32,
        has_native_model: bool,
    ) -> Result<Self> {
        if !has_native_model {
            return Ok(Self::modelless(configured_tokens));
        }
        let per_lane_tokens = if configured_tokens == 0 {
            512
        } else {
            configured_tokens
        };
        let pool_tokens = per_lane_tokens
            .checked_mul(effective_lanes.max(1))
            .context(
                "ctx_size multiplied by admitted lane_count exceeds the native context range",
            )?;
        Ok(Self {
            per_lane_tokens,
            pool_tokens,
        })
    }

    /// Modelless fixtures supply an aggregate pool directly, without native
    /// per-lane allocation or native defaults.
    pub(super) const fn modelless(pool_tokens: u32) -> Self {
        Self {
            per_lane_tokens: pool_tokens,
            pool_tokens,
        }
    }
}

impl RuntimeState {
    /// Aggregate KV cells reserved for all admitted execution lanes (`n_ctx`).
    /// Zero denotes unknown capacity in a modelless fixture.
    pub fn kv_pool_tokens(&self) -> u32 {
        self.context_capacity.pool_tokens
    }

    /// Context limit for one request, including its sampling penalty history.
    /// Shared KV capacity does not enlarge an individual request's window.
    pub fn context_tokens_per_lane(&self) -> u32 {
        self.context_capacity.per_lane_tokens
    }
}

/// Encoder output is context-global, so native admits only one execution lane
/// for encoder-decoder workloads. Dummy models have no native contract.
pub(super) fn effective_lane_count(configured: u32, model: &StageModel) -> Result<u32> {
    if !model.has_native_model() {
        return Ok(configured);
    }
    Ok(lane_count_for_workload(
        configured,
        model.workload_info()?.kind,
    ))
}

pub(super) fn lane_count_for_workload(configured: u32, workload: ModelWorkload) -> u32 {
    if workload == ModelWorkload::EncoderDecoder {
        1
    } else {
        configured.max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_pool_includes_every_lane_without_enlarging_request_history() {
        let capacity = ContextCapacity::for_loaded_model(131_072, 4, true).unwrap();
        let mut runtime = RuntimeState::new_modelless_for_test(4);
        runtime.context_capacity = capacity;
        assert_eq!(runtime.kv_pool_tokens(), 524_288);
        assert_eq!(runtime.context_tokens_per_lane(), 131_072);
        let sampling = skippy_runtime::SamplingConfig {
            penalty_last_n: i32::MAX,
            dry: skippy_runtime::DrySamplingConfig {
                penalty_last_n: i32::MAX,
                ..Default::default()
            },
            ..Default::default()
        }
        .with_penalty_windows_within(runtime.context_tokens_per_lane() as usize);
        assert_eq!(sampling.penalty_last_n, 131_072);
        assert_eq!(sampling.dry.penalty_last_n, 131_072);
    }

    #[test]
    fn multi_lane_admission_keeps_work_that_fits_the_shared_pool() {
        use skippy_protocol::{StageConfig, StageKvCacheConfig, StageKvCacheMode};
        use skippy_runtime::{MemoryCacheCapabilities, ModelStateKind};

        let config = StageConfig {
            ctx_size: 131_072,
            lane_count: 4,
            kv_cache: Some(StageKvCacheConfig {
                exact_max_bytes: None,
                mode: StageKvCacheMode::LookupRecord,
                max_entries: 64,
                max_bytes: 0,
                l2_max_bytes: 0,
                codec: skippy_protocol::StageKvCacheCodec::Native,
                min_tokens: 1,
                shared_prefix_stride_tokens: 1,
                shared_prefix_record_limit: 1,
            }),
            ..Default::default()
        };
        let cache = crate::kv_integration::KvStageIntegration::from_loaded_model(
            &config,
            Some(ModelStateKind::Dense),
            Some(MemoryCacheCapabilities {
                resident: true,
                resident_attention: true,
                resident_checkpoint: false,
                kv_recurrent: true,
                full_state: true,
            }),
            None,
        )
        .unwrap()
        .unwrap();
        let mut runtime = RuntimeState::new_modelless_for_test(4);
        runtime.context_capacity = ContextCapacity::for_loaded_model(131_072, 4, true).unwrap();
        let _reservations = (0..4)
            .map(|lane| {
                cache
                    .reserve_resident_capacity(&format!("lane-{lane}"), 94_000)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let decision = cache
            .admit_resident_capacity(&mut runtime, "next-wave", 94_000, 0, 0, None)
            .unwrap();
        assert!(decision.admitted);
        assert_eq!(decision.capacity_tokens, 524_288);
        assert_eq!(decision.inflight_outstanding_tokens, 376_000);
        assert_eq!(decision.evicted_entries, 0);
        assert_eq!(decision.projected_free_tokens, 148_288);

        runtime.context_capacity = ContextCapacity::modelless(config.ctx_size);
        let understated = cache
            .admit_resident_capacity(&mut runtime, "next-wave", 94_000, 0, 0, None)
            .unwrap();
        assert!(!understated.admitted);
        assert_eq!(understated.admission_deficit_tokens, 244_928);
    }

    #[test]
    fn native_pool_uses_effective_workload_lanes_and_native_defaults() {
        let lanes = lane_count_for_workload(4, ModelWorkload::EncoderDecoder);
        let capacity = ContextCapacity::for_loaded_model(2048, lanes, true).unwrap();
        assert_eq!(capacity.pool_tokens, 2048);
        assert_eq!(capacity.per_lane_tokens, 2048);
        let capacity = ContextCapacity::for_loaded_model(0, 0, true).unwrap();
        assert_eq!(capacity.pool_tokens, 512);
        assert_eq!(
            lane_count_for_workload(0, ModelWorkload::CausalGeneration),
            1
        );
    }

    #[test]
    fn native_pool_rejects_overflow_instead_of_understating_capacity() {
        assert!(ContextCapacity::for_loaded_model(u32::MAX, 2, true).is_err());
    }

    #[test]
    fn modelless_capacity_remains_the_explicit_aggregate_input() {
        for tokens in [0, 131_072] {
            let runtime = RuntimeState::new_modelless_with_capacity_for_test(4, tokens);
            assert_eq!(runtime.kv_pool_tokens(), tokens);
            let capacity = ContextCapacity::for_loaded_model(tokens, 4, false).unwrap();
            assert_eq!(capacity.pool_tokens, tokens);
        }
    }
}
