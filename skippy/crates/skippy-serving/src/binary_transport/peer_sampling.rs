//! Sampling configs that arrive from a peer stage.
//!
//! In split serving the final stage samples with the config the first stage
//! sends on the wire. The frontend validated it there, but this stage cannot
//! trust that the peer did, so the config is bounded here again before it
//! reaches the native sampler.

use skippy_protocol::binary::{StageSamplingConfig, sampling_flags};
use skippy_runtime::{LogitBias, MAX_LOGIT_BIAS, SamplingConfig};

/// Converts a peer's sampling config for the native sampler of a stage whose
/// context holds `ctx_size` tokens.
pub(in crate::binary_transport) fn runtime_sampling_config(
    sampling: Option<&StageSamplingConfig>,
    ctx_size: u32,
) -> Option<SamplingConfig> {
    let sampling = sampling?;
    let mut config = SamplingConfig {
        enabled: true,
        ignore_eos: sampling.ignore_eos || (sampling.flags & sampling_flags::IGNORE_EOS) != 0,
        seed: sampling.seed,
        temperature: sampling.temperature,
        top_p: sampling.top_p,
        top_k: sampling.top_k,
        min_p: sampling.min_p,
        presence_penalty: sampling.presence_penalty,
        frequency_penalty: sampling.frequency_penalty,
        repeat_penalty: sampling.repeat_penalty,
        penalty_last_n: sampling.penalty_last_n,
        typical_p: sampling.typical_p,
        top_nsigma: sampling.top_nsigma,
        dynatemp_range: sampling.dynatemp_range,
        dynatemp_exponent: sampling.dynatemp_exponent,
        dry: skippy_runtime::DrySamplingConfig {
            multiplier: sampling.dry_multiplier,
            base: sampling.dry_base,
            allowed_length: sampling.dry_allowed_length,
            penalty_last_n: sampling.dry_penalty_last_n,
            sequence_breakers: sampling.dry_sequence_breakers.clone(),
        },
        xtc: skippy_runtime::XtcSamplingConfig {
            probability: sampling.xtc_probability,
            threshold: sampling.xtc_threshold,
        },
        mirostat_mode: sampling.mirostat_mode,
        mirostat_entropy: sampling.mirostat_entropy,
        mirostat_learning_rate: sampling.mirostat_learning_rate,
        samplers: sampling.samplers.clone(),
        reasoning_budget: skippy_runtime::ReasoningBudget::Resolved(
            sampling.reasoning_budget_tokens,
        ),
        ..SamplingConfig::default()
    };
    config.logit_bias = sampling
        .logit_bias
        .iter()
        .take(MAX_LOGIT_BIAS)
        .map(|source| LogitBias {
            token_id: source.token_id,
            bias: source.bias,
        })
        .collect();
    let ctx_size = usize::try_from(ctx_size).unwrap_or(usize::MAX);
    sampling
        .enabled()
        .then(|| config.with_penalty_windows_within(ctx_size))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peer_penalty_windows_are_capped_at_the_context() {
        let sampling = StageSamplingConfig {
            flags: sampling_flags::ENABLED,
            penalty_last_n: i32::MAX,
            dry_penalty_last_n: i32::MAX,
            ..StageSamplingConfig::default()
        };

        let config = runtime_sampling_config(Some(&sampling), 4096).expect("sampling is enabled");
        assert_eq!(config.penalty_last_n, 4096);
        assert_eq!(config.dry.penalty_last_n, 4096);
    }

    #[test]
    fn peer_penalty_windows_within_the_context_are_kept() {
        let sampling = StageSamplingConfig {
            flags: sampling_flags::ENABLED,
            penalty_last_n: 256,
            dry_penalty_last_n: 0,
            ..StageSamplingConfig::default()
        };

        let config = runtime_sampling_config(Some(&sampling), 4096).expect("sampling is enabled");
        assert_eq!(config.penalty_last_n, 256);
        assert_eq!(config.dry.penalty_last_n, 0);
    }
}
