use sha2::{Digest, Sha256};
use skippy_runtime::SamplingConfig;

/// Returns whether a sampled output can be replayed without changing the
/// request's sampling semantics.
///
/// A positive temperature always leaves the output dependent on RNG state,
/// including when the caller supplied a seed. A disabled sampler and a
/// non-positive temperature both use deterministic greedy selection.
pub(super) fn sampling_replay_safe(sampling: &SamplingConfig) -> bool {
    !sampling.enabled || sampling.temperature <= 0.0
}

fn update_bytes(digest: &mut Sha256, field: &[u8], value: &[u8]) {
    digest.update((field.len() as u64).to_le_bytes());
    digest.update(field);
    digest.update((value.len() as u64).to_le_bytes());
    digest.update(value);
}

fn update_bool(digest: &mut Sha256, field: &[u8], value: bool) {
    update_bytes(digest, field, &[u8::from(value)]);
}

fn update_u32(digest: &mut Sha256, field: &[u8], value: u32) {
    update_bytes(digest, field, &value.to_le_bytes());
}

fn update_i32(digest: &mut Sha256, field: &[u8], value: i32) {
    update_bytes(digest, field, &value.to_le_bytes());
}

fn update_f32(digest: &mut Sha256, field: &[u8], value: f32) {
    update_u32(digest, field, value.to_bits());
}

fn update_string(digest: &mut Sha256, field: &[u8], value: &str) {
    update_bytes(digest, field, value.as_bytes());
}

/// Hashes every field that affects native sampling. The domain version must
/// change whenever sampling semantics change so old token records cannot
/// alias a new sampler contract.
pub(super) fn sampling_semantic_fingerprint(
    sampling: &SamplingConfig,
    chat_sampling_metadata: Option<&str>,
) -> String {
    use std::fmt::Write as _;

    // Intentionally destructure every field without `..`. Adding a sampler
    // control must make this fingerprint fail to compile until the new field
    // is assigned stable key semantics below.
    let SamplingConfig {
        enabled,
        ignore_eos,
        seed,
        temperature,
        top_p,
        top_k,
        min_p,
        presence_penalty,
        frequency_penalty,
        repeat_penalty,
        penalty_last_n,
        logit_bias,
        typical_p,
        top_nsigma,
        dynatemp_range,
        dynatemp_exponent,
        dry,
        xtc,
        mirostat_mode,
        mirostat_entropy,
        mirostat_learning_rate,
        samplers,
    } = sampling;

    let mut digest = Sha256::new();
    digest.update(b"skippy-sampling-fingerprint-v2");
    update_bool(&mut digest, b"enabled", *enabled);
    update_bool(&mut digest, b"ignore_eos", *ignore_eos);
    update_u32(&mut digest, b"seed", *seed);
    update_f32(&mut digest, b"temperature", *temperature);
    update_f32(&mut digest, b"top_p", *top_p);
    update_i32(&mut digest, b"top_k", *top_k);
    update_f32(&mut digest, b"min_p", *min_p);
    update_f32(&mut digest, b"presence_penalty", *presence_penalty);
    update_f32(&mut digest, b"frequency_penalty", *frequency_penalty);
    update_f32(&mut digest, b"repeat_penalty", *repeat_penalty);
    update_i32(&mut digest, b"penalty_last_n", *penalty_last_n);

    update_bytes(
        &mut digest,
        b"logit_bias_count",
        &(logit_bias.len() as u64).to_le_bytes(),
    );
    for (index, bias) in logit_bias.iter().enumerate() {
        let field = format!("logit_bias[{index}]");
        update_i32(&mut digest, field.as_bytes(), bias.token_id);
        update_f32(&mut digest, field.as_bytes(), bias.bias);
    }

    update_f32(&mut digest, b"typical_p", *typical_p);
    update_f32(&mut digest, b"top_nsigma", *top_nsigma);
    update_f32(&mut digest, b"dynatemp_range", *dynatemp_range);
    update_f32(&mut digest, b"dynatemp_exponent", *dynatemp_exponent);
    update_f32(&mut digest, b"dry.multiplier", dry.multiplier);
    update_f32(&mut digest, b"dry.base", dry.base);
    update_i32(&mut digest, b"dry.allowed_length", dry.allowed_length);
    update_i32(&mut digest, b"dry.penalty_last_n", dry.penalty_last_n);
    update_bytes(
        &mut digest,
        b"dry.sequence_breakers_count",
        &(dry.sequence_breakers.len() as u64).to_le_bytes(),
    );
    for (index, breaker) in dry.sequence_breakers.iter().enumerate() {
        update_string(
            &mut digest,
            format!("dry.sequence_breakers[{index}]").as_bytes(),
            breaker,
        );
    }

    update_f32(&mut digest, b"xtc.probability", xtc.probability);
    update_f32(&mut digest, b"xtc.threshold", xtc.threshold);
    update_i32(&mut digest, b"mirostat_mode", *mirostat_mode);
    update_f32(&mut digest, b"mirostat_entropy", *mirostat_entropy);
    update_f32(
        &mut digest,
        b"mirostat_learning_rate",
        *mirostat_learning_rate,
    );
    update_bytes(
        &mut digest,
        b"samplers_count",
        &(samplers.len() as u64).to_le_bytes(),
    );
    for (index, sampler) in samplers.iter().enumerate() {
        update_string(
            &mut digest,
            format!("samplers[{index}]").as_bytes(),
            sampler,
        );
    }
    update_bool(
        &mut digest,
        b"chat_sampling_metadata_present",
        chat_sampling_metadata.is_some(),
    );
    if let Some(metadata) = chat_sampling_metadata {
        update_string(&mut digest, b"chat_sampling_metadata", metadata);
    }

    let digest = digest.finalize();
    let mut fingerprint = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(&mut fingerprint, "{byte:02x}");
    }
    fingerprint
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_changes_for_every_sampling_field() {
        let baseline = SamplingConfig::default();
        let baseline_fingerprint = sampling_semantic_fingerprint(&baseline, None);
        let mut mutations = Vec::new();

        macro_rules! mutation {
            ($name:literal, $change:expr) => {{
                let mut sampling = baseline.clone();
                $change(&mut sampling);
                mutations.push(($name, sampling));
            }};
        }

        mutation!("enabled", |sampling: &mut SamplingConfig| sampling
            .enabled =
            true);
        mutation!("ignore_eos", |sampling: &mut SamplingConfig| sampling
            .ignore_eos =
            true);
        mutation!("seed", |sampling: &mut SamplingConfig| sampling.seed = 42);
        mutation!("temperature", |sampling: &mut SamplingConfig| sampling
            .temperature =
            0.5);
        mutation!("top_p", |sampling: &mut SamplingConfig| sampling.top_p =
            0.5);
        mutation!("top_k", |sampling: &mut SamplingConfig| sampling.top_k = 10);
        mutation!("min_p", |sampling: &mut SamplingConfig| sampling.min_p =
            0.1);
        mutation!("presence_penalty", |sampling: &mut SamplingConfig| {
            sampling.presence_penalty = 0.1
        });
        mutation!("frequency_penalty", |sampling: &mut SamplingConfig| {
            sampling.frequency_penalty = 0.1
        });
        mutation!("repeat_penalty", |sampling: &mut SamplingConfig| sampling
            .repeat_penalty =
            1.1);
        mutation!("penalty_last_n", |sampling: &mut SamplingConfig| sampling
            .penalty_last_n =
            10);
        mutation!("logit_bias", |sampling: &mut SamplingConfig| sampling
            .logit_bias
            .push(skippy_runtime::LogitBias {
                token_id: 42,
                bias: 0.5
            }));
        mutation!("typical_p", |sampling: &mut SamplingConfig| sampling
            .typical_p =
            0.5);
        mutation!("top_nsigma", |sampling: &mut SamplingConfig| sampling
            .top_nsigma =
            0.5);
        mutation!("dynatemp_range", |sampling: &mut SamplingConfig| sampling
            .dynatemp_range =
            0.5);
        mutation!("dynatemp_exponent", |sampling: &mut SamplingConfig| {
            sampling.dynatemp_exponent = 0.5
        });
        mutation!("dry.multiplier", |sampling: &mut SamplingConfig| sampling
            .dry
            .multiplier =
            0.5);
        mutation!("dry.base", |sampling: &mut SamplingConfig| sampling
            .dry
            .base = 2.0);
        mutation!("dry.allowed_length", |sampling: &mut SamplingConfig| {
            sampling.dry.allowed_length = 3
        });
        mutation!("dry.penalty_last_n", |sampling: &mut SamplingConfig| {
            sampling.dry.penalty_last_n = 10
        });
        mutation!("dry.sequence_breakers", |sampling: &mut SamplingConfig| {
            sampling.dry.sequence_breakers.push("|".to_string())
        });
        mutation!("xtc.probability", |sampling: &mut SamplingConfig| {
            sampling.xtc.probability = 0.5
        });
        mutation!("xtc.threshold", |sampling: &mut SamplingConfig| sampling
            .xtc
            .threshold =
            0.5);
        mutation!("mirostat_mode", |sampling: &mut SamplingConfig| sampling
            .mirostat_mode =
            1);
        mutation!("mirostat_entropy", |sampling: &mut SamplingConfig| {
            sampling.mirostat_entropy = 4.0
        });
        mutation!("mirostat_learning_rate", |sampling: &mut SamplingConfig| {
            sampling.mirostat_learning_rate = 0.2
        });
        mutation!("samplers", |sampling: &mut SamplingConfig| sampling
            .samplers
            .swap(0, 1));

        for (field, sampling) in mutations {
            assert_ne!(
                baseline_fingerprint,
                sampling_semantic_fingerprint(&sampling, None),
                "mutating {field} must change the sampling fingerprint"
            );
        }
    }

    #[test]
    fn fingerprint_distinguishes_chat_sampling_metadata() {
        let sampling = SamplingConfig::default();
        assert_ne!(
            sampling_semantic_fingerprint(&sampling, None),
            sampling_semantic_fingerprint(&sampling, Some("{}"))
        );
        assert_ne!(
            sampling_semantic_fingerprint(&sampling, None),
            sampling_semantic_fingerprint(&sampling, Some("<none>"))
        );
        assert_ne!(
            sampling_semantic_fingerprint(&sampling, Some("a")),
            sampling_semantic_fingerprint(&sampling, Some("ab"))
        );
    }
}
