#[cfg(test)]
use mesh_llm_types::router::strip_split_suffix;
/// Model selection from structural requirements and advertised capabilities.
pub use mesh_llm_types::router::{
    Classification, MediaRequirements, classify, media_requirements, strip_split_suffix_owned,
};

pub(crate) fn model_satisfies_media_requirements(
    caps: &crate::models::ModelCapabilities,
    media: &MediaRequirements,
) -> bool {
    (!media.needs_vision || caps.supports_vision_runtime())
        && (!media.needs_audio || caps.supports_audio_runtime())
}

pub(crate) fn filter_media_compatible_candidates<'a>(
    candidates: &[RoutingCandidate<'a>],
    media: &MediaRequirements,
) -> Option<Vec<RoutingCandidate<'a>>> {
    let media_available: Vec<_> = candidates
        .iter()
        .filter(|c| model_satisfies_media_requirements(&c.caps, media))
        .cloned()
        .collect();
    if media_available.is_empty() && media.requires_runtime_modality() {
        None
    } else if media_available.is_empty() {
        Some(candidates.to_vec())
    } else {
        Some(media_available)
    }
}

// ── Model selection ─────────────────────────────────────────────────

/// A candidate model in the auto-routing pool.
///
/// `capacity` and `replicas` summarise the model's routable placements (hosts
/// that serve it right now, this node included). Each placement contributes
/// [`placement_weight`] of its measured tok/s — neutral when unmeasured — so a
/// model's draw weight is its total serving capacity, not its name.
#[derive(Clone, Debug)]
pub struct RoutingCandidate<'a> {
    pub name: &'a str,
    pub caps: crate::models::ModelCapabilities,
    /// Total model parameter count in billions, when advertised.
    pub parameter_count_b: Option<f64>,
    /// Routable placements of this model (at least 1). Exploration draws a
    /// placement uniformly, so it is replica-aware too.
    pub replicas: usize,
    /// Sum of [`placement_weight`] over the placements; the model's draw weight.
    pub capacity: f64,
}

impl<'a> RoutingCandidate<'a> {
    /// Build a candidate for one unmeasured placement. Useful for pre-startup
    /// paths or test fixtures.
    #[cfg(test)]
    pub fn unscored(name: &'a str, caps: crate::models::ModelCapabilities) -> Self {
        Self {
            name,
            caps,
            parameter_count_b: None,
            replicas: 1,
            capacity: TPS_NEUTRAL_WEIGHT,
        }
    }
}

/// Minimum number of throughput samples before `tps_hint` is allowed
/// to influence weighting. Below this, the candidate is treated as
/// cold (neutral weight).
const TPS_MIN_SAMPLES: u64 = 3;

/// Lower clamp on tps used as a weight. Prevents catastrophic peers
/// (~1 tok/s) from being completely starved — they still get the
/// occasional request so they can re-prove themselves.
const TPS_WEIGHT_MIN: f64 = 5.0;

/// Upper clamp on tps used as a weight. Prevents a single very fast
/// outlier from monopolizing routing.
const TPS_WEIGHT_MAX: f64 = 100.0;

/// Neutral weight assigned to cold / unscored candidates so they get
/// a fair shot at picking up data. Set to the midpoint of the clamp
/// range so a cold model is treated as "average" against scored peers.
const TPS_NEUTRAL_WEIGHT: f64 = 25.0;

/// Probability of ignoring capacity and drawing a placement uniformly. Keeps
/// the system from locking onto stale rankings and gives cold placements a
/// guaranteed share of traffic so they accumulate data.
const EXPLORATION_PROBABILITY: f64 = 0.15;

/// Draw weight of one placement (one host serving one model) from its measured
/// throughput. Below `TPS_MIN_SAMPLES` the placement is cold and neutral; above,
/// the mean tok/s clamped to `TPS_WEIGHT_MIN..=TPS_WEIGHT_MAX` so a catastrophic
/// host is never fully starved and a single very fast host cannot monopolise.
pub(crate) fn placement_weight(tps: Option<f64>, samples: u64) -> f64 {
    if samples >= TPS_MIN_SAMPLES {
        tps.filter(|tps| tps.is_finite() && *tps > 0.0)
            .unwrap_or(TPS_NEUTRAL_WEIGHT)
            .clamp(TPS_WEIGHT_MIN, TPS_WEIGHT_MAX)
    } else {
        TPS_NEUTRAL_WEIGHT
    }
}

/// Weight of a target in the two-choice host draw: capacity scaled by distance.
/// Latency is a tiebreak factor only (direct/≤50 ms ×1 … ≥300 ms ×0.5, never 0):
/// LLM latency is dominated by generation, and a "nearest first" primary key
/// would concentrate every nearby client on the same host.
pub(crate) fn locality_factor(rtt_ms: Option<u32>) -> f64 {
    const NEAR_MS: f64 = 50.0;
    const FAR_MS: f64 = 300.0;
    match rtt_ms {
        None => 1.0,
        Some(rtt) => {
            let excess = ((f64::from(rtt) - NEAR_MS) / (FAR_MS - NEAR_MS)).clamp(0.0, 1.0);
            1.0 - 0.5 * excess
        }
    }
}

/// Pick the best model for a classified request using gossiped capabilities
/// and locally observed throughput.
///
/// Declared tool requirements prefer tool-capable models. Media compatibility
/// is checked structurally before selection; prompt keywords have no effect.
///
/// Falls back to all models if the filter matches nothing. Then biases
/// toward larger models by partitioning single-digit-B names to the
/// bottom tier. Within the chosen tier, picks weighted by observed
/// tok/s (with cold models treated as average), with a configurable
/// exploration probability that ignores weights and picks uniformly.
pub fn pick_model_classified<'a>(
    classification: &Classification,
    available_models: &[RoutingCandidate<'a>],
) -> Option<&'a str> {
    use crate::models::CapabilityLevel;

    if available_models.is_empty() {
        return None;
    }
    if available_models.len() == 1 {
        return Some(available_models[0].name);
    }

    // Capability filter based on what the request needs.
    let filtered: Vec<&RoutingCandidate<'a>> = available_models
        .iter()
        .filter(|c| !classification.needs_tools || c.caps.tool_use != CapabilityLevel::None)
        .collect();

    // Fall back to all models if the filter matched nothing.
    let candidates: Vec<&RoutingCandidate<'a>> = if filtered.is_empty() {
        available_models.iter().collect()
    } else {
        filtered
    };

    // Bias toward larger models: explicit metadata below 10B goes to the
    // bottom tier. When metadata is absent, fall back to the legacy
    // single-digit-B name heuristic.
    let (big, small): (Vec<_>, Vec<_>) = candidates
        .into_iter()
        .partition(|c| !is_small_parameter_model(c));

    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as u64;

    if !big.is_empty() {
        return Some(pick_weighted(&big, nanos));
    }
    if !small.is_empty() {
        return Some(pick_weighted(
            &small,
            nanos.wrapping_add(0x9E37_79B9_7F4A_7C15),
        ));
    }
    None
}

/// Routing weight of a model: its total placement capacity, floored so a
/// candidate can never drop out of the draw entirely.
fn candidate_weight(candidate: &RoutingCandidate<'_>) -> f64 {
    if candidate.capacity.is_finite() && candidate.capacity >= TPS_WEIGHT_MIN {
        candidate.capacity
    } else {
        TPS_WEIGHT_MIN
    }
}

/// Draw one index from `weights` (all finite, non-negative, not all zero).
pub(crate) fn weighted_index(weights: &[f64], roll: f64) -> usize {
    let total: f64 = weights.iter().sum();
    let pick = roll * total;
    let mut acc = 0.0;
    for (index, weight) in weights.iter().enumerate() {
        acc += weight;
        if pick < acc {
            return index;
        }
    }
    weights.len().saturating_sub(1)
}

/// Pick one candidate from a non-empty slice: with `EXPLORATION_PROBABILITY`
/// a uniformly drawn placement (so a model's exploration share is its replica
/// share), otherwise a capacity-weighted draw.
fn pick_weighted<'a>(candidates: &[&RoutingCandidate<'a>], seed: u64) -> &'a str {
    debug_assert!(!candidates.is_empty(), "pick_weighted requires non-empty");

    let mut rng = SplitMix64::new(seed);

    let weights: Vec<f64> = if rng.next_f64() < EXPLORATION_PROBABILITY {
        candidates
            .iter()
            .map(|c| c.replicas.max(1) as f64)
            .collect()
    } else {
        candidates.iter().map(|c| candidate_weight(c)).collect()
    };
    candidates[weighted_index(&weights, rng.next_f64())].name
}

fn is_small_parameter_model(candidate: &RoutingCandidate<'_>) -> bool {
    candidate
        .parameter_count_b
        .map(|count| count.is_finite() && count < 10.0)
        .unwrap_or_else(|| is_single_digit_b_name(candidate.name))
}

/// Small deterministic PRNG so a single seed drives both the
/// exploration coin flip and the weighted draw. Avoids pulling in a
/// rand dependency just for routing.
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        // Avoid the zero state which gives a degenerate sequence.
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub(crate) fn next_f64(&mut self) -> f64 {
        // Use the top 53 bits for a uniform float in [0, 1).
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Return true if `name` advertises a single-digit billion-parameter
/// count, e.g. "Qwen3.5-2B-Q4_K_M" or "llama-3-7b-instruct".
///
/// Accepts: a standalone digit 1-9 immediately followed by `b` or `B`,
/// with the digit *not* preceded by another digit or `.` (so "12B" and
/// "2.5B" don't count) and the `B` *not* followed by another digit (so
/// "BF16" isn't a match).
///
/// Names without any digit-B pattern return false — they are treated as
/// "probably strong" because small open-weight models almost always
/// advertise their size in the filename.
fn is_single_digit_b_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    for i in 0..bytes.len() {
        let c = bytes[i];
        if !c.is_ascii_digit() {
            continue;
        }
        // Must be a single digit run at a word boundary: previous char
        // must not be another digit, a '.', or an ASCII letter. That
        // last part rules out active-params tags like "A3B" where
        // the 3B is a subset of a larger total count advertised
        // elsewhere in the name (e.g. "Qwen3.6-35B-A3B").
        if i > 0 {
            let prev = bytes[i - 1];
            if prev.is_ascii_digit() || prev == b'.' || prev.is_ascii_alphabetic() {
                continue;
            }
        }
        // Digit must be 1-9 (0B would be nonsense, ignore)
        if c == b'0' {
            continue;
        }
        // Next byte must be b or B
        let Some(&next) = bytes.get(i + 1) else {
            continue;
        };
        if next != b'b' && next != b'B' {
            continue;
        }
        // And the byte after that must not be another digit (avoid BF16-like continuations)
        if let Some(&after) = bytes.get(i + 2)
            && after.is_ascii_digit()
        {
            continue;
        }
        return true;
    }
    false
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_classify_tool_call() {
        // Content that implies tool use + tools schema = ToolCall
        let body = json!({
            "messages": [{"role": "user", "content": "Run the tests and check the output"}],
            "tools": [{"type": "function", "function": {"name": "bash"}}]
        });
        assert!(classify(&body).needs_tools);
    }

    #[test]
    fn test_classify_chat_default() {
        let body = json!({
            "messages": [
                {"role": "user", "content": "What's the capital of France?"}
            ]
        });
        let cl = classify(&body);
        assert!(!cl.needs_tools);
        assert!(!cl.has_media_inputs);
    }

    #[test]
    fn test_classify_code_with_tools() {
        // Code request that happens to have tools — should be Code, not ToolCall
        let body = json!({
            "messages": [{"role": "user", "content": "Write a Python function to sort a list and debug it"}],
            "tools": [{"type": "function", "function": {"name": "bash"}}]
        });
        let cl = classify(&body);
        assert!(cl.needs_tools);
    }

    #[test]
    fn test_classify_tools_schema_always_needs_tools() {
        // Tools schema present = agentic session, always needs_tools
        // even if the message content is plain chat
        let body = json!({
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "function": {"name": "bash"}}]
        });
        let cl = classify(&body);
        assert!(cl.needs_tools);
    }

    #[test]
    fn test_classify_tools_schema_with_tool_content() {
        // Tools in schema AND content implies tool use — needs tools
        let body = json!({
            "messages": [{"role": "user", "content": "Read the file and fix the bug"}],
            "tools": [{"type": "function", "function": {"name": "read"}}]
        });
        let cl = classify(&body);
        assert!(cl.needs_tools);
    }

    #[test]
    fn test_classify_anthropic_text_blocks_with_tools() {
        // Anthropic-style content blocks should still be parsed as text
        // and trigger needs_tools when tool-intent is present.
        let body = json!({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "List files in this directory and read README.md"}
                    ]
                }
            ],
            "tools": [{"name": "shell"}]
        });
        let cl = classify(&body);
        assert!(cl.needs_tools);
    }

    #[test]
    fn test_classify_anthropic_tool_use_block_sets_needs_tools() {
        // If an explicit tool_use/tool_result block is present, mark as needs_tools.
        let body = json!({
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_123", "name": "shell", "input": {"command": "ls"}}
                    ]
                }
            ]
        });
        let cl = classify(&body);
        assert!(cl.needs_tools);
    }

    #[test]
    fn test_anthropic_tool_request_sets_needs_tools() {
        let body = json!({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "List files in this directory and read README.md"}
                    ]
                }
            ],
            "tools": [{"name": "shell"}]
        });
        let cl = classify(&body);
        assert!(cl.needs_tools);
    }

    #[test]
    fn test_media_requirements_detect_audio_block() {
        let body = json!({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe this clip"},
                        {"type": "audio_url", "audio_url": {"url": "mesh://blob/client-1/example"}}
                    ]
                }
            ]
        });
        let media = media_requirements(&body);
        assert!(media.has_media);
        assert!(media.needs_audio);
        assert!(!media.needs_vision);
        assert!(classify(&body).has_media_inputs);
    }

    #[test]
    fn test_media_requirements_detect_image_block() {
        let body = json!({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
                    ]
                }
            ]
        });
        let media = media_requirements(&body);
        assert!(media.has_media);
        assert!(media.needs_vision);
        assert!(!media.needs_audio);
        assert!(classify(&body).has_media_inputs);
    }

    #[test]
    fn test_model_satisfies_media_requirements_matches_required_modalities() {
        use crate::models::{CapabilityLevel, ModelCapabilities};

        let text_caps = ModelCapabilities::default();
        let vision_caps = ModelCapabilities {
            vision: CapabilityLevel::Supported,
            ..Default::default()
        };
        let audio_caps = ModelCapabilities {
            audio: CapabilityLevel::Supported,
            ..Default::default()
        };
        let vision_audio_caps = ModelCapabilities {
            vision: CapabilityLevel::Supported,
            audio: CapabilityLevel::Supported,
            ..Default::default()
        };

        let text_only = MediaRequirements::default();
        let image = MediaRequirements {
            has_media: true,
            needs_vision: true,
            needs_audio: false,
        };
        let audio = MediaRequirements {
            has_media: true,
            needs_vision: false,
            needs_audio: true,
        };
        let image_and_audio = MediaRequirements {
            has_media: true,
            needs_vision: true,
            needs_audio: true,
        };

        assert!(model_satisfies_media_requirements(&text_caps, &text_only));
        assert!(!model_satisfies_media_requirements(&text_caps, &image));
        assert!(model_satisfies_media_requirements(&vision_caps, &image));
        assert!(!model_satisfies_media_requirements(&vision_caps, &audio));
        assert!(model_satisfies_media_requirements(&audio_caps, &audio));
        assert!(!model_satisfies_media_requirements(
            &vision_caps,
            &image_and_audio
        ));
        assert!(model_satisfies_media_requirements(
            &vision_audio_caps,
            &image_and_audio
        ));

        let likely_vision_caps = ModelCapabilities {
            multimodal: true,
            vision: CapabilityLevel::Likely,
            ..Default::default()
        };
        assert!(!model_satisfies_media_requirements(
            &likely_vision_caps,
            &image
        ));
    }

    #[test]
    fn test_filter_media_candidates_blocks_hard_media_miss() {
        use crate::models::{CapabilityLevel, ModelCapabilities};

        let text_caps = ModelCapabilities::default();
        let vision_caps = ModelCapabilities {
            vision: CapabilityLevel::Supported,
            ..Default::default()
        };
        let image = MediaRequirements {
            has_media: true,
            needs_vision: true,
            needs_audio: false,
        };
        let text_only = MediaRequirements::default();

        let text_candidates = vec![RoutingCandidate::unscored("text", text_caps)];
        assert!(filter_media_compatible_candidates(&text_candidates, &image).is_none());

        let mixed_candidates = vec![
            RoutingCandidate::unscored("text", text_caps),
            RoutingCandidate::unscored("vision", vision_caps),
        ];
        let filtered = filter_media_compatible_candidates(&mixed_candidates, &image)
            .expect("vision candidate should satisfy image media request");
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].name, "vision");

        let unfiltered = filter_media_compatible_candidates(&text_candidates, &text_only)
            .expect("text-only requests should keep normal router fallback behavior");
        // Text-only requests with text-only candidates pass through unfiltered.
        assert_eq!(unfiltered.len(), text_candidates.len());
        assert_eq!(unfiltered[0].name, text_candidates[0].name);
    }

    #[test]
    fn test_pick_tools_filters_by_capability() {
        use crate::models::{CapabilityLevel, ModelCapabilities};

        let tool_caps = ModelCapabilities {
            tool_use: CapabilityLevel::Supported,
            ..Default::default()
        };
        let no_caps = ModelCapabilities::default();

        let available = vec![
            RoutingCandidate::unscored("reasoning-model", no_caps),
            RoutingCandidate::unscored("tool-model", tool_caps),
        ];
        let cl = Classification {
            needs_tools: true,
            has_media_inputs: false,
        };
        let result = pick_model_classified(&cl, &available);
        assert_eq!(result, Some("tool-model"));
    }

    #[test]
    fn test_pick_falls_back_when_no_capability_match() {
        use crate::models::ModelCapabilities;

        let no_caps = ModelCapabilities::default();
        let available = vec![
            RoutingCandidate::unscored("model-a", no_caps),
            RoutingCandidate::unscored("model-b", no_caps),
        ];
        let cl = Classification {
            needs_tools: true,
            has_media_inputs: false,
        };
        // No tool-capable model — falls back to all
        let result = pick_model_classified(&cl, &available);
        assert!(result == Some("model-a") || result == Some("model-b"));
    }

    #[test]
    fn test_pick_empty_returns_none() {
        let available: Vec<RoutingCandidate<'_>> = vec![];
        let cl = Classification {
            needs_tools: false,
            has_media_inputs: false,
        };
        assert_eq!(pick_model_classified(&cl, &available), None);
    }

    #[test]
    fn test_pick_single_model() {
        use crate::models::ModelCapabilities;

        let available = vec![RoutingCandidate::unscored(
            "only-model",
            ModelCapabilities::default(),
        )];
        let cl = Classification {
            needs_tools: false,
            has_media_inputs: false,
        };
        assert_eq!(pick_model_classified(&cl, &available), Some("only-model"));
    }

    #[test]
    fn test_pick_chat_no_filter() {
        use crate::models::ModelCapabilities;

        let no_caps = ModelCapabilities::default();
        let available = vec![
            RoutingCandidate::unscored("model-a", no_caps),
            RoutingCandidate::unscored("model-b", no_caps),
        ];
        let cl = Classification {
            needs_tools: false,
            has_media_inputs: false,
        };
        // Chat with no special needs — any model is valid
        let result = pick_model_classified(&cl, &available);
        assert!(result == Some("model-a") || result == Some("model-b"));
    }

    #[test]
    fn test_strip_split_suffix() {
        assert_eq!(
            strip_split_suffix("MiniMax-M2.5-Q4_K_M-00001-of-00004"),
            "MiniMax-M2.5-Q4_K_M"
        );
        assert_eq!(
            strip_split_suffix("Qwen3-Coder-Next-Q4_K_M-00001-of-00004"),
            "Qwen3-Coder-Next-Q4_K_M"
        );
        assert_eq!(
            strip_split_suffix("Hermes-2-Pro-Mistral-7B-Q4_K_M"),
            "Hermes-2-Pro-Mistral-7B-Q4_K_M"
        );
        assert_eq!(strip_split_suffix(""), "");
    }

    #[test]
    fn test_is_single_digit_b_name() {
        // Single-digit sizes — match
        assert!(is_single_digit_b_name("Qwen3.5-2B-Q4_K_M"));
        assert!(is_single_digit_b_name("Qwen3.5-9B-Q4_K_M"));
        assert!(is_single_digit_b_name("llama-3-7b-instruct"));
        assert!(is_single_digit_b_name("Mistral-7B-Instruct-v0.3"));
        assert!(is_single_digit_b_name("gemma-2-2b-it"));

        // Multi-digit sizes — not small
        assert!(!is_single_digit_b_name("gemma-4-31B-it-Q8_0"));
        assert!(!is_single_digit_b_name("Qwen3.6-35B-A3B-BF16"));
        assert!(!is_single_digit_b_name("llama-3.1-70B-Instruct"));
        assert!(!is_single_digit_b_name("deepseek-v3-671B"));

        // Decimal sizes — not single-digit (treat as unknown/big)
        assert!(!is_single_digit_b_name("phi-3.5-mini-3.8B"));
        assert!(!is_single_digit_b_name("Qwen2.5-1.5B"));

        // Unknown names — no match → treated as big
        assert!(!is_single_digit_b_name("MiniMax-M2.5-Q4_K_M"));
        assert!(!is_single_digit_b_name("Qwen3-Coder-Next-Q4_K_M"));
        assert!(!is_single_digit_b_name(""));

        // BF16 / FP16 substrings must not trigger
        assert!(!is_single_digit_b_name("some-model-BF16"));
        assert!(!is_single_digit_b_name("some-model-fp16"));

        // Digit-B embedded with later digits (versions) must not trigger
        assert!(!is_single_digit_b_name("foo-2b1-bar")); // 2b followed by 1
    }

    #[test]
    fn test_pick_prefers_multi_digit_over_single_digit() {
        use crate::models::ModelCapabilities;

        let no_caps = ModelCapabilities::default();
        let available = vec![
            RoutingCandidate::unscored("Qwen3.5-2B-Q4_K_M", no_caps),
            RoutingCandidate::unscored("Qwen3.5-9B-Q4_K_M", no_caps),
            RoutingCandidate::unscored("gemma-4-31B-it-Q8_0", no_caps),
            RoutingCandidate::unscored("Qwen3.6-35B-A3B-BF16", no_caps),
            RoutingCandidate::unscored("MiniMax-M2.5-Q4_K_M", no_caps),
            RoutingCandidate::unscored("Qwen3-Coder-Next-Q4_K_M", no_caps),
        ];
        let cl = Classification {
            needs_tools: false,
            has_media_inputs: false,
        };

        let smalls = ["Qwen3.5-2B-Q4_K_M", "Qwen3.5-9B-Q4_K_M"];
        // Across many picks, small-tier names must never win when big-tier is non-empty.
        for _ in 0..200 {
            let picked = pick_model_classified(&cl, &available).expect("some pick");
            assert!(
                !smalls.contains(&picked),
                "small-tier model {picked} was picked despite a non-empty big tier"
            );
        }
    }

    #[test]
    fn test_pick_uses_parameter_metadata_before_name_heuristic() {
        use crate::models::ModelCapabilities;

        let no_caps = ModelCapabilities::default();
        let mut misleading_big_name = RoutingCandidate::unscored("unknown-strong-name", no_caps);
        misleading_big_name.parameter_count_b = Some(7.0);
        let mut misleading_small_name = RoutingCandidate::unscored("tiny-looking-7B", no_caps);
        misleading_small_name.parameter_count_b = Some(32.0);
        let available = vec![misleading_big_name, misleading_small_name];
        let cl = Classification {
            needs_tools: false,
            has_media_inputs: false,
        };

        for _ in 0..100 {
            let picked = pick_model_classified(&cl, &available).expect("some pick");
            assert_eq!(picked, "tiny-looking-7B");
        }
    }

    #[test]
    fn test_pick_falls_back_to_small_when_no_big_tier() {
        use crate::models::ModelCapabilities;

        let no_caps = ModelCapabilities::default();
        let available = vec![
            RoutingCandidate::unscored("Qwen3.5-2B-Q4_K_M", no_caps),
            RoutingCandidate::unscored("Qwen3.5-9B-Q4_K_M", no_caps),
        ];
        let cl = Classification {
            needs_tools: false,
            has_media_inputs: false,
        };

        let picked = pick_model_classified(&cl, &available).expect("some pick");
        assert!(picked == "Qwen3.5-2B-Q4_K_M" || picked == "Qwen3.5-9B-Q4_K_M");
    }

    #[test]
    fn test_pick_spreads_across_big_tier() {
        use crate::models::ModelCapabilities;
        use std::collections::HashSet;

        let no_caps = ModelCapabilities::default();
        let available = vec![
            RoutingCandidate::unscored("gemma-4-31B-it-Q8_0", no_caps),
            RoutingCandidate::unscored("Qwen3.6-35B-A3B-BF16", no_caps),
            RoutingCandidate::unscored("MiniMax-M2.5-Q4_K_M", no_caps),
            RoutingCandidate::unscored("Qwen3-Coder-Next-Q4_K_M", no_caps),
        ];
        let cl = Classification {
            needs_tools: false,
            has_media_inputs: false,
        };

        let mut seen = HashSet::new();
        for _ in 0..500 {
            if let Some(m) = pick_model_classified(&cl, &available) {
                seen.insert(m);
            }
            // Sleep a nanosecond-scale amount so the seed changes between iterations
            std::thread::sleep(std::time::Duration::from_nanos(1));
        }
        // Over 500 picks with nanosecond-seeded shuffles, we should see
        // at least 3 of the 4 big-tier models. (Allowing 1 slop for the
        // rare case where timing quantization biases the seed.)
        assert!(
            seen.len() >= 3,
            "expected spread across big-tier models, only saw {seen:?}"
        );
    }

    // ── tok/s-aware weighting ────────────────────────────────────────

    /// Helper: build a candidate with one measured placement.
    fn scored<'a>(
        name: &'a str,
        caps: crate::models::ModelCapabilities,
        tps: f64,
        samples: u64,
    ) -> RoutingCandidate<'a> {
        RoutingCandidate {
            name,
            caps,
            parameter_count_b: None,
            replicas: 1,
            capacity: placement_weight(Some(tps), samples),
        }
    }

    /// Helper: `hosts` equal-speed placements of one model.
    fn replicated<'a>(name: &'a str, hosts: usize, tps: f64) -> RoutingCandidate<'a> {
        let mut candidate = scored(name, crate::models::ModelCapabilities::default(), tps, 10);
        candidate.replicas = hosts;
        candidate.capacity = placement_weight(Some(tps), 10) * hosts as f64;
        candidate
    }

    #[test]
    fn replica_count_multiplies_a_models_share() {
        // Same per-host speed and tier: the model served by four hosts should
        // be drawn about four times as often as the one on a single host, on
        // both the capacity draw and the (placement-uniform) exploration draw.
        let counts = count_picks(
            &[
                replicated("replicated-70b", 4, 20.0),
                replicated("lone-70b", 1, 20.0),
            ],
            4000,
        );
        let replicated = counts.get("replicated-70b") as f64;
        let lone = counts.get("lone-70b") as f64;
        let share = replicated / (replicated + lone);
        assert!(
            (0.76..0.84).contains(&share),
            "expected ~80% of picks for the replicated model, got {replicated}:{lone}"
        );
    }

    #[test]
    fn strongly_skewed_replicas_keep_per_host_traffic_even() {
        // Twenty equal-speed replicas against one lone host: each host should
        // see about the same traffic (1/21 each), so the lone host is not
        // handed a multiple of a replica's load by name-uniform exploration.
        let counts = count_picks(
            &[
                replicated("fleet-70b", 20, 20.0),
                replicated("lone-70b", 1, 20.0),
            ],
            6300,
        );
        let fleet = counts.get("fleet-70b") as f64;
        let lone = counts.get("lone-70b") as f64;
        let per_fleet_host = fleet / 20.0;
        let ratio = lone / per_fleet_host;
        assert!(
            (0.6..1.5).contains(&ratio),
            "lone host should get ~1x a replica's traffic, got {ratio:.2}x (lone={lone}, fleet={fleet})"
        );
    }

    #[test]
    fn locality_factor_is_a_bounded_tiebreak() {
        assert_eq!(locality_factor(None), 1.0);
        assert_eq!(locality_factor(Some(0)), 1.0);
        assert_eq!(locality_factor(Some(50)), 1.0);
        assert!((locality_factor(Some(175)) - 0.75).abs() < 1e-9);
        assert_eq!(locality_factor(Some(300)), 0.5);
        assert_eq!(locality_factor(Some(5000)), 0.5);
    }

    fn count_picks(available: &[RoutingCandidate<'_>], iterations: usize) -> HashMapCounts {
        use std::collections::HashMap;
        let cl = Classification {
            needs_tools: false,
            has_media_inputs: false,
        };
        let mut counts: HashMap<String, usize> = HashMap::new();
        for _ in 0..iterations {
            if let Some(name) = pick_model_classified(&cl, available) {
                *counts.entry(name.to_string()).or_insert(0) += 1;
            }
            // Bump the nanosecond seed between iterations.
            std::thread::sleep(std::time::Duration::from_nanos(1));
        }
        HashMapCounts(counts)
    }

    struct HashMapCounts(std::collections::HashMap<String, usize>);

    impl HashMapCounts {
        fn get(&self, name: &str) -> usize {
            self.0.get(name).copied().unwrap_or(0)
        }
        fn total(&self) -> usize {
            self.0.values().sum()
        }
    }

    #[test]
    fn weighted_pick_all_cold_is_roughly_uniform() {
        // Backwards-compat: when nothing has tps data, picks should be
        // roughly uniform — same effective shape as the old random shuffle.
        use crate::models::ModelCapabilities;
        let no_caps = ModelCapabilities::default();
        let available = vec![
            RoutingCandidate::unscored("alpha-31B", no_caps),
            RoutingCandidate::unscored("beta-31B", no_caps),
            RoutingCandidate::unscored("gamma-31B", no_caps),
        ];
        let counts = count_picks(&available, 600);
        let expected = counts.total() / 3;
        // Allow ±50% (300 picks across 3 models is loose statistical ground,
        // but we just need to see no model is starved).
        for name in ["alpha-31B", "beta-31B", "gamma-31B"] {
            let got = counts.get(name);
            assert!(
                got > expected / 2,
                "cold model {name} was starved: {got}/{expected} expected"
            );
        }
    }

    #[test]
    fn weighted_pick_fast_wins_majority_but_slow_still_gets_some() {
        // Core design claim: fast tok/s tilts routing without starving the slow.
        use crate::models::ModelCapabilities;
        let no_caps = ModelCapabilities::default();
        let available = vec![
            scored("fast-31B", no_caps, 80.0, 50),
            scored("slow-31B", no_caps, 6.0, 50),
        ];
        let counts = count_picks(&available, 600);
        let fast = counts.get("fast-31B");
        let slow = counts.get("slow-31B");
        // Fast should win clearly more often.
        assert!(
            fast > slow,
            "fast tok/s model should win majority: fast={fast} slow={slow}"
        );
        // Fast wins by a meaningful margin (≥ 1.5x).
        assert!(
            fast as f64 > 1.5 * slow as f64,
            "fast model should win by at least 1.5x: fast={fast} slow={slow}",
        );
        // Slow model still gets meaningful traffic (exploration + clamp keep it alive).
        assert!(
            slow > 30,
            "slow model must not be starved (exploration keeps it alive): got {slow}",
        );
    }

    #[test]
    fn weighted_pick_cold_model_competes_with_hot_fast() {
        // A brand-new peer (no samples) must still get meaningful traffic
        // against an established fast peer — otherwise it can never
        // accumulate the data it needs to be scored.
        use crate::models::ModelCapabilities;
        let no_caps = ModelCapabilities::default();
        let available = vec![
            scored("hot-fast-31B", no_caps, 80.0, 50),
            RoutingCandidate::unscored("cold-newcomer-31B", no_caps),
        ];
        let counts = count_picks(&available, 600);
        let cold = counts.get("cold-newcomer-31B");
        // Cold gets NEUTRAL_WEIGHT (25) vs hot's clamped 80 — so cold
        // should still see at least a healthy minority of traffic.
        assert!(
            cold > 100,
            "cold newcomer must get fair traffic to accumulate samples: got {cold}/600"
        );
    }

    #[test]
    fn weighted_pick_low_sample_count_treated_as_cold() {
        // A model with only 1-2 samples shouldn't have those samples
        // dominate routing — we want a few real measurements before tps
        // participates.
        use crate::models::ModelCapabilities;
        let no_caps = ModelCapabilities::default();
        let available = vec![
            // Both are 31B "big-tier" names — single-digit-B partition
            // doesn't separate them.
            scored("alpha-31B", no_caps, 100.0, 1), // 1 sample of "fast" — should be ignored
            scored("beta-31B", no_caps, 100.0, 1),
            scored("gamma-31B", no_caps, 100.0, 1),
        ];
        let counts = count_picks(&available, 600);
        // All three should land near uniform since none has enough samples.
        let expected = counts.total() / 3;
        for name in ["alpha-31B", "beta-31B", "gamma-31B"] {
            let got = counts.get(name);
            assert!(
                got > expected / 2,
                "low-sample model {name} was treated as scored instead of cold: {got}"
            );
        }
    }

    #[test]
    fn candidate_weight_clamps_extremes() {
        // Sanity: weight stays bounded so no peer can fully starve or monopolize.
        use crate::models::ModelCapabilities;
        let no_caps = ModelCapabilities::default();

        let glacial = scored("glacial", no_caps, 0.5, 100);
        let blazing = scored("blazing", no_caps, 500.0, 100);
        let cold = RoutingCandidate::unscored("cold", no_caps);

        let wg = candidate_weight(&glacial);
        let wb = candidate_weight(&blazing);
        let wc = candidate_weight(&cold);

        assert!(wg >= TPS_WEIGHT_MIN, "glacial weight floored: {wg}");
        assert!(wb <= TPS_WEIGHT_MAX, "blazing weight capped: {wb}");
        assert!(
            (wc - TPS_NEUTRAL_WEIGHT).abs() < f64::EPSILON,
            "cold weight should be neutral: {wc}",
        );
    }
}
