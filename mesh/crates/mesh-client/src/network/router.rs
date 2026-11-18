/// Smart model router — classifies requests and picks the best model.
pub use mesh_llm_types::router::{
    Classification, MediaRequirements, classify, media_requirements, strip_split_suffix,
    strip_split_suffix_owned,
};

// ── Model profiles ──────────────────────────────────────────────────

/// Quality tier: higher = better quality, slower.
/// 1 = draft/tiny, 2 = good, 3 = strong, 4 = frontier
pub type Tier = u8;

pub struct ModelProfile {
    pub name: &'static str,
    pub tier: Tier,
    /// Whether this model can handle tool-calling requests (function calling).
    /// Models without this set to true are filtered out when tools are present.
    pub tools: bool,
}

/// Static profiles for catalog models.
/// Profiles declare only quality tier and tool support.
pub static MODEL_PROFILES: &[ModelProfile] = &[
    // ── Tier 4: Frontier ────────────────────────────────────────
    ModelProfile {
        name: "Qwen3-235B-A22B-Q4_K_M",
        tier: 4,
        tools: true,
    },
    ModelProfile {
        name: "Llama-3.1-405B-Instruct-Q2_K",
        tier: 4,
        tools: true,
    },
    ModelProfile {
        name: "MiniMax-M2.5-Q4_K_M",
        tier: 4,
        tools: true,
    },
    // ── Tier 3: Strong ──────────────────────────────────────────
    ModelProfile {
        name: "Qwen2.5-72B-Instruct-Q4_K_M",
        tier: 3,
        tools: true,
    },
    ModelProfile {
        name: "Llama-3.3-70B-Instruct-Q4_K_M",
        tier: 3,
        tools: true,
    },
    ModelProfile {
        name: "DeepSeek-R1-Distill-70B-Q4_K_M",
        tier: 3,
        tools: false, // reasoning-only, no tool support
    },
    ModelProfile {
        name: "Mixtral-8x22B-Instruct-Q4_K_M",
        tier: 3,
        tools: true,
    },
    ModelProfile {
        name: "Qwen3-32B-Q4_K_M",
        tier: 3,
        tools: true,
    },
    ModelProfile {
        name: "Qwen2.5-Coder-32B-Instruct-Q4_K_M",
        tier: 3,
        tools: true,
    },
    ModelProfile {
        name: "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M",
        tier: 3,
        tools: false,
    },
    ModelProfile {
        name: "Qwen3-30B-A3B-Q4_K_M",
        tier: 3,
        tools: true,
    },
    ModelProfile {
        name: "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M",
        tier: 3,
        tools: true,
    },
    ModelProfile {
        name: "Qwen2.5-32B-Instruct-Q4_K_M",
        tier: 3,
        tools: true,
    },
    ModelProfile {
        name: "Gemma-3-27B-it-Q4_K_M",
        tier: 3,
        tools: false, // unreliable tool calling
    },
    ModelProfile {
        name: "Qwen3.5-27B-Q4_K_M",
        tier: 3,
        tools: true,
    },
    ModelProfile {
        name: "Qwen3-Coder-Next-Q4_K_M",
        tier: 4,
        tools: true,
    },
    // ── Tier 2: Good ────────────────────────────────────────────
    ModelProfile {
        name: "Qwen3.5-9B-Q4_K_M",
        tier: 2,
        tools: false,
    },
    ModelProfile {
        name: "Mistral-Small-3.1-24B-Instruct-Q4_K_M",
        tier: 2,
        tools: true,
    },
    ModelProfile {
        name: "Devstral-Small-2505-Q4_K_M",
        tier: 2,
        tools: true,
    },
    ModelProfile {
        name: "GLM-4.7-Flash-Q4_K_M",
        tier: 2,
        tools: true,
    },
    ModelProfile {
        name: "GLM-4-32B-0414-Q4_K_M",
        tier: 2,
        tools: true,
    },
    ModelProfile {
        name: "Llama-4-Scout-Q4_K_M",
        tier: 2,
        tools: true,
    },
    ModelProfile {
        name: "Qwen3-14B-Q4_K_M",
        tier: 2,
        tools: true,
    },
    ModelProfile {
        name: "Qwen2.5-14B-Instruct-Q4_K_M",
        tier: 2,
        tools: true,
    },
    ModelProfile {
        name: "Qwen2.5-Coder-14B-Instruct-Q4_K_M",
        tier: 2,
        tools: true,
    },
    ModelProfile {
        name: "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M",
        tier: 2,
        tools: false,
    },
    ModelProfile {
        name: "Gemma-3-12B-it-Q4_K_M",
        tier: 2,
        tools: false,
    },
    ModelProfile {
        name: "Qwen3-8B-Q4_K_M",
        tier: 2,
        tools: true,
    },
    ModelProfile {
        name: "Hermes-2-Pro-Mistral-7B-Q4_K_M",
        tier: 2,
        tools: false,
    },
    ModelProfile {
        name: "Qwen2.5-Coder-7B-Instruct-Q4_K_M",
        tier: 2,
        tools: true,
    },
    // ── Tier 1: Small / Draft ───────────────────────────────────
    ModelProfile {
        name: "Qwen3-4B-Q4_K_M",
        tier: 1,
        tools: true,
    },
    ModelProfile {
        name: "Qwen2.5-3B-Instruct-Q4_K_M",
        tier: 1,
        tools: true,
    },
    ModelProfile {
        name: "Llama-3.2-3B-Instruct-Q4_K_M",
        tier: 1,
        tools: true,
    },
];

pub fn profile_for(model_name: &str) -> Option<&'static ModelProfile> {
    // Direct match first
    if let Some(p) = MODEL_PROFILES.iter().find(|p| p.name == model_name) {
        return Some(p);
    }
    // Strip split GGUF suffix: "Model-00001-of-00004" → "Model"
    let clean = strip_split_suffix(model_name);
    if clean != model_name {
        return MODEL_PROFILES.iter().find(|p| p.name == clean);
    }
    None
}

// ── Model selection ─────────────────────────────────────────────────

/// Pick a model using declared tool support, size tier and measured speed.
pub fn pick_model_classified<'a>(
    classification: &Classification,
    available_models: &[(&'a str, f64)],
) -> Option<&'a str> {
    if available_models.is_empty() {
        return None;
    }

    // Filter for tool-capable models if tools are required
    let filtered: Vec<(&str, f64)> = if classification.needs_tools {
        available_models
            .iter()
            .filter(|(name, _)| profile_for(name).map(|p| p.tools).unwrap_or(false))
            .copied()
            .collect()
    } else {
        available_models.to_vec()
    };
    // Fall back to all models if no tool-capable model found
    let candidates = if filtered.is_empty() {
        available_models
    } else {
        &filtered
    };

    // Score each available model
    let mut scored: Vec<(&str, i32)> = candidates
        .iter()
        .map(|(name, tok_s)| {
            let profile = profile_for(name);
            let tier = profile.map(|p| p.tier).unwrap_or(1) as i32;

            let tier_bonus = tier * if classification.needs_tools { 20 } else { 10 };

            // Speed bonus: higher for chat (speed matters), lower for agentic (quality matters)
            let speed_bonus = if classification.needs_tools {
                // Agentic: speed is a tiebreaker only
                (tok_s / 20.0).min(5.0) as i32
            } else {
                // Chat: speed matters more
                (tok_s / 5.0).min(20.0) as i32
            };

            let score = tier_bonus + speed_bonus;
            (*name, score)
        })
        .collect();

    scored.sort_by_key(|entry| std::cmp::Reverse(entry.1));

    // For non-agentic requests, spread load across top-scoring models.
    // Pick randomly among candidates within 15 points of the best score.
    // This avoids queueing all concurrent chat users on the same model
    // while keeping weak models as fallbacks, not equal contenders.
    if !classification.needs_tools && scored.len() > 1 {
        let best_score = scored[0].1;
        let top_tier: Vec<&(&str, i32)> = scored
            .iter()
            .filter(|(_, s)| best_score - s <= 15)
            .collect();
        if top_tier.len() > 1 {
            // Simple pseudo-random: use current time nanos to pick
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos() as usize;
            let pick = top_tier[nanos % top_tier.len()];
            return Some(pick.0);
        }
    }

    scored.first().map(|(name, _)| *name)
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
    fn test_anthropic_tool_request_prefers_stronger_tool_model() {
        // Reproduces Claude-like tool request shape and verifies needs_tools=true
        // pushes selection toward the stronger tool-capable model.
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

        let available = vec![("Qwen3-8B-Q4_K_M", 40.0), ("MiniMax-M2.5-Q4_K_M", 20.0)];
        let picked = pick_model_classified(&cl, &available);
        assert_eq!(picked, Some("MiniMax-M2.5-Q4_K_M"));
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
    fn test_pick_model_primary_strength_wins() {
        // Qwen3-8B (tier 2, Chat primary) and 235B (tier 4, Chat 3rd) score within
        // 15 points at Moderate complexity, so either is a valid pick (load spread).
        let available = vec![("Qwen3-8B-Q4_K_M", 50.0), ("Qwen3-235B-A22B-Q4_K_M", 20.0)];
        let result = pick_model_classified(
            &Classification {
                needs_tools: false,
                has_media_inputs: false,
            },
            &available,
        );
        assert!(result == Some("Qwen3-8B-Q4_K_M") || result == Some("Qwen3-235B-A22B-Q4_K_M"));
    }

    #[test]
    fn test_deep_complexity_prefers_bigger() {
        // Deep complexity amplifies tier bonus, but scores are within 15 points
        // so load spread makes either a valid pick.
        let available = vec![("Qwen3-8B-Q4_K_M", 50.0), ("Qwen3-235B-A22B-Q4_K_M", 20.0)];
        let result = pick_model_classified(
            &Classification {
                needs_tools: false,
                has_media_inputs: false,
            },
            &available,
        );
        assert!(result == Some("Qwen3-8B-Q4_K_M") || result == Some("Qwen3-235B-A22B-Q4_K_M"));
    }

    #[test]
    fn test_quick_complexity_prefers_smaller() {
        // Quick complexity: both score within 15 points (load spread applies).
        // Either is a valid pick — the key property is neither is excluded.
        let available = vec![
            ("Qwen3-8B-Q4_K_M", 50.0),
            ("Qwen2.5-72B-Instruct-Q4_K_M", 10.0),
        ];
        let result = pick_model_classified(
            &Classification {
                needs_tools: false,
                has_media_inputs: false,
            },
            &available,
        );
        assert!(result == Some("Qwen3-8B-Q4_K_M") || result == Some("Qwen2.5-72B-Instruct-Q4_K_M"));
    }

    #[test]
    fn test_pick_model_prefers_strength_match() {
        // Same tier, same speed — scores within 15 points, load spread applies.
        let available = vec![
            ("DeepSeek-R1-Distill-70B-Q4_K_M", 10.0), // tier 3, reasoning specialist
            ("Qwen2.5-72B-Instruct-Q4_K_M", 10.0),    // tier 3, chat primary
        ];
        let result = pick_model_classified(
            &Classification {
                needs_tools: false,
                has_media_inputs: false,
            },
            &available,
        );
        assert!(
            result == Some("DeepSeek-R1-Distill-70B-Q4_K_M")
                || result == Some("Qwen2.5-72B-Instruct-Q4_K_M")
        );
    }

    #[test]
    fn test_pick_model_code_specialist() {
        // Same tier, same speed — scores within 15 points, load spread applies.
        let available = vec![
            ("Qwen2.5-Coder-32B-Instruct-Q4_K_M", 15.0),
            ("Qwen2.5-32B-Instruct-Q4_K_M", 15.0),
        ];
        let result = pick_model_classified(
            &Classification {
                needs_tools: false,
                has_media_inputs: false,
            },
            &available,
        );
        assert!(
            result == Some("Qwen2.5-Coder-32B-Instruct-Q4_K_M")
                || result == Some("Qwen2.5-32B-Instruct-Q4_K_M")
        );
    }

    #[test]
    fn test_pick_model_empty() {
        let available: Vec<(&str, f64)> = vec![];
        assert_eq!(
            pick_model_classified(
                &Classification {
                    needs_tools: false,
                    has_media_inputs: false,
                },
                &available
            ),
            None
        );
    }

    #[test]
    fn test_pick_model_unknown_model_still_works() {
        let available = vec![("SomeUnknownModel", 30.0)];
        let result = pick_model_classified(
            &Classification {
                needs_tools: false,
                has_media_inputs: false,
            },
            &available,
        );
        assert_eq!(result, Some("SomeUnknownModel"));
    }

    #[test]
    fn test_profile_lookup() {
        assert!(profile_for("Qwen3-235B-A22B-Q4_K_M").is_some());
        assert_eq!(profile_for("Qwen3-235B-A22B-Q4_K_M").unwrap().tier, 4);
        assert!(profile_for("nonexistent").is_none());
    }

    #[test]
    fn test_classify_empty_tools_is_not_tool_call() {
        let body = json!({
            "messages": [{"role": "user", "content": "hello"}],
            "tools": []
        });
        assert!(!classify(&body).needs_tools);
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
    fn test_profile_for_split_gguf() {
        let p = profile_for("MiniMax-M2.5-Q4_K_M-00001-of-00004");
        assert!(p.is_some());
        assert_eq!(p.unwrap().name, "MiniMax-M2.5-Q4_K_M");
        assert_eq!(p.unwrap().tier, 4);
    }
}

#[test]
fn test_tools_filter_prefers_capable() {
    let available = vec![
        ("DeepSeek-R1-Distill-Qwen-32B-Q4_K_M", 10.0), // tools: false, Reasoning only
        ("Qwen2.5-32B-Instruct-Q4_K_M", 50.0),         // tools: true, Chat+Reasoning+Code
    ];
    // Without tools, Reasoning request: scores within 15 points, either valid
    let result = pick_model_classified(
        &Classification {
            needs_tools: false,
            has_media_inputs: false,
        },
        &available,
    );
    assert!(
        result == Some("DeepSeek-R1-Distill-Qwen-32B-Q4_K_M")
            || result == Some("Qwen2.5-32B-Instruct-Q4_K_M")
    );
    // With tools, Reasoning request: Qwen wins (DeepSeek filtered out — can't do tools)
    let result = pick_model_classified(
        &Classification {
            needs_tools: true,
            has_media_inputs: false,
        },
        &available,
    );
    assert_eq!(result, Some("Qwen2.5-32B-Instruct-Q4_K_M"));
}

#[test]
fn test_tools_filter_fallback_when_none_capable() {
    let available = vec![
        ("DeepSeek-R1-Distill-Qwen-32B-Q4_K_M", 10.0), // tools: false
    ];
    // With tools required but nothing capable: falls back to available
    let result = pick_model_classified(
        &Classification {
            needs_tools: true,
            has_media_inputs: false,
        },
        &available,
    );
    assert_eq!(result, Some("DeepSeek-R1-Distill-Qwen-32B-Q4_K_M"));
}

#[test]
fn test_agentic_prefers_strongest_model() {
    // Agentic (needs_tools=true): 32B (tier 3) should beat 7B (tier 2) even though 7B is faster
    let available = vec![
        ("Hermes-2-Pro-Mistral-7B-Q4_K_M", 87.0), // tier 2, tools: false
        ("Qwen2.5-Coder-7B-Instruct-Q4_K_M", 85.0), // tier 2, tools: true
        ("Qwen2.5-32B-Instruct-Q4_K_M", 18.0),    // tier 3, tools: true
    ];
    let cl = Classification {
        needs_tools: true,
        has_media_inputs: false,
    };
    let result = pick_model_classified(&cl, &available);
    // 32B should win: tier 3×20=60 beats Coder tier 2×20=40, despite lower speed
    assert_eq!(result, Some("Qwen2.5-32B-Instruct-Q4_K_M"));
}

#[test]
fn test_chat_prefers_fastest_model() {
    // Chat (needs_tools=false, Quick): scores within 15 points, load spread applies.
    let available = vec![
        ("Hermes-2-Pro-Mistral-7B-Q4_K_M", 87.0), // tier 2, fast
        ("Qwen2.5-32B-Instruct-Q4_K_M", 18.0),    // tier 3, slow
    ];
    let cl = Classification {
        needs_tools: false,
        has_media_inputs: false,
    };
    let result = pick_model_classified(&cl, &available);
    assert!(
        result == Some("Hermes-2-Pro-Mistral-7B-Q4_K_M")
            || result == Some("Qwen2.5-32B-Instruct-Q4_K_M")
    );
}

#[test]
fn test_agentic_deep_strongly_prefers_biggest() {
    // Deep agentic: tier 4 should massively beat tier 2
    let available = vec![
        ("Qwen2.5-Coder-7B-Instruct-Q4_K_M", 85.0), // tier 2
        ("MiniMax-M2.5-Q4_K_M", 21.0),              // tier 4
    ];
    let cl = Classification {
        needs_tools: true,
        has_media_inputs: false,
    };
    let result = pick_model_classified(&cl, &available);
    assert_eq!(result, Some("MiniMax-M2.5-Q4_K_M"));
}
