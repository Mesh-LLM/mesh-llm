use crate::{
    ExactStateMobility, FamilyCapabilityRecord, LayerRange, LayerSpec, ReviewedCapabilityRecord,
    SidebandKind, SidebandRequirement, SplitConstraint, SplitConstraintKind,
    StageRuntimeFamilyExpectation,
};

pub const STAGE_RUNTIME_LLAMA_FAMILY_EXPECTATIONS: &[StageRuntimeFamilyExpectation] = &[
    StageRuntimeFamilyExpectation {
        llama_architecture: "baichuan",
        family_id: "baichuan",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "bloom",
        family_id: "bloom",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "cohere2",
        family_id: "cohere2",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "command-r",
        family_id: "command_r",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "deepseek2",
        family_id: "deepseek2",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "deepseek2-ocr",
        family_id: "deepseek2ocr",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "exaone",
        family_id: "exaone",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "exaone4",
        family_id: "exaone4",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "falcon",
        family_id: "falcon",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "falcon-h1",
        family_id: "falcon_h1",
        recurrent_or_hybrid: true,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "gemma",
        family_id: "gemma",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "gemma2",
        family_id: "gemma2",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "gemma3",
        family_id: "gemma3",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "gemma3n",
        family_id: "gemma3n",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "gemma4",
        family_id: "gemma4",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "glm4",
        family_id: "glm4",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "gpt2",
        family_id: "gpt2",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "gptneox",
        family_id: "gptneox",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "granite",
        family_id: "granite",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "granitehybrid",
        family_id: "granite_hybrid",
        recurrent_or_hybrid: true,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "granitemoe",
        family_id: "granite_moe",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "hunyuan-dense",
        family_id: "hunyuan_dense",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "hunyuan-moe",
        family_id: "hunyuan_moe",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "hunyuan-vl",
        family_id: "hunyuan_vl",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "inkling",
        family_id: "inkling",
        recurrent_or_hybrid: true,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "internlm2",
        family_id: "internlm2",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "jais",
        family_id: "jais",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "jais2",
        family_id: "jais2",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "jamba",
        family_id: "jamba",
        recurrent_or_hybrid: true,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "laguna",
        family_id: "laguna",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "lfm2",
        family_id: "lfm2",
        recurrent_or_hybrid: true,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "llama",
        family_id: "llama",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "mamba",
        family_id: "mamba",
        recurrent_or_hybrid: true,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "mamba2",
        family_id: "mamba2",
        recurrent_or_hybrid: true,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "minimax-m2",
        family_id: "minimax_m27",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "mistral3",
        family_id: "mistral",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "mpt",
        family_id: "mpt",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "olmo",
        family_id: "olmo",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "olmo2",
        family_id: "olmo2",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "olmoe",
        family_id: "olmoe",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "phi2",
        family_id: "phi2",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "phi3",
        family_id: "phi",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "phimoe",
        family_id: "phimoe",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "qwen2",
        family_id: "qwen2",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "qwen2moe",
        family_id: "qwen2moe",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "qwen2vl",
        family_id: "qwen2vl",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "qwen3",
        family_id: "qwen3_dense",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "qwen3moe",
        family_id: "qwen3moe",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "qwen3next",
        family_id: "qwen3next",
        recurrent_or_hybrid: true,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "qwen4exp",
        family_id: "qwen4exp",
        recurrent_or_hybrid: true,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "qwen3vl",
        family_id: "qwen3vl",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "qwen3vlmoe",
        family_id: "qwen3vlmoe",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "qwen35",
        family_id: "qwen35",
        recurrent_or_hybrid: true,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "qwen35moe",
        family_id: "qwen35moe",
        recurrent_or_hybrid: true,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "rwkv6",
        family_id: "rwkv6",
        recurrent_or_hybrid: true,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "rwkv7",
        family_id: "rwkv7",
        recurrent_or_hybrid: true,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "arwkv7",
        family_id: "rwkv7",
        recurrent_or_hybrid: true,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "afmoe",
        family_id: "afmoe",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "apertus",
        family_id: "apertus",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "arcee",
        family_id: "arcee",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "arctic",
        family_id: "arctic",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "bailingmoe",
        family_id: "bailingmoe",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "bailingmoe2",
        family_id: "bailingmoe2",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "bitnet",
        family_id: "bitnet",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "chatglm",
        family_id: "chatglm",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "codeshell",
        family_id: "codeshell",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "dbrx",
        family_id: "dbrx",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "deci",
        family_id: "deci",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "dots1",
        family_id: "dots1",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "dream",
        family_id: "dream",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "ernie4-5",
        family_id: "ernie4_5",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "ernie4-5-moe",
        family_id: "ernie4_5_moe",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "exaone-moe",
        family_id: "exaone_moe",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "grok",
        family_id: "grok",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "grovemoe",
        family_id: "grovemoe",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "kimi-linear",
        family_id: "kimi_linear",
        recurrent_or_hybrid: true,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "lfm2moe",
        family_id: "lfm2moe",
        recurrent_or_hybrid: true,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "llada",
        family_id: "llada",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "llada-moe",
        family_id: "llada_moe",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "maincoder",
        family_id: "maincoder",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "mimo2",
        family_id: "mimo2",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "minicpm",
        family_id: "minicpm",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "minicpm3",
        family_id: "minicpm3",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "nemotron",
        family_id: "nemotron",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "nemotron-h",
        family_id: "nemotron_h",
        recurrent_or_hybrid: true,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "nemotron-h-moe",
        family_id: "nemotron_h_moe",
        recurrent_or_hybrid: true,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "openai-moe",
        family_id: "openai_moe",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "openelm",
        family_id: "openelm",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "orion",
        family_id: "orion",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "plamo",
        family_id: "plamo",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "plamo2",
        family_id: "plamo2",
        recurrent_or_hybrid: true,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "plamo3",
        family_id: "plamo3",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "plm",
        family_id: "plm",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "qwen",
        family_id: "qwen",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "refact",
        family_id: "refact",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "rnd1",
        family_id: "rnd1",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "rwkv6qwen2",
        family_id: "rwkv6qwen2",
        recurrent_or_hybrid: true,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "seed-oss",
        family_id: "seed_oss",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "smallthinker",
        family_id: "smallthinker",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "smollm3",
        family_id: "smollm3",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "starcoder",
        family_id: "starcoder",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "step35",
        family_id: "step35",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "xverse",
        family_id: "xverse",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "stablelm",
        family_id: "stablelm",
        recurrent_or_hybrid: false,
    },
    StageRuntimeFamilyExpectation {
        llama_architecture: "starcoder2",
        family_id: "starcoder2",
        recurrent_or_hybrid: false,
    },
];
pub fn qwen3_dense_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    dense_family_capability(
        "qwen3_dense",
        layer_count,
        activation_width,
        ExactStateMobility::Accepted,
    )
}

pub fn qwen2moe_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    dense_family_capability(
        "qwen2moe",
        layer_count,
        activation_width,
        ExactStateMobility::Accepted,
    )
}

pub fn qwen3moe_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    dense_family_capability(
        "qwen3moe",
        layer_count,
        activation_width,
        ExactStateMobility::Accepted,
    )
}

pub fn laguna_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    dense_family_capability(
        "laguna",
        layer_count,
        activation_width,
        ExactStateMobility::Untested,
    )
}

pub fn dense_family_capability(
    family_id: impl Into<String>,
    layer_count: u32,
    activation_width: u32,
    exact_state_mobility: ExactStateMobility,
) -> FamilyCapabilityRecord {
    FamilyCapabilityRecord {
        family_id: family_id.into(),
        layer_count,
        activation_width,
        exact_state_mobility,
        recurrent_ranges: Vec::new(),
        split_constraints: Vec::new(),
        sidebands: Vec::new(),
    }
}

pub fn llama_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    dense_family_capability(
        "llama",
        layer_count,
        activation_width,
        ExactStateMobility::Untested,
    )
}

pub fn deepseek2_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    dense_family_capability(
        "deepseek2",
        layer_count,
        activation_width,
        ExactStateMobility::Untested,
    )
}

pub fn deepseek2ocr_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    dense_family_capability(
        "deepseek2ocr",
        layer_count,
        activation_width,
        ExactStateMobility::Accepted,
    )
}

pub fn deepseek3_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    dense_family_capability(
        "deepseek3",
        layer_count,
        activation_width,
        ExactStateMobility::Untested,
    )
}

pub fn glm47_flash_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    dense_family_capability(
        "glm47_flash",
        layer_count,
        activation_width,
        ExactStateMobility::Untested,
    )
}

pub fn glm4_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    dense_family_capability(
        "glm4",
        layer_count,
        activation_width,
        ExactStateMobility::Accepted,
    )
}

pub fn gemma2_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    dense_family_capability(
        "gemma2",
        layer_count,
        activation_width,
        ExactStateMobility::Accepted,
    )
}

pub fn gemma3_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    dense_family_capability(
        "gemma3",
        layer_count,
        activation_width,
        ExactStateMobility::Accepted,
    )
}

pub fn gemma3n_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    FamilyCapabilityRecord {
        family_id: "gemma3n".to_string(),
        layer_count,
        activation_width,
        exact_state_mobility: ExactStateMobility::Accepted,
        recurrent_ranges: Vec::new(),
        split_constraints: vec![SplitConstraint {
            kind: SplitConstraintKind::SharedKvProducerConsumer,
            range: LayerRange {
                start: layer_count / 2,
                end: layer_count,
            },
            forbidden_boundaries: vec![layer_count.saturating_mul(2) / 3],
            reject_boundary_inside: false,
            reason: "Gemma3n upper layers reuse KV owned by lower upper-stack layers; keep the final slice start on the reviewed KV-owner boundary unless KV replay or transfer is added".to_string(),
        }],
        sidebands: vec![SidebandRequirement {
            kind: SidebandKind::Gemma3nAltup,
            first_required_layer: layer_count,
            reason: "Gemma3n downstream slices require the full AltUp activation sideband in addition to the boundary hidden state".to_string(),
        }],
    }
}

pub fn gemma4_a4b_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    dense_family_capability(
        "gemma4_a4b",
        layer_count,
        activation_width,
        ExactStateMobility::Untested,
    )
}

pub fn olmo_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    dense_family_capability(
        "olmo",
        layer_count,
        activation_width,
        ExactStateMobility::Untested,
    )
}

pub fn minimax_m27_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    dense_family_capability(
        "minimax_m27",
        layer_count,
        activation_width,
        ExactStateMobility::Accepted,
    )
}

pub fn falcon_h1_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    FamilyCapabilityRecord {
        family_id: "falcon_h1".to_string(),
        layer_count,
        activation_width,
        exact_state_mobility: ExactStateMobility::RejectedTooLarge,
        recurrent_ranges: vec![LayerRange {
            start: 0,
            end: layer_count,
        }],
        split_constraints: Vec::new(),
        sidebands: Vec::new(),
    }
}

/// Capability for the Qwen3.5 series (llama.cpp `qwen35` / `qwen35moe`).
///
/// Qwen3.6 and Qwen3.8 releases share this architecture pair; there is no
/// separate `qwen36` or `qwen38` architecture in llama.cpp. Certified evidence for both
/// `mradermacher/UnifiedReward-Edit-qwen35-4b-i1-GGUF` (`qwen35`) and
/// `unsloth/Qwen3.6-35B-A3B-GGUF` (`qwen35moe`) records state mobility as
/// rejected-too-large, so exact full-state handoff must not be advertised.
pub fn qwen35_series_capability(
    family_id: &str,
    layer_count: u32,
    activation_width: u32,
) -> FamilyCapabilityRecord {
    FamilyCapabilityRecord {
        family_id: family_id.to_string(),
        layer_count,
        activation_width,
        exact_state_mobility: ExactStateMobility::RejectedTooLarge,
        recurrent_ranges: vec![LayerRange {
            start: 0,
            end: layer_count,
        }],
        split_constraints: Vec::new(),
        sidebands: Vec::new(),
    }
}

pub fn inkling_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    FamilyCapabilityRecord {
        family_id: "inkling".to_string(),
        layer_count,
        activation_width,
        exact_state_mobility: ExactStateMobility::RejectedTooLarge,
        recurrent_ranges: vec![LayerRange {
            start: 0,
            end: layer_count,
        }],
        split_constraints: Vec::new(),
        sidebands: Vec::new(),
    }
}

pub fn qwen3next_capability(
    layer_count: u32,
    activation_width: u32,
    recurrent_ranges: Vec<LayerRange>,
) -> FamilyCapabilityRecord {
    FamilyCapabilityRecord {
        family_id: "qwen3next".to_string(),
        layer_count,
        activation_width,
        exact_state_mobility: ExactStateMobility::RejectedTooLarge,
        recurrent_ranges,
        split_constraints: Vec::new(),
        sidebands: Vec::new(),
    }
}

/// Conservative Qwen4 experimental / Qwen3.8 Flash-Next capability.
///
/// QWEN4EXP combines indexed attention with recurrent Gated DeltaNet state.
/// Until per-layer ownership is derived from artifact metadata and certified,
/// keep the complete trunk sticky and reject exact state mobility.
pub fn qwen4exp_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    FamilyCapabilityRecord {
        family_id: "qwen4exp".to_string(),
        layer_count,
        activation_width,
        exact_state_mobility: ExactStateMobility::RejectedTooLarge,
        recurrent_ranges: vec![LayerRange {
            start: 0,
            end: layer_count,
        }],
        split_constraints: Vec::new(),
        sidebands: vec![SidebandRequirement {
            kind: SidebandKind::TokenIds,
            first_required_layer: 1,
            reason: "Qwen4Exp downstream PLE layers require original token ids to compute n-gram rows and preserve token history in attention cache cells".to_string(),
        }],
    }
}

pub fn kimi_linear_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    let mut recurrent_ranges = Vec::new();
    let mut start = 0;
    while start < layer_count {
        let end = start.saturating_add(3).min(layer_count.saturating_sub(1));
        if start < end {
            recurrent_ranges.push(LayerRange { start, end });
        }
        start = start.saturating_add(4);
    }

    FamilyCapabilityRecord {
        family_id: "kimi_linear".to_string(),
        layer_count,
        activation_width,
        exact_state_mobility: ExactStateMobility::RejectedTooLarge,
        recurrent_ranges,
        split_constraints: Vec::new(),
        sidebands: Vec::new(),
    }
}

pub fn recurrent_family_capability(
    family_id: &str,
    layer_count: u32,
    activation_width: u32,
) -> FamilyCapabilityRecord {
    FamilyCapabilityRecord {
        family_id: family_id.to_string(),
        layer_count,
        activation_width,
        exact_state_mobility: ExactStateMobility::Accepted,
        recurrent_ranges: vec![LayerRange {
            start: 0,
            end: layer_count,
        }],
        split_constraints: Vec::new(),
        sidebands: Vec::new(),
    }
}

pub fn rwkv6_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    FamilyCapabilityRecord {
        family_id: "rwkv6".to_string(),
        layer_count,
        activation_width,
        exact_state_mobility: ExactStateMobility::RejectedTooLarge,
        recurrent_ranges: vec![LayerRange {
            start: 0,
            end: layer_count,
        }],
        split_constraints: Vec::new(),
        sidebands: Vec::new(),
    }
}

pub fn rwkv7_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    FamilyCapabilityRecord {
        family_id: "rwkv7".to_string(),
        layer_count,
        activation_width,
        exact_state_mobility: ExactStateMobility::RejectedTooLarge,
        recurrent_ranges: vec![LayerRange {
            start: 0,
            end: layer_count,
        }],
        split_constraints: Vec::new(),
        sidebands: vec![SidebandRequirement {
            kind: SidebandKind::Rwkv7VFirst,
            first_required_layer: layer_count,
            reason: "RWKV7 downstream slices require the layer-0 v_first activation sideband in addition to the boundary hidden state".to_string(),
        }],
    }
}

pub fn gemma4_e4b_capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    FamilyCapabilityRecord {
        family_id: "gemma4_e4b".to_string(),
        layer_count,
        activation_width,
        exact_state_mobility: ExactStateMobility::Untested,
        recurrent_ranges: Vec::new(),
        split_constraints: vec![SplitConstraint {
            kind: SplitConstraintKind::SharedKvProducerConsumer,
            range: LayerRange { start: 0, end: 0 },
            forbidden_boundaries: vec![12, 14, 24, 28],
            reject_boundary_inside: false,
            reason: "known-bad Gemma4 E4B shared-KV producer/consumer boundary; keep this cut rejected unless KV replay or KV transfer is added".to_string(),
        }],
        sidebands: vec![SidebandRequirement {
            kind: SidebandKind::TokenIds,
            first_required_layer: layer_count,
            reason: "Gemma4 E4B downstream slices require token-id sideband to rebuild the auxiliary per-layer input path".to_string(),
        }],
    }
}

pub fn reviewed_capability_records() -> Vec<ReviewedCapabilityRecord> {
    serde_json::from_str(include_str!(
        "../capabilities/reviewed-family-capabilities.json"
    ))
    .expect("reviewed family capability registry must be valid JSON")
}

pub fn reviewed_capability_for_identity(
    model_identity: &str,
    layer_count: u32,
    activation_width: u32,
) -> Option<FamilyCapabilityRecord> {
    let normalized = model_identity.to_ascii_lowercase();
    reviewed_capability_records()
        .into_iter()
        .find(|record| reviewed_record_matches(record, &normalized))
        .map(|record| capability_for_request(record.capability, layer_count, activation_width))
}

pub fn infer_family_capability(
    model_identity: &str,
    layer_count: u32,
    activation_width: u32,
) -> Option<FamilyCapabilityRecord> {
    if let Some(capability) =
        reviewed_capability_for_identity(model_identity, layer_count, activation_width)
    {
        return Some(capability);
    }

    let normalized = model_identity.to_ascii_lowercase();
    let compact = normalized.replace(['_', '-', '/', ' '], "");
    // Release parsing needs the boundary after a dotted version, so it reads a
    // form that keeps `-` while still joining `qwen3_8` into `qwen38`.
    let release_form = normalized.replace(['_', '/', ' '], "");

    // A Qwen3 point release we have no evidence for must not be guessed into a
    // neighbouring Qwen family. Every known point release (3.5, 3.6, 3.8) is
    // hybrid, so the closest lexical match would advertise a non-recurrent
    // policy for a probably-recurrent model. Stop before the fallback chain.
    if unknown_qwen3_point_release(&release_form) {
        return None;
    }

    infer_granite_gemma_capability(&compact, layer_count, activation_width)
        .or_else(|| {
            infer_falcon_minimax_glm_deepseek_capability(&compact, layer_count, activation_width)
        })
        .or_else(|| infer_mistral_olmo_llama_capability(&compact, layer_count, activation_width))
        .or_else(|| infer_qwen_next_capability(&compact, layer_count, activation_width))
        .or_else(|| infer_recurrent_capability(&compact, layer_count, activation_width))
        .or_else(|| infer_qwen_capability(&release_form, &compact, layer_count, activation_width))
        .or_else(|| infer_remaining_family_capability(&compact, layer_count, activation_width))
        .or_else(|| {
            infer_stage_runtime_fallback_capability(&compact, layer_count, activation_width)
        })
}

fn infer_granite_gemma_capability(
    compact: &str,
    layer_count: u32,
    activation_width: u32,
) -> Option<FamilyCapabilityRecord> {
    if compact.contains("granitehybrid") {
        return Some(recurrent_family_capability(
            "granite_hybrid",
            layer_count,
            activation_width,
        ));
    }
    if compact.contains("granitemoe") {
        return Some(dense_family_capability(
            "granite_moe",
            layer_count,
            activation_width,
            ExactStateMobility::Accepted,
        ));
    }
    if compact.contains("granite") {
        return Some(dense_family_capability(
            "granite",
            layer_count,
            activation_width,
            ExactStateMobility::Accepted,
        ));
    }
    if compact.contains("gemma4") && compact.contains("e4b") {
        return Some(gemma4_e4b_capability(layer_count, activation_width));
    }
    if compact.contains("gemma4") && compact.contains("a4b") {
        return Some(gemma4_a4b_capability(layer_count, activation_width));
    }
    if compact.contains("gemma4") {
        return Some(dense_family_capability(
            "gemma4",
            layer_count,
            activation_width,
            ExactStateMobility::Untested,
        ));
    }
    if compact.contains("gemma3n") {
        return Some(gemma3n_capability(layer_count, activation_width));
    }
    if compact.contains("gemma3") {
        return Some(gemma3_capability(layer_count, activation_width));
    }
    if compact.contains("gemma2") {
        return Some(gemma2_capability(layer_count, activation_width));
    }
    if compact == "gemma" || compact.contains("gemmait") {
        return Some(dense_family_capability(
            "gemma",
            layer_count,
            activation_width,
            ExactStateMobility::Accepted,
        ));
    }

    None
}

fn infer_falcon_minimax_glm_deepseek_capability(
    compact: &str,
    layer_count: u32,
    activation_width: u32,
) -> Option<FamilyCapabilityRecord> {
    if compact.contains("falconh1") {
        return Some(falcon_h1_capability(layer_count, activation_width));
    }
    if compact.contains("minimaxm27")
        || compact.contains("minimaxm2.7")
        || compact.contains("minimaxm2")
    {
        return Some(minimax_m27_capability(layer_count, activation_width));
    }
    if compact.contains("glm47flash") || compact.contains("glm4.7flash") {
        return Some(glm47_flash_capability(layer_count, activation_width));
    }
    if compact.contains("glm4moe") {
        return Some(dense_family_capability(
            "glm4_moe",
            layer_count,
            activation_width,
            ExactStateMobility::Accepted,
        ));
    }
    if compact.contains("glm4") {
        return Some(glm4_capability(layer_count, activation_width));
    }
    if compact.contains("deepseek2ocr") || compact.contains("deepseekocr") {
        return Some(deepseek2ocr_capability(layer_count, activation_width));
    }
    if compact.contains("deepseekcoderv2")
        || compact.contains("deepseekv2")
        || compact.contains("deepseek2")
    {
        return Some(deepseek2_capability(layer_count, activation_width));
    }
    if compact.contains("deepseekv3") || compact.contains("deepseek3") {
        return Some(deepseek3_capability(layer_count, activation_width));
    }

    None
}

fn infer_mistral_olmo_llama_capability(
    compact: &str,
    layer_count: u32,
    activation_width: u32,
) -> Option<FamilyCapabilityRecord> {
    if compact.contains("mistral4") {
        return Some(dense_family_capability(
            "mistral4",
            layer_count,
            activation_width,
            ExactStateMobility::Untested,
        ));
    }
    if compact.contains("mistral3") || compact.contains("ministral3") {
        return Some(dense_family_capability(
            "mistral",
            layer_count,
            activation_width,
            ExactStateMobility::Accepted,
        ));
    }
    if compact.contains("olmoe") {
        return Some(dense_family_capability(
            "olmoe",
            layer_count,
            activation_width,
            ExactStateMobility::Accepted,
        ));
    }
    if compact.contains("olmo2") {
        return Some(dense_family_capability(
            "olmo2",
            layer_count,
            activation_width,
            ExactStateMobility::Accepted,
        ));
    }
    if compact.contains("olmo") {
        return Some(olmo_capability(layer_count, activation_width));
    }
    if compact.contains("laguna") {
        return Some(laguna_capability(layer_count, activation_width));
    }
    if compact.contains("llama") {
        return Some(llama_capability(layer_count, activation_width));
    }

    None
}

fn infer_qwen_next_capability(
    compact: &str,
    layer_count: u32,
    activation_width: u32,
) -> Option<FamilyCapabilityRecord> {
    if compact.contains("qwen3next") || compact.contains("qwen3codernext") {
        return Some(qwen3next_capability(
            layer_count,
            activation_width,
            vec![LayerRange {
                start: 0,
                end: layer_count,
            }],
        ));
    }

    None
}

fn infer_recurrent_capability(
    compact: &str,
    layer_count: u32,
    activation_width: u32,
) -> Option<FamilyCapabilityRecord> {
    if compact.contains("inkling") {
        return Some(inkling_capability(layer_count, activation_width));
    }
    if compact.contains("kimilinear") {
        return Some(kimi_linear_capability(layer_count, activation_width));
    }
    if compact.contains("jamba") {
        return Some(recurrent_family_capability(
            "jamba",
            layer_count,
            activation_width,
        ));
    }
    if compact.contains("lfm2moe") {
        return Some(recurrent_family_capability(
            "lfm2moe",
            layer_count,
            activation_width,
        ));
    }
    if compact.contains("lfm2") {
        return Some(recurrent_family_capability(
            "lfm2",
            layer_count,
            activation_width,
        ));
    }
    if compact.contains("mamba2") {
        return Some(recurrent_family_capability(
            "mamba2",
            layer_count,
            activation_width,
        ));
    }
    if compact.contains("mamba") {
        return Some(recurrent_family_capability(
            "mamba",
            layer_count,
            activation_width,
        ));
    }
    if compact.contains("rwkv6qwen2") {
        return Some(recurrent_family_capability(
            "rwkv6qwen2",
            layer_count,
            activation_width,
        ));
    }
    if compact.contains("rwkv6") {
        return Some(rwkv6_capability(layer_count, activation_width));
    }
    if compact.contains("rwkv7") {
        return Some(rwkv7_capability(layer_count, activation_width));
    }

    None
}

fn infer_qwen_capability(
    release_form: &str,
    compact: &str,
    layer_count: u32,
    activation_width: u32,
) -> Option<FamilyCapabilityRecord> {
    // Flash-Next's GGUF metadata names the upstream llama.cpp architecture
    // explicitly. Check it before release-name routing: other Qwen3.8 models
    // still load as qwen35/qwen35moe and must retain that separate policy.
    if compact.contains("qwen4exp")
        || compact.contains("qwen3.8flashnext")
        || compact.contains("qwen38flashnext")
    {
        return Some(qwen4exp_capability(layer_count, activation_width));
    }
    if compact.contains("qwen2moe") {
        return Some(qwen2moe_capability(layer_count, activation_width));
    }
    if let Some(family_id) = qwen35_series_family_id(release_form, compact) {
        return Some(qwen35_series_capability(
            family_id,
            layer_count,
            activation_width,
        ));
    }
    if compact.contains("qwen3moe") || is_qwen3_active_parameter_moe(compact) {
        return Some(qwen3moe_capability(layer_count, activation_width));
    }
    if compact.contains("qwen2vl") {
        return Some(dense_family_capability(
            "qwen2vl",
            layer_count,
            activation_width,
            ExactStateMobility::Untested,
        ));
    }
    if compact.contains("qwen3vlmoe") {
        return Some(dense_family_capability(
            "qwen3vlmoe",
            layer_count,
            activation_width,
            ExactStateMobility::Accepted,
        ));
    }
    if compact.contains("qwen3vl") {
        return Some(dense_family_capability(
            "qwen3vl",
            layer_count,
            activation_width,
            ExactStateMobility::Untested,
        ));
    }
    if compact.contains("qwen3") {
        return Some(qwen3_dense_capability(layer_count, activation_width));
    }
    if compact.contains("qwen2") {
        return Some(dense_family_capability(
            "qwen2",
            layer_count,
            activation_width,
            ExactStateMobility::Accepted,
        ));
    }

    None
}

fn infer_remaining_family_capability(
    compact: &str,
    layer_count: u32,
    activation_width: u32,
) -> Option<FamilyCapabilityRecord> {
    infer_hunyuan_phi_gpt_capability(compact, layer_count, activation_width)
        .or_else(|| infer_mid_remaining_capability(compact, layer_count, activation_width))
        .or_else(|| {
            infer_exaone_stable_starcoder_capability(compact, layer_count, activation_width)
        })
}

fn infer_hunyuan_phi_gpt_capability(
    compact: &str,
    layer_count: u32,
    activation_width: u32,
) -> Option<FamilyCapabilityRecord> {
    if compact.contains("hunyuanmoe") {
        return Some(dense_family_capability(
            "hunyuan_moe",
            layer_count,
            activation_width,
            ExactStateMobility::Accepted,
        ));
    }
    if compact.contains("hunyuanvl") {
        return Some(dense_family_capability(
            "hunyuan_vl",
            layer_count,
            activation_width,
            ExactStateMobility::Untested,
        ));
    }
    if compact.contains("hunyuandense") {
        return Some(dense_family_capability(
            "hunyuan_dense",
            layer_count,
            activation_width,
            ExactStateMobility::Accepted,
        ));
    }
    if compact.contains("phimoe") {
        return Some(dense_family_capability(
            "phimoe",
            layer_count,
            activation_width,
            ExactStateMobility::Accepted,
        ));
    }
    if compact.contains("phi2") {
        return Some(dense_family_capability(
            "phi2",
            layer_count,
            activation_width,
            ExactStateMobility::RejectedTooLarge,
        ));
    }
    if compact.contains("phi3") || compact == "phi" || compact.contains("phimini") {
        return Some(dense_family_capability(
            "phi",
            layer_count,
            activation_width,
            ExactStateMobility::Accepted,
        ));
    }
    if compact.contains("gptneox") {
        return Some(dense_family_capability(
            "gptneox",
            layer_count,
            activation_width,
            ExactStateMobility::Accepted,
        ));
    }
    if compact.contains("gpt2") {
        return Some(dense_family_capability(
            "gpt2",
            layer_count,
            activation_width,
            ExactStateMobility::Accepted,
        ));
    }

    None
}

fn infer_mid_remaining_capability(
    compact: &str,
    layer_count: u32,
    activation_width: u32,
) -> Option<FamilyCapabilityRecord> {
    if compact.contains("museglimmer") {
        return Some(dense_family_capability(
            "muse_glimmer",
            layer_count,
            activation_width,
            ExactStateMobility::Untested,
        ));
    }
    if compact.contains("mpt") {
        return Some(dense_family_capability(
            "mpt",
            layer_count,
            activation_width,
            ExactStateMobility::Accepted,
        ));
    }
    if compact.contains("bloom") {
        return Some(dense_family_capability(
            "bloom",
            layer_count,
            activation_width,
            ExactStateMobility::Accepted,
        ));
    }
    if compact.contains("baichuan") {
        return Some(dense_family_capability(
            "baichuan",
            layer_count,
            activation_width,
            ExactStateMobility::Accepted,
        ));
    }
    if compact.contains("commandr") {
        return Some(dense_family_capability(
            "command_r",
            layer_count,
            activation_width,
            ExactStateMobility::Accepted,
        ));
    }
    if compact.contains("cohere2") {
        return Some(dense_family_capability(
            "cohere2",
            layer_count,
            activation_width,
            ExactStateMobility::Accepted,
        ));
    }
    if compact.contains("falcon") {
        return Some(dense_family_capability(
            "falcon",
            layer_count,
            activation_width,
            ExactStateMobility::Accepted,
        ));
    }
    if compact.contains("internlm2") {
        return Some(dense_family_capability(
            "internlm2",
            layer_count,
            activation_width,
            ExactStateMobility::Accepted,
        ));
    }

    None
}

fn infer_exaone_stable_starcoder_capability(
    compact: &str,
    layer_count: u32,
    activation_width: u32,
) -> Option<FamilyCapabilityRecord> {
    if compact.contains("exaonemoe") {
        return Some(dense_family_capability(
            "exaone_moe",
            layer_count,
            activation_width,
            ExactStateMobility::RejectedTooLarge,
        ));
    }
    if compact.contains("exaone4") {
        return Some(dense_family_capability(
            "exaone4",
            layer_count,
            activation_width,
            ExactStateMobility::Accepted,
        ));
    }
    if compact.contains("exaone") {
        return Some(dense_family_capability(
            "exaone",
            layer_count,
            activation_width,
            ExactStateMobility::Accepted,
        ));
    }
    if compact.contains("stablelm") {
        return Some(dense_family_capability(
            "stablelm",
            layer_count,
            activation_width,
            ExactStateMobility::Accepted,
        ));
    }
    if compact.contains("starcoder2") {
        return Some(dense_family_capability(
            "starcoder2",
            layer_count,
            activation_width,
            ExactStateMobility::Accepted,
        ));
    }

    None
}

fn infer_stage_runtime_fallback_capability(
    compact: &str,
    layer_count: u32,
    activation_width: u32,
) -> Option<FamilyCapabilityRecord> {
    let mut fallback: Option<(&StageRuntimeFamilyExpectation, usize)> = None;
    for expected in STAGE_RUNTIME_LLAMA_FAMILY_EXPECTATIONS {
        let architecture = expected
            .llama_architecture
            .replace(['_', '-', '/', ' '], "");
        let family = expected.family_id.replace(['_', '-', '/', ' '], "");
        let matched_len = if compact.contains(&architecture) {
            architecture.len()
        } else if compact.contains(&family) {
            family.len()
        } else {
            continue;
        };
        if fallback.is_none_or(|(_, previous_len)| matched_len > previous_len) {
            fallback = Some((expected, matched_len));
        }
    }
    if let Some((expected, _)) = fallback {
        return Some(if expected.recurrent_or_hybrid {
            recurrent_family_capability(expected.family_id, layer_count, activation_width)
        } else {
            dense_family_capability(
                expected.family_id,
                layer_count,
                activation_width,
                ExactStateMobility::Accepted,
            )
        });
    }

    None
}

/// Resolves the Qwen3.5-series family id (`qwen35moe` or `qwen35`), if any.
///
/// Qwen3.5, Qwen3.6, and Qwen3.8 releases all load as llama.cpp `qwen35` /
/// `qwen35moe`; there is no separate `qwen36` or `qwen38` architecture.
/// Release names spell the series with a dot (`Qwen3.6-35B-A3B`), and the
/// compacted identity keeps that dot, so both the dotted release spelling and
/// the dotless architecture string have to be recognized here.
fn qwen35_series_family_id(release_form: &str, compact_identity: &str) -> Option<&'static str> {
    if !is_qwen35_series(release_form) {
        return None;
    }
    if compact_identity.contains("moe") || is_qwen3_active_parameter_moe(compact_identity) {
        return Some("qwen35moe");
    }
    Some("qwen35")
}

/// Point releases of Qwen3 that are known to load as the `qwen35` /
/// `qwen35moe` hybrid pair.
///
/// This list is deliberately explicit. Whether a new point release reuses an
/// existing llama.cpp architecture is a fact about that release, not something
/// derivable from its name, so a new entry must be justified by a real
/// artifact. `unknown_qwen3_point_release` keeps the failure mode safe for
/// releases that are not listed here yet.
const QWEN35_SERIES_POINT_RELEASES: &[char] = &['5', '6', '8'];

fn is_qwen35_series(release_form: &str) -> bool {
    matches!(
        qwen3_release(release_form),
        Some(Qwen3Release::PointRelease(release))
            if QWEN35_SERIES_POINT_RELEASES.contains(&release)
    )
}

/// What the `qwen3` marker in a compacted identity says about the release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Qwen3Release {
    /// A single-digit point release such as `Qwen3.6` or `qwen36`.
    PointRelease(char),
    /// A dotted release whose version is not a single digit, such as
    /// `Qwen3.50`. It names *some* release, but not one this code has ever
    /// reviewed, so it must never be narrowed to a reviewed neighbour.
    UnreviewedDotted,
}

/// Reports what the identity says about its Qwen3 release, if anything.
///
/// Both the dotted release spelling (`Qwen3.6-35B-A3B` -> `qwen3.6`) and the
/// dotless architecture spelling (`qwen36`) are recognized. Parameter counts
/// are not point releases: `Qwen3-8B` compacts to `qwen38b` and
/// `Qwen3-235B-A22B` to `qwen3235ba22b`, so a digit run followed by `b`
/// belongs to the model size.
fn qwen3_release(release_form: &str) -> Option<Qwen3Release> {
    release_form
        .match_indices("qwen3")
        .find_map(|(index, marker)| {
            let rest = &release_form[index + marker.len()..];
            if let Some(dotted) = rest.strip_prefix('.') {
                // `qwen3.6-35b` keeps the size right after the release digit,
                // so a reviewed dotted spelling is exactly one digit. Anything
                // else (`Qwen3.50`) is a release we know nothing about, and
                // truncating it to its first digit would silently claim the
                // reviewed `Qwen3.5` policy for an unreviewed architecture.
                let digits = dotted.chars().take_while(char::is_ascii_digit).count();
                if digits != 1 {
                    return Some(Qwen3Release::UnreviewedDotted);
                }
                return dotted.chars().next().map(Qwen3Release::PointRelease);
            }
            let digits = rest.chars().take_while(char::is_ascii_digit).count();
            // A multi-digit run is a parameter count (`qwen3235ba22b`), a
            // single digit followed by `b` is too (`qwen38b`), and one
            // followed by `.` is a fractional size (`Qwen3-0.6B` ->
            // `qwen30.6b`).
            if digits != 1 || rest[digits..].starts_with(['b', '.']) {
                return None;
            }
            rest.chars().next().map(Qwen3Release::PointRelease)
        })
}

/// Reports a Qwen3 point release that no family policy covers yet.
///
/// A release such as `Qwen3.9` almost certainly does not load as plain
/// `qwen3moe`, so guessing that family would advertise a non-recurrent policy
/// for a model that may well be hybrid. Returning no capability instead makes
/// the gap explicit at onboarding time rather than silently wrong at runtime.
fn unknown_qwen3_point_release(release_form: &str) -> bool {
    match qwen3_release(release_form) {
        Some(Qwen3Release::PointRelease(release)) => {
            !QWEN35_SERIES_POINT_RELEASES.contains(&release)
        }
        Some(Qwen3Release::UnreviewedDotted) => true,
        None => false,
    }
}

fn is_qwen3_active_parameter_moe(compact_identity: &str) -> bool {
    if !compact_identity.contains("qwen3") {
        return false;
    }

    let bytes = compact_identity.as_bytes();
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] != b'a' {
            index += 1;
            continue;
        }

        let mut cursor = index + 1;
        let mut saw_digit = false;
        while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
            saw_digit = true;
            cursor += 1;
        }

        if cursor < bytes.len() && bytes[cursor] == b'.' {
            cursor += 1;
            while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
                saw_digit = true;
                cursor += 1;
            }
        }

        if saw_digit && cursor < bytes.len() && bytes[cursor] == b'b' {
            return true;
        }
        index += 1;
    }

    false
}

fn reviewed_record_matches(record: &ReviewedCapabilityRecord, normalized_identity: &str) -> bool {
    [
        record.model_id.as_deref(),
        record.canonical_ref.as_deref(),
        record
            .distribution_id
            .as_deref()
            .filter(|value| value.len() >= 12),
    ]
    .into_iter()
    .flatten()
    .any(|value| !value.is_empty() && normalized_identity.contains(&value.to_ascii_lowercase()))
        || match (
            record.source_repo.as_deref(),
            record.source_revision.as_deref(),
            record.source_file.as_deref(),
        ) {
            (Some(repo), Some(revision), Some(file)) => {
                normalized_identity.contains(&repo.to_ascii_lowercase())
                    && normalized_identity.contains(&revision.to_ascii_lowercase())
                    && normalized_identity.contains(&file.to_ascii_lowercase())
            }
            (Some(repo), _, Some(file)) => {
                normalized_identity.contains(&repo.to_ascii_lowercase())
                    && normalized_identity.contains(&file.to_ascii_lowercase())
            }
            _ => false,
        }
}

fn capability_for_request(
    mut capability: FamilyCapabilityRecord,
    layer_count: u32,
    activation_width: u32,
) -> FamilyCapabilityRecord {
    let stored_layer_count = capability.layer_count;
    capability.layer_count = layer_count;
    if activation_width != 0 {
        capability.activation_width = activation_width;
    }
    for range in &mut capability.recurrent_ranges {
        if range.start == 0 && range.end == stored_layer_count {
            range.end = layer_count;
        }
    }
    for sideband in &mut capability.sidebands {
        if sideband.first_required_layer == stored_layer_count {
            sideband.first_required_layer = layer_count;
        }
    }
    capability
}

pub fn dense_attention_layers(count: u32, parameter_bytes: u64) -> Vec<LayerSpec> {
    (0..count)
        .map(|index| LayerSpec {
            index,
            attention: true,
            recurrent: false,
            parameter_bytes,
        })
        .collect()
}

pub fn falcon_h1_layers(count: u32, parameter_bytes: u64) -> Vec<LayerSpec> {
    (0..count)
        .map(|index| LayerSpec {
            index,
            attention: true,
            recurrent: true,
            parameter_bytes,
        })
        .collect()
}

pub fn qwen3next_layers(
    count: u32,
    recurrent_layers: impl IntoIterator<Item = u32>,
    parameter_bytes: u64,
) -> Vec<LayerSpec> {
    let recurrent_layers: std::collections::BTreeSet<u32> = recurrent_layers.into_iter().collect();
    (0..count)
        .map(|index| {
            let recurrent = recurrent_layers.contains(&index);
            LayerSpec {
                index,
                attention: !recurrent,
                recurrent,
                parameter_bytes,
            }
        })
        .collect()
}
