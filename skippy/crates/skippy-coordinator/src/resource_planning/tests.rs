use super::*;

fn gqa_metadata(context_length: u32) -> GgufCompactMeta {
    GgufCompactMeta {
        context_length,
        head_count: 32,
        kv_head_count: 8,
        layer_count: 32,
        key_length: 128,
        value_length: 128,
        ..Default::default()
    }
}

#[test]
fn explicit_overrides_are_preserved() {
    let metadata = gqa_metadata(32_768);
    let plan = plan_runtime_resources(RuntimeResourcePlanInput {
        ctx_size_override: Some(16_384),
        parallel_override: Some(7),
        model_bytes: 10_000_000_000,
        projector_bytes: 0,
        vram_bytes: 24_000_000_000,
        metadata: Some(&metadata),
        kv_cache_quant: GgufKvCacheQuant::Q8_0,
        local_layer_fraction: None,
        planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
        measured_buffers: None,
    });

    assert_eq!(plan.context_length, 16_384);
    assert_eq!(plan.slots, 7);
    let breakdown = plan.breakdown.unwrap();
    assert_eq!(
        breakdown.planned_kv_bytes,
        breakdown.kv_bytes_per_token * 16_384 * 7
    );
}

#[test]
fn zero_parallel_override_produces_valid_automatic_and_explicit_plans() {
    let metadata = gqa_metadata(131_072);
    // Check both paths even if the automatic planner panics, so the explicit
    // path must also demonstrate a nonzero, chargeable lane allocation.
    let valid_plans = [None, Some(8192)].map(|ctx_size_override| {
        std::panic::catch_unwind(|| {
            let plan = plan_runtime_resources(RuntimeResourcePlanInput {
                ctx_size_override,
                parallel_override: Some(0),
                model_bytes: 3 * 1024 * 1024 * 1024,
                projector_bytes: 0,
                vram_bytes: 16 * 1024 * 1024 * 1024,
                metadata: Some(&metadata),
                kv_cache_quant: GgufKvCacheQuant::F16,
                local_layer_fraction: None,
                planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
                measured_buffers: None,
            });
            assert!(plan.slots > 0, "zero-lane plan for {ctx_size_override:?}");
            assert!(plan.context_length > 0);
            let breakdown = plan.breakdown.unwrap();
            assert!(!breakdown.slots_auto, "zero was an explicit override");
            assert!(breakdown.planned_kv_bytes > 0);
            assert_eq!(breakdown.slots, plan.slots);
            if let Some(explicit_context) = ctx_size_override {
                assert_eq!(plan.context_length, explicit_context);
            }
        })
        .is_ok()
    });
    assert_eq!(valid_plans, [true, true]);
}

#[test]
fn auto_context_clamped_to_native() {
    let metadata = gqa_metadata(16_384);
    let plan = plan_runtime_resources(RuntimeResourcePlanInput {
        ctx_size_override: None,
        parallel_override: None,
        model_bytes: 5_000_000_000,
        projector_bytes: 0,
        vram_bytes: 80_000_000_000,
        metadata: Some(&metadata),
        kv_cache_quant: GgufKvCacheQuant::Q8_0,
        local_layer_fraction: None,
        planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
        measured_buffers: None,
    });

    assert_eq!(
        plan.context_length, 16_384,
        "should reach native context, not exceed it"
    );
}

#[test]
fn q8_cache_reaches_larger_context_than_f16() {
    // Tight VRAM so f16 can only reach 8K but q8_0 reaches 16K.
    // KV budget = (7.0 - 5.0) * 0.85 = 1.7 GB.
    // f16: 131072 B/tok → 1.7G / 131K ≈ 12K → snaps 8K
    // q8:   69632 B/tok → 1.7G / 69K  ≈ 24K → snaps 16K
    let metadata = gqa_metadata(131_072);
    let f16_plan = plan_runtime_resources(RuntimeResourcePlanInput {
        ctx_size_override: None,
        parallel_override: Some(1),
        model_bytes: 5_000_000_000,
        projector_bytes: 0,
        vram_bytes: 7_000_000_000,
        metadata: Some(&metadata),
        kv_cache_quant: GgufKvCacheQuant::F16,
        local_layer_fraction: None,
        planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
        measured_buffers: None,
    });
    let q8_plan = plan_runtime_resources(RuntimeResourcePlanInput {
        ctx_size_override: None,
        parallel_override: Some(1),
        model_bytes: 5_000_000_000,
        projector_bytes: 0,
        vram_bytes: 7_000_000_000,
        metadata: Some(&metadata),
        kv_cache_quant: GgufKvCacheQuant::Q8_0,
        local_layer_fraction: None,
        planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
        measured_buffers: None,
    });

    assert!(
        q8_plan.context_length > f16_plan.context_length,
        "q8_0 should afford more context: q8={}K, f16={}K",
        q8_plan.context_length / 1024,
        f16_plan.context_length / 1024
    );
}

#[test]
fn llama_31_8b_on_m2_reserves_four_lanes_in_one_f16_pool() {
    let metadata = gqa_metadata(131_072);
    let plan = plan_runtime_resources(RuntimeResourcePlanInput {
        ctx_size_override: None,
        parallel_override: None,
        model_bytes: 4_994_200_000,
        projector_bytes: 0,
        vram_bytes: 18_185 * 1024 * 1024,
        metadata: Some(&metadata),
        kv_cache_quant: GgufKvCacheQuant::F16,
        local_layer_fraction: None,
        planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
        measured_buffers: None,
    });

    assert_eq!(plan.context_length, 16_384);
    assert_eq!(plan.slots, 4);
    assert_eq!(
        plan.breakdown.unwrap().planned_kv_bytes,
        8 * 1024 * 1024 * 1024
    );
}

#[test]
fn fallback_defaults_without_metadata() {
    let plan = plan_runtime_resources(RuntimeResourcePlanInput {
        ctx_size_override: None,
        parallel_override: None,
        model_bytes: 5_000_000_000,
        projector_bytes: 0,
        vram_bytes: 16_000_000_000,
        metadata: None,
        kv_cache_quant: GgufKvCacheQuant::Q8_0,
        local_layer_fraction: None,
        planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
        measured_buffers: None,
    });

    assert_eq!(plan.context_length, 16_384);
    assert_eq!(plan.slots, 4);
}

#[test]
fn both_profiles_produce_identical_auto_plans() {
    // The old behavior traded context for concurrency on the shared-mesh
    // profile (shallower context, more lanes). The default is now uniform:
    // hold context at `min(native, 128k)` and run the capped lane count,
    // so both profiles plan the same context and lane count. The
    // profile axis is retained for the bandwidth-aware follow-up.
    let metadata = gqa_metadata(131_072);
    let input = |profile| RuntimeResourcePlanInput {
        ctx_size_override: None,
        parallel_override: None,
        model_bytes: 5_000_000_000,
        projector_bytes: 0,
        vram_bytes: 16_000_000_000,
        metadata: Some(&metadata),
        kv_cache_quant: GgufKvCacheQuant::Q8_0,
        local_layer_fraction: None,
        planning_profile: profile,
        measured_buffers: None,
    };
    let dedicated_plan =
        plan_runtime_resources(input(RuntimeResourcePlanningProfile::DedicatedLocal));
    let shared_plan = plan_runtime_resources(input(RuntimeResourcePlanningProfile::SharedMesh));

    assert_eq!(
        dedicated_plan, shared_plan,
        "both profiles hold the 128k floor and fill lanes identically"
    );
    assert!(dedicated_plan.context_length <= 131_072);
}

#[test]
fn auto_context_capped_at_128k_for_million_token_native() {
    // Nemotron-class 1M-token native on a fat node. Left unclamped the
    // planner would drive a multi-hundred-K context; the default ceiling
    // holds it at 128k and spends the rest of the budget on lanes.
    let metadata = gqa_metadata(1_048_576);
    let plan = plan_runtime_resources(RuntimeResourcePlanInput {
        ctx_size_override: None,
        parallel_override: None,
        model_bytes: 5_000_000_000,
        projector_bytes: 0,
        vram_bytes: 80_000_000_000,
        metadata: Some(&metadata),
        kv_cache_quant: GgufKvCacheQuant::Q8_0,
        local_layer_fraction: None,
        planning_profile: RuntimeResourcePlanningProfile::SharedMesh,
        measured_buffers: None,
    });

    assert_eq!(
        plan.context_length, 131_072,
        "1M native must clamp to the 128k default ceiling"
    );
    assert_eq!(
        plan.slots, 4,
        "fat node should keep the full 4 lanes at 128k"
    );
}

#[test]
fn tight_budget_reduces_per_lane_depth_for_four_lanes() {
    // Four lanes share one pool, but each reserves its requested depth.
    let metadata = gqa_metadata(262_144);
    let plan = plan_runtime_resources(RuntimeResourcePlanInput {
        ctx_size_override: None,
        parallel_override: None,
        model_bytes: 5_000_000_000,
        projector_bytes: 0,
        vram_bytes: 18_000_000_000,
        metadata: Some(&metadata),
        kv_cache_quant: GgufKvCacheQuant::Q8_0,
        local_layer_fraction: None,
        planning_profile: RuntimeResourcePlanningProfile::SharedMesh,
        measured_buffers: None,
    });

    assert_eq!(
        plan.context_length, 32_768,
        "context must fit four lane reservations"
    );
    assert_eq!(
        plan.slots, 4,
        "auto still selects the 4-lane target; got {} lanes",
        plan.slots
    );
}

#[test]
fn explicit_parallel_with_auto_context() {
    let metadata = gqa_metadata(32_768);
    let plan = plan_runtime_resources(RuntimeResourcePlanInput {
        ctx_size_override: None,
        parallel_override: Some(2),
        model_bytes: 5_000_000_000,
        projector_bytes: 0,
        vram_bytes: 80_000_000_000,
        metadata: Some(&metadata),
        kv_cache_quant: GgufKvCacheQuant::Q8_0,
        local_layer_fraction: None,
        planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
        measured_buffers: None,
    });

    assert_eq!(plan.context_length, 32_768);
    assert_eq!(plan.slots, 2);
}

#[test]
fn auto_slots_capped_at_llama_server_default() {
    // Regression: a small model on a huge-VRAM box used to plan
    // `slots = 16` because the VRAM-derived per-lane math pretended
    // each lane carved off its own `n_ctx × bytes/token` allocation.
    // With `kv_unified = true` (skippy patch 0034) those 16 lanes
    // race for the same `n_ctx` cell pool, and 3 concurrent agent
    // requests at ~14k tokens each blow it up with
    // `find_slot` failures → HTTP 502
    // `RuntimeError: llama_decode failed`.
    //
    // Match llama-server's auto default of 4 (see
    // `.deps/llama.cpp/tools/server/server.cpp`: "n_parallel is
    // set to auto, using n_parallel = 4 and kv_unified = true").
    let metadata = gqa_metadata(32_768);
    let plan = plan_runtime_resources(RuntimeResourcePlanInput {
        ctx_size_override: None,
        parallel_override: None,
        model_bytes: 5_000_000_000,
        projector_bytes: 0,
        // 128GB free — plenty for many "per-lane" slots under the
        // old broken math.
        vram_bytes: 128_000_000_000,
        metadata: Some(&metadata),
        kv_cache_quant: GgufKvCacheQuant::Q8_0,
        local_layer_fraction: None,
        planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
        measured_buffers: None,
    });

    assert_eq!(plan.context_length, 32_768);
    assert!(
        plan.slots <= 4,
        "auto-planner should not exceed llama-server's 4-lane unified-KV ceiling; got {}",
        plan.slots
    );
}

/// More lanes reserve more cells in the same unified pool, reducing the
/// affordable per-lane context at a fixed memory budget.
#[test]
fn explicit_parallel_reserves_capacity_for_every_lane() {
    let metadata = gqa_metadata(131_072);
    let input = |parallel_override| RuntimeResourcePlanInput {
        ctx_size_override: None,
        parallel_override,
        model_bytes: 32_000_000_000,
        projector_bytes: 0,
        vram_bytes: 122_000_000_000,
        metadata: Some(&metadata),
        kv_cache_quant: GgufKvCacheQuant::Q8_0,
        local_layer_fraction: None,
        planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
        measured_buffers: None,
    };
    let auto = plan_runtime_resources(input(None));
    let wide = plan_runtime_resources(input(Some(128)));
    assert_eq!(wide.context_length, 8192);
    assert_eq!(auto.context_length, 131_072);
    assert!(wide.breakdown.unwrap().planned_kv_bytes <= wide.breakdown.unwrap().kv_budget_bytes);
    assert_eq!(wide.slots, 128);
}

#[test]
fn explicit_parallel_can_exceed_auto_ceiling() {
    // Operators who know their workload can still go higher than
    // the auto ceiling via `parallel_override`.
    let metadata = gqa_metadata(131_072);
    let plan = plan_runtime_resources(RuntimeResourcePlanInput {
        ctx_size_override: None,
        parallel_override: Some(8),
        model_bytes: 5_000_000_000,
        projector_bytes: 0,
        vram_bytes: 128_000_000_000,
        metadata: Some(&metadata),
        kv_cache_quant: GgufKvCacheQuant::Q8_0,
        local_layer_fraction: None,
        planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
        measured_buffers: None,
    });

    assert_eq!(plan.slots, 8);
}

#[test]
fn split_model_uses_local_layer_fraction() {
    // 480B-class model: 94 layers, 264GB total, host holds 62/94 layers.
    let metadata = GgufCompactMeta {
        context_length: 131_072,
        head_count: 64,
        kv_head_count: 8,
        layer_count: 94,
        key_length: 128,
        value_length: 128,
        ..Default::default()
    };
    let total_model_bytes: u64 = 264_000_000_000;
    let local_fraction = 62.0 / 94.0;
    let local_model_bytes = (total_model_bytes as f64 * local_fraction) as u64;

    // Without split awareness: 206 GB VRAM, 264 GB model → negative budget → minimum
    let no_split = plan_runtime_resources(RuntimeResourcePlanInput {
        ctx_size_override: None,
        parallel_override: None,
        model_bytes: total_model_bytes,
        projector_bytes: 0,
        vram_bytes: 206_000_000_000,
        metadata: Some(&metadata),
        kv_cache_quant: GgufKvCacheQuant::Q8_0,
        local_layer_fraction: None,
        planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
        measured_buffers: None,
    });

    // With split awareness: local model ~174 GB, local KV fraction 0.66
    let split = plan_runtime_resources(RuntimeResourcePlanInput {
        ctx_size_override: None,
        parallel_override: None,
        model_bytes: local_model_bytes,
        projector_bytes: 0,
        vram_bytes: 206_000_000_000,
        metadata: Some(&metadata),
        kv_cache_quant: GgufKvCacheQuant::Q8_0,
        local_layer_fraction: Some(local_fraction),
        planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
        measured_buffers: None,
    });

    assert!(
        split.context_length > no_split.context_length,
        "split-aware should produce larger context: split={}K, no_split={}K",
        split.context_length / 1024,
        no_split.context_length / 1024
    );
    assert!(
        split.context_length >= 32_768,
        "480B split on 206+103 GB with q8_0 should get at least 32K per lane, got {}K",
        split.context_length / 1024
    );
}

#[test]
fn lane_count_is_independent_of_kv_quant() {
    // KV quant changes bytes per cell, but the default lane target remains
    // four. Context depth absorbs the memory difference when necessary.
    let metadata = gqa_metadata(131_072);
    let plan_with = |quant| {
        plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: None,
            model_bytes: 5_000_000_000,
            projector_bytes: 0,
            vram_bytes: 80_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: quant,
            local_layer_fraction: None,
            planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
            measured_buffers: None,
        })
    };
    let q8_plan = plan_with(GgufKvCacheQuant::Q8_0);
    let q4_plan = plan_with(GgufKvCacheQuant::Q4_0);

    assert_eq!(q8_plan.context_length, q4_plan.context_length);
    assert_eq!(
        q8_plan.slots, q4_plan.slots,
        "lane count must not depend on KV quant under unified KV: q4={}, q8={}",
        q4_plan.slots, q8_plan.slots
    );
}

#[test]
fn measured_total_kv_matches_equivalent_static_plan() {
    // One layer and one 64-element K/V head with F16 storage cost 256 bytes
    // per token per lane. Four 8192-token lanes allocate 8 MiB in total.
    let metadata = GgufCompactMeta {
        context_length: 131_072,
        head_count: 1,
        kv_head_count: 1,
        layer_count: 1,
        key_length: 64,
        value_length: 64,
        ..Default::default()
    };
    let input = RuntimeResourcePlanInput {
        ctx_size_override: None,
        parallel_override: Some(4),
        model_bytes: 0,
        projector_bytes: 0,
        vram_bytes: 10 * 1024 * 1024,
        metadata: Some(&metadata),
        kv_cache_quant: GgufKvCacheQuant::F16,
        local_layer_fraction: None,
        planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
        measured_buffers: None,
    };
    let static_plan = plan_runtime_resources(input);
    let measured_plan = plan_runtime_resources(RuntimeResourcePlanInput {
        measured_buffers: Some(MeasuredBufferFootprint {
            compute_bytes: 0,
            kv_bytes: 8 * 1024 * 1024,
            context_length: 8192,
            lane_count: 4,
        }),
        ..input
    });

    // The 85% static and 88% measured budgets both fit 8192 tokens per lane.
    assert_eq!(static_plan.context_length, 8192);
    assert_eq!(measured_plan.context_length, static_plan.context_length);
    let breakdown = measured_plan.breakdown.unwrap();
    assert_eq!(breakdown.kv_bytes_per_token, 256);
    assert_eq!(breakdown.planned_kv_bytes, 8 * 1024 * 1024);
    assert_eq!(breakdown.measured_fit, Some(true));
}

#[test]
fn measured_total_kv_replans_for_requested_lanes() {
    let metadata = GgufCompactMeta {
        context_length: 131_072,
        head_count: 1,
        kv_head_count: 1,
        layer_count: 1,
        key_length: 64,
        value_length: 64,
        ..Default::default()
    };
    // These historical allocations all contain 32768 token cells and cost
    // 8 MiB. Their per-lane depths vary with the measured lane count.
    for measured_lanes in [1, 2, 4, 8] {
        let footprint = MeasuredBufferFootprint {
            compute_bytes: 0,
            kv_bytes: 8 * 1024 * 1024,
            context_length: 32_768 / measured_lanes,
            lane_count: measured_lanes,
        };
        for (requested_lanes, expected_context) in [(1, 32_768), (2, 16_384), (4, 8192), (8, 4096)]
        {
            let plan = plan_runtime_resources(RuntimeResourcePlanInput {
                ctx_size_override: None,
                parallel_override: Some(requested_lanes),
                model_bytes: 0,
                projector_bytes: 0,
                vram_bytes: 10 * 1024 * 1024,
                metadata: Some(&metadata),
                kv_cache_quant: GgufKvCacheQuant::F16,
                local_layer_fraction: None,
                planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
                measured_buffers: Some(footprint),
            });
            assert_eq!(
                plan.context_length, expected_context,
                "measured {measured_lanes} lanes, requested {requested_lanes} lanes"
            );
            assert_eq!(plan.slots, requested_lanes);
            let breakdown = plan.breakdown.unwrap();
            assert_eq!(breakdown.kv_bytes_per_token, 256);
            assert_eq!(breakdown.planned_kv_bytes, 8 * 1024 * 1024);
            assert_eq!(breakdown.measured_fit, Some(true));
        }
    }
}

#[test]
fn budget_driven_context_uses_measured_kv_and_compute() {
    // Roomy node: 16 GiB VRAM, 3 GiB weights. The 2 GiB measurement is the
    // total for four 16384-token lanes, giving 32768 bytes/token/lane.
    // With measured compute charged, four lanes can each reach 65536 tokens.
    let metadata = gqa_metadata(131_072);
    let plan = plan_runtime_resources(RuntimeResourcePlanInput {
        ctx_size_override: None,
        parallel_override: None,
        model_bytes: 3 * 1024 * 1024 * 1024,
        projector_bytes: 0,
        vram_bytes: 16 * 1024 * 1024 * 1024,
        metadata: Some(&metadata),
        kv_cache_quant: GgufKvCacheQuant::Q8_0,
        local_layer_fraction: None,
        planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
        measured_buffers: Some(MeasuredBufferFootprint {
            compute_bytes: 512 * 1024 * 1024,
            kv_bytes: 2 * 1024 * 1024 * 1024,
            context_length: 16_384,
            lane_count: 4,
        }),
    });
    assert_eq!(plan.context_length, 65_536);
    let measured_breakdown = plan.breakdown.expect("measured planner breakdown");
    assert_eq!(
        measured_breakdown.planning_source,
        RuntimeResourcePlanSource::MeasuredFootprint
    );
    assert_eq!(measured_breakdown.kv_bytes_per_token, 32_768);
    assert_eq!(measured_breakdown.compute_charge_bytes, 512 * 1024 * 1024);
    assert_eq!(measured_breakdown.measured_fit, Some(true));
    assert_eq!(measured_breakdown.planned_kv_bytes, 65_536 * 32_768 * 4);

    // Tight node where the ladder's estimate binds: 5 GiB free, the q8
    // estimate (~69632 B/token/lane) caps the ladder at 16384 per lane.
    // Total measured KV of 512 MiB at 16384 tokens across four lanes costs
    // 8192 bytes/token/lane, permitting the native 131072-token window.
    let tight = plan_runtime_resources(RuntimeResourcePlanInput {
        ctx_size_override: None,
        parallel_override: None,
        model_bytes: 1024 * 1024 * 1024,
        projector_bytes: 0,
        vram_bytes: 6 * 1024 * 1024 * 1024,
        metadata: Some(&metadata),
        kv_cache_quant: GgufKvCacheQuant::Q8_0,
        local_layer_fraction: None,
        planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
        measured_buffers: Some(MeasuredBufferFootprint {
            compute_bytes: 128 * 1024 * 1024,
            kv_bytes: 512 * 1024 * 1024,
            context_length: 16_384,
            lane_count: 4,
        }),
    });
    assert_eq!(tight.context_length, 131_072);

    let tight_ladder = plan_runtime_resources(RuntimeResourcePlanInput {
        ctx_size_override: None,
        parallel_override: None,
        model_bytes: 1024 * 1024 * 1024,
        projector_bytes: 0,
        vram_bytes: 6 * 1024 * 1024 * 1024,
        metadata: Some(&metadata),
        kv_cache_quant: GgufKvCacheQuant::Q8_0,
        local_layer_fraction: None,
        planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
        measured_buffers: None,
    });
    assert_eq!(tight_ladder.context_length, 16_384);
}

#[test]
fn budget_driven_context_degrades_to_ladder_when_measurement_is_unusable() {
    let metadata = gqa_metadata(131_072);
    let base = RuntimeResourcePlanInput {
        ctx_size_override: None,
        parallel_override: None,
        model_bytes: 5_000_000_000,
        projector_bytes: 0,
        vram_bytes: 24_000_000_000,
        metadata: Some(&metadata),
        kv_cache_quant: GgufKvCacheQuant::Q8_0,
        local_layer_fraction: None,
        planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
        measured_buffers: None,
    };

    // Zero-context, zero-KV, or zero-lane measurements cannot drive the
    // budget model; the plan must equal the static-ladder answer.
    let ladder = plan_runtime_resources(base).context_length;
    for unusable in [
        MeasuredBufferFootprint {
            compute_bytes: 512 * 1024 * 1024,
            kv_bytes: 0,
            context_length: 16_384,
            lane_count: 4,
        },
        MeasuredBufferFootprint {
            compute_bytes: 512 * 1024 * 1024,
            kv_bytes: 2 * 1024 * 1024 * 1024,
            context_length: 0,
            lane_count: 4,
        },
        MeasuredBufferFootprint {
            compute_bytes: 512 * 1024 * 1024,
            kv_bytes: 2 * 1024 * 1024 * 1024,
            context_length: 16_384,
            lane_count: 0,
        },
    ] {
        let degraded = plan_runtime_resources(RuntimeResourcePlanInput {
            measured_buffers: Some(unusable),
            ..base
        });
        assert_eq!(degraded.context_length, ladder);
    }
}

#[test]
fn budget_driven_context_reports_exhausted_measurement_without_fallback() {
    let metadata = gqa_metadata(131_072);
    let plan = plan_runtime_resources(RuntimeResourcePlanInput {
        ctx_size_override: None,
        parallel_override: None,
        model_bytes: 5 * 1024 * 1024 * 1024,
        projector_bytes: 0,
        vram_bytes: 6 * 1024 * 1024 * 1024,
        metadata: Some(&metadata),
        kv_cache_quant: GgufKvCacheQuant::Q8_0,
        local_layer_fraction: None,
        planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
        measured_buffers: Some(MeasuredBufferFootprint {
            compute_bytes: 1024 * 1024 * 1024,
            kv_bytes: 2 * 1024 * 1024 * 1024,
            context_length: 16_384,
            lane_count: 4,
        }),
    });
    let breakdown = plan.breakdown.expect("planner breakdown");
    assert_eq!(
        breakdown.planning_source,
        RuntimeResourcePlanSource::MeasuredFootprint
    );
    assert_eq!(breakdown.measured_fit, Some(false));
    assert_eq!(plan.context_length, MIN_AUTO_CONTEXT_LENGTH);
    assert!(breakdown.planned_kv_bytes > breakdown.kv_budget_bytes);
}

#[test]
fn budget_driven_compute_charge_scales_with_lane_count() {
    // Compute buffers scale ~linearly with lanes (measured 399/783/1551
    // MiB at 2/4/8 on the 5080). A footprint measured at 4 lanes must be
    // charged at 2x when the plan resolves 8 lanes, and at 0.5x when it
    // resolves 2 — and the context depth must follow the charge.
    let metadata = gqa_metadata(131_072);
    let footprint = MeasuredBufferFootprint {
        compute_bytes: 512 * 1024 * 1024,
        kv_bytes: 2 * 1024 * 1024 * 1024,
        context_length: 16_384,
        lane_count: 4,
    };
    let plan_at = |parallel_override: Option<usize>| {
        plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override,
            model_bytes: 3 * 1024 * 1024 * 1024,
            projector_bytes: 0,
            vram_bytes: 16 * 1024 * 1024 * 1024,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::Q8_0,
            local_layer_fraction: None,
            planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
            measured_buffers: Some(footprint),
        })
        .context_length
    };

    let at2 = plan_at(Some(2));
    let at4 = plan_at(Some(4));
    let at8 = plan_at(Some(8));
    // Scaling the charge DOWN (fewer lanes than measured) frees budget and
    // can only deepen (or hold) the plan; scaling UP can only shallow it.
    assert!(at2 >= at4);
    assert!(at8 <= at4);
    // At the measured lane count the charge is exactly the measured value,
    // so this matches the roomy-node case of
    // budget_driven_context_uses_measured_kv_and_compute.
    assert_eq!(at4, 65_536);
}

fn projector_input(vram_bytes: u64, projector_bytes: u64) -> RuntimeResourcePlanInput<'static> {
    RuntimeResourcePlanInput {
        ctx_size_override: None,
        parallel_override: None,
        model_bytes: 5_000_000_000,
        projector_bytes,
        vram_bytes,
        metadata: None,
        kv_cache_quant: GgufKvCacheQuant::Q8_0,
        local_layer_fraction: None,
        planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
        measured_buffers: None,
    }
}

#[test]
fn projector_bytes_are_charged_to_the_plan_and_shrink_the_context() {
    // A 1 GiB projector shares the pool with the weights, so the fit has to
    // reserve it. Before mesh-llm#1166 nothing charged it, and the projector
    // was loaded after the text model into whatever the plan left over.
    let metadata = gqa_metadata(131_072);
    let plan_with = |projector_bytes: u64| {
        plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: None,
            model_bytes: 5_000_000_000,
            projector_bytes,
            vram_bytes: 16_000_000_000,
            metadata: Some(&metadata),
            kv_cache_quant: GgufKvCacheQuant::Q8_0,
            local_layer_fraction: None,
            planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
            measured_buffers: None,
        })
    };

    let without = plan_with(0);
    let with = plan_with(1_073_741_824);

    assert!(
        with.context_length < without.context_length,
        "charging the projector must leave it room: {} vs {}",
        with.context_length,
        without.context_length
    );
    let with_breakdown = with.breakdown.expect("plan carries a breakdown");
    let without_breakdown = without.breakdown.expect("plan carries a breakdown");
    assert_eq!(with_breakdown.projector_bytes, 1_073_741_824);
    assert_eq!(without_breakdown.projector_bytes, 0);
    assert!(
        with_breakdown.kv_budget_bytes < without_breakdown.kv_budget_bytes,
        "the projector is resident weight memory: {} vs {}",
        with_breakdown.kv_budget_bytes,
        without_breakdown.kv_budget_bytes
    );
}

#[test]
fn projector_larger_than_the_pool_keeps_the_minimum_context() {
    // A projector claim bigger than the whole pool must saturate, not
    // underflow into a huge budget.
    let metadata = gqa_metadata(131_072);
    let plan = plan_runtime_resources(RuntimeResourcePlanInput {
        metadata: Some(&metadata),
        ..projector_input(16_000_000_000, 900_000_000_000)
    });

    assert_eq!(plan.context_length, MIN_AUTO_CONTEXT_LENGTH);
    assert_eq!(
        plan.breakdown
            .expect("plan carries a breakdown")
            .kv_budget_bytes,
        0
    );
}

#[test]
fn zero_projector_bytes_keep_the_previous_budget() {
    // The field is additive: a model with no projector plans exactly as it
    // did before.
    let plan = plan_runtime_resources(projector_input(16_000_000_000, 0));

    assert_eq!(
        plan.breakdown
            .expect("plan carries a breakdown")
            .kv_budget_bytes,
        usable_kv_cache_budget(16_000_000_000, 5_000_000_000, 0)
    );
}
