use super::super::*;

#[test]
fn aggregator_preserves_backend_summary_lines() {
    let mut aggregator = NativeLogAggregator::default();
    assert_eq!(
        aggregator.process_line("backend_init succeeded"),
        vec![NativeLogEvent {
            message: "backend_init succeeded".to_string(),
            category: "backend",
            params: Vec::new(),
        }]
    );
    assert_eq!(
        aggregator.process_line("llama_backend_init: GGML_CUDA"),
        vec![NativeLogEvent {
            message: "llama_backend_init: GGML_CUDA".to_string(),
            category: "backend",
            params: Vec::new(),
        }]
    );
    assert_eq!(
        aggregator.process_line("llama_backend_init: GGML_HIP backend initialized"),
        vec![NativeLogEvent {
            message: "llama_backend_init: GGML_HIP backend initialized".to_string(),
            category: "backend",
            params: Vec::new(),
        }]
    );
    assert_eq!(
        aggregator.process_line("llama_backend_init: GGML_ROCM backend initialized"),
        vec![NativeLogEvent {
            message: "llama_backend_init: GGML_ROCM backend initialized".to_string(),
            category: "backend",
            params: Vec::new(),
        }]
    );
}

#[test]
fn aggregator_forwards_llama_context_config_lines() {
    let mut aggregator = NativeLogAggregator::default();
    assert_eq!(
        aggregator.process_line("llama_context: n_ubatch      = 512"),
        vec![NativeLogEvent {
            message: "llama_context: n_ubatch      = 512".to_string(),
            category: "runtime",
            params: Vec::new(),
        }]
    );
    assert_eq!(
        aggregator.process_line("llama_context: flash_attn    = enabled"),
        vec![NativeLogEvent {
            message: "llama_context: flash_attn    = enabled".to_string(),
            category: "runtime",
            params: Vec::new(),
        }]
    );
    assert!(
        aggregator
            .process_line("llama_context: n_ctx         = 8192")
            .is_empty()
    );
    assert!(
        aggregator
            .process_line("llama_context: causal_attn   = 1")
            .is_empty()
    );
}

#[test]
fn aggregator_ignores_non_backend_cuda_mentions() {
    let mut aggregator = NativeLogAggregator::default();
    assert!(
        aggregator
            .process_line("CUDA kernel launch for attention")
            .is_empty()
    );
    assert!(aggregator.process_line("offloading to CUDA").is_empty());
}

#[test]
fn aggregator_builds_metadata_summary_and_progress() {
    let mut aggregator = NativeLogAggregator::default();
    assert_eq!(
            aggregator.process_line(
                "llama_model_loader: loaded meta data with 10 key-value pairs and 100 tensors from model.gguf (version GGUF V3)"
            ),
            vec![NativeLogEvent {
                message: "model load plan: metadata rows=10, tensor rows=100".to_string(),
                category: "model",
                params: Vec::new(),
            }]
        );

    for (idx, line) in [
        "llama_model_loader: - kv   0: general.architecture str = qwen35",
        "llama_model_loader: - kv   1: general.name str = Qwen 3.5 4B",
        "llama_model_loader: - kv   2: general.type str = model",
        "llama_model_loader: - kv   3: general.size_label str = 4B",
        "llama_model_loader: - kv   4: qwen35.context_length u32 = 40960",
        "llama_model_loader: - kv   5: qwen35.block_count u32 = 36",
        "llama_model_loader: - kv   6: qwen35.embedding_length u32 = 2560",
        "llama_model_loader: - kv   7: qwen35.feed_forward_length u32 = 9728",
        "llama_model_loader: - kv   8: qwen35.attention.head_count u32 = 32",
        "llama_model_loader: - kv   9: qwen35.attention.head_count_kv u32 = 8",
    ]
    .iter()
    .enumerate()
    {
        let events = aggregator.process_line(line);
        assert!(
            events
                .iter()
                .any(|event| event.message.contains(&format!("{}%", (idx + 1) * 10))),
            "expected {}% metadata progress in {:?}",
            (idx + 1) * 10,
            events
        );
    }

    let flush_events = aggregator.process_line("llm_load_print_meta: version = 3");
    assert!(
        flush_events
            .iter()
            .any(|event| event.message == "llm_load_print_meta: version = 3")
    );
    assert!(
        flush_events
            .iter()
            .any(|event| event.message == "Reading model metadata...")
    );
    assert!(flush_events.iter().any(|event| {
        event.params.iter().any(|(key, value)| {
            key == "architecture" && value == &Value::String("qwen35".to_string())
        })
    }));
}

#[test]
fn aggregator_emits_tensor_progress_from_type_summaries() {
    let mut aggregator = NativeLogAggregator::default();
    aggregator.process_line(
            "llama_model_loader: loaded meta data with 46 key-value pairs and 100 tensors from model.gguf (version GGUF V3)",
        );

    let first = aggregator.process_line("llama_model_loader: - type  f32:  30 tensors");
    assert!(
        first
            .iter()
            .any(|event| event.message.contains("tensors 10%"))
    );
    assert!(
        first
            .iter()
            .any(|event| event.message.contains("tensors 30%"))
    );

    let second = aggregator.process_line("llama_model_loader: - type q4_k:  70 tensors");
    assert!(
        second
            .iter()
            .any(|event| event.message.contains("tensors 100%"))
    );
    assert!(second.iter().any(|event| {
        event.message == "Reading tensor groups..."
            && event
                .params
                .iter()
                .any(|(key, value)| key == "f32" && value == &Value::from(30_u64))
            && event
                .params
                .iter()
                .any(|(key, value)| key == "q4_K" && value == &Value::from(70_u64))
    }));
}

#[test]
fn aggregator_records_measured_buffers_for_snapshot_api() {
    let mut aggregator = NativeLogAggregator::default();
    aggregator.process_line("sched_reserve:        CUDA0 compute buffer size =   579.83 MiB");
    aggregator.process_line("llama_kv_cache:        CUDA0 KV buffer size =  1088.00 MiB");
    assert_eq!(
        aggregator.measured_snapshot(),
        MeasuredNativeBuffers {
            compute_mib: Some(579.83),
            kv_mib: Some(1088.00),
            host_memory_observed: false,
        }
    );

    // A later, smaller reserve for a shorter context must not lower the
    // recorded high-water mark for either kind.
    aggregator.process_line("sched_reserve:        CUDA0 compute buffer size =   512.00 MiB");
    aggregator.process_line("llama_kv_cache:        CUDA0 KV buffer size =  1024.00 MiB");
    assert_eq!(
        aggregator.measured_snapshot(),
        MeasuredNativeBuffers {
            compute_mib: Some(579.83),
            kv_mib: Some(1088.00),
            host_memory_observed: false,
        }
    );

    // CPU-offloaded buffers stay excluded from device totals, but make the
    // device-only snapshot ineligible for reuse against a mixed pool.
    aggregator.process_line("load_tensors: CPU_Mapped model buffer size =  2048.00 MiB");
    aggregator.process_line("llama_kv_cache:        CPU KV buffer size =  4096.00 MiB");
    aggregator.process_line("llama_kv_cache:        CPU compute buffer size =  8192.00 MiB");
    assert_eq!(
        aggregator.measured_snapshot(),
        MeasuredNativeBuffers {
            compute_mib: Some(579.83),
            kv_mib: Some(1088.00),
            host_memory_observed: true,
        }
    );

    // Model-agnostic summary lines carry no buffer size to record.
    aggregator.process_line("VRAM used: 12.4 GB");
    assert_eq!(
        aggregator.measured_snapshot(),
        MeasuredNativeBuffers {
            compute_mib: Some(579.83),
            kv_mib: Some(1088.00),
            host_memory_observed: true,
        }
    );

    // register/unregister reset clears the snapshot for the next model.
    aggregator.reset();
    assert_eq!(
        aggregator.measured_snapshot(),
        MeasuredNativeBuffers {
            compute_mib: None,
            kv_mib: None,
            host_memory_observed: false,
        }
    );
}

#[test]
fn aggregator_sums_measured_buffers_across_devices() {
    // One buffer line is printed per backend device: on multi-GPU the
    // measured footprint must SUM across devices (a per-kind max would
    // under-measure by the device count and the planner would buy
    // roughly N x too much context).
    let mut aggregator = NativeLogAggregator::default();
    aggregator.process_line("sched_reserve:        CUDA0 compute buffer size =   544.00 MiB");
    aggregator.process_line("sched_reserve:        CUDA1 compute buffer size =   544.00 MiB");
    aggregator.process_line("llama_kv_cache:        CUDA0 KV buffer size =  544.00 MiB");
    aggregator.process_line("llama_kv_cache:        CUDA1 KV buffer size =  544.00 MiB");
    assert_eq!(
        aggregator.measured_snapshot(),
        MeasuredNativeBuffers {
            compute_mib: Some(1088.00),
            kv_mib: Some(1088.00),
            host_memory_observed: false,
        }
    );
}

#[test]
fn aggregator_excludes_host_pinned_buffers() {
    // CUDA_Host is host RAM pinned for device transfers, not device
    // memory — charging it against a VRAM budget would over-reserve.
    let mut aggregator = NativeLogAggregator::default();
    aggregator.process_line("sched_reserve:        CUDA0 compute buffer size =   400.00 MiB");
    aggregator.process_line("sched_reserve:      CUDA_Host compute buffer size =  128.00 MiB");
    aggregator.process_line("llama_kv_cache:      CUDA_Host KV buffer size =   64.00 MiB");
    assert_eq!(
        aggregator.measured_snapshot(),
        MeasuredNativeBuffers {
            compute_mib: Some(400.00),
            kv_mib: None,
            host_memory_observed: false,
        }
    );
}

#[test]
fn aggregator_reset_clears_stale_measured_buffers_on_model_load() {
    // Review blocker (PR #1719): the model-load "loaded meta data" line
    // fires reset_model_loading_state mid-open. A new model load
    // invalidates the previous model's measured buffer sizes (buffer
    // scales are model/shape-specific), so the reset clears them; buffer
    // lines emitted after the reset are measured normally, and the host
    // plan tuple (model/context/lanes) lives outside the aggregator
    // entirely so it is untouched by the reset.
    let mut aggregator = NativeLogAggregator::default();
    aggregator.process_line("sched_reserve:        CUDA0 compute buffer size =   579.83 MiB");
    // Use a fully parsable line so `process_line` actually invokes
    // reset_model_loading_state (see parse_loaded_metadata_counts).
    aggregator.process_line(
        "llama_model_loader: loaded meta data with 26 key-value pairs and 291 tensors from model.gguf (version GGUF V3)",
    );
    // The pre-reset measurement belongs to the previous model and must be
    // cleared, not stranded into the new model's footprint.
    assert_eq!(
        aggregator.measured_snapshot(),
        MeasuredNativeBuffers {
            compute_mib: None,
            kv_mib: None,
            host_memory_observed: false,
        }
    );
    aggregator.process_line("llama_kv_cache:        CUDA0 KV buffer size =  1088.00 MiB");
    assert_eq!(
        aggregator.measured_snapshot(),
        MeasuredNativeBuffers {
            compute_mib: None,
            kv_mib: Some(1088.00),
            host_memory_observed: false,
        }
    );
}

#[test]
fn aggregator_parses_measured_compute_buffer_size() {
    let mut aggregator = NativeLogAggregator::default();
    assert_eq!(
        aggregator.process_line("sched_reserve:        CUDA0 compute buffer size =   579.83 MiB"),
        vec![NativeLogEvent {
            message: "sched_reserve:        CUDA0 compute buffer size =   579.83 MiB".to_string(),
            category: "memory",
            params: vec![
                ("buffer_mib".to_string(), Value::from(579.83_f64)),
                (
                    "backend_device".to_string(),
                    Value::String("CUDA0".to_string())
                ),
            ],
        }]
    );
}

#[test]
fn aggregator_parses_measured_kv_buffer_size() {
    let mut aggregator = NativeLogAggregator::default();
    assert_eq!(
        aggregator.process_line("llama_kv_cache:        CUDA0 KV buffer size =  1088.00 MiB"),
        vec![NativeLogEvent {
            message: "llama_kv_cache:        CUDA0 KV buffer size =  1088.00 MiB".to_string(),
            category: "kv_cache",
            params: vec![
                ("buffer_mib".to_string(), Value::from(1088.00_f64)),
                (
                    "backend_device".to_string(),
                    Value::String("CUDA0".to_string())
                ),
            ],
        }]
    );
}

#[test]
fn aggregator_parses_metal_compute_buffer_size() {
    let mut aggregator = NativeLogAggregator::default();
    assert_eq!(
        aggregator.process_line("sched_reserve:        Metal compute buffer size =  312.50 MiB"),
        vec![NativeLogEvent {
            message: "sched_reserve:        Metal compute buffer size =  312.50 MiB".to_string(),
            category: "memory",
            params: vec![
                ("buffer_mib".to_string(), Value::from(312.5_f64)),
                (
                    "backend_device".to_string(),
                    Value::String("Metal".to_string())
                ),
            ],
        }]
    );
}

#[test]
fn aggregator_preserves_memory_summary_lines() {
    let mut aggregator = NativeLogAggregator::default();
    assert_eq!(
        aggregator.process_line("VRAM used: 12.4 GB"),
        vec![NativeLogEvent {
            message: "VRAM used: 12.4 GB".to_string(),
            category: "memory",
            params: Vec::new(),
        }]
    );
}

#[test]
fn aggregator_preserves_model_buffers_for_all_devices() {
    let mut aggregator = NativeLogAggregator::default();
    for device in ["CUDA0", "CUDA1", "Metal", "ROCm0", "Vulkan0", "CPU_Mapped"] {
        let line = format!("load_tensors: {device} model buffer size = 4321.00 MiB");
        let events = aggregator.process_line(&line);
        assert_eq!(events.len(), 1, "missing buffer for {device}");
        assert_eq!(events[0].message, line);
        assert_eq!(events[0].category, "memory");
    }
}

#[test]
fn aggregator_summarizes_unique_layer_assignments_including_output_layer() {
    let mut aggregator = NativeLogAggregator::default();
    aggregator.process_line("llama_model_loader: - kv 0: qwen35.block_count u32 = 4");
    for (layer, device) in [
        (0, "CUDA0"),
        (1, "CPU"),
        (2, "CUDA0"),
        (2, "CUDA0"),
        (3, "CUDA1"),
        (4, "CPU"),
    ] {
        let events = aggregator.process_line(&format!(
            "load_tensors: layer {layer} assigned to device {device}, is_swa = 0"
        ));
        assert!(
            events
                .iter()
                .all(|event| event.message != "Model layers by device")
        );
    }

    assert_eq!(
        aggregator.process_line("load_tensors: finished"),
        vec![NativeLogEvent {
            message: "Model layers by device".to_string(),
            category: "model",
            params: vec![
                ("CPU".to_string(), Value::from(2_u64)),
                ("CUDA0".to_string(), Value::from(2_u64)),
                ("CUDA1".to_string(), Value::from(1_u64)),
            ],
        }]
    );
    assert!(aggregator.process_line("load_tensors: finished").is_empty());
}

#[test]
fn aggregator_emits_only_changed_layer_device_counts() {
    let mut aggregator = NativeLogAggregator::default();
    aggregator.process_line("load_tensors: layer 0 assigned to device CUDA0");
    assert_eq!(aggregator.process_line("load_tensors: finished").len(), 1);
    aggregator.process_line("load_tensors: layer 0 assigned to device CUDA0");
    assert!(aggregator.process_line("load_tensors: finished").is_empty());

    aggregator.process_line("load_tensors: layer 0 assigned to device CPU");
    assert_eq!(
        aggregator.process_line("load_tensors: finished"),
        vec![NativeLogEvent {
            message: "Model layers by device".to_string(),
            category: "model",
            params: vec![("CPU".to_string(), Value::from(1_u64))],
        }]
    );
}

#[test]
fn aggregator_resets_layer_devices_for_each_model() {
    let mut aggregator = NativeLogAggregator::default();
    aggregator.process_line("load_tensors: layer 0 assigned to device CUDA0");
    let next_model = aggregator.process_line(
        "llama_model_loader: loaded meta data with 1 key-value pairs and 1 tensors from next.gguf (version GGUF V3)",
    );
    assert!(next_model.iter().any(|event| {
        event.message == "Model layers by device"
            && event.params == vec![("CUDA0".to_string(), Value::from(1_u64))]
    }));
    aggregator.process_line("load_tensors: layer 0 assigned to device Metal");
    assert_eq!(
        aggregator.process_line("load_tensors: finished"),
        vec![NativeLogEvent {
            message: "Model layers by device".to_string(),
            category: "model",
            params: vec![("Metal".to_string(), Value::from(1_u64))],
        }]
    );
}

#[test]
fn aggregator_tags_cpu_offload_evidence_without_capacity_facts() {
    let mut aggregator = NativeLogAggregator::default();
    let model_buffer =
        aggregator.process_line("load_tensors:   CPU_Mapped model buffer size = 47492.37 MiB");
    assert_eq!(
        model_buffer,
        vec![NativeLogEvent {
            message: "load_tensors:   CPU_Mapped model buffer size = 47492.37 MiB".to_string(),
            category: "memory",
            params: vec![
                (
                    "offload_device".to_string(),
                    Value::String("CPU".to_string())
                ),
                (
                    "offload_surface".to_string(),
                    Value::String("model_buffer".to_string())
                ),
            ],
        }]
    );
    assert_no_capacity_params(&model_buffer);

    assert_eq!(
        aggregator.process_line("llama_kv_cache:        CPU KV buffer size =  3264.00 MiB"),
        vec![NativeLogEvent {
            message: "llama_kv_cache:        CPU KV buffer size =  3264.00 MiB".to_string(),
            category: "kv_cache",
            params: vec![
                (
                    "offload_device".to_string(),
                    Value::String("CPU".to_string())
                ),
                (
                    "offload_surface".to_string(),
                    Value::String("kv_buffer".to_string())
                ),
            ],
        }]
    );
    assert_eq!(
        aggregator.process_line("sched_reserve:        CPU compute buffer size =   856.29 MiB"),
        vec![NativeLogEvent {
            message: "sched_reserve:        CPU compute buffer size =   856.29 MiB".to_string(),
            category: "memory",
            params: vec![
                (
                    "offload_device".to_string(),
                    Value::String("CPU".to_string())
                ),
                (
                    "offload_surface".to_string(),
                    Value::String("compute_buffer".to_string())
                ),
            ],
        }]
    );
}

fn assert_no_capacity_params(events: &[NativeLogEvent]) {
    const CAPACITY_KEYS: &[&str] = &[
        "backend_device",
        "capacity_gb",
        "gpu_count",
        "gpu_vram",
        "vram_bytes",
    ];
    assert!(events.iter().all(|event| {
        event
            .params
            .iter()
            .all(|(key, _)| !CAPACITY_KEYS.contains(&key.as_str()))
    }));
}

#[test]
fn aggregator_tracks_kv_cache_layer_progress_without_double_counting() {
    let mut aggregator = NativeLogAggregator::default();
    let plan = aggregator.process_line(
            "llama_kv_cache: size = 4096.00 MiB (131072 cells,   8 layers,  2/1 seqs), K (f16): 2048.00 MiB, V (f16): 2048.00 MiB",
        );
    assert_eq!(
        plan,
        vec![NativeLogEvent {
            message: "kv cache plan: layer rows=8".to_string(),
            category: "kv_cache",
            params: Vec::new(),
        }]
    );

    let first = aggregator.process_line("llama_kv_cache: layer   0: filtered");
    assert!(
        first
            .iter()
            .any(|event| event.message.contains("kv cache 10%"))
    );

    let duplicate = aggregator.process_line("llama_kv_cache: layer   0: dev = MTL0");
    assert!(duplicate.is_empty());

    for layer in 1..8 {
        aggregator.process_line(&format!("llama_kv_cache: layer   {layer}: filtered"));
    }

    let summary = aggregator.process_line("llama_kv_cache: attn_rot = 128");
    assert_eq!(
        summary,
        vec![NativeLogEvent {
            message: "llama_kv_cache: attn_rot = 128".to_string(),
            category: "kv_cache",
            params: Vec::new(),
        }]
    );
}

#[test]
fn aggregator_preserves_tokenizer_summary_lines() {
    let mut aggregator = NativeLogAggregator::default();
    assert_eq!(
        aggregator.process_line("init_tokenizer: initializing tokenizer for type 2"),
        vec![NativeLogEvent {
            message: "init_tokenizer: initializing tokenizer for type 2".to_string(),
            category: "tokenizer",
            params: Vec::new(),
        }]
    );
}

#[test]
fn aggregator_suppresses_print_info_lines() {
    let mut aggregator = NativeLogAggregator::default();
    assert!(
        aggregator
            .process_line("print_info: n_vocab               = 248320")
            .is_empty()
    );
}

#[test]
fn aggregator_rejects_empty_and_whitespace_lines() {
    let mut aggregator = NativeLogAggregator::default();
    assert!(aggregator.process_line("").is_empty());
    assert!(aggregator.process_line("   ").is_empty());
}

#[test]
fn aggregator_suppresses_raw_noise_lines() {
    let mut aggregator = NativeLogAggregator::default();
    assert!(
        aggregator
            .process_line("clip_model_loader: tensor[0]: n_dims = 1, name = v.blk.0.attn_out.bias")
            .is_empty()
    );
    assert!(
        aggregator
            .process_line("tokenizer.ggml.tokens arr[str,248320] = [\"!\", ...]")
            .is_empty()
    );
}

#[test]
fn parse_layer_assignment_extracts_layer_and_device() {
    assert_eq!(
        parse_layer_assignment("load_tensors: layer   0 assigned to device CUDA0"),
        Some((0, "CUDA0"))
    );
    assert_eq!(
        parse_layer_assignment("load_tensors: layer  63 assigned to device CUDA0, is_swa = 0"),
        Some((63, "CUDA0"))
    );
    assert_eq!(
        parse_layer_assignment("load_tensors: layer   5 assigned to device CPU, is_swa = 1"),
        Some((5, "CPU"))
    );
    assert_eq!(
        parse_layer_assignment("llm_load_tensors: offloaded 64/65 layers"),
        None
    );
    assert_eq!(
        parse_layer_assignment("load_tensors: layer   0 computation graph"),
        None
    );
    assert_eq!(
        parse_layer_assignment("load_tensors: layer x assigned to device CUDA0"),
        None
    );
    assert_eq!(
        parse_layer_assignment("load_tensors: layer 0 assigned to device , is_swa = 0"),
        None
    );
}

#[test]
fn aggregator_tracks_layer_assign_progress_using_block_count() {
    let mut aggregator = NativeLogAggregator::default();

    aggregator.process_line(
            "llama_model_loader: loaded meta data with 10 key-value pairs and 100 tensors from model.gguf (version GGUF V3)"
        );
    for line in [
        "llama_model_loader: - kv   0: general.architecture str = qwen35",
        "llama_model_loader: - kv   1: general.name str = Qwen 3.5 4B",
        "llama_model_loader: - kv   2: general.type str = model",
        "llama_model_loader: - kv   3: general.size_label str = 4B",
        "llama_model_loader: - kv   4: qwen35.context_length u32 = 40960",
        "llama_model_loader: - kv   5: qwen35.block_count u32 = 4",
        "llama_model_loader: - kv   6: qwen35.embedding_length u32 = 2560",
        "llama_model_loader: - kv   7: qwen35.feed_forward_length u32 = 9728",
        "llama_model_loader: - kv   8: qwen35.attention.head_count u32 = 32",
        "llama_model_loader: - kv   9: qwen35.attention.head_count_kv u32 = 8",
    ] {
        aggregator.process_line(line);
    }
    aggregator.process_line("llm_load_print_meta: version = 3");

    let e0 = aggregator.process_line("load_tensors: layer   0 assigned to device CUDA0");
    assert!(e0.iter().any(|event| event.message.contains("layers 10%")));
    assert!(e0.iter().any(|event| event.message.contains("layers 20%")));

    let e1 = aggregator.process_line("load_tensors: layer   1 assigned to device CUDA0");
    assert!(e1.iter().any(|event| event.message.contains("layers 50%")));

    let e2 = aggregator.process_line("load_tensors: layer   2 assigned to device CUDA0");
    assert!(e2.iter().any(|event| event.message.contains("layers 70%")));

    let e3 = aggregator.process_line("load_tensors: layer   3 assigned to device CUDA0");
    let pcts: Vec<&str> = e3
        .iter()
        .filter_map(|ev| {
            if ev.message.contains("layers") && ev.message.contains('%') {
                Some(ev.message.as_str())
            } else {
                None
            }
        })
        .collect();
    assert!(
        pcts.iter().any(|m| m.contains("100%")),
        "expected layers 100% at final layer, got {:?}",
        pcts
    );
}
