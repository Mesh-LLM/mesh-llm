#[test]
fn test_build_gpus_both_none() {
    let result = build_gpus(None, None, None, None, None, None);
    assert!(result.is_empty(), "expected empty vec when no gpu_name");
}
#[test]
fn test_build_gpus_single_no_vram() {
    let result = build_gpus(Some("NVIDIA RTX 5090"), None, None, None, None, None);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].name, "NVIDIA RTX 5090");
    assert_eq!(result[0].vram_bytes, 0);
}

#[test]
fn test_build_gpus_single_with_vram() {
    let result = build_gpus(
        Some("NVIDIA RTX 5090"),
        Some("34359738368"),
        None,
        None,
        None,
        None,
    );
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].name, "NVIDIA RTX 5090");
    assert_eq!(result[0].vram_bytes, 34_359_738_368);
}

#[test]
fn test_build_gpus_multi_full_vram() {
    let result = build_gpus(
        Some("NVIDIA RTX 5090, NVIDIA RTX 3080"),
        Some("34359738368,10737418240"),
        None,
        None,
        None,
        None,
    );
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].name, "NVIDIA RTX 5090");
    assert_eq!(result[0].vram_bytes, 34_359_738_368);
    assert_eq!(result[1].name, "NVIDIA RTX 3080");
    assert_eq!(result[1].vram_bytes, 10_737_418_240);
}

#[test]
fn test_build_gpus_multi_full_vram_without_space_after_comma() {
    let result = build_gpus(
        Some("NVIDIA RTX 5090,NVIDIA RTX 3080"),
        Some("34359738368,10737418240"),
        None,
        None,
        None,
        None,
    );
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].name, "NVIDIA RTX 5090");
    assert_eq!(result[1].name, "NVIDIA RTX 3080");
    assert_eq!(result[0].vram_bytes, 34_359_738_368);
    assert_eq!(result[1].vram_bytes, 10_737_418_240);
}

#[test]
fn test_build_gpus_multi_names_trim_whitespace() {
    let result = build_gpus(
        Some(" GPU0 ,GPU1 ,  GPU2  "),
        Some("100,200,300"),
        None,
        None,
        None,
        None,
    );
    assert_eq!(result.len(), 3);
    assert_eq!(result[0].name, "GPU0");
    assert_eq!(result[1].name, "GPU1");
    assert_eq!(result[2].name, "GPU2");
}

#[test]
fn test_build_gpus_expands_summarized_identical_names() {
    let result = build_gpus(
        Some("2× NVIDIA A100"),
        Some("85899345920,85899345920"),
        None,
        Some("1948.70,1948.70"),
        None,
        None,
    );
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].name, "NVIDIA A100");
    assert_eq!(result[1].name, "NVIDIA A100");
    assert_eq!(result[0].vram_bytes, 85_899_345_920);
    assert_eq!(result[1].vram_bytes, 85_899_345_920);
    assert_eq!(result[0].mem_bandwidth_gbps, Some(1948.70));
    assert_eq!(result[1].mem_bandwidth_gbps, Some(1948.70));
}

#[test]
fn test_build_gpus_multi_partial_vram() {
    let result = build_gpus(
        Some("NVIDIA RTX 5090, NVIDIA RTX 3080"),
        Some("34359738368"),
        None,
        None,
        None,
        None,
    );
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].vram_bytes, 34_359_738_368);
    assert_eq!(
        result[1].vram_bytes, 0,
        "missing VRAM entry should default to 0"
    );
}

#[test]
fn test_build_gpus_vram_no_gpu_name() {
    let result = build_gpus(None, Some("34359738368"), None, None, None, None);
    assert!(
        result.is_empty(),
        "no gpu_name means no entries even if vram present"
    );
}

#[test]
fn test_build_gpus_vram_whitespace_trimmed() {
    let result = build_gpus(
        Some("NVIDIA RTX 4090"),
        Some(" 25769803776 "),
        None,
        None,
        None,
        None,
    );
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].vram_bytes, 25_769_803_776);
}

#[test]
fn test_build_gpus_with_bandwidth() {
    let result = build_gpus(
        Some("NVIDIA A100, NVIDIA A6000"),
        Some("85899345920,51539607552"),
        None,
        Some("1948.70,780.10"),
        None,
        None,
    );
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].mem_bandwidth_gbps, Some(1948.70));
    assert_eq!(result[1].mem_bandwidth_gbps, Some(780.10));
}

#[test]
fn test_build_gpus_unparsable_vram_preserves_index() {
    let result = build_gpus(
        Some("GPU0, GPU1, GPU2"),
        Some("100,foo,300"),
        None,
        None,
        None,
        None,
    );
    assert_eq!(result.len(), 3);
    assert_eq!(result[0].vram_bytes, 100);
    assert_eq!(
        result[1].vram_bytes, 0,
        "unparsable vram should default to 0, not shift indices"
    );
    assert_eq!(result[2].vram_bytes, 300);
}

#[test]
fn test_build_gpus_unparsable_bandwidth_preserves_index() {
    let result = build_gpus(
        Some("GPU0, GPU1, GPU2"),
        Some("100,200,300"),
        None,
        Some("1.0,bad,3.0"),
        None,
        None,
    );
    assert_eq!(result.len(), 3);
    assert_eq!(result[0].mem_bandwidth_gbps, Some(1.0));
    assert_eq!(
        result[1].mem_bandwidth_gbps, None,
        "unparsable bandwidth should be None, not shift indices"
    );
    assert_eq!(result[2].mem_bandwidth_gbps, Some(3.0));
}

#[test]
fn test_build_gpus_with_both_tflops_precisions() {
    let result = build_gpus(
        Some("GPU0, GPU1"),
        Some("100,200"),
        None,
        None,
        Some("312.5,419.5"),
        Some("625.0,839.0"),
    );
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].compute_tflops_fp32, Some(312.5));
    assert_eq!(result[0].compute_tflops_fp16, Some(625.0));
    assert_eq!(result[1].compute_tflops_fp32, Some(419.5));
    assert_eq!(result[1].compute_tflops_fp16, Some(839.0));
}

#[test]
fn test_build_gpus_fp32_only_fp16_absent() {
    let result = build_gpus(
        Some("GPU0, GPU1"),
        Some("100,200"),
        None,
        None,
        Some("312.5,bad"),
        None,
    );
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].compute_tflops_fp32, Some(312.5));
    assert_eq!(result[1].compute_tflops_fp32, None);
    assert!(result.iter().all(|gpu| gpu.compute_tflops_fp16.is_none()));
}

#[test]
fn test_gpu_entry_omits_tflops_when_none() {
    let value = serde_json::to_value(build_gpus(
        Some("NVIDIA A100"),
        Some("85899345920"),
        None,
        Some("1948.70"),
        None,
        None,
    ))
    .unwrap();

    let first = value.as_array().unwrap().first().unwrap();
    assert!(first.get("compute_tflops_fp32").is_none());
    assert!(first.get("compute_tflops_fp16").is_none());
    assert!(first.get("mem_bandwidth_gbps").is_some());
}

#[test]
fn test_api_status_gpu_entry_uses_new_name() {
    let value = serde_json::to_value(build_gpus(
        Some("NVIDIA A100"),
        Some("85899345920"),
        None,
        Some("1948.70"),
        None,
        None,
    ))
    .unwrap();

    let first = value.as_array().unwrap().first().unwrap();
    assert_eq!(first.get("mem_bandwidth_gbps").unwrap(), &json!(1948.7));
    assert!(
        first.get("bandwidth_gbps").is_none(),
        "API status JSON should use mem_bandwidth_gbps"
    );
}

#[test]
fn test_build_gpus_with_reserved_bytes_preserves_index() {
    let result = build_gpus(
        Some("GPU0, GPU1, GPU2"),
        Some("100,200,300"),
        Some("10,,30"),
        None,
        None,
        None,
    );
    assert_eq!(result.len(), 3);
    assert_eq!(result[0].reserved_bytes, Some(10));
    assert_eq!(result[1].reserved_bytes, None);
    assert_eq!(result[2].reserved_bytes, Some(30));
}

#[test]
fn test_gpu_entry_omits_reserved_bytes_when_none() {
    let value = serde_json::to_value(build_gpus(
        Some("NVIDIA A100"),
        Some("85899345920"),
        None,
        Some("1948.70"),
        None,
        None,
    ))
    .unwrap();

    let first = value.as_array().unwrap().first().unwrap();
    assert!(first.get("reserved_bytes").is_none());
}

#[test]
fn test_http_body_text_extracts_body() {
    let raw = b"POST /api/plugins/x/tools/y HTTP/1.1\r\nHost: localhost\r\nContent-Length: 7\r\n\r\n{\"a\":1}";
    assert_eq!(http_body_text(raw), "{\"a\":1}");
}
