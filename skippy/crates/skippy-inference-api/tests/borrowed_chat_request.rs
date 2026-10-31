//! Preserve the plugin's exact wire bytes without copying large request bodies.
use serde_json::{Value, json};
use skippy_inference_api::{
    chat::{ChatCompletionRequest, request_view::ChatCompletionRequestRef},
    generation::GenerationRequest,
};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

thread_local! {
    static ALLOCATIONS: Cell<Option<usize>> = const { Cell::new(None) };
}

struct CountingAllocator;

// SAFETY: all allocation operations delegate unchanged to the system allocator.
// The counter is thread-local and does not allocate or retain pointers.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|count| count.set(count.get().map(|n| n + 1)));
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCATIONS.with(|count| count.set(count.get().map(|n| n + 1)));
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn assert_wire_parity(wire: ChatCompletionRequest) {
    let expected = serde_json::to_vec(&wire).unwrap();
    let request = GenerationRequest::from(wire);
    let actual = serde_json::to_vec(&ChatCompletionRequestRef(&request)).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn borrowed_projection_preserves_characterized_backend_requests() {
    let cases: Value =
        serde_json::from_str(include_str!("fixtures/generation-wire-main.json")).unwrap();
    let mut requests = 0;
    for case in cases.as_array().unwrap() {
        for backend in case["backend"].as_array().unwrap() {
            let Some(request) = backend.get("request") else {
                continue;
            };
            assert_wire_parity(serde_json::from_value(request.clone()).unwrap());
            requests += 1;
        }
    }
    assert!(
        requests >= 20,
        "the golden corpus must exercise real requests"
    );
}

#[test]
fn borrowed_projection_preserves_options_extensions_and_duplicate_keys() {
    let mut wire: ChatCompletionRequest = serde_json::from_value(json!({
        "model":"model", "messages":[
            {"role":"assistant","tool_calls":[]},
            {"role":"assistant","content":null,"reasoning_content":"reason"},
            {"role":"user","content":"quote: \" slash: \\ newline: \n"}
        ],
        "stream":true, "max_tokens":12, "max_completion_tokens":13,
        "temperature":0.3, "top_p":0.9, "n":1, "logprobs":true, "top_logprobs":2,
        "presence_penalty":0.1, "frequency_penalty":0.2, "logit_bias":{"42":-2},
        "response_format":{"type":"json_object"}, "tools":[], "tool_choice":"auto",
        "parallel_tool_calls":false, "user":"user", "stop":["end"], "seed":3,
        "reasoning":{"effort":"high"}, "reasoning_effort":"low",
        "prompt_cache_key":"key", "prompt_cache_retention":"in_memory",
        "stream_options":{"include_usage":true}, "extension":{"opaque":[1,null]}
    }))
    .unwrap();
    // Programmatic backend extensions can collide with known fields. Preserve
    // the existing flatten order and duplicate bytes used by the wire digest.
    wire.extra.insert("model".into(), json!("extension-model"));
    assert_wire_parity(wire);
    assert_wire_parity(ChatCompletionRequest::default());
}

#[test]
fn serializing_multimodal_payloads_borrows_without_heap_allocations() {
    let payload = "A".repeat(1024 * 1024);
    let wire: ChatCompletionRequest = serde_json::from_value(json!({
        "model":"model", "messages":[{"role":"user","content":[
            {"type":"image_url","image_url":{"url":format!("data:image/png;base64,{payload}")}},
            {"type":"input_audio","input_audio":{"data":payload,"format":"wav"}}
        ]}],
        "tools":[{"type":"function","function":{"name":"example","parameters":{"type":"object"}}}],
        "opaque":{"nested":["preserve",null]}
    }))
    .unwrap();
    let expected = serde_json::to_vec(&wire).unwrap();
    let request = GenerationRequest::from(wire);
    let mut output = Vec::with_capacity(expected.len());
    ALLOCATIONS.with(|count| count.set(Some(0)));
    let result = serde_json::to_writer(&mut output, &ChatCompletionRequestRef(&request));
    let allocations = ALLOCATIONS.with(|count| count.replace(None).unwrap());
    result.unwrap();
    assert_eq!(output, expected);
    assert_eq!(
        allocations, 0,
        "serialization must borrow the payload and extensions"
    );
}

#[test]
fn wire_validation_adds_no_allocations_over_generation_validation() {
    let payload = "A".repeat(1024 * 1024);
    let mut wire: ChatCompletionRequest = serde_json::from_value(json!({
        "model":"model", "messages":[{"role":"user","content":[
            {"type":"image_url","image_url":{"url":format!("data:image/png;base64,{payload}")}},
            {"type":"input_audio","input_audio":{"data":payload,"format":"wav"}}
        ]}],
        "tools":[{"type":"function","function":{"name":"example","parameters":{"type":"object"}}}],
        "opaque":{"nested":["preserve",null]}
    }))
    .unwrap();
    for valid in [true, false] {
        if !valid {
            wire.model.clear();
        }
        let request = GenerationRequest::from(wire.clone());
        ALLOCATIONS.with(|count| count.set(Some(0)));
        let generation_result = request.validate();
        let generation_allocations = ALLOCATIONS.with(|count| count.replace(None).unwrap());
        ALLOCATIONS.with(|count| count.set(Some(0)));
        let wire_result = wire.validate();
        let wire_allocations = ALLOCATIONS.with(|count| count.replace(None).unwrap());
        assert_eq!(generation_result.is_ok(), valid);
        assert_eq!(wire_result.is_ok(), valid);
        assert_eq!(
            wire_allocations, generation_allocations,
            "wire validation must not clone messages, media, tools or extensions"
        );
    }
}
