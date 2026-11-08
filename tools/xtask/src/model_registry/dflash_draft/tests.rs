use std::path::Path;

use serde_json::json;

use super::gguf::{TensorType, f32_to_f16, quantize_q8_0};
use super::laguna::{Source, plan, size_label};
use super::safetensors::{Safetensors, f16_to_f32};
use super::*;

#[test]
fn f16_conversion_rounds_to_nearest_even() {
    assert_eq!(f32_to_f16(1.0), 0x3c00);
    assert_eq!(f32_to_f16(-2.0), 0xc000);
    assert_eq!(f32_to_f16(65504.0), 0x7bff);
    assert_eq!(f32_to_f16(1.0e6), 0x7c00);
    assert_eq!(f32_to_f16(0.0), 0);
    // Halfway between 1.0 and the next f16 rounds to even (down)...
    assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3c00);
    // ...and halfway above an odd mantissa rounds up.
    assert_eq!(f32_to_f16(1.0 + 3.0 * 2f32.powi(-11)), 0x3c02);
    // The smallest subnormal survives; half of it rounds to zero.
    assert_eq!(f32_to_f16(2f32.powi(-24)), 0x0001);
    assert_eq!(f32_to_f16(2f32.powi(-25)), 0x0000);
    // Every finite f16 converts to f32 and back unchanged.
    for bits in 0..=u16::MAX {
        if bits & 0x7c00 == 0x7c00 {
            continue;
        }
        assert_eq!(f32_to_f16(f16_to_f32(bits)), bits, "{bits:#06x}");
    }
}

#[test]
fn q8_0_matches_the_reference_rounding() {
    let mut block: Vec<f32> = (0..32).map(|i| i as f32 - 16.0).collect();
    block[0] = -127.0;
    let out = quantize_q8_0(&block).unwrap();
    assert_eq!(out.len(), 34);
    // d = 127 / 127 = 1, so every value is stored as itself.
    assert_eq!(u16::from_le_bytes([out[0], out[1]]), f32_to_f16(1.0));
    assert_eq!(out[2] as i8, -127);
    assert_eq!(out[3] as i8, -15);
    assert_eq!(out[33] as i8, 15);

    // Halves round away from zero: amax 254 gives d = 2.
    let mut halves = vec![0.0f32; 32];
    halves[0] = 254.0;
    halves[1] = 5.0;
    halves[2] = -5.0;
    let out = quantize_q8_0(&halves).unwrap();
    assert_eq!((out[3] as i8, out[4] as i8), (3, -3));

    // An all-zero block stores d = 0 and zeros.
    assert!(quantize_q8_0(&[0.0; 32]).unwrap().iter().all(|b| *b == 0));
    assert!(quantize_q8_0(&[1.0; 31]).is_err());
    assert!(quantize_q8_0(&[f32::NAN; 32]).is_err());
}

#[test]
fn size_labels_follow_llama_cpp() {
    assert_eq!(size_label(462_064_896), "462M");
    assert_eq!(size_label(7_241_732_096), "7.2B");
    assert_eq!(size_label(1_500), "1.5K");
    assert_eq!(size_label(30_000_000_000), "30B");
}

/// Writes a SafeTensors file of BF16 tensors whose values are their indices.
fn write_safetensors(path: &Path, tensors: &[(&str, Vec<u64>)]) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape) in tensors {
        let count: u64 = shape.iter().product();
        let start = data.len() as u64;
        for i in 0..count {
            let value = ((i % 64) as f32 - 32.0) / 8.0;
            data.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
        }
        header.insert(
            name.to_string(),
            json!({"dtype": "BF16", "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(data);
    std::fs::write(path, bytes).unwrap();
}

const HIDDEN: u64 = 64;
const HEADS: u64 = 2;
const KV_HEADS: u64 = 1;
const HEAD_DIM: u64 = 32;

fn draft_config() -> serde_json::Value {
    json!({
        "architectures": ["DFlashLagunaForCausalLM"],
        "model_type": "laguna",
        "attention_bias": false,
        "hidden_act": "silu",
        "head_dim": HEAD_DIM,
        "hidden_size": HIDDEN,
        "intermediate_size": 128,
        "max_position_embeddings": 4096,
        "num_attention_heads": HEADS,
        "num_hidden_layers": 1,
        "num_key_value_heads": KV_HEADS,
        "rms_norm_eps": 1e-6,
        "rope_theta": 500000.0,
        "sliding_window": 512,
        "layer_types": ["sliding_attention"],
        "gating": "per-head",
        "vocab_size": 8,
        "draft_vocab_size": 8,
        "num_experts": 0,
        "dflash_config": {"block_size": 16, "mask_token_id": 4, "target_layer_ids": [0, 2], "causal": true}
    })
}

/// A one-layer Laguna drafter and a byte-level BPE target tokenizer.
fn fixture(dir: &Path, config: &serde_json::Value) -> (std::path::PathBuf, std::path::PathBuf) {
    let source = dir.join("Tiny-Laguna-DFlash");
    let target = dir.join("target");
    std::fs::create_dir_all(&source).unwrap();
    std::fs::create_dir_all(&target).unwrap();
    std::fs::write(source.join("config.json"), config.to_string()).unwrap();
    let qkv = HEADS * HEAD_DIM + 2 * KV_HEADS * HEAD_DIM;
    write_safetensors(
        &source.join("model.safetensors"),
        &[
            ("aux_hidden_norms.0.weight", vec![HIDDEN]),
            ("aux_hidden_norms.1.weight", vec![HIDDEN]),
            ("fc.weight", vec![HIDDEN, 2 * HIDDEN]),
            ("hidden_norm.weight", vec![HIDDEN]),
            ("layers.0.input_layernorm.weight", vec![HIDDEN]),
            ("layers.0.mlp.down_proj.weight", vec![HIDDEN, 128]),
            ("layers.0.mlp.gate_proj.weight", vec![128, HIDDEN]),
            ("layers.0.mlp.up_proj.weight", vec![128, HIDDEN]),
            ("layers.0.post_attention_layernorm.weight", vec![HIDDEN]),
            ("layers.0.self_attn.g_proj.weight", vec![HEADS, HIDDEN]),
            ("layers.0.self_attn.k_norm.weight", vec![HEAD_DIM]),
            (
                "layers.0.self_attn.o_proj.weight",
                vec![HIDDEN, HEADS * HEAD_DIM],
            ),
            ("layers.0.self_attn.q_norm.weight", vec![HEAD_DIM]),
            ("layers.0.self_attn.qkv_proj.weight", vec![qkv, HIDDEN]),
            ("norm.weight", vec![HIDDEN]),
        ],
    );
    std::fs::write(
        target.join("config.json"),
        json!({"model_type": "laguna"}).to_string(),
    )
    .unwrap();
    write_tokenizer(&target, |_| {});
    std::fs::write(
        target.join("tokenizer_config.json"),
        json!({"bos_token": "<EOS>", "eos_token": "<EOS>", "mask_token": "<MASK>", "chat_template": "{{ x }}"})
            .to_string(),
    )
    .unwrap();
    (source, target)
}

/// The fixture's target files, as a manifest would list them.
fn target_files() -> Vec<String> {
    ["config.json", "tokenizer.json", "tokenizer_config.json"]
        .map(String::from)
        .to_vec()
}

/// Writes a small byte-level BPE tokenizer with Laguna's preprocessing,
/// after `edit`.
fn write_tokenizer(target: &Path, edit: impl FnOnce(&mut serde_json::Value)) {
    let mut tokenizer = json!({
            "model": {"type": "BPE", "vocab": {"a": 2, "b": 3, "ab": 5}, "merges": [["a", "b"]]},
            "decoder": {"type": "ByteLevel"},
            "added_tokens": [
                {"id": 0, "content": "<EOS>", "special": true, "normalized": false},
                {"id": 1, "content": "<|tool|>", "special": false, "normalized": false},
                {"id": 4, "content": "<MASK>", "special": true, "normalized": false},
                {"id": 6, "content": "x\u{2581}y", "special": false, "normalized": false}
            ],
            "post_processor": {
                "type": "TemplateProcessing",
                "single": [{"SpecialToken": {"id": "<EOS>"}}, {"Sequence": {"id": "A"}}],
                "pair": [{"SpecialToken": {"id": "<EOS>"}}, {"Sequence": {"id": "A"}},
                         {"SpecialToken": {"id": "<EOS>"}}, {"Sequence": {"id": "B"}}]
            }
    });
    let preprocessing: serde_json::Value =
        serde_json::from_str(vocab::LAGUNA_PREPROCESSING).unwrap();
    for key in ["normalizer", "pre_tokenizer"] {
        tokenizer[key] = preprocessing[key].clone();
    }
    edit(&mut tokenizer);
    std::fs::write(target.join("tokenizer.json"), tokenizer.to_string()).unwrap();
}

/// A tensor as a GGUF lists it: name, dims and type id.
type TensorInfo = (String, Vec<u64>, u32);

/// Header keys and tensors of a written GGUF.
fn read_header(path: &Path) -> (Vec<String>, Vec<TensorInfo>) {
    let bytes = std::fs::read(path).unwrap();
    let mut at = 8;
    let u64_at = |bytes: &[u8], at: &mut usize| {
        let v = u64::from_le_bytes(bytes[*at..*at + 8].try_into().unwrap());
        *at += 8;
        v
    };
    let tensors = u64_at(&bytes, &mut at);
    let kvs = u64_at(&bytes, &mut at);
    fn string(bytes: &[u8], at: &mut usize) -> String {
        let len = u64::from_le_bytes(bytes[*at..*at + 8].try_into().unwrap()) as usize;
        *at += 8;
        let s = String::from_utf8(bytes[*at..*at + len].to_vec()).unwrap();
        *at += len;
        s
    }
    fn skip(bytes: &[u8], at: &mut usize, kind: u32) {
        match kind {
            4..=6 => *at += 4,
            7 => *at += 1,
            8 => {
                string(bytes, at);
            }
            9 => {
                let inner = u32::from_le_bytes(bytes[*at..*at + 4].try_into().unwrap());
                *at += 4;
                let n = u64::from_le_bytes(bytes[*at..*at + 8].try_into().unwrap());
                *at += 8;
                (0..n).for_each(|_| skip(bytes, at, inner));
            }
            other => panic!("type {other}"),
        }
    }
    let mut keys = Vec::new();
    for _ in 0..kvs {
        keys.push(string(&bytes, &mut at));
        let kind = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        at += 4;
        skip(&bytes, &mut at, kind);
    }
    let mut infos = Vec::new();
    for _ in 0..tensors {
        let name = string(&bytes, &mut at);
        let n = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        at += 4;
        let dims = (0..n).map(|_| u64_at(&bytes, &mut at)).collect();
        let kind = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        at += 12;
        infos.push((name, dims, kind));
    }
    (keys, infos)
}

#[test]
fn a_laguna_drafter_converts_to_the_dflash_layout() {
    let dir = tempfile::tempdir().unwrap();
    let (source, target) = fixture(dir.path(), &draft_config());
    let output = dir.path().join("out.gguf");
    convert(
        &source,
        &target,
        &target_files(),
        "Tiny Laguna DFlash",
        &output,
    )
    .unwrap();
    let (keys, tensors) = read_header(&output);
    assert_eq!(
        keys,
        [
            "general.architecture",
            "general.type",
            "general.name",
            "general.size_label",
            "dflash.block_count",
            "dflash.context_length",
            "dflash.embedding_length",
            "dflash.feed_forward_length",
            "dflash.attention.head_count",
            "dflash.attention.head_count_kv",
            "dflash.rope.freq_base",
            "dflash.attention.layer_norm_rms_epsilon",
            "dflash.attention.key_length",
            "dflash.attention.value_length",
            "general.file_type",
            "dflash.block_size",
            "dflash.target_layers",
            "dflash.attention.causal",
            "dflash.decoder_arch",
            "dflash.attention.sliding_window",
            "dflash.attention.sliding_window_pattern",
            "general.quantization_version",
            "tokenizer.ggml.model",
            "tokenizer.ggml.pre",
            "tokenizer.ggml.tokens",
            "tokenizer.ggml.token_type",
            "tokenizer.ggml.merges",
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.mask_token_id",
            "tokenizer.ggml.add_bos_token",
            "tokenizer.ggml.add_sep_token",
            "tokenizer.chat_template",
        ]
    );
    let names: Vec<(&str, &[u64], u32)> = tensors
        .iter()
        .map(|(n, d, k)| (n.as_str(), d.as_slice(), *k))
        .collect();
    assert_eq!(
        names,
        [
            ("enc.aux_norm.weight", &[64, 2][..], 0),
            ("fc.weight", &[128, 64][..], 8),
            ("enc.output_norm.weight", &[64][..], 0),
            ("blk.0.attn_norm.weight", &[64][..], 0),
            ("blk.0.ffn_down.weight", &[128, 64][..], 8),
            ("blk.0.ffn_gate.weight", &[64, 128][..], 8),
            ("blk.0.ffn_up.weight", &[64, 128][..], 8),
            ("blk.0.ffn_norm.weight", &[64][..], 0),
            ("blk.0.attn_gate.weight", &[64, 2][..], 8),
            ("blk.0.attn_k_norm.weight", &[32][..], 0),
            ("blk.0.attn_output.weight", &[64, 64][..], 8),
            ("blk.0.attn_q_norm.weight", &[32][..], 0),
            ("blk.0.attn_q.weight", &[64, 64][..], 8),
            ("blk.0.attn_k.weight", &[64, 32][..], 8),
            ("blk.0.attn_v.weight", &[64, 32][..], 8),
            ("output_norm.weight", &[64][..], 0),
        ]
    );
    // The file ends with the last tensor, padded to the alignment.
    assert_eq!(std::fs::metadata(&output).unwrap().len() % 32, 0);
}

#[test]
fn the_fused_qkv_splits_by_rows() {
    let dir = tempfile::tempdir().unwrap();
    let (source, _) = fixture(dir.path(), &draft_config());
    let checkpoint = Safetensors::open(&source.join("model.safetensors")).unwrap();
    let plan = plan(&source, &checkpoint, "x").unwrap();
    let rows: Vec<(u64, u64)> = plan
        .tensors
        .iter()
        .filter_map(|(_, source)| match source {
            Source::Rows { start, rows, .. } => Some((*start, *rows)),
            _ => None,
        })
        .collect();
    assert_eq!(rows, [(0, 64), (64, 32), (96, 32)]);
    assert!(
        plan.tensors
            .iter()
            .all(|(entry, _)| entry.kind == TensorType::F32 || entry.dims[0] % 32 == 0)
    );
}

#[test]
fn the_tokenizer_follows_llama_cpp_types_and_special_tokens() {
    let dir = tempfile::tempdir().unwrap();
    let (_, target) = fixture(dir.path(), &draft_config());
    let metadata = vocab::metadata(&target, &target_files(), 8).unwrap();
    let get = |key: &str| {
        metadata
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.clone())
    };
    assert_eq!(
        get("tokenizer.ggml.tokens"),
        Some(gguf::Value::Strs(
            [
                "<EOS>", "<|tool|>", "a", "b", "<MASK>", "ab", "x y", "[PAD7]"
            ]
            .map(String::from)
            .to_vec()
        ))
    );
    // Special and special-looking added tokens are control tokens; other
    // added tokens are user-defined; a missing id is unused.
    assert_eq!(
        get("tokenizer.ggml.token_type"),
        Some(gguf::Value::I32s(vec![3, 3, 1, 1, 3, 1, 4, 5]))
    );
    assert_eq!(
        get("tokenizer.ggml.merges"),
        Some(gguf::Value::Strs(vec!["a b".into()]))
    );
    assert_eq!(
        get("tokenizer.ggml.pre"),
        Some(gguf::Value::Str("laguna".into()))
    );
    assert_eq!(
        get("tokenizer.ggml.bos_token_id"),
        Some(gguf::Value::U32(0))
    );
    assert_eq!(
        get("tokenizer.ggml.add_bos_token"),
        Some(gguf::Value::Bool(true))
    );
    assert_eq!(
        get("tokenizer.ggml.add_sep_token"),
        Some(gguf::Value::Bool(true))
    );
    assert_eq!(
        get("tokenizer.chat_template"),
        Some(gguf::Value::Str("{{ x }}".into()))
    );
    assert!(
        vocab::metadata(&target, &target_files(), 6).is_err(),
        "ids beyond vocab_size"
    );
}

#[test]
fn an_unknown_tokenizer_or_config_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = draft_config();
    config["partial_rotary_factor"] = json!(0.5);
    let (source, target) = fixture(dir.path(), &config);
    let checkpoint = Safetensors::open(&source.join("model.safetensors")).unwrap();
    let error = plan(&source, &checkpoint, "x").err().unwrap().to_string();
    assert!(error.contains("partial_rotary_factor"), "{error}");

    std::fs::write(
        target.join("config.json"),
        json!({"model_type": "qwen3"}).to_string(),
    )
    .unwrap();
    let error = vocab::metadata(&target, &target_files(), 8)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("qwen3"), "{error}");
}

fn manifest(
    dir: &Path,
    source: &Path,
    target: &Path,
    sha256: &str,
    bytes: u64,
) -> std::path::PathBuf {
    let hash = |path: &Path| file_sha256(path).unwrap().0;
    let manifest = json!({"schema_version": 1, "drafts": [{
        "name": "tiny",
        "source": {"repo": "x/Tiny-Laguna-DFlash", "revision": "0".repeat(40), "files": [
            {"path": "config.json", "sha256": hash(&source.join("config.json"))},
            {"path": "model.safetensors", "sha256": hash(&source.join("model.safetensors"))}]},
        "target_model": {"repo": "x/target", "revision": "0".repeat(40), "files": [
            {"path": "config.json", "sha256": hash(&target.join("config.json"))},
            {"path": "tokenizer.json", "sha256": hash(&target.join("tokenizer.json"))},
            {"path": "tokenizer_config.json", "sha256": hash(&target.join("tokenizer_config.json"))}]},
        "convert": {"converter": "dflash-laguna", "outtype": "q8_0"},
        "output": {"repo": "meshllm/tiny", "file": "tiny.gguf", "sha256": sha256, "bytes": bytes}
    }]});
    let path = dir.join("manifest.json");
    std::fs::write(&path, manifest.to_string()).unwrap();
    path
}

fn args(manifest: &Path, source: &Path, target: &Path, output: &Path) -> Vec<String> {
    [
        "--manifest",
        manifest.to_str().unwrap(),
        "--draft",
        "tiny",
        "--source",
        source.to_str().unwrap(),
        "--target-model",
        target.to_str().unwrap(),
        "--output",
        output.to_str().unwrap(),
    ]
    .map(String::from)
    .to_vec()
}

#[test]
fn the_output_is_kept_only_if_it_matches_the_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let (source, target) = fixture(dir.path(), &draft_config());
    let reference = dir.path().join("reference.gguf");
    convert(
        &source,
        &target,
        &target_files(),
        "Tiny Laguna DFlash",
        &reference,
    )
    .unwrap();
    let (sha256, bytes) = file_sha256(&reference).unwrap();
    let output = dir.path().join("tiny.gguf");

    // A conversion that differs from the manifest is refused and removed.
    let wrong = manifest(dir.path(), &source, &target, &"0".repeat(64), bytes);
    let error = run(&args(&wrong, &source, &target, &output))
        .unwrap_err()
        .to_string();
    assert!(error.contains(&sha256), "{error}");
    assert!(!output.exists());
    assert!(!output.with_extension("gguf.partial").exists());

    // The same sources reproduce the same bytes.
    let right = manifest(dir.path(), &source, &target, &sha256, bytes);
    run(&args(&right, &source, &target, &output)).unwrap();
    assert_eq!(file_sha256(&output).unwrap(), (sha256, bytes));
}

#[test]
fn a_source_that_differs_from_the_manifest_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let (source, target) = fixture(dir.path(), &draft_config());
    let manifest = manifest(dir.path(), &source, &target, &"0".repeat(64), 1);
    std::fs::write(target.join("tokenizer_config.json"), "{}").unwrap();
    let output = dir.path().join("tiny.gguf");
    let error = run(&args(&manifest, &source, &target, &output))
        .unwrap_err()
        .to_string();
    assert!(error.contains("tokenizer_config.json"), "{error}");
    assert!(!output.exists());
}

#[test]
fn a_checkpoint_that_differs_from_its_config_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let (source, _) = fixture(dir.path(), &draft_config());
    let checkpoint = Safetensors::open(&source.join("model.safetensors")).unwrap();

    // A layer the config names but the checkpoint lacks.
    let mut config = draft_config();
    config["num_hidden_layers"] = json!(2);
    config["layer_types"] = json!(["sliding_attention", "sliding_attention"]);
    std::fs::write(source.join("config.json"), config.to_string()).unwrap();
    let error = plan(&source, &checkpoint, "x").err().unwrap().to_string();
    assert!(error.contains("layers.1."), "{error}");

    // A width the tensors do not have.
    let mut config = draft_config();
    config["hidden_size"] = json!(32);
    std::fs::write(source.join("config.json"), config.to_string()).unwrap();
    let error = plan(&source, &checkpoint, "x").err().unwrap().to_string();
    assert!(error.contains("shape"), "{error}");

    // A mask token outside the vocabulary, and a layer id that overflows.
    for (key, value) in [
        ("mask_token_id", json!(8)),
        ("target_layer_ids", json!([0, 2147483647])),
    ] {
        let mut config = draft_config();
        config["dflash_config"][key] = value;
        std::fs::write(source.join("config.json"), config.to_string()).unwrap();
        assert!(plan(&source, &checkpoint, "x").is_err(), "{key}");
    }
}

#[test]
fn tokenizers_llama_cpp_would_treat_differently_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let (_, target) = fixture(dir.path(), &draft_config());

    // Another splitting expression is not llama.cpp's laguna pre-tokenizer.
    write_tokenizer(&target, |t| {
        t["pre_tokenizer"]["pretokenizers"][1]["pattern"]["Regex"] = json!("\\s+");
    });
    let error = vocab::metadata(&target, &target_files(), 8)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("pre-tokenizer"), "{error}");

    // An interior special token in the single template, and a pair that is
    // only the boundary token, are refused rather than mis-converted.
    write_tokenizer(&target, |t| {
        t["post_processor"]["single"] = json!([
            {"SpecialToken": {"id": "<EOS>"}}, {"Sequence": {"id": "A"}},
            {"SpecialToken": {"id": "<MASK>"}}, {"Sequence": {"id": "A"}}
        ]);
    });
    assert!(vocab::metadata(&target, &target_files(), 8).is_err());
    write_tokenizer(&target, |t| {
        t["post_processor"]["single"] = json!([
            {"SpecialToken": {"id": "<EOS>"}}, {"Sequence": {"id": "A"}}, {"SpecialToken": {"id": "<EOS>"}}
        ]);
        t["post_processor"]["pair"] = json!([{"SpecialToken": {"id": "<EOS>"}}]);
    });
    assert!(vocab::metadata(&target, &target_files(), 8).is_err());

    // A template that adds nothing is accepted.
    write_tokenizer(&target, |t| {
        t["post_processor"]["single"] = json!([{"Sequence": {"id": "A"}}]);
        t["post_processor"]["pair"] = json!([{"Sequence": {"id": "A"}}, {"SpecialToken": {"id": "<EOS>"}}, {"Sequence": {"id": "B"}}]);
    });
    let metadata = vocab::metadata(&target, &target_files(), 8).unwrap();
    assert!(
        !metadata
            .iter()
            .any(|(key, _)| key == "tokenizer.ggml.add_bos_token")
    );
}

#[test]
fn an_unlisted_chat_template_file_is_not_read() {
    let dir = tempfile::tempdir().unwrap();
    let (_, target) = fixture(dir.path(), &draft_config());
    std::fs::write(
        target.join("tokenizer_config.json"),
        json!({"bos_token": "<EOS>", "eos_token": "<EOS>"}).to_string(),
    )
    .unwrap();
    std::fs::write(target.join("chat_template.jinja"), "{{ y }}").unwrap();
    let template = |listed: &[String]| {
        vocab::metadata(&target, listed, 8)
            .unwrap()
            .into_iter()
            .find(|(key, _)| key == "tokenizer.chat_template")
            .map(|(_, value)| value)
    };
    assert_eq!(template(&target_files()), None);
    let mut listed = target_files();
    listed.push("chat_template.jinja".into());
    assert_eq!(template(&listed), Some(gguf::Value::Str("{{ y }}".into())));
}
