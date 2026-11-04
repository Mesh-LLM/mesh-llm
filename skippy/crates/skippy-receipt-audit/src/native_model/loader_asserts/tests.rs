use anyhow::bail;
use skippy_receipts::weights::ByteSource;
use skippy_receipts::weights::limits::CannotAudit;

use super::*;

const UINT32: u32 = 4;
const INT32: u32 = 5;
const FLOAT32: u32 = 6;
const STRING: u32 = 8;
const ARRAY: u32 = 9;
const GGML_F32: u32 = 0;

/// A metadata value of a test GGUF.
#[derive(Clone)]
enum Value {
    U32(u32),
    F32(f32),
    Str(String),
    Ints(Vec<i32>),
    Strings(Vec<Vec<u8>>),
}

struct Memory(Vec<u8>);

impl ByteSource for Memory {
    fn read_at(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        let (start, end) = (offset as usize, (offset + len) as usize);
        match self.0.get(start..end) {
            Some(bytes) => Ok(bytes.to_vec()),
            None => bail!("past the end"),
        }
    }

    fn size(&self) -> u64 {
        self.0.len() as u64
    }
}

fn string(out: &mut Vec<u8>, s: &[u8]) {
    out.extend((s.len() as u64).to_le_bytes());
    out.extend(s);
}

/// A GGUF of `metadata` and f32 `tensors` of the given shapes.
fn gguf(metadata: &[(String, Value)], tensors: &[(String, Vec<u64>)]) -> WeightFile {
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend((tensors.len() as u64).to_le_bytes());
    out.extend((metadata.len() as u64).to_le_bytes());
    for (key, value) in metadata {
        string(&mut out, key.as_bytes());
        match value {
            Value::U32(v) => {
                out.extend(UINT32.to_le_bytes());
                out.extend(v.to_le_bytes());
            }
            Value::F32(v) => {
                out.extend(FLOAT32.to_le_bytes());
                out.extend(v.to_le_bytes());
            }
            Value::Str(v) => {
                out.extend(STRING.to_le_bytes());
                string(&mut out, v.as_bytes());
            }
            Value::Ints(values) => {
                out.extend(ARRAY.to_le_bytes());
                out.extend(INT32.to_le_bytes());
                out.extend((values.len() as u64).to_le_bytes());
                for v in values {
                    out.extend(v.to_le_bytes());
                }
            }
            Value::Strings(values) => {
                out.extend(ARRAY.to_le_bytes());
                out.extend(STRING.to_le_bytes());
                out.extend((values.len() as u64).to_le_bytes());
                for v in values {
                    string(&mut out, v);
                }
            }
        }
    }
    let mut offset = 0u64;
    for (name, dims) in tensors {
        string(&mut out, name.as_bytes());
        out.extend((dims.len() as u32).to_le_bytes());
        for d in dims {
            out.extend(d.to_le_bytes());
        }
        out.extend(GGML_F32.to_le_bytes());
        out.extend(offset.to_le_bytes());
        offset += (dims.iter().product::<u64>() * 4).div_ceil(32) * 32;
    }
    out.resize(out.len().div_ceil(32) * 32 + offset as usize, 0);
    WeightFile::open(Box::new(Memory(out))).expect("a test GGUF")
}

/// A small model the loader takes: two layers, two query heads per KV
/// head, a byte-level vocabulary of three tokens.
struct Model {
    metadata: Vec<(String, Value)>,
    tensors: Vec<(String, Vec<u64>)>,
}

impl Model {
    fn new(arch: &str) -> Self {
        let key = |k: &str| format!("{arch}.{k}");
        Self {
            metadata: vec![
                ("general.architecture".into(), Value::Str(arch.into())),
                (key("context_length"), Value::U32(4096)),
                (key("embedding_length"), Value::U32(8)),
                (key("block_count"), Value::U32(2)),
                (key("attention.head_count"), Value::U32(2)),
                (key("attention.head_count_kv"), Value::U32(1)),
                (key("rope.dimension_count"), Value::U32(4)),
                (key("attention.layer_norm_rms_epsilon"), Value::F32(1e-6)),
                ("tokenizer.ggml.model".into(), Value::Str("gpt2".into())),
                (
                    "tokenizer.ggml.tokens".into(),
                    Value::Strings(vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()]),
                ),
            ],
            tensors: vec![("token_embd.weight".into(), vec![8, 3])],
        }
    }

    /// Sets `key` (under the architecture when it starts with `.`).
    fn with(mut self, key: &str, value: Value) -> Self {
        let key = match key.strip_prefix('.') {
            Some(rest) => format!("{}.{rest}", self.arch()),
            None => key.to_owned(),
        };
        self.metadata.retain(|(k, _)| *k != key);
        self.metadata.push((key, value));
        self
    }

    fn tensor(mut self, name: &str, dims: Vec<u64>) -> Self {
        self.tensors.retain(|(n, _)| n != name);
        self.tensors.push((name.into(), dims));
        self
    }

    fn arch(&self) -> String {
        match &self.metadata[0].1 {
            Value::Str(arch) => arch.clone(),
            _ => unreachable!(),
        }
    }

    fn open(&self) -> WeightFile {
        gguf(&self.metadata, &self.tensors)
    }

    /// Why the checks refuse the model, which must be a [`CannotAudit`].
    fn refused(&self, what: &str) -> String {
        let error = check(&self.open())
            .err()
            .unwrap_or_else(|| panic!("{what} is not refused"));
        assert!(CannotAudit::caused(&error), "{what}: {error:#}");
        error.to_string()
    }
}

fn byte_tokens() -> Vec<Vec<u8>> {
    (0..=255u8)
        .map(|b| format!("<0x{b:02X}>").into_bytes())
        .collect()
}

#[test]
fn a_model_the_loader_takes_passes() {
    assert_eq!(check(&Model::new("llama").open()).unwrap().vocabulary, 3);
    // per-layer counts, rope sections and experts that add up
    let model = Model::new("qwen3moe")
        .with(".attention.head_count", Value::Ints(vec![2, 4]))
        .with(".expert_count", Value::U32(8))
        .with(".expert_used_count", Value::Ints(vec![2, 2]))
        .with(".rope.dimension_sections", Value::Ints(vec![1, 1, 0, 0]))
        .with(".expert_gating_func", Value::U32(1));
    check(&model.open()).unwrap();
    let spm = Model::new("llama")
        .with("tokenizer.ggml.model", Value::Str("llama".into()))
        .with("tokenizer.ggml.tokens", Value::Strings(byte_tokens()))
        .tensor("token_embd.weight", vec![8, 256]);
    check(&spm.open()).unwrap();
}

/// Each value an assert or abort of the native code takes, refused before.
#[test]
fn the_values_the_native_code_asserts_on_are_refused() {
    let llama = || Model::new("llama");
    let cases = [
        // llama-model.cpp, load_hparams
        ("no layers", llama().with(".block_count", Value::U32(0))),
        ("513 layers", llama().with(".block_count", Value::U32(513))),
        (
            "-1 layers",
            llama().with(".block_count", Value::U32(u32::MAX)),
        ),
        (
            "all next-token layers",
            llama().with(".nextn_predict_layers", Value::U32(2)),
        ),
        (
            "a next-token layer without its tensors",
            llama().with(".nextn_predict_layers", Value::U32(1)),
        ),
        (
            "an unknown classifier",
            llama().with(".classifier.activation", Value::Str("relu6".into())),
        ),
        (
            "an unknown rope scaling",
            llama().with(".rope.scaling.type", Value::Str("ntk".into())),
        ),
        (
            "too many experts",
            llama()
                .with(".expert_count", Value::U32(1025))
                .with(".expert_used_count", Value::U32(1)),
        ),
        (
            "more used than experts",
            llama()
                .with(".expert_count", Value::U32(4))
                .with(".expert_used_count", Value::U32(5)),
        ),
        (
            "experts none used",
            llama().with(".expert_count", Value::U32(4)),
        ),
        (
            "used without experts",
            llama().with(".expert_used_count", Value::U32(1)),
        ),
        (
            "a layer that uses more",
            llama()
                .with(".expert_count", Value::U32(4))
                .with(".expert_used_count", Value::Ints(vec![2, 5])),
        ),
        (
            "groups that do not divide",
            llama()
                .with(".expert_count", Value::U32(6))
                .with(".expert_used_count", Value::U32(1))
                .with(".expert_group_count", Value::U32(4))
                .with(".expert_group_used_count", Value::U32(1)),
        ),
        (
            "groups none used",
            llama()
                .with(".expert_count", Value::U32(8))
                .with(".expert_used_count", Value::U32(1))
                .with(".expert_group_count", Value::U32(4)),
        ),
        // llama-graph.cpp, the graph builders and the CPU kernels
        (
            "an unknown gating",
            llama()
                .with(".expert_count", Value::U32(4))
                .with(".expert_used_count", Value::U32(1))
                .with(".expert_gating_func", Value::U32(9)),
        ),
        (
            "a pooled model",
            llama().with(".pooling_type", Value::U32(7)),
        ),
        (
            "query heads not whole per KV head",
            llama()
                .with(".attention.head_count", Value::U32(3))
                .with(".attention.head_count_kv", Value::U32(2)),
        ),
        (
            "a layer's heads not whole",
            llama()
                .with(".attention.head_count", Value::Ints(vec![2, 3]))
                .with(".attention.head_count_kv", Value::U32(2)),
        ),
        (
            "heads for more layers",
            llama().with(".attention.head_count", Value::Ints(vec![2, 2, 2])),
        ),
        (
            "an odd rope",
            llama().with(".rope.dimension_count", Value::U32(3)),
        ),
        (
            "a rope wider than a head",
            llama().with(".rope.dimension_count", Value::U32(6)),
        ),
        (
            "no rope sections",
            llama().with(".rope.dimension_sections", Value::Ints(vec![0, 0, 0, 0])),
        ),
        (
            "a negative rope section",
            llama().with(".rope.dimension_sections", Value::Ints(vec![-2, 3, 0, 0])),
        ),
        (
            "rope sections past half a head",
            llama().with(".rope.dimension_sections", Value::Ints(vec![2, 1, 0, 0])),
        ),
        (
            "an epsilon below zero",
            llama().with(".attention.layer_norm_rms_epsilon", Value::F32(-1.0)),
        ),
        (
            "an epsilon that is no number",
            llama().with(".attention.layer_norm_epsilon", Value::F32(f32::NAN)),
        ),
        (
            "a step of 0",
            llama().with(".interleave_moe_layer_step", Value::U32(0)),
        ),
        ("no context", llama().with(".context_length", Value::U32(0))),
        (
            "no heads in gemma2",
            Model::new("gemma2")
                .with(".attention.head_count", Value::U32(0))
                .with(".attention.head_count_kv", Value::U32(0)),
        ),
        (
            "all layers share a KV cache",
            Model::new("gemma4").with(".attention.shared_kv_layers", Value::U32(1)),
        ),
        (
            "layers that read another's cache",
            Model::new("mistral-large4"),
        ),
        // llama-vocab.cpp
        (
            "no vocabulary",
            llama().with("tokenizer.ggml.model", Value::Str("no_vocab".into())),
        ),
        (
            "a WordPiece vocabulary",
            llama().with("tokenizer.ggml.model", Value::Str("bert".into())),
        ),
        (
            "a token twice",
            llama().with(
                "tokenizer.ggml.tokens",
                Value::Strings(vec![b"a".to_vec(), b"b".to_vec(), b"a".to_vec()]),
            ),
        ),
        (
            "an empty token named as another",
            llama().with(
                "tokenizer.ggml.tokens",
                Value::Strings(vec![b"[EMPTY_1]".to_vec(), b"".to_vec(), b"c".to_vec()]),
            ),
        ),
        (
            "a SentencePiece vocabulary short of a byte",
            llama().with("tokenizer.ggml.model", Value::Str("llama".into())),
        ),
        (
            "fewer embedding rows than tokens",
            llama().tensor("token_embd.weight", vec![8, 2]),
        ),
    ];
    for (what, model) in cases {
        let why = model.refused(what);
        assert!(!why.is_empty(), "{what}");
    }
}

#[test]
fn the_values_a_model_may_have_pass() {
    let llama = || Model::new("llama");
    for (what, model) in [
        (
            "an unspecified pooling",
            llama().with(".pooling_type", Value::U32(u32::MAX)),
        ),
        (
            "deepseek2 reads no gating as softmax",
            Model::new("deepseek2")
                .with(".expert_count", Value::U32(4))
                .with(".expert_used_count", Value::U32(1))
                .with(".expert_gating_func", Value::U32(0)),
        ),
        (
            "a key length that sets the head",
            llama()
                .with(".attention.key_length", Value::U32(8))
                .with(".rope.dimension_count", Value::U32(8)),
        ),
        (
            "a next-token layer with its tensors",
            llama()
                .with(".nextn_predict_layers", Value::U32(1))
                .tensor("blk.1.nextn.eh_proj.weight", vec![8, 8]),
        ),
    ] {
        if let Err(error) = check(&model.open()) {
            panic!("{what}: {error:#}");
        }
    }
}

#[test]
fn a_layer_that_reads_another_layers_cache_is_not_loaded_alone() {
    let gemma3n = Model::new("gemma3n")
        .with(".block_count", Value::U32(30))
        .open();
    check_layer(&gemma3n, 19).unwrap();
    assert!(CannotAudit::caused(&check_layer(&gemma3n, 20).unwrap_err()));
    let gemma4 = Model::new("gemma4")
        .with(".block_count", Value::U32(30))
        .with(".attention.shared_kv_layers", Value::U32(10))
        .open();
    check_layer(&gemma4, 19).unwrap();
    assert!(check_layer(&gemma4, 20).is_err());
    check_layer(&Model::new("llama").open(), 1).unwrap();
}
