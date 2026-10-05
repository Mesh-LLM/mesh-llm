use std::fs;
use std::path::Path;

use crate::package::ExplicitSourceIdentity;

#[derive(Clone)]
pub(crate) struct FixtureTensor<'a> {
    pub(crate) name: &'a str,
    pub(crate) dimensions: Vec<u64>,
    pub(crate) dtype: u32,
    pub(crate) offset: u64,
}

pub(crate) fn tensor(name: &str, offset: u64) -> FixtureTensor<'_> {
    FixtureTensor {
        name,
        dimensions: vec![2, 2],
        dtype: 0,
        offset,
    }
}

fn string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

/// How [`write_fixture`] serializes the optional `llama.nextn_predict_layers` key.
enum NextnPredictLayers {
    /// Omit the key entirely.
    Absent,
    /// Well-formed unsigned-integer value.
    Layers(u32),
    /// Non-integer value (GGUF type 8 = string), to exercise rejection.
    NonInteger(&'static str),
}

impl NextnPredictLayers {
    fn is_present(&self) -> bool {
        !matches!(self, Self::Absent)
    }
}

pub(crate) fn fixture(path: &Path, tensors: &[FixtureTensor<'_>], split: Option<(u16, u16, u64)>) {
    write_fixture(path, 2, NextnPredictLayers::Absent, tensors, split, true);
}

pub(crate) fn fixture_without_alignment(path: &Path, tensors: &[FixtureTensor<'_>]) {
    write_fixture(path, 2, NextnPredictLayers::Absent, tensors, None, false);
}

pub(crate) fn fixture_with_nextn(
    path: &Path,
    block_count: u32,
    nextn_predict_layers: Option<u32>,
    tensors: &[FixtureTensor<'_>],
    split: Option<(u16, u16, u64)>,
) {
    write_fixture(
        path,
        block_count,
        nextn_predict_layers.map_or(NextnPredictLayers::Absent, NextnPredictLayers::Layers),
        tensors,
        split,
        true,
    );
}

pub(crate) fn fixture_with_nextn_noninteger(
    path: &Path,
    block_count: u32,
    tensors: &[FixtureTensor<'_>],
    split: Option<(u16, u16, u64)>,
) {
    write_fixture(
        path,
        block_count,
        NextnPredictLayers::NonInteger("one"),
        tensors,
        split,
        true,
    );
}

fn write_fixture(
    path: &Path,
    block_count: u32,
    nextn: NextnPredictLayers,
    tensors: &[FixtureTensor<'_>],
    split: Option<(u16, u16, u64)>,
    include_alignment: bool,
) {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    let metadata_count = 3
        + u64::from(include_alignment)
        + u64::from(nextn.is_present())
        + if split.is_some() { 3 } else { 0 };
    bytes.extend_from_slice(&metadata_count.to_le_bytes());
    string(&mut bytes, "general.architecture");
    bytes.extend_from_slice(&8_u32.to_le_bytes());
    string(&mut bytes, "llama");
    string(&mut bytes, "llama.block_count");
    bytes.extend_from_slice(&4_u32.to_le_bytes());
    bytes.extend_from_slice(&block_count.to_le_bytes());
    match nextn {
        NextnPredictLayers::Absent => {}
        NextnPredictLayers::Layers(layers) => {
            string(&mut bytes, "llama.nextn_predict_layers");
            bytes.extend_from_slice(&4_u32.to_le_bytes());
            bytes.extend_from_slice(&layers.to_le_bytes());
        }
        NextnPredictLayers::NonInteger(value) => {
            string(&mut bytes, "llama.nextn_predict_layers");
            bytes.extend_from_slice(&8_u32.to_le_bytes());
            string(&mut bytes, value);
        }
    }
    if include_alignment {
        string(&mut bytes, "general.alignment");
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&32_u32.to_le_bytes());
    }
    string(&mut bytes, "tokenizer.ggml.tokens");
    bytes.extend_from_slice(&9_u32.to_le_bytes());
    bytes.extend_from_slice(&8_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u64.to_le_bytes());
    string(&mut bytes, "fixture-token");
    if let Some((number, count, total)) = split {
        for (key, value) in [("split.no", number), ("split.count", count)] {
            string(&mut bytes, key);
            bytes.extend_from_slice(&2_u32.to_le_bytes());
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        string(&mut bytes, "split.tensors.count");
        bytes.extend_from_slice(&10_u32.to_le_bytes());
        bytes.extend_from_slice(&total.to_le_bytes());
    }
    for t in tensors {
        string(&mut bytes, t.name);
        bytes.extend_from_slice(&(t.dimensions.len() as u32).to_le_bytes());
        for dimension in &t.dimensions {
            bytes.extend_from_slice(&dimension.to_le_bytes());
        }
        bytes.extend_from_slice(&t.dtype.to_le_bytes());
        bytes.extend_from_slice(&t.offset.to_le_bytes());
    }
    bytes.resize(bytes.len().div_ceil(32) * 32, 0);
    // All fixtures use a 32-byte padded extent per tensor, including quantized Q8_0
    // fixtures below which explicitly extend their payload.
    bytes.resize(bytes.len() + tensors.len() * 32, 0x3f);
    fs::write(path, bytes).unwrap();
}

pub(crate) fn explicit(source: &Path) -> ExplicitSourceIdentity {
    ExplicitSourceIdentity {
        model_id: Some("fixture/model:Q8_0".to_string()),
        source_revision: Some("immutable-source".to_string()),
        source_file: Some(source.file_name().unwrap().to_str().unwrap().to_string()),
        ..ExplicitSourceIdentity::default()
    }
}
