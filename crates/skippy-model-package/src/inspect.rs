use std::path::PathBuf;

use anyhow::Result;
use serde::Serialize;
use skippy_ffi::TensorRole;
use skippy_runtime::TensorInfo;

use crate::write::ModelSource;

#[derive(Debug, Serialize)]
struct InspectOutput {
    tensor_count: usize,
    tensors: Vec<TensorOutput>,
}

#[derive(Debug, Serialize)]
struct TensorOutput {
    name: String,
    layer_index: Option<u32>,
    role: String,
    ggml_type: u32,
    byte_size: u64,
}

pub(crate) fn inspect(model: PathBuf) -> Result<()> {
    let source = ModelSource::open(&model)?;
    let tensors = source.tensors;
    let output = InspectOutput {
        tensor_count: tensors.len(),
        tensors: tensors.into_iter().map(tensor_output).collect(),
    };
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn tensor_output(tensor: TensorInfo) -> TensorOutput {
    TensorOutput {
        name: tensor.name,
        layer_index: tensor.layer_index,
        role: role_name(tensor.role).to_string(),
        ggml_type: tensor.ggml_type,
        byte_size: tensor.byte_size,
    }
}

fn role_name(role: TensorRole) -> &'static str {
    match role {
        TensorRole::Unknown => "unknown",
        TensorRole::Metadata => "metadata",
        TensorRole::Tokenizer => "tokenizer",
        TensorRole::Embedding => "embedding",
        TensorRole::Layer => "layer",
        TensorRole::FinalNorm => "final_norm",
        TensorRole::Output => "output",
    }
}
