//! Ergonomic aliases for the generated upstream quantization ABI types.

use crate::bindings;

pub type LlamaFileType = bindings::llama::llama_ftype;
pub type GgmlType = bindings::llama::ggml_type;
pub type LlamaModelKvOverrideType = bindings::llama::llama_model_kv_override_type;
pub type LlamaModelKvOverrideValue = bindings::llama::llama_model_kv_override__bindgen_ty_1;
pub type LlamaModelKvOverride = bindings::llama::llama_model_kv_override;
pub type LlamaModelTensorOverride = bindings::llama::llama_model_tensor_override;
pub type LlamaModelImatrixData = bindings::llama::llama_model_imatrix_data;
pub type LlamaModelQuantizeParams = bindings::llama::llama_model_quantize_params;
