use std::ffi::c_char;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum LlamaFileType {
    AllF32 = 0,
    MostlyF16 = 1,
    MostlyQ4_0 = 2,
    MostlyQ4_1 = 3,
    MostlyQ8_0 = 7,
    MostlyQ5_0 = 8,
    MostlyQ5_1 = 9,
    MostlyQ2K = 10,
    MostlyQ3KS = 11,
    MostlyQ3KM = 12,
    MostlyQ3KL = 13,
    MostlyQ4KS = 14,
    MostlyQ4KM = 15,
    MostlyQ5KS = 16,
    MostlyQ5KM = 17,
    MostlyQ6K = 18,
    MostlyIQ2XXS = 19,
    MostlyIQ2XS = 20,
    MostlyQ2KS = 21,
    MostlyIQ3XS = 22,
    MostlyIQ3XXS = 23,
    MostlyIQ1S = 24,
    MostlyIQ4NL = 25,
    MostlyIQ3S = 26,
    MostlyIQ3M = 27,
    MostlyIQ2S = 28,
    MostlyIQ2M = 29,
    MostlyIQ4XS = 30,
    MostlyIQ1M = 31,
    MostlyBf16 = 32,
    MostlyTQ1_0 = 36,
    MostlyTQ2_0 = 37,
    MostlyMxfp4Moe = 38,
    MostlyNvfp4 = 39,
    MostlyQ1_0 = 40,
    MostlyQ2_0 = 41,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1M = 29,
    Bf16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    Mxfp4 = 39,
    Nvfp4 = 40,
    Q1_0 = 41,
    Q2_0 = 42,
    Count = 43,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum LlamaModelKvOverrideType {
    Int = 0,
    Float = 1,
    Bool = 2,
    Str = 3,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union LlamaModelKvOverrideValue {
    pub val_i64: i64,
    pub val_f64: f64,
    pub val_bool: bool,
    pub val_str: [c_char; 128],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct LlamaModelKvOverride {
    pub tag: LlamaModelKvOverrideType,
    pub key: [c_char; 128],
    pub value: LlamaModelKvOverrideValue,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LlamaModelTensorOverride {
    pub pattern: *const c_char,
    pub tensor_type: GgmlType,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LlamaModelImatrixData {
    pub name: *const c_char,
    pub data: *const f32,
    pub size: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LlamaModelQuantizeParams {
    pub nthread: i32,
    pub ftype: LlamaFileType,
    pub output_tensor_type: GgmlType,
    pub token_embedding_type: GgmlType,
    pub allow_requantize: bool,
    pub quantize_output_tensor: bool,
    pub only_copy: bool,
    pub pure: bool,
    pub keep_split: bool,
    pub dry_run: bool,
    pub imatrix: *const LlamaModelImatrixData,
    pub kv_overrides: *const LlamaModelKvOverride,
    pub tt_overrides: *const LlamaModelTensorOverride,
    pub prune_layers: *const i32,
}
