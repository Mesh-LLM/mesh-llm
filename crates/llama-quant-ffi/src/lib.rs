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

#[derive(Debug)]
pub enum NativeRuntimeLoadError {
    Load(String),
    AlreadyLoaded,
}

impl std::fmt::Display for NativeRuntimeLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Load(message) => write!(f, "{message}"),
            Self::AlreadyLoaded => write!(f, "native runtime library is already loaded"),
        }
    }
}

impl std::error::Error for NativeRuntimeLoadError {}

#[cfg(not(feature = "dynamic-runtime"))]
pub fn native_runtime_loaded() -> bool {
    true
}

#[cfg(not(feature = "dynamic-runtime"))]
/// No-op for statically linked builds.
///
/// # Safety
///
/// Static builds resolve the native ABI at process link/load time, so this
/// function does not dereference the supplied path or mutate loader state.
pub unsafe fn load_native_runtime_library(
    _path: impl AsRef<std::path::Path>,
) -> Result<(), NativeRuntimeLoadError> {
    Ok(())
}

#[cfg(not(feature = "dynamic-runtime"))]
/// No-op for statically linked builds.
///
/// # Safety
///
/// Static builds resolve the native ABI at process link/load time, so this
/// function does not dereference the supplied paths or mutate loader state.
pub unsafe fn load_native_runtime_libraries<I, P>(_paths: I) -> Result<(), NativeRuntimeLoadError>
where
    I: IntoIterator<Item = P>,
    P: AsRef<std::path::Path>,
{
    Ok(())
}

#[cfg(feature = "dynamic-runtime")]
mod dynamic {
    use super::*;
    use libloading::Library;
    use std::sync::OnceLock;

    static SYMBOLS: OnceLock<Symbols> = OnceLock::new();

    pub fn native_runtime_loaded() -> bool {
        SYMBOLS.get().is_some()
    }

    /// Load a native llama.cpp runtime library and resolve quantization symbols.
    ///
    /// # Safety
    ///
    /// The caller must ensure the library belongs to the same pinned llama.cpp
    /// build and exposes an ABI-compatible `llama_model_quantize` surface.
    pub unsafe fn load_native_runtime_library(
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), NativeRuntimeLoadError> {
        let symbols = unsafe { Symbols::load_paths(&[path.as_ref()]) }?;
        SYMBOLS
            .set(symbols)
            .map_err(|_| NativeRuntimeLoadError::AlreadyLoaded)
    }

    /// Load native runtime libraries and resolve quantization symbols.
    ///
    /// Libraries are searched from last to first so dependencies can be passed
    /// before the primary `libllama`/`llama.dll` library.
    ///
    /// # Safety
    ///
    /// The caller must ensure every library belongs to the same pinned llama.cpp
    /// build and exposes an ABI-compatible quantization surface.
    pub unsafe fn load_native_runtime_libraries<I, P>(
        paths: I,
    ) -> Result<(), NativeRuntimeLoadError>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<std::path::Path>,
    {
        let collected = paths
            .into_iter()
            .map(|path| path.as_ref().to_path_buf())
            .collect::<Vec<_>>();
        let symbols = unsafe { Symbols::load_paths(&collected) }?;
        SYMBOLS
            .set(symbols)
            .map_err(|_| NativeRuntimeLoadError::AlreadyLoaded)
    }

    fn symbols() -> &'static Symbols {
        SYMBOLS
            .get()
            .expect("llama quant native runtime library has not been loaded")
    }

    struct Symbols {
        _libraries: Vec<Library>,
        llama_model_quantize_default_params: unsafe extern "C" fn() -> LlamaModelQuantizeParams,
        llama_model_quantize: unsafe extern "C" fn(
            *const c_char,
            *const c_char,
            *const LlamaModelQuantizeParams,
        ) -> u32,
    }

    impl Symbols {
        unsafe fn load_paths<P>(paths: &[P]) -> Result<Self, NativeRuntimeLoadError>
        where
            P: AsRef<std::path::Path>,
        {
            if paths.is_empty() {
                return Err(NativeRuntimeLoadError::Load(
                    "native runtime did not provide any libraries".to_string(),
                ));
            }
            let mut libraries = Vec::with_capacity(paths.len());
            for path in paths {
                libraries.push(
                    unsafe { Library::new(path.as_ref()) }
                        .map_err(|err| NativeRuntimeLoadError::Load(err.to_string()))?,
                );
            }
            let llama_model_quantize_default_params = lookup_symbol(
                &libraries,
                b"llama_model_quantize_default_params\0",
                "llama_model_quantize_default_params",
            )?;
            let llama_model_quantize = lookup_symbol(
                &libraries,
                b"llama_model_quantize\0",
                "llama_model_quantize",
            )?;
            Ok(Self {
                _libraries: libraries,
                llama_model_quantize_default_params,
                llama_model_quantize,
            })
        }
    }

    fn lookup_symbol<Sym>(
        libraries: &[Library],
        name: &[u8],
        label: &str,
    ) -> Result<Sym, NativeRuntimeLoadError>
    where
        Sym: Copy + 'static,
    {
        for library in libraries.iter().rev() {
            if let Ok(symbol) = unsafe { library.get::<Sym>(name) } {
                return Ok(*symbol);
            }
        }
        Err(NativeRuntimeLoadError::Load(format!(
            "native runtime symbol not found: {label}"
        )))
    }

    /// Return llama.cpp quantization default parameters.
    ///
    /// # Safety
    ///
    /// The loaded native runtime must expose an ABI-compatible implementation.
    pub unsafe fn llama_model_quantize_default_params() -> LlamaModelQuantizeParams {
        unsafe { (symbols().llama_model_quantize_default_params)() }
    }

    /// Quantize a GGUF model through llama.cpp.
    ///
    /// # Safety
    ///
    /// `fname_inp`, `fname_out`, and `params` must be valid pointers matching
    /// llama.cpp's `llama_model_quantize` contract for the loaded runtime.
    pub unsafe fn llama_model_quantize(
        fname_inp: *const c_char,
        fname_out: *const c_char,
        params: *const LlamaModelQuantizeParams,
    ) -> u32 {
        unsafe { (symbols().llama_model_quantize)(fname_inp, fname_out, params) }
    }
}

#[cfg(feature = "dynamic-runtime")]
pub use dynamic::*;

#[cfg(not(feature = "dynamic-runtime"))]
#[allow(clippy::missing_safety_doc)]
unsafe extern "C" {
    pub fn llama_model_quantize_default_params() -> LlamaModelQuantizeParams;

    pub fn llama_model_quantize(
        fname_inp: *const c_char,
        fname_out: *const c_char,
        params: *const LlamaModelQuantizeParams,
    ) -> u32;
}
