use std::ffi::{CString, c_char};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, ensure};

use crate::QuantRunnerArgs;
use crate::backend::BackendRunStatus;
use crate::imatrix::NativeImatrix;
use crate::manifest::Manifest;
use crate::quantize::normalize_tensor_type_entry;
use crate::splits::SplitWindow;
use crate::types::{QuantType, TensorType};

const KV_QUANTIZE_IMATRIX_FILE: &str = "quantize.imatrix.file";
const KV_QUANTIZE_IMATRIX_DATASET: &str = "quantize.imatrix.dataset";
const KV_QUANTIZE_IMATRIX_N_ENTRIES: &str = "quantize.imatrix.entries_count";
const KV_QUANTIZE_IMATRIX_N_CHUNKS: &str = "quantize.imatrix.chunks_count";

pub(crate) fn build_native_quantize_command(
    args: &QuantRunnerArgs,
    manifest: &Manifest,
    staged_first_shard: &Path,
    output_prefix: &Path,
    window: SplitWindow,
) -> Result<Vec<String>> {
    ensure_native_quantize_supported(args)?;
    ensure_full_split_window(manifest, window)?;
    let quant = manifest
        .quant
        .as_deref()
        .context("quantize manifest is missing quant type")?;
    let mut command = vec![format!("{}-quantize", args.backend.as_str())];
    for library in &args.native_runtime_libraries {
        command.push("--native-runtime-library".to_string());
        command.push(library.display().to_string());
    }
    if args.allow_requantize {
        command.push("--allow-requantize".to_string());
    }
    if args.pure {
        command.push("--pure".to_string());
    }
    if args.dry_run {
        command.push("--dry-run".to_string());
    }
    if args.leave_output_tensor {
        command.push("--leave-output-tensor".to_string());
    }
    if let Some(imatrix) = args.imatrix.as_deref() {
        command.push("--imatrix".to_string());
        command.push(imatrix.display().to_string());
    }
    for include in &args.include_weights {
        command.push("--include-weights".to_string());
        command.push(include.clone());
    }
    for exclude in &args.exclude_weights {
        command.push("--exclude-weights".to_string());
        command.push(exclude.clone());
    }
    push_optional(
        &mut command,
        "--output-tensor-type",
        &args.output_tensor_type,
    );
    push_optional(
        &mut command,
        "--token-embedding-type",
        &args.token_embedding_type,
    );
    for entry in &args.tensor_type {
        let normalized_entry = normalize_tensor_type_entry(entry)?;
        command.push("--tensor-type".to_string());
        command.push(normalized_entry);
    }
    if let Some(tensor_type_file) = manifest.tensor_type_file.as_deref() {
        command.push("--tensor-type-file".to_string());
        command.push(tensor_type_file.display().to_string());
    }
    if let Some(prune_layers) = args.prune_layers.as_deref() {
        command.push("--prune-layers".to_string());
        command.push(prune_layers.to_string());
    }
    for override_kv in &args.override_kv {
        command.push("--override-kv".to_string());
        command.push(override_kv.clone());
    }
    command.extend([
        "--keep-split".to_string(),
        staged_first_shard.display().to_string(),
        llama_split_output_prefix(output_prefix)
            .display()
            .to_string(),
        quant.to_string(),
    ]);
    if let Some(nthreads) = args.nthreads {
        command.push(nthreads.to_string());
    }
    Ok(command)
}

pub(crate) fn run_native_quantize(
    args: &QuantRunnerArgs,
    manifest: &Manifest,
    staged_first_shard: &Path,
    output_prefix: &Path,
    window: SplitWindow,
) -> Result<BackendRunStatus> {
    ensure_native_quantize_supported(args)?;
    ensure_full_split_window(manifest, window)?;
    let native_inputs = NativeQuantizeInputs::build(args, manifest)?;
    let mut params = unsafe { llama_quant_ffi::llama_model_quantize_default_params() };
    let quant = manifest
        .quant
        .as_deref()
        .context("quantize manifest is missing quant type")?
        .parse::<QuantType>()
        .map_err(anyhow::Error::msg)?;
    params.nthread = args.nthreads.map_or(0, |value| value as i32);
    params.ftype = quant.as_llama_file_type();
    params.allow_requantize = args.allow_requantize;
    params.quantize_output_tensor = !args.leave_output_tensor;
    params.only_copy = quant == QuantType::Copy;
    params.pure = args.pure;
    params.keep_split = true;
    params.dry_run = args.dry_run;
    params.output_tensor_type =
        optional_ggml_type(args.output_tensor_type.as_deref(), "--output-tensor-type")?
            .unwrap_or(params.output_tensor_type);
    params.token_embedding_type = optional_ggml_type(
        args.token_embedding_type.as_deref(),
        "--token-embedding-type",
    )?
    .unwrap_or(params.token_embedding_type);
    params.tt_overrides = native_inputs.tensor_overrides_ptr();
    params.prune_layers = native_inputs.prune_layers_ptr();
    params.kv_overrides = native_inputs.kv_overrides_ptr();
    params.imatrix = native_inputs.imatrix_ptr();

    let input = path_to_cstring(staged_first_shard)?;
    let native_output_prefix = llama_split_output_prefix(output_prefix);
    let output = path_to_cstring(&native_output_prefix)?;
    let code =
        unsafe { llama_quant_ffi::llama_model_quantize(input.as_ptr(), output.as_ptr(), &params) };
    Ok(BackendRunStatus::from_code(code as i32))
}

fn ensure_native_quantize_supported(args: &QuantRunnerArgs) -> Result<()> {
    load_llama_quant_runtime(&args.native_runtime_libraries)?;
    ensure!(
        llama_quant_ffi::native_runtime_loaded(),
        "llama quant runtime is not linked; pass --native-runtime-library or build the standalone static target"
    );
    ensure!(
        args.include_weights.is_empty() || args.exclude_weights.is_empty(),
        "--include-weights and --exclude-weights cannot be used together"
    );
    ensure!(
        args.max_memory.is_none(),
        "--max-memory is not supported by the native llama quantize backend now that mesh-llm no longer patches llama-quantize memory chunking"
    );
    Ok(())
}

fn ensure_full_split_window(manifest: &Manifest, window: SplitWindow) -> Result<()> {
    ensure!(
        window.first_split == 1 && window.last_split == manifest.expected_splits,
        "native llama quantize backend no longer supports partial split windows after removing mesh-llm's patched llama-quantize split-window support; requested {}..{} of {}",
        window.first_split,
        window.last_split,
        manifest.expected_splits
    );
    Ok(())
}

fn load_llama_quant_runtime(libraries: &[PathBuf]) -> Result<()> {
    if libraries.is_empty() || llama_quant_ffi::native_runtime_loaded() {
        return Ok(());
    }
    match unsafe { llama_quant_ffi::load_native_runtime_libraries(libraries) } {
        Ok(()) | Err(llama_quant_ffi::NativeRuntimeLoadError::AlreadyLoaded) => Ok(()),
        Err(error) => Err(anyhow!("load native llama quant runtime: {error}")),
    }
}

struct NativeQuantizeInputs {
    _tensor_patterns: Vec<CString>,
    tensor_overrides: Vec<llama_quant_ffi::LlamaModelTensorOverride>,
    prune_layers: Vec<i32>,
    kv_overrides: Vec<llama_quant_ffi::LlamaModelKvOverride>,
    imatrix: Option<NativeImatrix>,
}

impl NativeQuantizeInputs {
    fn build(args: &QuantRunnerArgs, manifest: &Manifest) -> Result<Self> {
        let (_tensor_patterns, tensor_overrides) = tensor_overrides(args, manifest)?;
        let imatrix = args
            .imatrix
            .as_deref()
            .map(|path| NativeImatrix::load(path, &args.include_weights, &args.exclude_weights))
            .transpose()?;
        Ok(Self {
            _tensor_patterns,
            tensor_overrides,
            prune_layers: prune_layers(args.prune_layers.as_deref())?,
            kv_overrides: kv_overrides(&args.override_kv, imatrix.as_ref())?,
            imatrix,
        })
    }

    fn tensor_overrides_ptr(&self) -> *const llama_quant_ffi::LlamaModelTensorOverride {
        if self.tensor_overrides.is_empty() {
            std::ptr::null()
        } else {
            self.tensor_overrides.as_ptr()
        }
    }

    fn prune_layers_ptr(&self) -> *const i32 {
        if self.prune_layers.is_empty() {
            std::ptr::null()
        } else {
            self.prune_layers.as_ptr()
        }
    }

    fn kv_overrides_ptr(&self) -> *const llama_quant_ffi::LlamaModelKvOverride {
        if self.kv_overrides.is_empty() {
            std::ptr::null()
        } else {
            self.kv_overrides.as_ptr()
        }
    }

    fn imatrix_ptr(&self) -> *const llama_quant_ffi::LlamaModelImatrixData {
        self.imatrix
            .as_ref()
            .map_or(std::ptr::null(), NativeImatrix::as_ptr)
    }
}

fn tensor_overrides(
    args: &QuantRunnerArgs,
    manifest: &Manifest,
) -> Result<(Vec<CString>, Vec<llama_quant_ffi::LlamaModelTensorOverride>)> {
    let mut entries = args.tensor_type.clone();
    if let Some(path) = manifest.tensor_type_file.as_deref() {
        let text = fs::read_to_string(path)
            .with_context(|| format!("read tensor type file {}", path.display()))?;
        entries.extend(text.split_whitespace().map(ToString::to_string));
    }
    if entries.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let mut patterns = Vec::with_capacity(entries.len());
    let mut overrides = Vec::with_capacity(entries.len() + 1);
    for entry in entries {
        let (raw_pattern, raw_type) = entry
            .split_once('=')
            .ok_or_else(|| anyhow!("malformed tensor type entry {entry:?}"))?;
        ensure!(
            !raw_pattern.is_empty(),
            "tensor type entry has empty tensor name"
        );
        let tensor_type = TensorType::parse(raw_type)
            .with_context(|| format!("unsupported raw ggml tensor type {raw_type:?}"))?
            .as_ggml_type()
            .with_context(|| {
                format!("tensor type override requires raw ggml_type, got {raw_type:?}")
            })?;
        patterns.push(CString::new(raw_pattern.to_ascii_lowercase())?);
        overrides.push(llama_quant_ffi::LlamaModelTensorOverride {
            pattern: patterns.last().expect("just pushed").as_ptr(),
            tensor_type,
        });
    }
    overrides.push(llama_quant_ffi::LlamaModelTensorOverride {
        pattern: std::ptr::null(),
        tensor_type: llama_quant_ffi::GgmlType::Count,
    });
    Ok((patterns, overrides))
}

fn prune_layers(raw: Option<&str>) -> Result<Vec<i32>> {
    let Some(raw) = raw else {
        return Ok(Vec::new());
    };
    let mut layers = raw
        .split(',')
        .map(|value| {
            let layer = value
                .parse::<i32>()
                .with_context(|| format!("invalid layer id {value:?}"))?;
            ensure!(layer >= 0, "invalid negative layer id {layer}");
            Ok(layer)
        })
        .collect::<Result<Vec<_>>>()?;
    layers.sort_unstable();
    layers.dedup();
    layers.push(-1);
    Ok(layers)
}

fn kv_overrides(
    raw_overrides: &[String],
    imatrix: Option<&NativeImatrix>,
) -> Result<Vec<llama_quant_ffi::LlamaModelKvOverride>> {
    if raw_overrides.is_empty() && imatrix.is_none() {
        return Ok(Vec::new());
    }
    let imatrix_extra = if imatrix.is_some() { 4 } else { 0 };
    let mut overrides = Vec::with_capacity(raw_overrides.len() + imatrix_extra + 1);
    for raw in raw_overrides {
        overrides.push(parse_kv_override(raw)?);
    }
    if let Some(imatrix) = imatrix {
        overrides.push(string_kv_override(
            KV_QUANTIZE_IMATRIX_FILE,
            &imatrix.source_path().display().to_string(),
        )?);
        if let Some(dataset) = imatrix.dataset() {
            overrides.push(string_kv_override(KV_QUANTIZE_IMATRIX_DATASET, dataset)?);
        }
        overrides.push(int_kv_override(
            KV_QUANTIZE_IMATRIX_N_ENTRIES,
            imatrix.entry_count() as i64,
        )?);
        if imatrix.chunk_count() > 0 {
            overrides.push(int_kv_override(
                KV_QUANTIZE_IMATRIX_N_CHUNKS,
                imatrix.chunk_count() as i64,
            )?);
        }
    }
    overrides.push(llama_quant_ffi::LlamaModelKvOverride {
        tag: llama_quant_ffi::LlamaModelKvOverrideType::Int,
        key: [0; 128],
        value: llama_quant_ffi::LlamaModelKvOverrideValue { val_i64: 0 },
    });
    Ok(overrides)
}

fn int_kv_override(key: &str, value: i64) -> Result<llama_quant_ffi::LlamaModelKvOverride> {
    Ok(llama_quant_ffi::LlamaModelKvOverride {
        tag: llama_quant_ffi::LlamaModelKvOverrideType::Int,
        key: fixed_c_char_array(key, "KV override key")?,
        value: llama_quant_ffi::LlamaModelKvOverrideValue { val_i64: value },
    })
}

fn string_kv_override(key: &str, value: &str) -> Result<llama_quant_ffi::LlamaModelKvOverride> {
    Ok(llama_quant_ffi::LlamaModelKvOverride {
        tag: llama_quant_ffi::LlamaModelKvOverrideType::Str,
        key: fixed_c_char_array(key, "KV override key")?,
        value: llama_quant_ffi::LlamaModelKvOverrideValue {
            val_str: fixed_c_char_array(value, "KV override string value")?,
        },
    })
}

fn parse_kv_override(raw: &str) -> Result<llama_quant_ffi::LlamaModelKvOverride> {
    let (key, value) = raw
        .split_once('=')
        .ok_or_else(|| anyhow!("malformed KV override {raw:?}"))?;
    ensure!(!key.is_empty(), "KV override has empty key");
    let key = fixed_c_char_array(key, "KV override key")?;
    if let Some(rest) = value.strip_prefix("int:") {
        return Ok(llama_quant_ffi::LlamaModelKvOverride {
            tag: llama_quant_ffi::LlamaModelKvOverrideType::Int,
            key,
            value: llama_quant_ffi::LlamaModelKvOverrideValue {
                val_i64: rest.parse::<i64>()?,
            },
        });
    }
    if let Some(rest) = value.strip_prefix("float:") {
        return Ok(llama_quant_ffi::LlamaModelKvOverride {
            tag: llama_quant_ffi::LlamaModelKvOverrideType::Float,
            key,
            value: llama_quant_ffi::LlamaModelKvOverrideValue {
                val_f64: rest.parse::<f64>()?,
            },
        });
    }
    if let Some(rest) = value.strip_prefix("bool:") {
        let val_bool = match rest {
            "true" => true,
            "false" => false,
            _ => return Err(anyhow!("invalid bool KV override value {rest:?}")),
        };
        return Ok(llama_quant_ffi::LlamaModelKvOverride {
            tag: llama_quant_ffi::LlamaModelKvOverrideType::Bool,
            key,
            value: llama_quant_ffi::LlamaModelKvOverrideValue { val_bool },
        });
    }
    if let Some(rest) = value.strip_prefix("str:") {
        return Ok(llama_quant_ffi::LlamaModelKvOverride {
            tag: llama_quant_ffi::LlamaModelKvOverrideType::Str,
            key,
            value: llama_quant_ffi::LlamaModelKvOverrideValue {
                val_str: fixed_c_char_array(rest, "KV override string value")?,
            },
        });
    }
    Err(anyhow!("invalid KV override type in {raw:?}"))
}

fn fixed_c_char_array(raw: &str, label: &str) -> Result<[c_char; 128]> {
    ensure!(raw.len() < 128, "{label} cannot exceed 127 bytes");
    let cstring = CString::new(raw).with_context(|| format!("{label} contains NUL byte"))?;
    let mut out = [0 as c_char; 128];
    for (dst, src) in out.iter_mut().zip(cstring.as_bytes_with_nul()) {
        *dst = *src as c_char;
    }
    Ok(out)
}

fn optional_ggml_type(raw: Option<&str>, flag: &str) -> Result<Option<llama_quant_ffi::GgmlType>> {
    raw.map(|value| {
        let tensor_type = TensorType::parse(value)
            .with_context(|| format!("{flag} has unsupported ggml type {value:?}"))?;
        tensor_type
            .as_ggml_type()
            .with_context(|| format!("{flag} requires a raw ggml_type, got {value:?}"))
    })
    .transpose()
}

fn path_to_cstring(path: &Path) -> Result<CString> {
    let text = path
        .to_str()
        .with_context(|| format!("path is not valid UTF-8: {}", path.display()))?;
    CString::new(text).with_context(|| format!("path contains NUL byte: {}", path.display()))
}

fn llama_split_output_prefix(path: &Path) -> PathBuf {
    if path.extension().and_then(|value| value.to_str()) != Some("gguf") {
        return path.to_path_buf();
    }
    let Some(stem) = path.file_stem() else {
        return path.to_path_buf();
    };
    path.with_file_name(stem)
}

fn push_optional(command: &mut Vec<String>, flag: &str, value: &Option<String>) {
    if let Some(value) = value {
        command.push(flag.to_string());
        command.push(value.clone());
    }
}

#[cfg(test)]
mod tests {
    use crate::MANIFEST_VERSION;
    use crate::backend::BackendKind;
    use crate::types::JobKind;

    use super::*;

    #[test]
    fn builds_native_quantize_command() {
        let mut args = native_args();
        args.tensor_type = vec!["mtp_head.weight=NVFP4".to_string()];
        args.prune_layers = Some("2,1,2".to_string());
        args.override_kv = vec!["general.name=str:test".to_string()];
        let manifest = manifest(None);
        let command = build_native_quantize_command(
            &args,
            &manifest,
            Path::new("/tmp/in/model-00001-of-00002.gguf"),
            Path::new("/tmp/out/model-q2"),
            SplitWindow {
                first_split: 1,
                last_split: 2,
            },
        )
        .unwrap();

        assert_eq!(command[0], "llama-api-quantize");
        assert!(!command.contains(&"--native-runtime-library".to_string()));
        assert!(command.contains(&"--keep-split".to_string()));
        assert!(command.contains(&"--tensor-type".to_string()));
        assert!(command.contains(&"--prune-layers".to_string()));
        assert!(command.contains(&"--override-kv".to_string()));
        assert!(!command.contains(&"--max-memory".to_string()));
        assert!(command.contains(&"Q2_K".to_string()));
    }

    #[test]
    fn passes_explicit_tensor_type_file_to_command() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).unwrap();
        let recipe = root.join("glm-5.2-q2-k-mtp-q8.tensor-types.txt");
        fs::write(&recipe, "(^|\\.)nextn\\.=Q8_0").unwrap();
        let args = native_args();
        let manifest = manifest(Some(recipe.clone()));

        let command = build_native_quantize_command(
            &args,
            &manifest,
            Path::new("/tmp/in/model-00001-of-00002.gguf"),
            Path::new("/tmp/out/model-q2-mtp-q8"),
            SplitWindow {
                first_split: 1,
                last_split: 2,
            },
        )
        .unwrap();

        assert!(command.contains(&"--tensor-type-file".to_string()));
        assert!(command.contains(&recipe.display().to_string()));
        assert!(!command.contains(&"(^|\\.)nextn\\.=Q8_0".to_string()));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn builds_skippy_abi_quantize_command_label() {
        let mut args = native_args();
        args.backend = BackendKind::SkippyAbi;
        let manifest = manifest(None);

        let command = build_native_quantize_command(
            &args,
            &manifest,
            Path::new("/tmp/in/model-00001-of-00002.gguf"),
            Path::new("/tmp/out/model-q2"),
            SplitWindow {
                first_split: 1,
                last_split: 2,
            },
        )
        .unwrap();

        assert_eq!(command[0], "skippy-abi-quantize");
        assert!(!command.contains(&"--native-runtime-library".to_string()));
    }

    #[test]
    fn strips_gguf_extension_from_llama_split_output_prefix() {
        let args = native_args();
        let manifest = manifest(None);

        let command = build_native_quantize_command(
            &args,
            &manifest,
            Path::new("/tmp/in/model.gguf"),
            Path::new("/tmp/out/model-q4.gguf"),
            SplitWindow {
                first_split: 1,
                last_split: 2,
            },
        )
        .unwrap();

        assert!(command.contains(&"/tmp/out/model-q4".to_string()));
        assert!(!command.contains(&"/tmp/out/model-q4.gguf".to_string()));
        assert_eq!(
            llama_split_output_prefix(Path::new("/tmp/out/model-q4.gguf")),
            PathBuf::from("/tmp/out/model-q4")
        );
    }

    #[test]
    fn rejects_partial_split_window_after_dropping_patched_llama_quantize() {
        let args = native_args();
        let manifest = manifest(None);

        let error = build_native_quantize_command(
            &args,
            &manifest,
            Path::new("/tmp/in/model-00001-of-00002.gguf"),
            Path::new("/tmp/out/model-q2"),
            SplitWindow {
                first_split: 1,
                last_split: 1,
            },
        )
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("no longer supports partial split windows")
        );
    }

    #[test]
    fn builds_native_inputs_for_tensor_prune_and_kv_overrides() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).unwrap();
        let recipe = root.join("tensors.txt");
        fs::write(&recipe, "blk.0.weight=Q8_0").unwrap();
        let mut args = native_args();
        args.tensor_type = vec!["mtp_head.weight=NVFP4".to_string()];
        args.prune_layers = Some("2,1,2".to_string());
        args.override_kv = vec![
            "general.name=str:test".to_string(),
            "custom.count=int:7".to_string(),
            "custom.enabled=bool:true".to_string(),
        ];
        let manifest = manifest(Some(recipe));

        let inputs = NativeQuantizeInputs::build(&args, &manifest).unwrap();

        assert_eq!(inputs.tensor_overrides.len(), 3);
        assert_eq!(inputs.prune_layers, vec![1, 2, -1]);
        assert_eq!(inputs.kv_overrides.len(), 4);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn appends_tensor_type_file_entries_after_explicit_overrides() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).unwrap();
        let recipe = root.join("glm-5.2-q2-k-mtp-q8.tensor-types.txt");
        fs::write(
            &recipe,
            "^token_embd\\.weight$=Q8_0\n(^|\\.)nextn\\.=Q8_0\n",
        )
        .unwrap();
        let mut args = native_args();
        args.tensor_type = vec!["nextn\\.pre_projection\\.weight=F16".to_string()];
        let manifest = manifest(Some(recipe));

        let inputs = NativeQuantizeInputs::build(&args, &manifest).unwrap();

        assert_eq!(inputs.tensor_overrides.len(), 4);
        assert_eq!(
            inputs._tensor_patterns[0].to_str().unwrap(),
            "nextn\\.pre_projection\\.weight"
        );
        assert_eq!(
            inputs._tensor_patterns[1].to_str().unwrap(),
            "^token_embd\\.weight$"
        );
        assert_eq!(
            inputs._tensor_patterns[2].to_str().unwrap(),
            "(^|\\.)nextn\\."
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn loads_legacy_imatrix_with_include_filter() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).unwrap();
        let imatrix_path = root.join("imatrix.dat");
        write_legacy_imatrix(
            &imatrix_path,
            &[
                ("blk.0.attn_q.weight", 2, &[2.0, 4.0]),
                ("blk.0.ffn_down.weight", 1, &[9.0, 12.0]),
            ],
        );
        let mut args = native_args();
        args.imatrix = Some(imatrix_path);
        args.include_weights = vec!["attn_q".to_string()];
        let manifest = manifest(None);

        let inputs = NativeQuantizeInputs::build(&args, &manifest).unwrap();

        assert_eq!(inputs.imatrix.as_ref().unwrap().entry_count(), 1);
        assert!(inputs.kv_overrides.len() >= 3);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn loads_gguf_imatrix_for_native_inputs() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).unwrap();
        let imatrix_path = root.join("imatrix.gguf");
        write_gguf_imatrix(&imatrix_path);
        let mut args = native_args();
        args.imatrix = Some(imatrix_path);
        args.include_weights = vec!["attn_q".to_string()];
        let manifest = manifest(None);

        let inputs = NativeQuantizeInputs::build(&args, &manifest).unwrap();
        let imatrix = inputs.imatrix.as_ref().unwrap();

        assert_eq!(imatrix.entry_count(), 1);
        assert_eq!(imatrix.dataset(), Some("calibration.txt"));
        assert_eq!(imatrix.chunk_count(), 7);
        assert!(inputs.kv_overrides.len() >= 3);
        fs::remove_dir_all(root).unwrap();
    }

    fn native_args() -> QuantRunnerArgs {
        QuantRunnerArgs {
            backend: BackendKind::LlamaApi,
            native_runtime_libraries: Vec::new(),
            work_dir: PathBuf::from("/tmp/work"),
            print_only: false,
            dry_run: false,
            allow_requantize: true,
            pure: false,
            imatrix: None,
            include_weights: Vec::new(),
            exclude_weights: Vec::new(),
            output_tensor_type: None,
            token_embedding_type: None,
            tensor_type: Vec::new(),
            prune_layers: None,
            override_kv: Vec::new(),
            nthreads: Some(8),
            leave_output_tensor: true,
            no_stage_source: false,
            keep_staged_source: false,
            spool_dir: None,
            keep_spool: false,
            watchdog_seconds: None,
            max_memory: None,
            memory_policy: crate::memory_budget::MemoryPolicy::Hard,
            record_dir: None,
            json_event_file: None,
            json_event_interval_seconds: 120,
            json_event_window: 8,
        }
    }

    fn manifest(tensor_type_file: Option<PathBuf>) -> Manifest {
        Manifest {
            schema_version: MANIFEST_VERSION,
            kind: JobKind::QuantizeGguf,
            source: PathBuf::from("/tmp/source"),
            source_prefix: Some("BF16".to_string()),
            target: PathBuf::from("/tmp/target"),
            target_prefix: "Q2_K".to_string(),
            output_basename: "model-q2".to_string(),
            expected_splits: 2,
            window_size: 1,
            quant: Some("Q2_K".to_string()),
            output_type: None,
            tensor_type_file,
        }
    }

    fn unique_temp_dir() -> PathBuf {
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("skippy-native-quantize-{nanos}-{id}"))
    }

    fn write_legacy_imatrix(path: &Path, entries: &[(&str, i32, &[f32])]) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(entries.len() as i32).to_le_bytes());
        for (name, ncall, values) in entries {
            bytes.extend_from_slice(&(name.len() as i32).to_le_bytes());
            bytes.extend_from_slice(name.as_bytes());
            bytes.extend_from_slice(&ncall.to_le_bytes());
            bytes.extend_from_slice(&(values.len() as i32).to_le_bytes());
            for value in *values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        bytes.extend_from_slice(&3_i32.to_le_bytes());
        let dataset = "dataset.txt";
        bytes.extend_from_slice(&(dataset.len() as i32).to_le_bytes());
        bytes.extend_from_slice(dataset.as_bytes());
        fs::write(path, bytes).unwrap();
    }

    fn write_gguf_imatrix(path: &Path) {
        const GGUF_MAGIC: &[u8; 4] = b"GGUF";
        const GGML_TYPE_F32: u32 = 0;
        const GGUF_TYPE_UINT32: u32 = 4;
        const GGUF_TYPE_STRING: u32 = 8;
        const GGUF_TYPE_ARRAY: u32 = 9;
        const KV_GENERAL_ALIGNMENT: &str = "general.alignment";
        const KV_IMATRIX_DATASETS: &str = "imatrix.datasets";
        const KV_IMATRIX_CHUNK_COUNT: &str = "imatrix.chunk_count";

        let sums_name = "blk.0.attn_q.weight.in_sum2";
        let counts_name = "blk.0.attn_q.weight.counts";
        let mut bytes = Vec::new();
        bytes.extend_from_slice(GGUF_MAGIC);
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        bytes.extend_from_slice(&4_u64.to_le_bytes());
        write_gguf_kv_string(&mut bytes, "general.type", "imatrix", GGUF_TYPE_STRING);
        write_gguf_kv_u32(&mut bytes, KV_GENERAL_ALIGNMENT, 32, GGUF_TYPE_UINT32);
        write_gguf_kv_u32(&mut bytes, KV_IMATRIX_CHUNK_COUNT, 7, GGUF_TYPE_UINT32);
        write_gguf_kv_string_array(
            &mut bytes,
            KV_IMATRIX_DATASETS,
            &["calibration.txt"],
            GGUF_TYPE_ARRAY,
            GGUF_TYPE_STRING,
        );
        write_gguf_tensor_info(&mut bytes, sums_name, &[2, 2], GGML_TYPE_F32, 0);
        write_gguf_tensor_info(&mut bytes, counts_name, &[1, 2], GGML_TYPE_F32, 16);
        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        for value in [2.0_f32, 4.0, 9.0, 11.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in [2.0_f32, 0.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(path, bytes).unwrap();
    }

    fn write_gguf_kv_string(bytes: &mut Vec<u8>, key: &str, value: &str, string_type: u32) {
        write_gguf_string(bytes, key);
        bytes.extend_from_slice(&string_type.to_le_bytes());
        write_gguf_string(bytes, value);
    }

    fn write_gguf_kv_u32(bytes: &mut Vec<u8>, key: &str, value: u32, uint32_type: u32) {
        write_gguf_string(bytes, key);
        bytes.extend_from_slice(&uint32_type.to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn write_gguf_kv_string_array(
        bytes: &mut Vec<u8>,
        key: &str,
        values: &[&str],
        array_type: u32,
        string_type: u32,
    ) {
        write_gguf_string(bytes, key);
        bytes.extend_from_slice(&array_type.to_le_bytes());
        bytes.extend_from_slice(&string_type.to_le_bytes());
        bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            write_gguf_string(bytes, value);
        }
    }

    fn write_gguf_tensor_info(
        bytes: &mut Vec<u8>,
        name: &str,
        dims: &[u64],
        tensor_type: u32,
        offset: u64,
    ) {
        write_gguf_string(bytes, name);
        bytes.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for dim in dims {
            bytes.extend_from_slice(&dim.to_le_bytes());
        }
        bytes.extend_from_slice(&tensor_type.to_le_bytes());
        bytes.extend_from_slice(&offset.to_le_bytes());
    }

    fn write_gguf_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }
}
