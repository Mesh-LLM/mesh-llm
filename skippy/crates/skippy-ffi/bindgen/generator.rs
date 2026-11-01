//! Deterministic bindgen generation for Skippy's consumed native ABI surface.
//!
//! This file is shared verbatim between `build.rs` (via `#[path]` inclusion)
//! and nothing else; keeping a single implementation means the committed
//! bindings under `src/bindings/` and the CI regeneration check can never
//! drift apart in configuration.
//!
//! Three translation units are generated, one per ABI owner:
//!
//! - `skippy.rs` from the patched llama.cpp checkout's public `skippy.h`
//!   umbrella (Skippy's own versioned native-runtime C ABI).
//! - `llama.rs` from the pinned upstream `llama.h` surface Skippy consumes
//!   directly (log hooks, quantization, vocabulary probes, embeddings,
//!   metadata reads, graph-reuse counters).
//! - `mtmd.rs` from the pinned `mtmd.h` / `mtmd-helper.h` multimodal
//!   surface.
//!
//! Every allowlist below is intentional. Widening it widens the supported ABI
//! surface and must be reviewed as an ABI change.

use std::path::{Path, PathBuf};

use bindgen::callbacks::{EnumVariantValue, ParseCallbacks};

/// One generated bindings file per ABI owner.
pub struct GeneratedBindings {
    pub skippy: String,
    pub llama: String,
    pub mtmd: String,
}

/// Renames rustified C enum variants to the spellings the Rust-facing API has
/// always used, so type aliases in this crate keep downstream code stable.
///
/// The mechanical rule strips the enum's own SCREAMING_SNAKE prefix, keeps
/// digit-carrying tokens verbatim, title-cases the rest, and retains `_`
/// before pure-digit tokens (`Q4_0`, not `Q40`). [`VARIANT_RENAMES`] holds the
/// upstream spellings no rule can derive (`BF16` -> `Bf16` but `IQ2_XXS` ->
/// `IQ2XXS`); extend that table, never the rule.
#[derive(Debug)]
struct EnumRenamer;

/// Explicit `(enum name, C variant) -> Rust variant` overrides for spellings
/// that are not derivable mechanically. Keep this table small and reviewable.
const VARIANT_RENAMES: &[(&str, &str, &str)] = &[
    (
        "skippy_backend_device_type",
        "SKIPPY_BACKEND_DEVICE_TYPE_IGPU",
        "IGpu",
    ),
    (
        "skippy_activation_dtype",
        "SKIPPY_ACTIVATION_DTYPE_BF16",
        "Bf16",
    ),
    (
        "ggml_backend_dev_type",
        "GGML_BACKEND_DEVICE_TYPE_IGPU",
        "IGpu",
    ),
    (
        "llama_model_kv_override_type",
        "LLAMA_KV_OVERRIDE_TYPE_INT",
        "Int",
    ),
    (
        "llama_model_kv_override_type",
        "LLAMA_KV_OVERRIDE_TYPE_FLOAT",
        "Float",
    ),
    (
        "llama_model_kv_override_type",
        "LLAMA_KV_OVERRIDE_TYPE_BOOL",
        "Bool",
    ),
    (
        "llama_model_kv_override_type",
        "LLAMA_KV_OVERRIDE_TYPE_STR",
        "Str",
    ),
    ("ggml_type", "GGML_TYPE_BF16", "Bf16"),
    ("ggml_type", "GGML_TYPE_MXFP4", "Mxfp4"),
    ("ggml_type", "GGML_TYPE_NVFP4", "Nvfp4"),
    ("ggml_type", "GGML_TYPE_IQ1_S", "IQ1S"),
    ("ggml_type", "GGML_TYPE_IQ1_M", "IQ1M"),
    ("ggml_type", "GGML_TYPE_IQ2_XS", "IQ2XS"),
    ("ggml_type", "GGML_TYPE_IQ2_XXS", "IQ2XXS"),
    ("ggml_type", "GGML_TYPE_IQ2_S", "IQ2S"),
    ("ggml_type", "GGML_TYPE_IQ2_M", "IQ2M"),
    ("ggml_type", "GGML_TYPE_IQ3_XS", "IQ3XS"),
    ("ggml_type", "GGML_TYPE_IQ3_XXS", "IQ3XXS"),
    ("ggml_type", "GGML_TYPE_IQ3_S", "IQ3S"),
    ("ggml_type", "GGML_TYPE_IQ3_M", "IQ3M"),
    ("ggml_type", "GGML_TYPE_IQ4_XS", "IQ4XS"),
    ("ggml_type", "GGML_TYPE_IQ4_NL", "IQ4NL"),
    ("llama_ftype", "LLAMA_FTYPE_MOSTLY_BF16", "MostlyBf16"),
    (
        "llama_ftype",
        "LLAMA_FTYPE_MOSTLY_MXFP4_MOE",
        "MostlyMxfp4Moe",
    ),
    ("llama_ftype", "LLAMA_FTYPE_MOSTLY_NVFP4", "MostlyNvfp4"),
    ("llama_ftype", "LLAMA_FTYPE_MOSTLY_IQ1_S", "MostlyIQ1S"),
    ("llama_ftype", "LLAMA_FTYPE_MOSTLY_IQ1_M", "MostlyIQ1M"),
    ("llama_ftype", "LLAMA_FTYPE_MOSTLY_IQ2_XS", "MostlyIQ2XS"),
    ("llama_ftype", "LLAMA_FTYPE_MOSTLY_IQ2_XXS", "MostlyIQ2XXS"),
    ("llama_ftype", "LLAMA_FTYPE_MOSTLY_IQ2_S", "MostlyIQ2S"),
    ("llama_ftype", "LLAMA_FTYPE_MOSTLY_IQ2_M", "MostlyIQ2M"),
    ("llama_ftype", "LLAMA_FTYPE_MOSTLY_IQ3_XS", "MostlyIQ3XS"),
    ("llama_ftype", "LLAMA_FTYPE_MOSTLY_IQ3_XXS", "MostlyIQ3XXS"),
    ("llama_ftype", "LLAMA_FTYPE_MOSTLY_IQ3_S", "MostlyIQ3S"),
    ("llama_ftype", "LLAMA_FTYPE_MOSTLY_IQ3_M", "MostlyIQ3M"),
    ("llama_ftype", "LLAMA_FTYPE_MOSTLY_IQ4_XS", "MostlyIQ4XS"),
    ("llama_ftype", "LLAMA_FTYPE_MOSTLY_IQ4_NL", "MostlyIQ4NL"),
    (
        "mtmd_gen_audio_type",
        "MTMD_GEN_AUDIO_TYPE_QWEN3TTS",
        "Qwen3Tts",
    ),
    (
        "mtmd_gen_audio_type",
        "MTMD_GEN_AUDIO_TYPE_POCKETTTS",
        "PocketTts",
    ),
];

impl ParseCallbacks for EnumRenamer {
    fn enum_variant_name(
        &self,
        enum_name: Option<&str>,
        original_variant_name: &str,
        _variant_value: EnumVariantValue,
    ) -> Option<String> {
        let enum_name = enum_name
            .and_then(|name| name.strip_prefix("enum "))
            .filter(|name| !name.starts_with('('))?;
        for (owner, variant, renamed) in VARIANT_RENAMES {
            if *owner == enum_name && *variant == original_variant_name {
                return Some((*renamed).to_string());
            }
        }
        let enum_screaming = enum_name.to_ascii_uppercase();
        let prefix_tokens: Vec<&str> = enum_screaming.split('_').collect();
        // Prefer the full enum-name prefix, then progressively shorter ones:
        // upstream declares `SKIPPY_CACHEGEN_RECORD_F16` inside
        // `skippy_cachegen_record_kind`, omitting the trailing `KIND`.
        let stripped = (1..=prefix_tokens.len())
            .rev()
            .map(|count| prefix_tokens[..count].join("_") + "_")
            .find_map(|prefix| original_variant_name.strip_prefix(&prefix))
            .unwrap_or(original_variant_name);
        Some(camel_case_variant(stripped))
    }
}

/// `IO_ERROR` -> `IoError`, `Q4_0` -> `Q4_0`, `TOKEN_MAJOR` -> `TokenMajor`.
fn camel_case_variant(stripped: &str) -> String {
    let mut out = String::new();
    let tokens: Vec<&str> = stripped.split('_').collect();
    for (index, token) in tokens.iter().enumerate() {
        if token.is_empty() {
            continue;
        }
        // Keep `_` only between a digit-carrying token and a pure-digit
        // token: `Q4_0` keeps its separator, `Q3_K_S` and `IQ2_XXS` drop it.
        if index > 0
            && tokens[index - 1].chars().any(|c| c.is_ascii_digit())
            && token.chars().all(|c| c.is_ascii_digit())
        {
            out.push('_');
        }
        if token.chars().any(|c| c.is_ascii_digit()) {
            out.push_str(token);
        } else {
            let mut chars = token.chars();
            if let Some(first) = chars.next() {
                out.push(first.to_ascii_uppercase());
                out.push_str(&chars.as_str().to_ascii_lowercase());
            }
        }
    }
    out
}

/// Generates the three bindings files from a prepared llama.cpp checkout.
pub struct BindingGenerator {
    llama_root: PathBuf,
}

impl BindingGenerator {
    pub fn new(llama_root: impl Into<PathBuf>) -> Self {
        Self {
            llama_root: llama_root.into(),
        }
    }

    pub fn generate(&self) -> Result<GeneratedBindings, String> {
        let skippy = self.generate_one(
            "skippy_abi.h",
            &[
                self.llama_root.join("include"),
                self.llama_root.join("ggml/include"),
            ],
            |builder| {
                builder
                    .allowlist_function("skippy_.*")
                    .allowlist_type("skippy_.*")
                    .allowlist_var("SKIPPY_.*")
                    .allowlist_var("skippy_ffi_bridge_.*")
            },
        )?;
        let llama = self.generate_one(
            "llama_upstream.h",
            &[
                self.llama_root.join("include"),
                self.llama_root.join("ggml/include"),
            ],
            |builder| {
                builder
                    .allowlist_function(
                        "^(llama_model_quantize|llama_model_quantize_default_params|\
                         llama_perf_context|llama_model_is_recurrent|llama_model_is_hybrid|\
                         llama_model_is_diffusion|llama_model_get_vocab|llama_vocab_n_tokens|\
                         llama_vocab_mask|llama_get_embeddings_ith|llama_set_embeddings|\
                         llama_model_meta_val_str|llama_log_set|ggml_log_set|\
                         llama_model_default_params|llama_context_default_params|\
                         llama_model_load_from_file|llama_model_free|llama_init_from_model|\
                         llama_free|llama_decode|llama_get_model|llama_get_memory|\
                         llama_memory_seq_rm|llama_get_logits_ith|llama_n_ctx|llama_n_batch|\
                         llama_n_ubatch|llama_model_n_embd|llama_model_n_layer|\
                         llama_model_rope_type|llama_set_causal_attn|\
                         ggml_backend_dev_by_name)$",
                    )
                    .allowlist_type(
                        "^(llama_model_quantize_params|llama_model_kv_override|\
                         llama_model_kv_override_type|llama_model_tensor_override|\
                         llama_model_imatrix_data|llama_ftype|llama_log_callback|\
                         llama_perf_context_data|llama_perf_context|llama_token|\
                         llama_logit_bias|llama_vocab|llama_model|llama_context|ggml_type|\
                         ggml_log_callback|ggml_backend_dev_t|ggml_backend_device|\
                         llama_batch|llama_model_params|llama_context_params|\
                         llama_progress_callback|ggml_sched_eval_callback|\
                         ggml_abort_callback)$",
                    )
                    .allowlist_var("^LLAMA_FTYPE_.*")
            },
        )?;
        let mtmd = self.generate_one(
            "mtmd_abi.h",
            &[
                self.llama_root.join("include"),
                self.llama_root.join("ggml/include"),
                self.llama_root.join("tools/mtmd"),
            ],
            |builder| {
                builder
                    .allowlist_function("mtmd_.*")
                    .allowlist_type("mtmd_.*")
                    .allowlist_var("MTMD_.*")
            },
        )?;
        Ok(GeneratedBindings {
            skippy,
            llama,
            mtmd,
        })
    }

    fn wrapper_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("bindgen/wrapper")
    }

    fn generate_one(
        &self,
        wrapper: &str,
        include_dirs: &[PathBuf],
        configure: impl FnOnce(bindgen::Builder) -> bindgen::Builder,
    ) -> Result<String, String> {
        let mut builder = bindgen::Builder::default()
            // Keep generated code free of style lints. One attribute per
            // line stays stable under `cargo fmt` so the committed bindings
            // and the CI freshness check cannot drift on formatting.
            // PartialEq derives on callback-bearing structs exist for field
            // comparison in tests; function-pointer fields are never
            // compared by identity, so quiet the derive-site lint too.
            .raw_line("#![allow(non_camel_case_types)]")
            .raw_line("#![allow(non_upper_case_globals)]")
            .raw_line("#![allow(unpredictable_function_pointer_comparisons)]")
            .header(Self::wrapper_dir().join(wrapper).display().to_string())
            .parse_callbacks(Box::new(EnumRenamer))
            // Rustified enums only for named C enums (all carry an owner
            // prefix; bindgen full-matches these against the enum name).
            // Anonymous C enums stay in consts style so their constants keep
            // the plain integer type the surrounding typedefs (e.g.
            // `skippy_runtime_event_category = u32`) expect. Parenthesized
            // alternation does not match here; keep the flat `|`-form.
            .rustified_enum("llama_.*|ggml_.*|mtmd_.*|skippy_.*")
            // Deterministic, reviewable output: no target-specific layout
            // tests in committed files (cross-platform consumers compile the
            // same bindings), no bitfield or layout-trick noise.
            .layout_tests(false)
            .derive_partialeq(true)
            .derive_default(false)
            .generate_comments(true)
            .fit_macro_constants(false)
            .size_t_is_usize(true);
        for dir in include_dirs {
            builder = builder.clang_arg(format!("-I{}", dir.display()));
        }
        builder = configure(builder);
        let bindings = builder
            .generate()
            .map_err(|error| format!("bindgen failed for {wrapper}: {error}"))?;
        Ok(bindings.to_string())
    }
}

/// Writes `content` to `path`, replacing the file only when the content
/// differs so regeneration never perturbs mtimes for unchanged bindings.
pub fn write_if_changed(path: &Path, content: &str) -> Result<bool, String> {
    if let Ok(existing) = std::fs::read_to_string(path)
        && existing == content
    {
        return Ok(false);
    }
    std::fs::write(path, content)
        .map_err(|error| format!("failed to write {}: {error}", path.display()))?;
    Ok(true)
}
