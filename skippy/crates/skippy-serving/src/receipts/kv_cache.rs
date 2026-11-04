//! Whether a stage's KV cache is one an audit covers.
//!
//! A stage with a quantized KV cache (`cache_type_k` or `cache_type_v` of
//! q8_0 or q4_0) issues no receipts: an honest one fails the loose attention
//! check in about 1 audit in 4.5 ([`skippy_receipts::cache_types`]). So its
//! recorder does not start, the stage serves every request as usual, and
//! an embedder that asks whether the runtime issues receipts hears that it
//! issues none. A turn that stored quantized rows all the same is refused
//! again when it commits (`skippy_receipts::recorder`).

use skippy_protocol::StageConfig;
use skippy_receipts::cache_types;

/// Why a stage with `config`'s KV cache issues no receipts, when it issues
/// none.
pub(super) fn unaudited(config: &StageConfig) -> Option<String> {
    unaudited_types(&config.cache_type_k, &config.cache_type_v)
}

/// [`unaudited`] for the cache types of the keys and the values, as a stage
/// config names them. A name the runtime does not know is not judged here:
/// the model does not load with it.
fn unaudited_types(cache_type_k: &str, cache_type_v: &str) -> Option<String> {
    [
        ("cache_type_k", cache_type_k),
        ("cache_type_v", cache_type_v),
    ]
    .into_iter()
    .find_map(|(setting, name)| {
        let ggml_type = i32::try_from(skippy_runtime::parse_cache_type(name).ok()?).ok()?;
        cache_types::audited(ggml_type).is_none().then(|| {
            format!(
                "{setting} is {}, and no audit covers a quantized KV cache",
                cache_types::name(ggml_type)
            )
        })
    })
}

#[cfg(test)]
mod tests {
    use super::unaudited_types;

    #[test]
    fn only_a_cache_that_is_not_quantized_gets_receipts() {
        for (k, v) in [("f16", "f16"), ("", ""), ("F16", "f16")] {
            assert_eq!(unaudited_types(k, v), None, "{k} {v}");
        }
        for (k, v, named) in [
            ("q8_0", "q8_0", "cache_type_k is q8_0"),
            ("f16", "q8_0", "cache_type_v is q8_0"),
            ("q4_0", "f16", "cache_type_k is q4_0"),
            ("f16", "q4", "cache_type_v is q4_0"),
            ("q8", "f16", "cache_type_k is q8_0"),
        ] {
            let reason = unaudited_types(k, v).unwrap();
            assert!(reason.contains(named), "{k} {v}: {reason}");
            assert!(reason.contains("quantized KV cache"), "{reason}");
        }
        // the runtime refuses such a name when it loads the model
        assert_eq!(unaudited_types("auto", "f16"), None);
    }
}
