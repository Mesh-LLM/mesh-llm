//! The inference receipts options of a model load.

use skippy_serving::receipts::ReceiptsOptions;

/// What a model's runtime needs to issue inference receipts, as the load
/// options carry it. It holds the seed of the node's signing key, which
/// must never reach a log or an error message: `Debug` prints everything
/// but the key, of which only its presence is shown.
#[derive(Clone)]
pub struct ReceiptsLoadOptions(ReceiptsOptions);

impl From<ReceiptsOptions> for ReceiptsLoadOptions {
    fn from(options: ReceiptsOptions) -> Self {
        Self(options)
    }
}

impl From<ReceiptsLoadOptions> for ReceiptsOptions {
    fn from(options: ReceiptsLoadOptions) -> Self {
        options.0
    }
}

impl std::fmt::Debug for ReceiptsLoadOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Only the fields named here are printed, so one Skippy adds later
        // stays out until it is known to be safe to show.
        let options = &self.0;
        f.debug_struct("ReceiptsOptions")
            .field("listen", &options.listen)
            .field("allow_remote", &options.allow_remote)
            .field("downstream", &options.downstream)
            .field("claimed_model", &options.claimed_model)
            .field("signing_key", &options.signing_key.map(|_| "<redacted>"))
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use crate::SkippyModelLoadOptions;

    #[test]
    fn debug_never_prints_the_receipts_signing_key() {
        let seed: [u8; 32] = std::array::from_fn(|index| 200 + index as u8);
        let options = SkippyModelLoadOptions::for_direct_gguf("m", "/models/m.gguf").with_receipts(
            Some(skippy_serving::receipts::ReceiptsOptions::embedded(
                "org/repo@abc/m.gguf".into(),
                Some(seed),
            )),
        );
        let key_bytes = seed.map(|byte| byte.to_string()).join(",");
        for printed in [format!("{options:?}"), format!("{options:#?}")] {
            let compact: String = printed.split_whitespace().collect();
            assert!(!compact.contains(&key_bytes), "{printed}");
            assert!(
                compact.contains("signing_key:Some(\"<redacted>\""),
                "{printed}"
            );
            // the rest of the options is still there to read
            assert!(printed.contains("org/repo@abc/m.gguf"), "{printed}");
            assert!(printed.contains("ctx_size"), "{printed}");
        }
        // and the check above can fail: this is how the key would print
        let leaked: String = format!("{seed:#?}").split_whitespace().collect();
        assert!(leaked.contains(&key_bytes));
    }

    #[test]
    fn debug_shows_options_without_receipts() {
        let options = SkippyModelLoadOptions::for_direct_gguf("m", "/models/m.gguf");
        assert!(format!("{options:?}").contains("receipts: None"));
    }
}
