//! Pure model-size label parsing shared by catalog and mesh selection code.

/// Parse catalog labels such as `20GB`, `4.4GB`, or `491MB` into GB.
#[must_use]
pub fn parse_size_gb(label: &str) -> f64 {
    let label = label.trim();
    if let Some(gb) = label.strip_suffix("GB") {
        gb.trim().parse().unwrap_or(0.0)
    } else if let Some(mb) = label.strip_suffix("MB") {
        mb.trim().parse::<f64>().unwrap_or(0.0) / 1000.0
    } else {
        0.0
    }
}

/// Parse the legacy Nostr model-pack size form, which recognizes only a GB suffix.
///
/// This intentionally preserves the historical `trim_end_matches("GB")`
/// behavior used by both Nostr model-pack implementations; catalog labels use
/// [`parse_size_gb`] instead because they also support MB.
#[must_use]
pub fn parse_gb_suffix_size(label: &str) -> f64 {
    label.trim_end_matches("GB").parse::<f64>().unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::{parse_gb_suffix_size, parse_size_gb};

    #[test]
    fn parses_catalog_gb_and_mb_labels() {
        assert_eq!(parse_size_gb(" 4.4GB "), 4.4);
        assert_eq!(parse_size_gb("491MB"), 0.491);
        assert_eq!(parse_size_gb("unknown"), 0.0);
    }

    #[test]
    fn preserves_legacy_gb_only_parser_behavior() {
        assert_eq!(parse_gb_suffix_size("4.4GB"), 4.4);
        assert_eq!(parse_gb_suffix_size(" 4.4GB "), 0.0);
        assert_eq!(parse_gb_suffix_size("491MB"), 0.0);
    }
}
