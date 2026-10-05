//! Configured inference-fit safety margin and unit conversion.

/// Margin the local fit withholds on top of the driver reserve when the owner
/// configures none.
pub const BUILTIN_SAFETY_MARGIN_GB: f64 = 2.0;

/// The configured safety margin in bytes, rounded to whole MiB the way the fit
/// target rounds it, so the advertised reserve matches the memory the fit
/// actually withholds. `None` means the owner configured no margin and the
/// built-in default applies. Absurd or negative values saturate to zero rather
/// than overflowing.
pub fn safety_margin_bytes(safety_margin_gb: Option<f64>) -> u64 {
    let gb = safety_margin_gb.unwrap_or(BUILTIN_SAFETY_MARGIN_GB);
    safety_margin_mib(gb).saturating_mul(1024 * 1024)
}

/// Whole MiB withheld for a margin expressed in GB.
pub fn safety_margin_mib(safety_margin_gb: f64) -> u64 {
    (safety_margin_gb * 1024.0).round().max(0.0) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_built_in_margin_is_withheld_when_the_owner_configures_none() {
        assert_eq!(safety_margin_bytes(None), 2 * 1024 * 1024 * 1024);
        assert_eq!(safety_margin_bytes(Some(0.5)), 512 * 1024 * 1024);
        assert_eq!(safety_margin_bytes(Some(0.0)), 0);
    }

    #[test]
    fn an_impossible_margin_saturates_instead_of_overflowing() {
        assert_eq!(safety_margin_bytes(Some(-1.0)), 0);
        assert_eq!(safety_margin_bytes(Some(f64::MAX)), u64::MAX);
    }
}
