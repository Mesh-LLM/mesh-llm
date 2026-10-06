//! Finite supplied workspace/size observations; no filesystem or runtime capacity claim.
use anyhow::{Result, bail};
pub(super) fn estimate(source: u64) -> Result<u64> {
    source
        .checked_add(32 * 1024_u64.pow(3))
        .ok_or_else(|| anyhow::anyhow!("workspace estimate overflow"))
}
pub(super) fn format(size: u64) -> String {
    let mut value = size as f64;
    for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"] {
        if value < 1024.0 || unit == "PiB" {
            return if unit == "B" {
                format!("{size} B")
            } else {
                format!("{value:.1} {unit}")
            };
        }
        value /= 1024.0;
    }
    unreachable!()
}
pub(super) fn generation(path: &std::path::Path) -> Result<String> {
    let bytes = crate::snapshot_promotion::local_publisher::regular_input::read(
        &mut crate::snapshot_promotion::local_publisher::regular_input::open(
            path,
            1024 * 1024,
            false,
        )?,
        1024 * 1024,
    )?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|_| anyhow::anyhow!("generation defaults JSON refused"))?;
    if !value.is_object() {
        bail!("generation defaults object required");
    }
    Ok(serde_json::to_string_pretty(&value)?)
}
#[cfg(test)]
mod tests {
    #[test]
    fn layer_workspace_observations_preserve_headroom_units_and_refuse_overflow() {
        assert_eq!(super::estimate(7).unwrap(), 34359738375);
        assert!(super::estimate(u64::MAX).is_err());
        assert_eq!(super::format(0), "0 B");
        assert_eq!(super::format(1024), "1.0 KiB");
        assert_eq!(super::format(1024 * 1024), "1.0 MiB");
    }
}
