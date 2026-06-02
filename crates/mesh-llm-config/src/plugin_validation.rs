use anyhow::{Result, bail};

use crate::PluginConfigEntry;

pub(crate) fn validate_plugin_entries(entries: &[PluginConfigEntry]) -> Result<()> {
    for (index, entry) in entries.iter().enumerate() {
        validate_plugin_startup(entry, index)?;
    }
    Ok(())
}

fn validate_plugin_startup(entry: &PluginConfigEntry, index: usize) -> Result<()> {
    if matches!(entry.startup.connect_timeout_secs, Some(0)) {
        bail!("plugin[{index}].startup.connect_timeout_secs must be at least 1 when set");
    }
    if matches!(entry.startup.init_timeout_secs, Some(0)) {
        bail!("plugin[{index}].startup.init_timeout_secs must be at least 1 when set");
    }
    Ok(())
}
