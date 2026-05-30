use std::time::Duration;

use mesh_llm_config::PluginStartupConfig;
use serde::Serialize;

pub(crate) const DEFAULT_PLUGIN_CONNECT_TIMEOUT_SECS: u64 = 10;
pub(crate) const DEFAULT_PLUGIN_INIT_TIMEOUT_SECS: u64 = 30;

#[derive(Clone, Debug)]
pub struct PluginStartupOptions {
    pub connect_timeout: Duration,
    pub init_timeout: Duration,
    pub optional: bool,
    pub lazy_start: bool,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct PluginStartupSummary {
    pub connect_timeout_secs: u64,
    pub init_timeout_secs: u64,
    pub optional: bool,
    pub lazy_start: bool,
}

impl Default for PluginStartupOptions {
    fn default() -> Self {
        Self {
            connect_timeout: Duration::from_secs(DEFAULT_PLUGIN_CONNECT_TIMEOUT_SECS),
            init_timeout: Duration::from_secs(DEFAULT_PLUGIN_INIT_TIMEOUT_SECS),
            optional: false,
            lazy_start: false,
        }
    }
}

impl PluginStartupOptions {
    pub fn from_config(config: &PluginStartupConfig) -> Self {
        let defaults = Self::default();
        Self {
            connect_timeout: config
                .connect_timeout_secs
                .map(Duration::from_secs)
                .unwrap_or(defaults.connect_timeout),
            init_timeout: config
                .init_timeout_secs
                .map(Duration::from_secs)
                .unwrap_or(defaults.init_timeout),
            optional: config.optional,
            lazy_start: config.lazy_start,
        }
    }

    pub fn connect_timeout(&self) -> Duration {
        self.connect_timeout
    }

    pub fn init_timeout(&self) -> Duration {
        self.init_timeout
    }

    pub fn summary(&self) -> PluginStartupSummary {
        PluginStartupSummary {
            connect_timeout_secs: self.connect_timeout.as_secs(),
            init_timeout_secs: self.init_timeout.as_secs(),
            optional: self.optional,
            lazy_start: self.lazy_start,
        }
    }
}
