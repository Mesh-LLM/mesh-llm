/// Half-open absolute block range `[start, end)` whose routed-expert tensors
/// (`ffn_{up,down,gate}_exps`) load into host memory. Placement only changes
/// where the weights live and which kernels run them; routing and precision
/// are unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostExpertLayers {
    pub start: u32,
    pub end: u32,
}

impl HostExpertLayers {
    /// Build a range of at most 4096 blocks, returning `None` when invalid.
    pub fn new(start: u32, end: u32) -> Option<Self> {
        (start < end && end - start <= 4096).then_some(Self { start, end })
    }

    pub fn layer_count(self) -> u32 {
        self.end.saturating_sub(self.start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RuntimeConfig;

    #[test]
    fn runtime_config_raw_defaults_host_expert_layers_to_empty_range() -> anyhow::Result<()> {
        let raw = RuntimeConfig::default().as_raw()?.raw;
        assert_eq!(raw.host_expert_layer_start, 0);
        assert_eq!(raw.host_expert_layer_end, 0);
        Ok(())
    }

    #[test]
    fn runtime_config_raw_carries_host_expert_layer_range() -> anyhow::Result<()> {
        let raw = RuntimeConfig {
            host_expert_layers: HostExpertLayers::new(25, 48),
            ..RuntimeConfig::default()
        }
        .as_raw()?
        .raw;
        assert_eq!(raw.host_expert_layer_start, 25);
        assert_eq!(raw.host_expert_layer_end, 48);
        Ok(())
    }

    #[test]
    fn host_expert_layers_rejects_empty_ranges() {
        assert_eq!(HostExpertLayers::new(4, 4), None);
        assert_eq!(HostExpertLayers::new(5, 4), None);
        assert_eq!(
            HostExpertLayers::new(0, 3).map(|layers| layers.layer_count()),
            Some(3)
        );
    }

    #[test]
    fn host_expert_layers_enforces_native_span_limit() -> anyhow::Result<()> {
        let layers = HostExpertLayers::new(17, 4113).expect("4096-block range");
        let raw = RuntimeConfig {
            host_expert_layers: Some(layers),
            ..RuntimeConfig::default()
        }
        .as_raw()?
        .raw;
        assert_eq!(raw.host_expert_layer_start, 17);
        assert_eq!(raw.host_expert_layer_end, 4113);
        assert_eq!(HostExpertLayers::new(17, 4114), None);
        for layers in [
            HostExpertLayers {
                start: 17,
                end: 4114,
            },
            HostExpertLayers { start: 4, end: 4 },
            HostExpertLayers { start: 5, end: 4 },
        ] {
            let config = RuntimeConfig {
                host_expert_layers: Some(layers),
                ..RuntimeConfig::default()
            };
            assert!(config.validate().is_err());
            assert!(config.as_raw().is_err());
        }
        Ok(())
    }
}
