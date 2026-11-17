//! Preserve requested host placement or reject it before model loading.
use anyhow::{Context, Result};

pub(super) fn from_stage_config(
    layers: Option<skippy_protocol::HostExpertLayers>,
) -> Result<Option<skippy_runtime::HostExpertLayers>> {
    layers
        .map(|layers| {
            skippy_runtime::HostExpertLayers::new(layers.start, layers.end)
                .context("host_expert_layers must contain between 1 and 4096 blocks")
        })
        .transpose()
}

#[cfg(test)]
mod tests {
    use super::super::{RuntimeLaunchOverrides, runtime_config_from_stage_config};
    use skippy_protocol::{HostExpertLayers, StageConfig};

    #[test]
    fn stage_conversion_preserves_host_placement_or_reports_invalid_range() -> anyhow::Result<()> {
        let convert = |layers| {
            runtime_config_from_stage_config(
                &StageConfig {
                    layer_start: 0,
                    layer_end: 2,
                    host_expert_layers: layers,
                    ..StageConfig::default()
                },
                &RuntimeLaunchOverrides::default(),
            )
        };
        assert!(convert(None)?.host_expert_layers.is_none());
        let valid = convert(Some(HostExpertLayers {
            start: 17,
            end: 4113,
        }))?
        .host_expert_layers
        .expect("host placement retained");
        assert_eq!((valid.start, valid.end), (17, 4113));
        for layers in [
            HostExpertLayers {
                start: 17,
                end: 4114,
            },
            HostExpertLayers { start: 4, end: 4 },
            HostExpertLayers { start: 5, end: 4 },
        ] {
            assert!(
                convert(Some(layers))
                    .unwrap_err()
                    .to_string()
                    .contains("host_expert_layers")
            );
        }
        Ok(())
    }
}
