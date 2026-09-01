use super::BoolOrAuto;

#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MultimodalConfig {
    #[serde(default)]
    pub mmproj: Option<String>,
    #[serde(default)]
    pub mmproj_url: Option<String>,
    #[serde(default)]
    pub mmproj_offload: Option<BoolOrAuto>,
    #[serde(default)]
    pub image_min_tokens: Option<u32>,
    #[serde(default)]
    pub image_max_tokens: Option<u32>,
    #[serde(default)]
    pub image_marker: Option<String>,
    #[serde(default)]
    pub media_marker: Option<String>,
    #[serde(default)]
    pub batch_max_tokens: Option<u32>,
    #[serde(default)]
    pub glm_dsa_policy: Option<String>,
    #[serde(default)]
    pub generation_signal_window: Option<u32>,
    #[serde(default)]
    pub embeddings: Option<toml::Value>,
    #[serde(default)]
    pub reranking: Option<toml::Value>,
    #[serde(default)]
    pub pooling: Option<toml::Value>,
    #[serde(default)]
    pub vocoder: Option<toml::Value>,
}

pub(crate) fn merge_multimodal(
    current: Option<MultimodalConfig>,
    mmproj: Option<String>,
) -> Option<MultimodalConfig> {
    let mut config = current.unwrap_or_default();
    config.mmproj = config.mmproj.or(mmproj);
    if config == MultimodalConfig::default() {
        None
    } else {
        Some(config)
    }
}
