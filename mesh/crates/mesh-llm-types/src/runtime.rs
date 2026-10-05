use serde::{Deserialize, Deserializer, Serialize, de};

const MODEL_RUNTIME_KIND_VARIANTS: &[&str] = &["auto", "cpu", "cuda", "rocm", "metal", "vulkan"];

#[derive(Clone, Copy, Debug, Default, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ModelRuntimeKind {
    #[default]
    Auto,
    Cpu,
    Cuda,
    Rocm,
    Metal,
    Vulkan,
}

impl ModelRuntimeKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Rocm => "rocm",
            Self::Metal => "metal",
            Self::Vulkan => "vulkan",
        }
    }

    pub fn parse_str(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "cpu" => Some(Self::Cpu),
            "cuda" => Some(Self::Cuda),
            "rocm" => Some(Self::Rocm),
            "metal" => Some(Self::Metal),
            "vulkan" => Some(Self::Vulkan),
            _ => None,
        }
    }
}

impl<'de> Deserialize<'de> for ModelRuntimeKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::parse_str(&value)
            .ok_or_else(|| de::Error::unknown_variant(&value, MODEL_RUNTIME_KIND_VARIANTS))
    }
}
