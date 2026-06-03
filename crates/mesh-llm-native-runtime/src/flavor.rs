use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{fmt, str::FromStr};

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum NativeRuntimeBackendKind {
    Cpu,
    Metal,
    Cuda,
    Rocm,
    Vulkan,
    Other(String),
}

pub type NativeRuntimeFlavor = NativeRuntimeBackendKind;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeRuntimeFlavorParseError {
    value: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeRuntimeBackend {
    pub kind: NativeRuntimeBackendKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda: Option<CudaRuntimeRequirements>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rocm: Option<RocmRuntimeRequirements>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vulkan: Option<VulkanRuntimeRequirements>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CudaRuntimeRequirements {
    pub toolkit_major: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_driver: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gpu_arches: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RocmRuntimeRequirements {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gpu_arches: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct VulkanRuntimeRequirements {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_api_version: Option<String>,
}

impl fmt::Display for NativeRuntimeFlavorParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid native runtime backend '{}'", self.value)
    }
}

impl std::error::Error for NativeRuntimeFlavorParseError {}

impl NativeRuntimeBackendKind {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
            Self::Rocm => "rocm",
            Self::Vulkan => "vulkan",
            Self::Other(value) => value.as_str(),
        }
    }

    pub fn default_rank(&self) -> i64 {
        match self {
            Self::Cuda => 650,
            Self::Rocm => 600,
            Self::Metal => 600,
            Self::Vulkan => 350,
            Self::Cpu => 100,
            Self::Other(_) => 0,
        }
    }
}

impl NativeRuntimeBackend {
    pub fn cpu() -> Self {
        Self {
            kind: NativeRuntimeBackendKind::Cpu,
            cuda: None,
            rocm: None,
            vulkan: None,
        }
    }

    pub fn metal() -> Self {
        Self {
            kind: NativeRuntimeBackendKind::Metal,
            cuda: None,
            rocm: None,
            vulkan: None,
        }
    }

    pub fn cuda(toolkit_major: u32, gpu_arches: Vec<String>) -> Self {
        Self {
            kind: NativeRuntimeBackendKind::Cuda,
            cuda: Some(CudaRuntimeRequirements {
                toolkit_major,
                min_driver: None,
                gpu_arches,
            }),
            rocm: None,
            vulkan: None,
        }
    }

    pub fn rocm(gpu_arches: Vec<String>) -> Self {
        Self {
            kind: NativeRuntimeBackendKind::Rocm,
            cuda: None,
            rocm: Some(RocmRuntimeRequirements {
                version: None,
                gpu_arches,
            }),
            vulkan: None,
        }
    }

    pub fn vulkan() -> Self {
        Self {
            kind: NativeRuntimeBackendKind::Vulkan,
            cuda: None,
            rocm: None,
            vulkan: Some(VulkanRuntimeRequirements {
                min_api_version: None,
            }),
        }
    }
}

impl fmt::Display for NativeRuntimeBackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for NativeRuntimeBackendKind {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for NativeRuntimeBackendKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(Self::from(value.as_str()))
    }
}

impl FromStr for NativeRuntimeBackendKind {
    type Err = NativeRuntimeFlavorParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let normalized = value.trim().to_ascii_lowercase();
        if normalized.is_empty() {
            return Err(NativeRuntimeFlavorParseError {
                value: value.to_string(),
            });
        }
        Ok(match normalized.as_str() {
            "cpu" => Self::Cpu,
            "metal" => Self::Metal,
            "cuda" | "cuda-blackwell" | "blackwell" => Self::Cuda,
            "rocm" | "hip" => Self::Rocm,
            "vulkan" => Self::Vulkan,
            _ => Self::Other(normalized),
        })
    }
}

impl From<&str> for NativeRuntimeBackendKind {
    fn from(value: &str) -> Self {
        value
            .parse()
            .unwrap_or_else(|_| Self::Other(value.trim().to_ascii_lowercase()))
    }
}
