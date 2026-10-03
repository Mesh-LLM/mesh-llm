use std::{error::Error, fmt};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginTarget {
    triple: String,
    archive_ext: ArchiveExt,
}

impl PluginTarget {
    pub fn current() -> Result<Self, UnsupportedTarget> {
        Self::from_os_arch(std::env::consts::OS, std::env::consts::ARCH)
    }

    pub fn from_os_arch(os: &str, arch: &str) -> Result<Self, UnsupportedTarget> {
        let (triple, archive_ext) = match (os, arch) {
            ("macos", "aarch64") => ("aarch64-apple-darwin", ArchiveExt::TarGz),
            ("macos", "x86_64") => ("x86_64-apple-darwin", ArchiveExt::TarGz),
            ("linux", "x86_64") => ("x86_64-unknown-linux-gnu", ArchiveExt::TarGz),
            ("linux", "aarch64") => ("aarch64-unknown-linux-gnu", ArchiveExt::TarGz),
            ("windows", "x86_64") => ("x86_64-pc-windows-msvc", ArchiveExt::Zip),
            ("windows", "aarch64") => ("aarch64-pc-windows-msvc", ArchiveExt::Zip),
            _ => {
                return Err(UnsupportedTarget {
                    os: os.to_string(),
                    arch: arch.to_string(),
                });
            }
        };
        Ok(Self {
            triple: triple.to_string(),
            archive_ext,
        })
    }

    pub fn triple(&self) -> &str {
        &self.triple
    }

    pub fn archive_ext(&self) -> ArchiveExt {
        self.archive_ext
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArchiveExt {
    #[serde(rename = "tar.gz")]
    TarGz,
    #[serde(rename = "zip")]
    Zip,
}

impl ArchiveExt {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TarGz => "tar.gz",
            Self::Zip => "zip",
        }
    }
}

impl fmt::Display for ArchiveExt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedTarget {
    os: String,
    arch: String,
}

impl fmt::Display for UnsupportedTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unsupported plugin target: {}/{}", self.os, self.arch)
    }
}

impl Error for UnsupportedTarget {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_supported_targets() {
        let linux = PluginTarget::from_os_arch("linux", "x86_64").unwrap();
        assert_eq!(linux.triple(), "x86_64-unknown-linux-gnu");
        assert_eq!(linux.archive_ext(), ArchiveExt::TarGz);

        let windows = PluginTarget::from_os_arch("windows", "aarch64").unwrap();
        assert_eq!(windows.triple(), "aarch64-pc-windows-msvc");
        assert_eq!(windows.archive_ext(), ArchiveExt::Zip);
    }

    #[test]
    fn rejects_unsupported_targets() {
        let err = PluginTarget::from_os_arch("linux", "arm").unwrap_err();
        assert_eq!(err.to_string(), "unsupported plugin target: linux/arm");
    }
}
