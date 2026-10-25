use super::error::{Checked, Rejected};
use crate::ci_plan::document::Json;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum ProductRow {
    LinuxCpu,
    LinuxCuda,
    MacosMetal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub(super) enum ProducerWorkflow {
    #[serde(rename = ".github/workflows/main_linux.yml")]
    Linux,
    #[serde(rename = ".github/workflows/main_macos.yml")]
    Macos,
}

impl ProducerWorkflow {
    pub(super) fn parse(name: &str, path: &str) -> Checked<Self> {
        match (name, path) {
            ("Main \u{b7} Linux", ".github/workflows/main_linux.yml") => Ok(Self::Linux),
            ("Main \u{b7} macOS", ".github/workflows/main_macos.yml") => Ok(Self::Macos),
            _ => Err(Rejected::Workflow),
        }
    }

    pub(super) const fn rows(self) -> &'static [ProductRow] {
        match self {
            Self::Linux => &[ProductRow::LinuxCpu, ProductRow::LinuxCuda],
            Self::Macos => &[ProductRow::MacosMetal],
        }
    }
}

impl ProductRow {
    pub(super) const fn id(self) -> &'static str {
        match self {
            Self::LinuxCpu => "linux-cpu",
            Self::LinuxCuda => "linux-cuda",
            Self::MacosMetal => "macos-metal",
        }
    }

    pub(super) const fn identity(self) -> [&'static str; 4] {
        match self {
            Self::LinuxCpu => ["linux", "amd64", "cpu", "x86_64-unknown-linux-gnu"],
            Self::LinuxCuda => ["linux", "amd64", "cuda", "x86_64-unknown-linux-gnu"],
            Self::MacosMetal => ["macos", "arm64", "metal", "aarch64-apple-darwin"],
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct RuntimeRow {
    pub(super) id: String,
    pub(super) platform: String,
    pub(super) architecture: String,
    pub(super) backend: String,
    pub(super) target: String,
}

pub(super) struct Catalog {
    rows: Vec<RuntimeRow>,
}

impl Catalog {
    pub(super) fn parse(ownership: &[u8], slices: &[u8]) -> Checked<Self> {
        let ownership = Json::parse(ownership).map_err(Rejected::Input)?;
        let slices = Json::parse(slices).map_err(Rejected::Input)?;
        let rows = crate::ci_plan::exhaustive_runtime_rows(&ownership, &slices)
            .map_err(Rejected::Catalog)?
            .into_iter()
            .map(serde_json::from_value)
            .collect::<Result<Vec<RuntimeRow>, _>>()
            .map_err(Rejected::Input)?;
        Ok(Self { rows })
    }

    pub(super) fn row(&self, requested: ProductRow) -> Checked<&RuntimeRow> {
        let mut matches = self.rows.iter().filter(|row| row.id == requested.id());
        let row = matches.next().ok_or(Rejected::MissingRow)?;
        if matches.next().is_some() {
            return Err(Rejected::MissingRow);
        }
        let actual = [
            row.platform.as_str(),
            row.architecture.as_str(),
            row.backend.as_str(),
            row.target.as_str(),
        ];
        if actual != requested.identity() {
            return Err(Rejected::RowIdentity);
        }
        Ok(row)
    }
}
