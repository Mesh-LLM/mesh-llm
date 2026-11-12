//! Auditable checkpoint identities; operator paths never select different weights.
use mesh_llm_plugin::{PluginContext, models::AssetPlan};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{io::Read, path::PathBuf};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Model {
    YuFeng,
    Nemotron,
}

pub struct Artifact {
    pub filename: &'static str,
    pub sha256: &'static str,
    pub bytes: u64,
}

const YUFENG: Artifact = Artifact {
    filename: "YuFeng-XGuard-Reason-0.6B.Q8_0.gguf",
    sha256: "3c0d0f65c71945c2ac6c306aa031b77ee8a65874d30c010ccd0ddbcf51306f23",
    bytes: 804_751_488,
};
const NEMOTRON: Artifact = Artifact {
    filename: "Nemotron-3.5-Content-Safety.Q8_0.gguf",
    sha256: "217c703e74074fa57116fee18eb4531e4c7cbbf1e2ed7d626076004900cb7884",
    bytes: 4_130_408_096,
};
const PROJECTOR: Artifact = Artifact {
    filename: "Nemotron-3.5-Content-Safety.mmproj-f16.gguf",
    sha256: "ae8dd1fb405d8acf4ea41bc0172ed72c5de283e0c35b594eb3718d498b2e86d0",
    bytes: 851_252_000,
};

impl Model {
    pub fn id(self) -> &'static str {
        match self {
            Self::YuFeng => "Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B",
            Self::Nemotron => "nvidia/Nemotron-3.5-Content-Safety",
        }
    }
    pub fn repo(self) -> &'static str {
        match self {
            Self::YuFeng => "mradermacher/YuFeng-XGuard-Reason-0.6B-GGUF",
            Self::Nemotron => "mradermacher/Nemotron-3.5-Content-Safety-GGUF",
        }
    }
    pub fn revision(self) -> &'static str {
        match self {
            Self::YuFeng => "a457e581bb00997ff1eb1f9ae0bf21488c6a632c",
            Self::Nemotron => "4fcb2495f250aa0dba62d7a3b1a18bc6fb5f002a",
        }
    }
    pub fn weights(self) -> &'static Artifact {
        match self {
            Self::YuFeng => &YUFENG,
            Self::Nemotron => &NEMOTRON,
        }
    }
    pub fn projector(self) -> Option<&'static Artifact> {
        (self == Self::Nemotron).then_some(&PROJECTOR)
    }
    pub fn layers(self) -> u32 {
        match self {
            Self::YuFeng => 28,
            Self::Nemotron => 34,
        }
    }
    pub fn template(self) -> &'static str {
        match self {
            Self::YuFeng => include_str!("../assets/yufeng.jinja"),
            Self::Nemotron => include_str!("../assets/nemotron.jinja"),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    pub screen_received_images: bool,
    pub model_path: Option<PathBuf>,
    pub projector_path: Option<PathBuf>,
    pub n_gpu_layers: i32,
    pub runtime_selection: Option<String>,
    pub n_threads: Option<usize>,
    pub ctx_size: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            screen_received_images: false,
            model_path: None,
            projector_path: None,
            n_gpu_layers: -1,
            runtime_selection: None,
            n_threads: None,
            ctx_size: 8192,
        }
    }
}

impl Settings {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            (2048..=32768).contains(&self.ctx_size),
            "ctx_size must be 2048..32768"
        );
        anyhow::ensure!(
            self.n_gpu_layers >= -1,
            "n_gpu_layers must be -1 or nonnegative"
        );
        anyhow::ensure!(
            self.n_threads.is_none_or(|value| value > 0),
            "n_threads must be positive"
        );
        anyhow::ensure!(
            self.model_path.is_some() || self.projector_path.is_none(),
            "projector_path requires model_path"
        );
        skippy_runtime_install::RuntimeSelection::parse(self.runtime_selection.as_deref())?;
        Ok(())
    }
}

pub struct Paths {
    pub weights: PathBuf,
    pub projector: Option<PathBuf>,
}

pub async fn acquire(
    model: Model,
    settings: &Settings,
    context: &PluginContext<'_>,
) -> anyhow::Result<Paths> {
    if let Some(weights) = &settings.model_path {
        anyhow::ensure!(
            model.projector().is_some() || settings.projector_path.is_none(),
            "YuFeng does not accept a projector"
        );
        anyhow::ensure!(
            model.projector().is_none() || settings.projector_path.is_some(),
            "Nemotron requires its pinned projector_path"
        );
        return Ok(Paths {
            weights: weights.clone(),
            projector: settings.projector_path.clone(),
        });
    }
    let files = std::iter::once(model.weights())
        .chain(model.projector())
        .map(|asset| asset.filename.to_owned())
        .collect();
    let plan = AssetPlan {
        model: model.id().to_owned(),
        repo: model.repo().to_owned(),
        revision: model.revision().to_owned(),
        files,
    };
    let bundle = context.prepare_model_assets(&plan).await?;
    anyhow::ensure!(
        bundle.revision == model.revision(),
        "asset revision mismatch"
    );
    let path = |asset: &Artifact| {
        bundle
            .files
            .get(asset.filename)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("required pinned asset missing"))
    };
    Ok(Paths {
        weights: path(model.weights())?,
        projector: model.projector().map(path).transpose()?,
    })
}

pub fn verify(path: &std::path::Path, artifact: &Artifact) -> anyhow::Result<()> {
    let mut file = std::fs::File::open(path)?;
    anyhow::ensure!(
        file.metadata()?.len() == artifact.bytes,
        "guard asset size mismatch"
    );
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    anyhow::ensure!(
        hex::encode(hash.finalize()) == artifact.sha256,
        "guard asset digest mismatch"
    );
    Ok(())
}
