//! UI build profile projection, output reuse and dependency freshness.
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, io, path::Path, time::SystemTime};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Profile {
    Debug,
    Dev,
    Release,
}

impl Profile {
    pub(super) fn parse(value: &str) -> Result<Self, &'static str> {
        match value.to_ascii_lowercase().as_str() {
            "debug" | "" => Ok(Self::Debug),
            "dev" => Ok(Self::Dev),
            "release" => Ok(Self::Release),
            _ => Err("expected debug, dev, or release build profile"),
        }
    }

    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Debug => "debug",
            Self::Dev => "dev",
            Self::Release => "release",
        }
    }
}

pub(super) struct Environment {
    pub profile: Profile,
    pub variables: BTreeMap<String, String>,
}

impl Environment {
    pub(super) fn debug_ui(&self) -> &str {
        if self.profile == Profile::Release {
            "false"
        } else {
            self.variables
                .get("VITE_MESH_LLM_DEBUG_UI")
                .map(String::as_str)
                .filter(|value| !value.is_empty())
                .unwrap_or("true")
        }
    }

    pub(super) fn projected_variables(&self) -> BTreeMap<String, String> {
        let mut variables = self.variables.clone();
        variables.insert("VITE_MESH_LLM_DEBUG_UI".into(), self.debug_ui().into());
        variables
    }

    pub(super) fn stamp_for(&self, ui: &Path) -> io::Result<String> {
        let mut stamp: serde_json::Value = serde_json::from_str(&self.stamp())?;
        let mut dotenv = BTreeMap::new();
        for name in DOTENV_INPUTS {
            let digest = match fs::read(ui.join(name)) {
                Ok(bytes) => Some(hex::encode(Sha256::digest(bytes))),
                Err(error) if error.kind() == io::ErrorKind::NotFound => None,
                Err(error) => return Err(error),
            };
            dotenv.insert(*name, digest);
        }
        stamp["dotenv"] = serde_json::to_value(dotenv)?;
        Ok(stamp.to_string())
    }

    pub(super) fn stamp(&self) -> String {
        // Versioned JSON distinguishes absent and empty settings and safely
        // represents arbitrary UTF-8 values without ambiguous line delimiters.
        serde_json::json!({
            "schema": 2,
            "profile": self.profile.name(),
            "variables": self.projected_variables(),
        })
        .to_string()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Decision {
    Reuse,
    Build,
    InstallAndBuild,
}

const DOTENV_INPUTS: &[&str] = &[
    ".env",
    ".env.local",
    ".env.production",
    ".env.production.local",
];

const INPUTS: &[&str] = &[
    "package.json",
    "pnpm-lock.yaml",
    "vite.config.ts",
    "tsconfig.json",
    "tsconfig.app.json",
    "tsconfig.node.json",
    "biome.json",
    "index.html",
    ".env",
    ".env.local",
    ".env.production",
    ".env.production.local",
    "src",
    "public",
];

pub(super) fn decide(ui: &Path, environment: &Environment) -> io::Result<Decision> {
    let dist = ui.join("dist");
    let Some(dist_metadata) = existing(&dist)? else {
        return install_decision(ui);
    };
    if !dist_metadata.is_dir() || !has_regular_file(&dist)? {
        return install_decision(ui);
    }
    let stamp = dist.join(".mesh-llm-ui-build-env");
    let recorded = match fs::read(&stamp) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return install_decision(ui),
        Err(error) => return Err(error),
    };
    if recorded != environment.stamp_for(ui)?.as_bytes() {
        return install_decision(ui);
    }
    let built = dist_metadata.modified()?;
    for name in INPUTS {
        if newer_input(&ui.join(name), built)? {
            return install_decision(ui);
        }
    }
    Ok(Decision::Reuse)
}

fn install_decision(ui: &Path) -> io::Result<Decision> {
    let Some(modules) = existing(&ui.join("node_modules"))? else {
        return Ok(Decision::InstallAndBuild);
    };
    if !modules.is_dir() {
        return Ok(Decision::InstallAndBuild);
    }
    let installed = modules.modified()?;
    for name in ["package.json", "pnpm-lock.yaml"] {
        if let Some(manifest) = existing(&ui.join(name))?
            && manifest.modified()? > installed
        {
            return Ok(Decision::InstallAndBuild);
        }
    }
    Ok(Decision::Build)
}

fn existing(path: &Path) -> io::Result<Option<fs::Metadata>> {
    match fs::metadata(path) {
        Ok(metadata) => Ok(Some(metadata)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

pub(super) fn has_regular_file(path: &Path) -> io::Result<bool> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        if entry.file_name() == ".mesh-llm-ui-build-env" {
            continue;
        }
        let kind = entry.file_type()?;
        if kind.is_file() || (kind.is_dir() && has_regular_file(&entry.path())?) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn newer_input(path: &Path, cutoff: SystemTime) -> io::Result<bool> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error),
    };
    if metadata.is_file() {
        return Ok(metadata.modified()? > cutoff);
    }
    if metadata.is_dir() {
        // Directory changes capture removals and renames even when surviving
        // regular-file mtimes predate the current build.
        if metadata.modified()? > cutoff {
            return Ok(true);
        }
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let kind = entry.file_type()?;
            if kind.is_file() && entry.metadata()?.modified()? > cutoff {
                return Ok(true);
            }
            if kind.is_dir() && newer_input(&entry.path(), cutoff)? {
                return Ok(true);
            }
        }
    }
    Ok(false)
}
