use super::manifest_preflight::Requirements;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Deserialize, Serialize)]
pub(super) struct Input {
    pub manifest: PathBuf,
    pub requirements: Requirements,
    #[serde(default)]
    pub builds: Vec<Build>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub build_jobs: Vec<BuildJob>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_config: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_config_sha256: Option<String>,
    #[serde(skip)]
    pub validated_engine_config: Option<super::external_config::Config>,
    #[serde(default, skip_serializing_if = "ContextQualification::is_mesh")]
    pub context_qualification: ContextQualification,
    #[serde(default, skip_serializing_if = "super::replay_profile::Mode::is_all")]
    pub replay_mode: super::replay_profile::Mode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hf_home: Option<PathBuf>,
    pub model: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_reference: Option<String>,
    #[serde(default)]
    pub model_sha256: String,
    #[serde(default)]
    pub minimum_context_tokens: u64,
    #[serde(default)]
    pub minimum_session_prompt_tokens: u64,
    #[serde(default)]
    pub require_recurrent_restores: bool,
    pub passes: u32,
    pub max_output_tokens: u64,
    pub request_timeout_seconds: u64,
    pub startup_timeout_seconds: u64,
    pub timeout_seconds: u64,
    pub output: PathBuf,
    pub prompt_token_range: Option<[u64; 2]>,
    pub min_cache_pct: Option<f64>,
    #[serde(default)]
    pub require_output_match: bool,
    pub max_ttft_regression_pct: Option<f64>,
    #[serde(default, skip_serializing)]
    pub resume: bool,
    pub dataset: Option<super::run_input::Dataset>,
}

impl Input {
    pub(super) fn validate(&self) -> crate::command::DynResult<()> {
        if self
            .hf_home
            .as_ref()
            .is_some_and(|home| !home.is_absolute())
            || (self.replay_mode != super::replay_profile::Mode::All
                && self.context_qualification.is_mesh())
        {
            return Err("selected replay profiles require captured context qualification and absolute HF-home".into());
        }
        if self.model_reference.as_ref().is_some_and(|reference| {
            reference.is_empty() || reference.chars().any(char::is_control)
        }) {
            return Err(
                "model reference must be nonempty and contain no control characters".into(),
            );
        }
        if self
            .prompt_token_range
            .is_some_and(|[minimum, maximum]| minimum == 0 || minimum > maximum)
            || self
                .min_cache_pct
                .is_some_and(|value| !value.is_finite() || !(0.0..=100.0).contains(&value))
            || self
                .max_ttft_regression_pct
                .is_some_and(|value| !value.is_finite())
        {
            return Err("invalid replay acceptance budget".into());
        }
        if !self.manifest.is_absolute()
            || ((self.context_qualification.is_mesh()
                || (!self.model_sha256.is_empty()
                    && (!self.build_jobs.is_empty()
                        || self
                            .builds
                            .iter()
                            .any(|build| matches!(build, Build::Mesh(_))))))
                && !self.model.is_absolute())
            || !self.output.is_absolute()
            || !(1..=1000).contains(&self.passes)
            || self.max_output_tokens == 0
            || !(1..=86400).contains(&self.request_timeout_seconds)
            || !(1..=86400).contains(&self.startup_timeout_seconds)
            || !(1..=86400).contains(&self.timeout_seconds)
            || self.startup_timeout_seconds >= self.timeout_seconds
        {
            return Err("run requires absolute paths, positive output budget and bounded request/startup/execution deadlines".into());
        }
        if self.model.as_os_str().is_empty() {
            return Err("run requires a nonempty model identity".into());
        }
        if self.context_qualification == ContextQualification::Captured
            && (self.minimum_context_tokens != 0
                || self.minimum_session_prompt_tokens != 0
                || self.require_recurrent_restores)
        {
            return Err("captured profile must explicitly omit Mesh context/session/recurrent qualification".into());
        }
        if self.context_qualification.is_mesh()
            && (self.engine_config.is_some()
                || self.builds.iter().any(|build| !build.engine().is_mesh()))
        {
            return Err("runtime context qualification currently requires mesh arms".into());
        }
        Ok(())
    }

    pub(super) fn record_selection(
        &self,
        document: &mut serde_json::Value,
    ) -> crate::command::DynResult<()> {
        document["config"]["replay_mode"] = serde_json::to_value(self.replay_mode)?;
        if let Some(home) = &self.hf_home {
            document["config"]["hf_home"] = serde_json::to_value(home)?;
        }
        document["config"]["minimum_worker_waves"] = self.requirements.minimum_worker_waves.into();
        document["config"]["required_frameworks"] =
            serde_json::to_value(&self.requirements.required_frameworks)?;
        if self.context_qualification.is_mesh()
            || (!self.model_sha256.is_empty()
                && self
                    .builds
                    .iter()
                    .any(|build| matches!(build, Build::Mesh(_))))
        {
            document["config"]["model_file"] = serde_json::to_value(&self.model)?;
            document["config"]["model_sha256"] = self.model_sha256.clone().into();
        }
        if self.context_qualification == ContextQualification::Captured {
            document["config"]["context_qualification"] = "captured".into();
        }
        if let Some(reference) = &self.model_reference {
            document["config"]["model"] = reference.clone().into();
        }
        Ok(())
    }

    pub(super) fn verify_builds(
        &self,
        budget: &super::run_budget::Budget,
    ) -> crate::command::DynResult<()> {
        let mut labels = std::collections::BTreeSet::new();
        for build in &self.builds {
            if build.label().is_empty()
                || [".", ".."].contains(&build.label())
                || !build
                    .label()
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"_.-".contains(&byte))
                || !labels.insert(build.label())
            {
                return Err("build labels must be unique safe path components".into());
            }
            super::run_transport::verify_build(build, budget)?;
        }
        Ok(())
    }
}

#[derive(Deserialize, Serialize)]
#[serde(untagged)]
pub(super) enum Build {
    Mesh(Box<MeshBuild>),
    External(Box<super::external_probe::Verified>),
}

#[derive(Deserialize, Serialize)]
pub(super) struct MeshBuild {
    #[serde(default)]
    pub engine: Engine,
    pub label: String,
    #[serde(rename = "ref")]
    pub reference: String,
    pub commit: String,
    pub binary: PathBuf,
    pub binary_sha256: String,
    pub runtime_root: PathBuf,
    pub runtime: PathBuf,
    pub runtime_sha256: String,
    pub backend: Option<String>,
    pub worktree: Option<PathBuf>,
}

impl Build {
    pub(super) fn label(&self) -> &str {
        match self {
            Self::Mesh(build) => &build.label,
            Self::External(build) => &build.label,
        }
    }
    pub(super) fn reference(&self) -> &str {
        match self {
            Self::Mesh(build) => &build.reference,
            Self::External(build) => &build.reference,
        }
    }
    pub(super) fn commit(&self) -> &str {
        match self {
            Self::Mesh(build) => &build.commit,
            Self::External(build) => &build.commit,
        }
    }
    pub(super) fn engine(&self) -> Engine {
        match self {
            Self::Mesh(build) => build.engine,
            Self::External(build) => match build.engine {
                super::external_config::Engine::Llama => Engine::Llama,
                super::external_config::Engine::Vllm => Engine::Vllm,
                super::external_config::Engine::Sglang => Engine::Sglang,
            },
        }
    }
}

#[derive(Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(super) enum Engine {
    #[default]
    Mesh,
    #[serde(rename = "llama.cpp")]
    Llama,
    Vllm,
    Sglang,
}

impl Engine {
    pub(super) fn is_mesh(self) -> bool {
        self == Self::Mesh
    }
}

#[derive(Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(super) enum ContextQualification {
    #[default]
    Mesh,
    Captured,
}
impl ContextQualification {
    pub(super) fn is_mesh(&self) -> bool {
        *self == Self::Mesh
    }
}

#[derive(Deserialize, Serialize)]
pub(super) struct BuildJob {
    pub repo: PathBuf,
    pub worktree_root: PathBuf,
    pub label: String,
    #[serde(rename = "ref")]
    pub reference: String,
    pub backend: String,
    pub git: PathBuf,
    pub just: PathBuf,
    pub timeout_seconds: u64,
    pub logs: PathBuf,
    #[serde(default)]
    pub skip_build: bool,
}
