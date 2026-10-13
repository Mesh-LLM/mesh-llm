use super::super::{acquisition, admission, bootstrap};
use crate::command::DynResult;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub(super) enum Workflow {
    Certification,
    NemotronCompose,
}
/// Omits only the binary and source/profile fields returned by actual bootstrap.
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Certification {
    pub mode: admission::Mode,
    pub projector: admission::Artifact,
    pub target_parts: Vec<admission::Artifact>,
    pub expected_parts: usize,
    pub mtp_draft: Option<admission::Artifact>,
    pub layer_count: u32,
    pub mtp_layer_count: Option<u32>,
    pub ctx_size: u32,
}
impl Certification {
    pub(super) fn bind(
        &self,
        observed: &bootstrap::execution::ObservedBootstrap,
        timeout_secs: u64,
    ) -> DynResult<admission::Input> {
        let input = admission::Input {
            schema_version: 1,
            mode: self.mode,
            binary: observed.binary.clone(),
            supplied_mesh_revision: observed.mesh_commit.clone(),
            native_profile: "standalone-static-skippy-quantize-cpu".into(),
            projector: self.projector.clone(),
            target_parts: self.target_parts.clone(),
            expected_parts: self.expected_parts,
            mtp_draft: self.mtp_draft.clone(),
            layer_count: self.layer_count,
            mtp_layer_count: self.mtp_layer_count,
            ctx_size: self.ctx_size,
            timeout_secs,
        };
        input.validate()?;
        Ok(input)
    }
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub schema_version: u32,
    pub workflow: Workflow,
    pub timeout_secs: u64,
    pub runner: admission::Artifact,
    pub bootstrap: bootstrap::contract::Input,
    pub certification: Certification,
    pub projector: acquisition::Projector,
    #[serde(default)]
    pub receipt_export: Option<super::receipt_export::Config>,
}
impl Input {
    pub(super) fn validate(&self) -> DynResult<()> {
        if self.workflow != Workflow::Certification {
            return Err("Nemotron compose delivery requires qualified G2 conversion and G4 caller; unsupported before launch".into());
        }
        if self.schema_version != 1
            || !(30..=86400).contains(&self.timeout_secs)
            || self.bootstrap.timeout_seconds != self.timeout_secs
            || !self.runner.path.is_absolute()
            || !bootstrap::contract::hex(&self.runner.sha256, 64)
        {
            return Err("native job schema/shared budget/runner pin refused".into());
        }
        self.bootstrap.validate()?;
        if let Some(export) = &self.receipt_export {
            export.validate()?;
            if self.timeout_secs <= export.export_budget_secs + 5 {
                return Err("whole job budget cannot reserve export and native execution".into());
            }
        }
        // Admission uses a structural placeholder only; execute replaces it with
        // the typed actual bootstrap return, never a caller-provided done flag.
        let structural = bootstrap::execution::ObservedBootstrap {
            binary: admission::Artifact {
                path: self
                    .runner
                    .path
                    .with_file_name("__bootstrap_output__")
                    .join("skippy-quantize"),
                sha256: "0".repeat(64),
            },
            mesh_commit: self.bootstrap.mesh_commit.clone(),
            prepared_llama_commit: self.bootstrap.llama_commit.clone(),
        };
        let request = acquisition::Request {
            schema_version: 1,
            certification: self
                .certification
                .bind(&structural, self.timeout_secs.min(3600))?,
            projector: self.projector.clone(),
        };
        request.validate()
    }
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
enum CompositionWorkflow {
    SuppliedConvertedCompose,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CompositionInput {
    pub schema_version: u32,
    workflow: CompositionWorkflow,
    pub timeout_secs: u64,
    pub runner: admission::Artifact,
    pub bootstrap: bootstrap::contract::Input,
    pub composition: crate::automation::hf_mtp_compose::job_phase::Template,
    #[serde(default)]
    pub receipt_export: Option<super::receipt_export::Config>,
}
impl CompositionInput {
    fn validate(&self) -> DynResult<()> {
        if self.schema_version != 1
            || !(30..=86400).contains(&self.timeout_secs)
            || self.bootstrap.timeout_seconds != self.timeout_secs
            || !self.runner.path.is_absolute()
            || !bootstrap::contract::hex(&self.runner.sha256, 64)
        {
            return Err("supplied compose schema/shared allowance/runner refused".into());
        }
        self.bootstrap.validate()?;
        self.composition.validate()?;
        if let Some(export) = &self.receipt_export {
            export.validate()?;
            if self.timeout_secs <= export.export_budget_secs + 5 {
                return Err("compose whole allowance cannot reserve evidence export".into());
            }
        }
        Ok(())
    }
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
enum NativeWorkflow {
    NativeNemotronCompose,
}
/// Local checkpoint custody and tokenizer-profile bytes are observed by the conversion owner.
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct NativeInput {
    pub schema_version: u32,
    workflow: NativeWorkflow,
    pub timeout_secs: u64,
    pub runner: admission::Artifact,
    pub bootstrap: bootstrap::contract::Input,
    pub conversion: crate::automation::hf_mtp_compose::raw_conversion::Template,
    #[serde(default)]
    pub receipt_export: Option<super::receipt_export::Config>,
}
impl NativeInput {
    fn validate(&self) -> DynResult<()> {
        if self.schema_version != 1
            || !(30..=86400).contains(&self.timeout_secs)
            || self.bootstrap.timeout_seconds != self.timeout_secs
            || !self.runner.path.is_absolute()
            || !bootstrap::contract::hex(&self.runner.sha256, 64)
        {
            return Err("native conversion worker schema/shared budget/runner refused".into());
        }
        self.bootstrap.validate()?;
        self.conversion.validate()?;
        if let Some(export) = &self.receipt_export {
            export.validate()?;
            if self.timeout_secs <= export.export_budget_secs + 5 {
                return Err("native conversion cannot reserve durable evidence allowance".into());
            }
        }
        Ok(())
    }
}

/// All bodies deny unknown fields; the distinct composition tag cannot silently consume certification fields.
#[derive(Deserialize, Serialize)]
#[serde(untagged)]
pub(super) enum JobInput {
    Certification(Input),
    Composition(CompositionInput),
    Native(NativeInput),
}
impl JobInput {
    pub(super) fn validate(&self) -> DynResult<()> {
        match self {
            Self::Certification(input) => input.validate(),
            Self::Composition(input) => input.validate(),
            Self::Native(input) => input.validate(),
        }
    }
    pub(super) fn timeout_secs(&self) -> u64 {
        match self {
            Self::Certification(i) => i.timeout_secs,
            Self::Composition(i) => i.timeout_secs,
            Self::Native(i) => i.timeout_secs,
        }
    }
    pub(super) fn receipt_export(&self) -> Option<&super::receipt_export::Config> {
        match self {
            Self::Certification(i) => i.receipt_export.as_ref(),
            Self::Composition(i) => i.receipt_export.as_ref(),
            Self::Native(i) => i.receipt_export.as_ref(),
        }
    }
    pub(super) fn native_status(&self) -> &'static str {
        match self {
            Self::Certification(_) => "CERTIFIED",
            Self::Composition(_) | Self::Native(_) => "COMPOSED",
        }
    }
}
