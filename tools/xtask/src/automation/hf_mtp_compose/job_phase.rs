//! Supplied converted MTP composition after actual worker bootstrap.
//! SafeTensors conversion remains an explicit separate family capability.
use super::{contract, execution};
use crate::{
    automation::hf_certify::admission::Artifact, command::DynResult, process::Cancellation,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(in crate::automation) enum MtpSource {
    SuppliedConverted { artifact: Artifact },
    NemotronCheckpoint { directory: PathBuf },
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(in crate::automation) struct Template {
    pub target_parts: Vec<Artifact>,
    pub mtp: MtpSource,
    pub target_basename: String,
    pub composite_basename: String,
    pub expected_parts: usize,
    pub mtp_block: u32,
    pub composite_repo: String,
}
/// The worker supplies these from its actual bootstrap return and owning budget.
/// No caller-provided completion flag or independent build attestation is accepted.
pub(in crate::automation) struct Context<'a> {
    pub binary: &'a Artifact,
    pub mesh_revision: &'a str,
    pub deadline: Instant,
    pub cancellation: &'a Cancellation,
}
impl Template {
    fn bind(&self, binary: &Artifact, revision: &str) -> DynResult<contract::Input> {
        let MtpSource::SuppliedConverted { artifact } = &self.mtp else {
            return Err("native Nemotron SafeTensors conversion is not implemented; supply a separately converted pinned MTP GGUF, without claiming native conversion".into());
        };
        let input = contract::Input {
            schema_version: 1,
            binary: binary.clone(),
            target_parts: self.target_parts.clone(),
            mtp_gguf: artifact.clone(),
            target_basename: self.target_basename.clone(),
            composite_basename: self.composite_basename.clone(),
            expected_parts: self.expected_parts,
            mtp_block: self.mtp_block,
            supplied_mesh_revision: revision.into(),
            native_profile: "standalone-static-skippy-quantize-cpu".into(),
            timeout_secs: 3600,
            composite_repo: self.composite_repo.clone(),
        };
        input.validate()?;
        Ok(input)
    }
    pub(in crate::automation) fn validate(&self) -> DynResult<()> {
        // Only structural admission: real identity is observed by the supervised
        // compose identity worker after binding the actual bootstrap output.
        let placeholder = Artifact {
            path: std::path::absolute(std::env::temp_dir().join("__compose_bootstrap_output__"))?,
            sha256: "0".repeat(64),
        };
        self.bind(&placeholder, &"0".repeat(40)).map(|_| ())
    }
    pub(in crate::automation) fn execute(
        &self,
        root: &Path,
        context: &Context<'_>,
        evidence: &mut Value,
    ) -> DynResult<Value> {
        self.validate()?;
        if context.cancellation.is_cancelled() || Instant::now() >= context.deadline {
            return Err("compose worker phase cancelled/deadline before identity or launch".into());
        }
        let bound = self.bind(context.binary, context.mesh_revision)?;
        let phase_deadline = context
            .deadline
            .min(Instant::now() + Duration::from_secs(3600));
        let plan =
            execution::execute(&bound, root, phase_deadline, context.cancellation, evidence)?;
        // Return an in-memory pending plan only. The outer worker owns final
        // signal/deadline/runner checks and correlated terminal publication.
        if context.cancellation.is_cancelled() || Instant::now() >= phase_deadline {
            return Err("compose worker phase cancelled/deadline after native validation".into());
        }
        Ok(plan)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn template(root: &Path) -> Template {
        let artifact = |path| Artifact {
            path,
            sha256: "a".repeat(64),
        };
        Template {
            target_parts: (1..=3)
                .map(|i| artifact(root.join(format!("Target-{i:05}-of-00003.gguf"))))
                .collect(),
            mtp: MtpSource::SuppliedConverted {
                artifact: artifact(root.join("mtp.gguf")),
            },
            target_basename: "Target".into(),
            composite_basename: "Composite".into(),
            expected_parts: 3,
            mtp_block: 88,
            composite_repo: "fixture/composite".into(),
        }
    }
    #[test]
    fn supplied_compose_phase_binds_actual_bootstrap_and_complete_middle_roster() {
        let root = tempfile::tempdir().unwrap();
        let base = root.path().canonicalize().unwrap();
        let template = template(&base);
        template.validate().unwrap();
        let binary = Artifact {
            path: base.join("actual-built-tool"),
            sha256: "b".repeat(64),
        };
        let input = template.bind(&binary, &"c".repeat(40)).unwrap();
        assert!(input.binary == binary);
        assert_eq!(input.supplied_mesh_revision, "c".repeat(40));
        assert!(input.target_parts == template.target_parts);
        assert_eq!(input.mtp_block, 88);
        assert_eq!(input.remote_name(1), "Composite-00002-of-00003.gguf");
        let mut invalid = template.clone();
        invalid.target_parts.swap(0, 1);
        assert!(invalid.validate().is_err());
        root.close().unwrap();
    }
    #[test]
    fn unconverted_checkpoint_and_unknown_fields_refuse_without_claiming_conversion() {
        let root = tempfile::tempdir().unwrap();
        let base = root.path().canonicalize().unwrap();
        let mut template = template(&base);
        template.mtp = MtpSource::NemotronCheckpoint {
            directory: base.join("checkpoint"),
        };
        assert!(
            template
                .validate()
                .unwrap_err()
                .to_string()
                .contains("not implemented")
        );
        let mut value = serde_json::to_value(template).unwrap();
        value["conversion_completed"] = serde_json::json!(true);
        assert!(serde_json::from_value::<Template>(value).is_err());
        root.close().unwrap();
    }
    #[test]
    fn supplied_compose_phase_cancel_and_deadline_prevent_any_identity_or_child_output() {
        let root = tempfile::tempdir().unwrap();
        let base = root.path().canonicalize().unwrap();
        let template = template(&base);
        let binary = Artifact {
            path: base.join("not-launched"),
            sha256: "b".repeat(64),
        };
        for cancelled in [false, true] {
            let cancellation = Cancellation::default();
            if cancelled {
                cancellation.cancel();
            }
            let context = Context {
                binary: &binary,
                mesh_revision: &"c".repeat(40),
                deadline: if cancelled {
                    Instant::now() + Duration::from_secs(60)
                } else {
                    Instant::now()
                },
                cancellation: &cancellation,
            };
            assert!(
                template
                    .execute(&base, &context, &mut serde_json::json!({}))
                    .is_err()
            );
            assert_eq!(std::fs::read_dir(&base).unwrap().count(), 0);
        }
        root.close().unwrap();
    }
}
