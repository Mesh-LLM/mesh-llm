use super::*;

pub(super) struct Candidate {
    pub(super) head: String,
    package: String,
    identity: String,
    branch: String,
}
impl Candidate {
    pub(super) fn from_outputs(value: &JobOutputs) -> DynResult<Self> {
        Ok(Self {
            head: input::head(&value.head)?,
            identity: input::digest(&value.identity)?,
            package: safe_line(&value.package, "package")?,
            branch: safe_line(&value.branch, "branch")?,
        })
    }
    fn outputs(self) -> BTreeMap<&'static str, String> {
        BTreeMap::from([
            ("package", self.package),
            ("identity", self.identity),
            ("head", self.head),
            ("branch", self.branch),
        ])
    }
}
pub(super) struct Resume {
    pub(super) head: String,
    package: String,
    identity: String,
    feedback: String,
    failure: Failure,
}
pub(super) struct Failure {
    pub(super) class: String,
    pub(super) stage: String,
}
impl Failure {
    pub(super) fn from_outputs(value: &JobOutputs, class: &str, stage: &str) -> DynResult<Self> {
        Ok(Self {
            class: safe_line(
                if value.failure_class.is_empty() {
                    class
                } else {
                    &value.failure_class
                },
                "failure_class",
            )?,
            stage: safe_line(
                if value.failure_stage.is_empty() {
                    stage
                } else {
                    &value.failure_stage
                },
                "failure_stage",
            )?,
        })
    }
    fn outputs(self) -> BTreeMap<&'static str, String> {
        BTreeMap::from([("failure_class", self.class), ("failure_stage", self.stage)])
    }
}
pub(super) enum AttemptDecision {
    Green(Candidate),
    Repairable(Resume),
    Failed(Failure),
}
impl AttemptDecision {
    pub(super) fn outputs(self) -> BTreeMap<&'static str, String> {
        match self {
            Self::Green(candidate) => {
                let mut result = candidate.outputs();
                result.extend([
                    ("state", "green".into()),
                    ("green", "true".into()),
                    ("repairable", "false".into()),
                ]);
                result
            }
            Self::Repairable(resume) => {
                let mut result = resume.failure.outputs();
                result.extend([
                    ("state", "repairable".into()),
                    ("repairable", "true".into()),
                    ("resume_package", resume.package),
                    ("resume_identity", resume.identity),
                    ("resume_head", resume.head),
                    ("resume_feedback", resume.feedback),
                ]);
                result
            }
            Self::Failed(failure) => {
                let mut result = failure.outputs();
                result.extend([("state", "failed".into()), ("repairable", "false".into())]);
                result
            }
        }
    }
}
pub(super) fn failed_or_resume(
    changed: bool,
    value: &JobOutputs,
    stage: &str,
) -> DynResult<AttemptDecision> {
    if changed && value.repairable == "true" {
        let failure = Failure::from_outputs(value, "candidate", "family-certification")?;
        if failure.class == "candidate" {
            return Ok(AttemptDecision::Repairable(Resume {
                head: input::head(&value.head)?,
                identity: input::digest(&value.identity)?,
                package: safe_line(&value.package, "package")?,
                feedback: safe_line(&value.feedback, "feedback")?,
                failure,
            }));
        }
        return Ok(AttemptDecision::Failed(failure));
    }
    Ok(AttemptDecision::Failed(Failure::from_outputs(
        value,
        "infrastructure",
        stage,
    )?))
}
pub(super) fn identity_failure() -> AttemptDecision {
    AttemptDecision::Failed(Failure {
        class: "candidate".into(),
        stage: "independent-verification-identity".into(),
    })
}
pub(super) enum FinalDecision {
    Noop,
    Certified,
    Publish(Candidate),
}
impl FinalDecision {
    pub(super) fn outputs(self) -> BTreeMap<&'static str, String> {
        match self {
            Self::Noop => BTreeMap::from([("state", "noop".into()), ("publish", "false".into())]),
            Self::Certified => {
                BTreeMap::from([("state", "green".into()), ("publish", "false".into())])
            }
            Self::Publish(candidate) => {
                let mut result = candidate.outputs();
                result.extend([("state", "green".into()), ("publish", "true".into())]);
                result
            }
        }
    }
}
