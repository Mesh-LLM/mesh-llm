use super::{
    Digest, Error, ErrorKind, ProducerIdentity, RunAttempt, SourceFamilyPlan, WorkflowRun,
};
use serde_json::Value;
use std::path::Path;

const ARTIFACTS: [(&str, &str); 5] = [
    ("plan.json", "plan_sha256"),
    ("binaries.tar", "binaries_sha256"),
    ("workload-oracles.tar", "workload_oracles_sha256"),
    ("llama-source.bundle", "llama_bundle_sha256"),
    ("llama-source.json", "llama_provenance_sha256"),
];

pub(crate) struct PackageVerification {
    pub(crate) expected_identity_sha256: Digest,
    pub(crate) current_run_id: String,
    pub(crate) current_run_attempt: String,
    pub(crate) controller_revision: Option<String>,
    pub(crate) selected_source: String,
}

pub(crate) struct VerifiedPackage {
    inputs: super::VerifiedPackageInputs,
    current: WorkflowRun,
    candidate_bundle: bool,
}

impl VerifiedPackage {
    pub(crate) fn publication_context(&self) -> (&ProducerIdentity, usize, bool) {
        (
            &self.inputs.identity,
            self.inputs.plan.models.len(),
            self.candidate_bundle,
        )
    }

    pub(super) fn into_parts(self) -> (super::VerifiedPackageInputs, WorkflowRun) {
        (self.inputs, self.current)
    }
}

pub(crate) fn verify_package(
    package: &Path,
    verification: PackageVerification,
) -> Result<VerifiedPackage, Error> {
    let identity_path = package.join("identity.json");
    let identity_bytes = std::fs::read(&identity_path)?;
    let identity = identity_from_captured(&identity_bytes, &verification.expected_identity_sha256)?;
    let identity_sha256 = verification.expected_identity_sha256.clone();
    let plan_bytes = std::fs::read(package.join("plan.json"))?;
    let plan = plan_from_captured(&plan_bytes, &identity)?;

    if !identity
        .get("schema")
        .is_some_and(|schema| schema.as_u64() == Some(3))
        || identity.get("platform").and_then(Value::as_str) != Some("macos-arm64-metal")
    {
        return Err(package_identity_error("unknown build identity"));
    }

    let candidate = source_identity(&identity, "candidate")?;
    let base = source_identity(&identity, "base")?;
    if verification
        .controller_revision
        .as_deref()
        .is_some_and(|revision| !revision.is_empty())
        && identity.get("controller").and_then(Value::as_str)
            != verification.controller_revision.as_deref()
    {
        return Err(package_identity_error("controller revision mismatch"));
    }

    if !selected_source_matches(&identity, &verification.selected_source) {
        return Err(package_identity_error("selected source identity mismatch"));
    }
    if !verification.selected_source.is_empty()
        && (candidate != verification.selected_source
            || base != verification.selected_source
            || optional_digest(&identity, "bundle_sha256")?.is_some()
            || string_field(&identity, "pass_id")? != "repair-1")
    {
        return Err(package_identity_error(
            "certify-only package changed selected source",
        ));
    }

    if identity.get("run_id").and_then(Value::as_str) != Some(verification.current_run_id.as_str())
    {
        return Err(package_identity_error("foreign workflow run or attempt"));
    }
    let producer_attempt: RunAttempt = identity
        .get("run_attempt")
        .and_then(Value::as_str)
        .ok_or_else(|| package_identity_error("invalid workflow run attempt"))?
        .to_owned()
        .try_into()
        .map_err(|_| package_identity_error("invalid workflow run attempt"))?;
    let current_run_attempt: RunAttempt = verification
        .current_run_attempt
        .as_str()
        .to_owned()
        .try_into()
        .map_err(|_| package_identity_error("invalid workflow run attempt"))?;
    if producer_attempt > current_run_attempt {
        return Err(package_identity_error("foreign workflow run or attempt"));
    }

    for (name, key) in ARTIFACTS {
        verify_artifact(package, &identity, name, key)?;
    }
    if optional_digest(&identity, "summary_sha256")?.is_some() {
        verify_artifact(package, &identity, "upstream-summary.md", "summary_sha256")?;
    }
    if optional_digest(&identity, "bundle_sha256")?.is_some() {
        verify_artifact(package, &identity, "candidate.bundle", "bundle_sha256")?;
    }

    let producer_identity: ProducerIdentity = serde_json::from_slice(&identity_bytes)?;
    Ok(VerifiedPackage {
        candidate_bundle: optional_digest(&identity, "bundle_sha256")?.is_some(),
        inputs: super::VerifiedPackageInputs {
            identity: producer_identity,
            identity_sha256,
            plan,
        },
        current: WorkflowRun {
            run_id: verification.current_run_id,
            run_attempt: current_run_attempt,
        },
    })
}

// Verify the same captured bytes that all parsing and retained projections consume.
// A later path hash cannot establish custody of an earlier captured buffer.
fn identity_from_captured(bytes: &[u8], expected: &Digest) -> Result<Value, Error> {
    if &Digest::of_bytes(bytes) != expected {
        return Err(package_identity_error("build identity digest mismatch"));
    }
    Ok(serde_json::from_slice(bytes)?)
}

fn plan_from_captured(bytes: &[u8], identity: &Value) -> Result<SourceFamilyPlan, Error> {
    let expected = Digest::try_from(string_field(identity, "plan_sha256")?.to_owned())
        .map_err(package_identity_error)?;
    if Digest::of_bytes(bytes) != expected {
        return Err(Error::new(
            ErrorKind::PackageArtifact,
            "plan.json digest mismatch",
        ));
    }
    SourceFamilyPlan::parse(bytes)
}

fn source_identity<'a>(identity: &'a Value, key: &str) -> Result<&'a str, Error> {
    let source = string_field(identity, key)?;
    if source.len() != 40
        || !source
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(package_identity_error("invalid source identity"));
    }
    Ok(source)
}

fn selected_source_matches(identity: &Value, selected_source: &str) -> bool {
    match identity.get("mesh_source") {
        Some(Value::String(source)) => source == selected_source,
        None => selected_source.is_empty(),
        Some(_) => false,
    }
}

fn verify_artifact(package: &Path, identity: &Value, name: &str, key: &str) -> Result<(), Error> {
    let actual = Digest::of_file(&package.join(name))?;
    let expected = required_field(identity, key)?;
    if expected.as_str() != Some(actual.as_str()) {
        return Err(Error::new(
            ErrorKind::PackageArtifact,
            format!("{name} digest mismatch"),
        ));
    }
    Ok(())
}

fn required_field<'a>(identity: &'a Value, key: &str) -> Result<&'a Value, Error> {
    identity
        .get(key)
        .ok_or_else(|| package_identity_error(format!("missing identity field: {key}")))
}

fn string_field<'a>(identity: &'a Value, key: &str) -> Result<&'a str, Error> {
    required_field(identity, key)?
        .as_str()
        .ok_or_else(|| package_identity_error(format!("invalid identity field: {key}")))
}

fn package_identity_error(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::PackageIdentity, message)
}

fn optional_digest<'a>(identity: &'a Value, key: &str) -> Result<Option<&'a str>, Error> {
    match identity.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) if text.is_empty() => Ok(None),
        Some(Value::String(text)) => Ok(Some(text)),
        Some(_) => Err(package_identity_error(format!(
            "invalid identity field: {key}"
        ))),
    }
}

#[cfg(test)]
#[path = "package_tests.rs"]
mod tests;
