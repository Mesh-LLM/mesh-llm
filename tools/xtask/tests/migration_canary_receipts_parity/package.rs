use crate::canary_receipts::{
    Digest, Family, PackageVerification, ReceiptContext, aggregate, verify_package,
};
use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
};

const PLAN: &[u8] = include_bytes!("../migration_canary_receipts/fixtures/plan.json");
const DENSE: &[u8] = include_bytes!("../migration_canary_receipts/fixtures/dense.jsonl");
const HYBRID: &[u8] = include_bytes!("../migration_canary_receipts/fixtures/hybrid.jsonl");

pub(crate) struct CaseFixture {
    _root: super::fixture_files::TestRoot,
    package: PathBuf,
    evidence: PathBuf,
    identity_sha256: Digest,
    context: ReceiptContext,
}

pub(crate) struct RustAggregate {
    pub(crate) green: bool,
    pub(crate) passed_count: usize,
    pub(crate) dense_attempt: Option<String>,
    pub(crate) summary: String,
    pub(crate) outputs: Option<String>,
}

impl CaseFixture {
    pub(super) fn package(&self) -> &Path {
        &self.package
    }

    pub(crate) fn reverify_package(&self) -> Result<(), crate::canary_receipts::Error> {
        self.verify_package_with_identity(self.identity_sha256.clone())
    }

    pub(crate) fn verify_package_with_identity(
        &self,
        expected_identity_sha256: Digest,
    ) -> Result<(), crate::canary_receipts::Error> {
        verify_package(
            &self.package,
            self.package_verification(expected_identity_sha256),
        )
        .map(|_| ())
    }

    fn package_verification(&self, expected_identity_sha256: Digest) -> PackageVerification {
        PackageVerification {
            expected_identity_sha256,
            current_run_id: "123".to_owned(),
            current_run_attempt: "4".to_owned(),
            controller_revision: Some("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_owned()),
            selected_source: String::new(),
        }
    }

    pub(crate) fn complete() -> Result<Self, Box<dyn Error>> {
        Self::new(false)
    }

    pub(crate) fn newest_failure() -> Result<Self, Box<dyn Error>> {
        Self::new(true)
    }

    fn new(newest_failure: bool) -> Result<Self, Box<dyn Error>> {
        let root = super::fixture_files::TestRoot::new()?;
        let package = root.0.join("package");
        let evidence = root.0.join("evidence");
        fs::create_dir(&package)?;
        fs::create_dir(&evidence)?;
        fs::write(package.join("plan.json"), PLAN)?;

        super::fixture_files::write_artifacts(&package)?;
        let identity_bytes = super::fixture_files::source_identity(&package)?;
        let identity_digest = Digest::of_bytes(&identity_bytes);
        let identity_sha256 = identity_digest.clone();
        fs::write(package.join("identity.json"), &identity_bytes)?;

        let verified = verify_package(
            &package,
            PackageVerification {
                expected_identity_sha256: identity_digest,
                current_run_id: "123".to_owned(),
                current_run_attempt: "4".to_owned(),
                controller_revision: Some("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_owned()),
                selected_source: String::new(),
            },
        )?;
        let context = ReceiptContext::from_verified_package(verified);

        super::fixture_files::write_worker(
            &evidence,
            "dense",
            "2",
            "success",
            DENSE,
            identity_sha256.as_str(),
        )?;
        super::fixture_files::write_worker(
            &evidence,
            "hybrid",
            "2",
            "success",
            HYBRID,
            identity_sha256.as_str(),
        )?;
        if newest_failure {
            super::fixture_files::write_worker(
                &evidence,
                "new-dense",
                "3",
                "failure",
                DENSE,
                identity_sha256.as_str(),
            )?;
        }

        Ok(Self {
            _root: root,
            package,
            evidence,
            identity_sha256,
            context,
        })
    }

    pub(crate) fn input_hashes(&self) -> Result<Vec<(PathBuf, String)>, Box<dyn Error>> {
        super::fixture_files::hash_paths(super::fixture_files::case_paths(
            &self.package,
            &self.evidence,
        )?)
    }

    pub(crate) fn run_rust(&self) -> Result<RustAggregate, Box<dyn Error>> {
        let report = aggregate(&self.context, &self.evidence)?;
        let dense = serde_json::from_str::<Family>(r#""dense""#)?;
        Ok(RustAggregate {
            green: report.is_green(),
            passed_count: report.passed.len(),
            dense_attempt: report
                .selected_attempts
                .get(&dense)
                .map(|attempt| attempt.as_str().to_owned()),
            summary: report.summary(),
            outputs: report.github_outputs(),
        })
    }
}
