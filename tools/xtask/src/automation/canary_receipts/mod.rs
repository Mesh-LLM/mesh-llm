//! Receipt checks over a hash-verified producer package and its source-owned plan.
//! Package hashes do not establish cryptographic authenticity or authorize publication.

mod aggregate;
mod boundary;
mod error;
mod identity;
mod package;
mod plan;
mod receipt;
mod results;
mod storage;

pub(crate) use aggregate::aggregate;
pub(crate) use error::{Error, ErrorKind};
pub(crate) use identity::{Digest, Family, ProducerIdentity, RunAttempt, WorkflowRun};
pub(crate) use package::{PackageVerification, VerifiedPackage, verify_package};
pub(crate) use plan::{FamilyModel, SourceFamilyPlan};
pub(crate) use receipt::{WorkerOutcome, WorkerReceipt, WorkerResult, write_receipt};
pub(crate) use results::validate_results;

struct VerifiedPackageInputs {
    identity: ProducerIdentity,
    identity_sha256: Digest,
    plan: SourceFamilyPlan,
}

#[cfg(test)]
pub(crate) struct TestPackageInputs {
    pub(crate) identity: ProducerIdentity,
    pub(crate) identity_sha256: Digest,
    pub(crate) plan: SourceFamilyPlan,
}

pub(crate) struct ReceiptContext {
    package: VerifiedPackageInputs,
    current: WorkflowRun,
}

impl ReceiptContext {
    pub(crate) fn from_verified_package(package: VerifiedPackage) -> Self {
        let (package, current) = package.into_parts();
        Self { package, current }
    }

    #[cfg(test)]
    pub(crate) fn from_test_package(
        package: TestPackageInputs,
        current: WorkflowRun,
    ) -> Result<Self, Error> {
        if package.identity.run_id != current.run_id
            || package.identity.run_attempt > current.run_attempt
        {
            return Err(Error::new(
                ErrorKind::ReceiptIdentity,
                "foreign workflow run or attempt",
            ));
        }
        Ok(Self {
            package: VerifiedPackageInputs {
                identity: package.identity,
                identity_sha256: package.identity_sha256,
                plan: package.plan,
            },
            current,
        })
    }

    pub(crate) fn model(&self, family: &Family) -> Result<&FamilyModel, Error> {
        self.package.plan.model(family)
    }
}
