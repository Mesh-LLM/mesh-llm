//! Export the exact independently certified selected commit without executing its code.
use super::{process, source};
use crate::{automation::canary_receipts::ProducerIdentity, command::DynResult};
use std::path::Path;

pub(crate) fn prepare(
    root: &Path,
    package: &Path,
    controller: &str,
    identity: &ProducerIdentity,
) -> DynResult<()> {
    process::operation(|| {
        source::revision(controller)?;
        source::revision(&identity.candidate)?;
        source::branch(&identity.branch)?;
        if !root.is_absolute() || identity.pass_id != "verify-1" {
            return Err(
                "selected publication requires an absolute source and independent verifier".into(),
            );
        }
        if process::text(root, &["rev-parse", "HEAD"])? != controller {
            return Err("publication checkout differs from frozen controller".into());
        }
        let candidate = &identity.candidate;
        let base = process::text(root, &["merge-base", controller, candidate])?;
        source::revision(&base)?;
        if process::text(root, &["rev-parse", &format!("{base}^{{tree}}")])?
            == process::text(root, &["rev-parse", &format!("{candidate}^{{tree}}")])?
        {
            return Err("selected source has no changes to publish against main".into());
        }
        let reference = format!("refs/heads/{}", identity.branch);
        process::text(
            root,
            &["update-ref", &reference, candidate, &"0".repeat(40)],
        )?;
        let bundle = package.join("candidate.bundle");
        if bundle.exists() {
            return Err("selected publication bundle already exists".into());
        }
        process::git(
            root,
            &[
                "bundle".into(),
                "create".into(),
                bundle.into(),
                reference.into(),
                format!("^{base}").into(),
            ],
            None,
        )?;
        Ok(())
    })
}
