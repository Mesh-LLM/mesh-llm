#[path = "source_plan/admission.rs"]
mod admission;
#[path = "source_plan/battery.rs"]
mod battery;
#[cfg(test)]
#[path = "source_plan/cache_tests.rs"]
mod cache_tests;
#[path = "source_plan/contract.rs"]
mod contract;
#[path = "source_plan/gguf.rs"]
mod gguf;
#[cfg(test)]
#[path = "source_plan/gguf_tests.rs"]
mod gguf_tests;
#[path = "source_plan/placement.rs"]
pub(crate) mod placement;
#[cfg(test)]
#[path = "source_plan/placement_tests.rs"]
mod placement_tests;
#[path = "source_plan/preflight_receipt.rs"]
mod preflight_receipt;

use super::canary_receipts::{Digest, SourceFamilyPlan};
use crate::command::DynResult;
use crate::repository::{check_args::Grammar, check_report::CheckReport};
use contract::{Input, PlanIdentity};
use std::{fs, io::Write, path::Path};

pub(crate) fn run(args: &[String], verify: bool) -> DynResult<()> {
    command(args, verify, false)
}

pub(crate) fn preflight(args: &[String]) -> DynResult<()> {
    command(args, false, true)
}

fn command(args: &[String], verify: bool, production: bool) -> DynResult<()> {
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation canary-receipts {source-plan|verify-source-plan|preflight} --input PATH",
        values: &["--input"],
        flags: &["--help"],
    };
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage)).emit();
    }
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error("unexpected positional arguments").emit();
    }
    let input: Input =
        serde_json::from_slice(&fs::read(parsed.last("--input").ok_or("missing --input")?)?)?;
    match verify {
        true => verify_existing(&input),
        false => generate(&input, production),
    }
}

fn generate(input: &Input, production: bool) -> DynResult<()> {
    generate_with_cancellation(input, production, &crate::process::Cancellation::default())
}

fn generate_with_cancellation(
    input: &Input,
    production: bool,
    cancellation: &crate::process::Cancellation,
) -> DynResult<()> {
    if production && !matches!(&input.cache, contract::CachePolicy::GgufMetadata { .. }) {
        return Err("production canary preflight requires gguf_metadata cache admission".into());
    }
    let resolved = input.resolve()?;
    battery::verify_revisions_with_cancellation(input, &resolved, cancellation)?;
    let manifest_sha256 = Digest::of_file(&resolved.manifest)?;
    let bytes = crate::ci_plan::family::controller_plan(&resolved.source, &resolved.manifest)?;
    SourceFamilyPlan::parse(&bytes)?;
    let cache_admission = admission::verify(&bytes, &input.cache)?;
    let scheduling = if production {
        Some(placement::project(&bytes)?)
    } else {
        None
    };
    fs::create_dir(&resolved.output)?;
    let plan_path = resolved.output.join("plan.json");
    create_file(&plan_path, &bytes)?;
    crate::ci_plan::family::verify_controller_plan(
        &resolved.source,
        &resolved.manifest,
        &plan_path,
    )?;
    battery::preflight_with_cancellation(&resolved, &plan_path, cancellation)?;
    battery::verify_revisions_with_cancellation(input, &resolved, cancellation)?;
    if Digest::of_file(&resolved.manifest)? != manifest_sha256 {
        return Err("selected manifest changed during controller preflight".into());
    }
    let identity = PlanIdentity {
        schema: 1,
        controller_revision: input.controller_revision.clone(),
        selected_revision: input.selected_revision.clone(),
        manifest_sha256,
        plan_sha256: Digest::of_bytes(&bytes),
        cache_admission,
        gguf_admission: cache_admission.gguf_admission(),
    };
    if Digest::of_file(&plan_path)? != identity.plan_sha256 {
        return Err("selected battery changed the immutable controller plan".into());
    }
    create_file(
        &resolved.output.join("source-plan.json"),
        &serde_json::to_vec_pretty(&identity)?,
    )?;
    if let Some(matrix) = scheduling {
        preflight_receipt::write(&resolved.output, &identity, &matrix)?;
    }
    Ok(())
}

fn verify_existing(input: &Input) -> DynResult<()> {
    let resolved = input.resolve()?;
    battery::verify_revisions(input, &resolved)?;
    let identity: PlanIdentity =
        serde_json::from_slice(&fs::read(resolved.output.join("source-plan.json"))?)?;
    let plan_path = resolved.output.join("plan.json");
    let bytes = fs::read(&plan_path)?;
    if identity.schema != 1
        || identity.controller_revision != input.controller_revision
        || identity.selected_revision != input.selected_revision
        || identity.manifest_sha256 != Digest::of_file(&resolved.manifest)?
        || identity.plan_sha256 != Digest::of_bytes(&bytes)
    {
        return Err("controller source-plan identity mismatch".into());
    }
    SourceFamilyPlan::parse(&bytes)?;
    let canonical = crate::ci_plan::family::controller_plan(&resolved.source, &resolved.manifest)?;
    if canonical != bytes {
        return Err("controller plan differs from the complete selected roster".into());
    }
    crate::ci_plan::family::verify_controller_plan(
        &resolved.source,
        &resolved.manifest,
        &plan_path,
    )?;
    let admitted = admission::verify(&bytes, &input.cache)?;
    if admitted != identity.cache_admission {
        return Err("cache admission policy differs from source-plan identity".into());
    }
    if identity.gguf_admission != admitted.gguf_admission() {
        return Err("GGUF admission state differs from cache admission policy".into());
    }
    Ok(())
}

fn create_file(path: &Path, bytes: &[u8]) -> DynResult<()> {
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?
        .write_all(bytes)?;
    Ok(())
}

pub(crate) fn generate_document(bytes: &[u8]) -> DynResult<()> {
    let input: Input = serde_json::from_slice(bytes)?;
    generate(&input, false)
}

pub(crate) fn admit_document(
    bytes: &[u8],
    cancellation: &crate::process::Cancellation,
) -> DynResult<()> {
    let input: Input = serde_json::from_slice(bytes)?;
    generate_with_cancellation(&input, true, cancellation)
}
