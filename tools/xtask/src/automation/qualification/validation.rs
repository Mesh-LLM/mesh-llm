use super::schema::{Backend, FailureKind, Platform, ProbeKind, Receipt, Scenario};
use super::{Error, hex_identity, verify_artifact, verify_execution};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Deserialize)]
struct Contracts {
    qualification: Matrix,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Matrix {
    schema_version: u32,
    required_roots: Vec<String>,
    required_backends: BTreeMap<Platform, Vec<Backend>>,
}

pub(crate) fn validate(receipt: &Receipt, source_sha: &str) -> Result<(), Error> {
    if receipt.schema_version != 1
        || !hex_identity(&receipt.source_sha, 40)
        || receipt.source_sha != source_sha
    {
        return Err(Error::Invalid(
            "schema version or candidate source SHA mismatch",
        ));
    }
    verify_artifact(&receipt.source_snapshot)?;
    verify_artifact(&receipt.contracts)?;
    let contracts: Contracts =
        serde_json::from_reader(std::fs::File::open(&receipt.contracts.path)?)?;
    let roots: BTreeSet<_> = contracts.qualification.required_roots.iter().collect();
    if contracts.qualification.schema_version != 1
        || roots.is_empty()
        || roots.len() != contracts.qualification.required_roots.len()
        || roots != receipt.roots.keys().collect()
    {
        return Err(Error::Invalid("frozen required-root matrix mismatch"));
    }
    for execution in receipt.roots.values() {
        verify_execution(execution)?;
    }
    verify_interpreters(receipt)?;
    let required_backends = contracts
        .qualification
        .required_backends
        .get(&receipt.platform)
        .ok_or(Error::Invalid("platform absent from frozen product matrix"))?;
    verify_products(receipt, required_backends)?;
    verify_protocol(receipt)?;
    let expected = match receipt.platform {
        Platform::Windows => vec![
            Scenario::ProductReadiness,
            Scenario::CorruptRuntime,
            Scenario::ReadinessTimeout,
        ],
        Platform::Linux | Platform::Macos => vec![
            Scenario::ProductReadiness,
            Scenario::ProtocolPair,
            Scenario::CorruptRuntime,
            Scenario::ReadinessTimeout,
        ],
    };
    let scenarios: BTreeSet<_> = receipt.scenarios.iter().map(|row| row.scenario).collect();
    if scenarios.len() != receipt.scenarios.len() || scenarios != expected.into_iter().collect() {
        return Err(Error::Invalid("missing, duplicate or unsupported scenario"));
    }
    for row in &receipt.scenarios {
        match row.scenario {
            Scenario::ProductReadiness | Scenario::ProtocolPair => {
                if row.expected_failure_observed || row.failure_kind.is_some() {
                    return Err(Error::Invalid("positive scenario claims expected failure"));
                }
                verify_execution(&row.execution)?;
                let required_cases = match row.scenario {
                    Scenario::ProductReadiness => receipt.products.len(),
                    Scenario::ProtocolPair => receipt.protocol_cases.len(),
                    Scenario::CorruptRuntime | Scenario::ReadinessTimeout => 0,
                };
                if row.execution.case_count
                    < u64::try_from(required_cases)
                        .map_err(|_| Error::Invalid("case count overflow"))?
                {
                    return Err(Error::Invalid("scenario does not cover every required row"));
                }
            }
            Scenario::CorruptRuntime | Scenario::ReadinessTimeout => {
                let expected_kind = negative_kind(row.scenario)?;
                if !row.expected_failure_observed
                    || row.failure_kind != Some(expected_kind)
                    || row.execution.exit_code == 0
                    || row.execution.argv.is_empty()
                    || row.execution.case_count == 0
                    || !row.execution.cleanup_complete
                {
                    return Err(Error::Invalid(
                        "negative scenario lacks exact rejection and cleanup",
                    ));
                }
                verify_artifact(&row.execution.evidence)?;
            }
        }
    }
    Ok(())
}

fn negative_kind(scenario: Scenario) -> Result<FailureKind, Error> {
    match scenario {
        Scenario::CorruptRuntime => Ok(FailureKind::DigestMismatch),
        Scenario::ReadinessTimeout => Ok(FailureKind::ReadinessTimeout),
        Scenario::ProductReadiness | Scenario::ProtocolPair => {
            Err(Error::Invalid("positive scenario has no rejection kind"))
        }
    }
}

fn verify_interpreters(receipt: &Receipt) -> Result<(), Error> {
    let proof = &receipt.interpreters;
    let kinds: BTreeSet<_> = proof.probes.iter().map(|probe| probe.kind).collect();
    let expected = [
        ProbeKind::Path,
        ProbeKind::Absolute,
        ProbeKind::Shebang,
        ProbeKind::Versioned,
    ];
    if proof.path.is_empty()
        || proof.attempts != 0
        || proof.probes.len() != 4
        || kinds != expected.into_iter().collect()
    {
        return Err(Error::Invalid("incomplete interpreter-unavailable proof"));
    }
    for probe in &proof.probes {
        if probe.candidates.is_empty()
            || probe.candidates.iter().any(String::is_empty)
            || !probe.found.is_empty()
        {
            return Err(Error::Invalid(
                "interpreter found or probe has no candidates",
            ));
        }
        verify_artifact(&probe.evidence)?;
    }
    Ok(())
}

fn verify_products(receipt: &Receipt, required: &[Backend]) -> Result<(), Error> {
    let backends: BTreeSet<_> = receipt.products.iter().map(|row| row.backend).collect();
    let expected: BTreeSet<_> = required.iter().copied().collect();
    if expected.is_empty()
        || expected.len() != required.len()
        || backends.len() != receipt.products.len()
        || backends != expected
    {
        return Err(Error::Invalid(
            "required hardware/backend product row missing or duplicated",
        ));
    }
    match receipt.platform {
        Platform::Linux
            if !backends.contains(&Backend::Cpu) || !backends.contains(&Backend::Cuda) =>
        {
            return Err(Error::Invalid("Linux CPU/CUDA qualification required"));
        }
        Platform::Macos if !backends.contains(&Backend::Metal) => {
            return Err(Error::Invalid("macOS Metal qualification required"));
        }
        Platform::Linux | Platform::Macos | Platform::Windows => {}
    }
    for product in &receipt.products {
        match (receipt.platform, product.backend) {
            (Platform::Macos, Backend::Metal)
            | (
                Platform::Linux | Platform::Windows,
                Backend::Cpu | Backend::Cuda | Backend::Rocm | Backend::Vulkan,
            ) => {}
            _ => return Err(Error::Invalid("backend is unsupported on receipt platform")),
        }
        for artifact in [
            &product.hardware_evidence,
            &product.host_manifest,
            &product.runtime_manifest,
            &product.product_manifest,
        ] {
            verify_artifact(artifact)?;
        }
        if product.files.is_empty() {
            return Err(Error::Invalid(
                "product has no immutable executable/runtime inputs",
            ));
        }
        for artifact in &product.files {
            verify_artifact(artifact)?;
        }
    }
    Ok(())
}

fn verify_protocol(receipt: &Receipt) -> Result<(), Error> {
    let expected_backends = match receipt.platform {
        Platform::Windows => {
            if !receipt.protocol_cases.is_empty() || !receipt.models.is_empty() {
                return Err(Error::Invalid(
                    "Windows has no invented inference qualification",
                ));
            }
            return Ok(());
        }
        Platform::Linux => vec![Backend::Cpu, Backend::Cuda],
        Platform::Macos => vec![Backend::Metal],
    };
    let model_ids = ["smollm2-q8-inference", "family-granite-hybrid"];
    let actual: BTreeSet<_> = receipt
        .models
        .iter()
        .map(|model| model.artifact_id.as_str())
        .collect();
    if receipt.models.len() != 2 || actual != model_ids.into_iter().collect() {
        return Err(Error::Invalid(
            "required pinned dense/recurrent models missing",
        ));
    }
    for model in &receipt.models {
        if !hex_identity(&model.revision, 40) || model.files.is_empty() {
            return Err(Error::Invalid(
                "model revision or immutable model files missing",
            ));
        }
        for artifact in &model.files {
            verify_artifact(artifact)?;
        }
    }
    let required: BTreeSet<_> = expected_backends
        .iter()
        .flat_map(|backend| model_ids.iter().map(move |model| (*backend, *model)))
        .collect();
    let cases: BTreeSet<_> = receipt
        .protocol_cases
        .iter()
        .map(|case| (case.backend, case.model_id.as_str()))
        .collect();
    if cases.len() != receipt.protocol_cases.len() || cases != required {
        return Err(Error::Invalid(
            "required protocol pair/backend case missing or duplicated",
        ));
    }
    for case in &receipt.protocol_cases {
        verify_execution(&case.execution)?;
    }
    Ok(())
}
