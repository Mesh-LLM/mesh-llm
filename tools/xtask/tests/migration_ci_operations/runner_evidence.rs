//! `ci-ops runner-identity bind` and evidence validation parity, using the
//! synthetic producer cohort from the legacy evidence tests: a proposal is
//! written only for a matching reviewed anchor, and untrusted producers,
//! substituted bytes and missing evidence fail closed.

use super::support::{Stage, TestResult, assert_case, check_case, fixture_dir};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs;

const COHORT: &str = "synthetic_cohort.json";

fn hash_bytes(raw: &[u8]) -> String {
    format!("sha256:{}", hex::encode(Sha256::digest(raw)))
}

/// A stage holding an unqualified catalog whose `public-cpu` image points
/// at the synthetic cohort's index, the cohort, and a reviewed anchor.
struct EvidenceStage {
    stage: Stage,
    raw: Vec<u8>,
}

impl EvidenceStage {
    fn new(label: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let stage = Stage::checkout(label)?;
        let raw = fs::read(fixture_dir().join(COHORT))?;
        let cohort: Value = serde_json::from_slice(&raw)?;
        stage.edit_catalog(|catalog| {
            for image in catalog["images"]
                .as_object_mut()
                .into_iter()
                .flat_map(|images| images.values_mut())
            {
                image["receipt"] = Value::Null;
                image["provenance"] = Value::Null;
            }
            let digest = cohort["candidates"]["candidate-index-public-cpu"]["digest"]
                .as_str()
                .unwrap_or_default();
            catalog["images"]["public-cpu"]["reference"] =
                format!("ghcr.io/mesh-llm/mesh-llm-cuda-runner@{digest}").into();
            catalog["images"]["public-cpu"]["native_toolchain_epoch"] =
                format!("mesh-llm-cuda-runner-sha256-{}", "8".repeat(64)).into();
            catalog["compiler_seed"]["key_prefix"] =
                "mesh-llm-sccache-seed-linux-x86_64-img-88888888-epoch-88888888-v2-".into();
        })?;
        stage.write("input.json", &raw)?;
        let evidence = Self { stage, raw };
        evidence.write_anchor(&evidence.anchor(&cohort["origin"]))?;
        Ok(evidence)
    }

    fn anchor(&self, origin: &Value) -> Value {
        let sha = hash_bytes(&self.raw);
        json!({
            "receipt": {"schema": 1, "cohort_sha256": sha, "index_candidate_key": "candidate-index-public-cpu"},
            "provenance": {
                "schema": 1,
                "scope": "reviewed_producer_admission",
                "validation": "offline_binding_only",
                "cohort_sha256": sha,
                "origin": origin,
                "admission_validator_revision": "f".repeat(40),
            },
        })
    }

    fn cohort(&self) -> Result<Value, Box<dyn std::error::Error>> {
        Ok(serde_json::from_slice(&self.raw)?)
    }

    fn write_anchor(&self, anchor: &Value) -> TestResult {
        self.stage
            .write("anchor.json", serde_json::to_string(anchor)?.as_bytes())
    }

    fn bind_args(&self, output: &str) -> Vec<String> {
        let root = self.stage.root_arg();
        [
            "--root",
            &root,
            "bind",
            "--image-id",
            "public-cpu",
            "--cohort",
            "input.json",
            "--anchor",
            "anchor.json",
            "--output",
            output,
        ]
        .iter()
        .map(|arg| (*arg).to_owned())
        .collect()
    }

    fn bind(&self, name: &str) -> Result<super::support::Outcome, Box<dyn std::error::Error>> {
        let proposal = self.stage.path().join("proposal");
        let reset = || -> TestResult {
            if proposal.exists() {
                fs::remove_dir_all(&proposal)?;
            }
            Ok(())
        };
        assert_case(name, &self.stage, &self.bind_args("proposal"), &reset)
    }
}

#[test]
fn bind_writes_a_fresh_proposal_that_validates() -> TestResult {
    let evidence = EvidenceStage::new("bind-ok")?;
    let before = evidence.stage.read("ci/runner-images.json")?;
    let outcome = evidence.bind("bind_proposal")?;
    assert_eq!(outcome.code, 0, "{}", outcome.stderr);
    assert_eq!(
        evidence.stage.read("ci/runner-images.json")?,
        before,
        "input catalog is never adopted"
    );
    let sha = hash_bytes(&evidence.raw);
    let stored = evidence.stage.path().join(format!(
        "proposal/ci/runner-image-evidence/{}.json",
        &sha[7..]
    ));
    assert_eq!(fs::read(stored)?, evidence.raw);
    let proposal = Stage::empty("bind-proposal-root")?;
    let written = evidence.stage.read("proposal/ci/runner-images.json")?;
    proposal.write("ci/runner-images.json", written.as_bytes())?;
    proposal.write(
        &format!("ci/runner-image-evidence/{}.json", &sha[7..]),
        &evidence.raw,
    )?;
    let outcome = check_case(
        "bind_proposal_validates",
        &proposal,
        &["lookup", "rust-clippy", "--field", "provenance"],
    )?;
    assert_eq!(outcome.code, 0, "{}", outcome.stderr);
    let root = evidence.stage.root_arg();
    let args: Vec<String> = [
        "--root",
        &root,
        "bind",
        "--image-id",
        "public-cpu",
        "--cohort",
        "input.json",
        "--anchor",
        "anchor.json",
        "--output",
        "proposal",
    ]
    .iter()
    .map(|arg| (*arg).to_owned())
    .collect();
    let outcome = assert_case("bind_output_exists", &evidence.stage, &args, &|| Ok(()))?;
    assert_eq!(outcome.code, 1);
    Ok(())
}

type AnchorEdit = fn(&mut Value);

#[test]
fn bind_rejects_denied_producers_and_substituted_evidence() -> TestResult {
    let cases: [(&str, AnchorEdit); 6] = [
        ("bind_denied_producer_event", |a| {
            a["provenance"]["origin"]["event"] = "pull_request".into()
        }),
        ("bind_denied_producer_repository", |a| {
            a["provenance"]["origin"]["repository"] = "fork/mesh-llm-runner-images".into()
        }),
        ("bind_invalid_timestamp", |a| {
            a["provenance"]["origin"]["timestamp"] = "20250229000000".into()
        }),
        ("bind_unsupported_trust_claim", |a| {
            a["provenance"]["validation"] = "authenticated".into()
        }),
        ("bind_substituted_bytes", |a| {
            let other = format!("sha256:{}", "0".repeat(64));
            a["receipt"]["cohort_sha256"] = other.clone().into();
            a["provenance"]["cohort_sha256"] = other.into();
        }),
        ("bind_candidate_alias", |a| {
            a["receipt"]["index_candidate_key"] = "candidate-index-public-cuda12".into()
        }),
    ];
    for (name, edit) in cases {
        let evidence = EvidenceStage::new(name)?;
        let mut anchor = evidence.anchor(&evidence.cohort()?["origin"]);
        edit(&mut anchor);
        evidence.write_anchor(&anchor)?;
        let outcome = evidence.bind(name)?;
        assert_eq!((outcome.code, outcome.stdout.as_str()), (1, ""), "{name}");
        assert!(
            !evidence.stage.path().join("proposal").exists(),
            "{name}: no proposal on failure"
        );
    }
    Ok(())
}

#[test]
fn bind_rejects_immutable_conflicts_and_missing_platforms() -> TestResult {
    let evidence = EvidenceStage::new("bind-conflict")?;
    let mut other = evidence.anchor(&evidence.cohort()?["origin"]);
    other["provenance"]["admission_validator_revision"] = "e".repeat(40).into();
    evidence.stage.edit_catalog(|catalog| {
        let image = &mut catalog["images"]["public-cpu"];
        image["receipt"] = other["receipt"].clone();
        image["provenance"] = other["provenance"].clone();
    })?;
    let sha = hash_bytes(&evidence.raw);
    evidence.stage.write(
        &format!("ci/runner-image-evidence/{}.json", &sha[7..]),
        &evidence.raw,
    )?;
    assert_eq!(evidence.bind("bind_immutable_conflict")?.code, 1);

    let evidence = EvidenceStage::new("bind-platform")?;
    evidence.stage.edit_catalog(|catalog| {
        catalog["runtime_rows"]["linux-cpu"]["architecture"] = "arm64".into()
    })?;
    assert_eq!(evidence.bind("bind_missing_runtime_platform")?.code, 1);
    Ok(())
}

#[test]
fn qualified_images_without_retained_evidence_fail_every_command() -> TestResult {
    let evidence = EvidenceStage::new("missing-evidence")?;
    let anchor = evidence.anchor(&evidence.cohort()?["origin"]);
    evidence.stage.edit_catalog(|catalog| {
        let image = &mut catalog["images"]["public-cpu"];
        image["receipt"] = anchor["receipt"].clone();
        image["provenance"] = anchor["provenance"].clone();
    })?;
    let hash = "a".repeat(64);
    for (name, args) in [
        ("missing_evidence_validate", vec!["validate"]),
        ("missing_evidence_lookup", vec!["lookup", "rust-clippy"]),
        (
            "missing_evidence_seed_key",
            vec!["seed-key", "--recipe-hash", &hash],
        ),
        ("missing_evidence_check", vec!["check"]),
    ] {
        let outcome = check_case(name, &evidence.stage, &args)?;
        assert_eq!((outcome.code, outcome.stdout.as_str()), (1, ""), "{name}");
    }
    let outcome = super::support::xtask_in(
        evidence.stage.path(),
        &[
            "ci-ops".into(),
            "runner-identity".into(),
            "--root".into(),
            evidence.stage.root_arg(),
            "diagnose".into(),
        ],
    )?;
    assert_ne!(outcome.code, 0);
    assert!(outcome.stdout.is_empty());
    Ok(())
}

#[test]
fn reanchored_cohort_still_rejects_structural_and_platform_corruption() -> TestResult {
    for pointer in [
        "/schema",
        "/extra",
        "/candidates/candidate-index-public-cpu/digest",
        "/candidates/candidate-index-public-cpu/backend/id",
        "/candidates/candidate-index-public-cpu/backend/name",
        "/platforms/public-cpu-amd64/receipt/platform/architecture",
        "/platforms/public-cpu-amd64/receipt/runtime/source/mesh_revision",
        "/platforms/public-cpu-amd64/receipt/oci/manifest/digest",
        "/platforms/public-cpu-amd64/receipt/runtime/cache",
        "/platforms/public-cpu-amd64/candidate/schema",
        "/platforms/public-cpu-amd64/candidate/extra",
        "/candidates/candidate-index-public-cpu/children",
    ] {
        let evidence = EvidenceStage::new("reanchored")?;
        let mut cohort = evidence.cohort()?;
        match pointer {
            "/extra" => cohort["extra"] = "unknown".into(),
            "/platforms/public-cpu-amd64/candidate/extra" => {
                cohort["platforms"]["public-cpu-amd64"]["candidate"]["extra"] = "unknown".into()
            }
            "/candidates/candidate-index-public-cpu/children" => {
                *cohort.pointer_mut(pointer).ok_or("pointer")? = json!([])
            }
            _ => *cohort.pointer_mut(pointer).ok_or("pointer")? = Value::Null,
        }
        let raw = serde_json::to_vec(&cohort)?;
        evidence.stage.write("input.json", &raw)?;
        let mut anchor = evidence.anchor(&cohort["origin"]);
        anchor["receipt"]["cohort_sha256"] = hash_bytes(&raw).into();
        anchor["provenance"]["cohort_sha256"] = hash_bytes(&raw).into();
        evidence.write_anchor(&anchor)?;
        let outcome = super::support::xtask_in(
            evidence.stage.path(),
            &[
                "ci-ops".into(),
                "runner-identity".into(),
                "--root".into(),
                evidence.stage.root_arg(),
                "bind".into(),
                "--image-id".into(),
                "public-cpu".into(),
                "--cohort".into(),
                "input.json".into(),
                "--anchor".into(),
                "anchor.json".into(),
                "--output".into(),
                "proposal".into(),
            ],
        )?;
        assert_ne!(outcome.code, 0, "{pointer}");
        assert!(!evidence.stage.path().join("proposal").exists());
    }
    Ok(())
}

#[test]
fn sequential_proposals_retain_shared_and_distinct_evidence() -> TestResult {
    for shared in [true, false] {
        let evidence = EvidenceStage::new("sequential")?;
        let invoke =
            |root: &std::path::Path, image: &str, cohort: &str, anchor: &str, output: &str| {
                super::support::xtask_in(
                    evidence.stage.path(),
                    &[
                        "ci-ops".into(),
                        "runner-identity".into(),
                        "--root".into(),
                        root.display().to_string(),
                        "bind".into(),
                        "--image-id".into(),
                        image.into(),
                        "--cohort".into(),
                        cohort.into(),
                        "--anchor".into(),
                        anchor.into(),
                        "--output".into(),
                        output.into(),
                    ],
                )
            };
        assert_eq!(
            invoke(
                evidence.stage.path(),
                "public-cpu",
                "input.json",
                "anchor.json",
                "first"
            )?
            .code,
            0
        );
        let first = evidence.stage.path().join("first");
        let mut catalog: Value =
            serde_json::from_slice(&fs::read(first.join("ci/runner-images.json"))?)?;
        catalog["images"]["public-vulkan"]["reference"] = format!(
            "ghcr.io/mesh-llm/mesh-llm-cuda-runner@sha256:{}",
            "7".repeat(64)
        )
        .into();
        catalog["images"]["public-vulkan"]["native_toolchain_epoch"] =
            format!("mesh-llm-cuda-runner-sha256-{}", "7".repeat(64)).into();
        fs::write(
            first.join("ci/runner-images.json"),
            serde_json::to_vec(&catalog)?,
        )?;
        let mut raw = evidence.raw.clone();
        if !shared {
            raw.push(b' ');
        }
        let mut anchor = evidence.anchor(&evidence.cohort()?["origin"]);
        anchor["receipt"]["index_candidate_key"] = "candidate-index-public-vulkan".into();
        anchor["receipt"]["cohort_sha256"] = hash_bytes(&raw).into();
        anchor["provenance"]["cohort_sha256"] = hash_bytes(&raw).into();
        evidence.stage.write("second-input.json", &raw)?;
        evidence
            .stage
            .write("second-anchor.json", &serde_json::to_vec(&anchor)?)?;
        assert_eq!(
            invoke(
                &first,
                "public-vulkan",
                "second-input.json",
                "second-anchor.json",
                "second"
            )?
            .code,
            0
        );
        let second = evidence.stage.path().join("second");
        assert_eq!(
            fs::read_dir(second.join("ci/runner-image-evidence"))?.count(),
            if shared { 1 } else { 2 }
        );
        let result = super::support::xtask_in(
            evidence.stage.path(),
            &[
                "ci-ops".into(),
                "runner-identity".into(),
                "--root".into(),
                second.display().to_string(),
                "validate".into(),
            ],
        )?;
        assert_eq!(result.code, 0, "{}", result.stderr);
    }
    Ok(())
}
