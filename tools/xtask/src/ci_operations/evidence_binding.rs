//! Offline binding of a catalog image to maintainer-reviewed producer
//! admission (`validate_binding`, `provenance`, `origin` in
//! `scripts/runner-image-evidence.py`). Checks membership and identity only;
//! nothing here authenticates a producer.

use crate::ci_operations::evidence_input::{
    IMAGE, MAX_SAFE, decode, digest, fields, hash_bytes, require, revision, text,
};
use crate::ci_operations::evidence_timestamp::check_timestamp;
use crate::ci_operations::python_access::{
    Outcome, eq, is_int_one, is_str, item, string, type_name,
};
use crate::ci_plan::document::Json;
use crate::prepared_input::python_value::display;

pub(crate) fn origin(value: &Json) -> Outcome<()> {
    fields(
        value,
        "repository repository_id workflow_id workflow_path run_id run_attempt event runner_images_revision mesh_revision timestamp",
    )?;
    let at = |key: &str| value.get(key).unwrap_or(&Json::Null);
    require(
        is_str(at("repository"), "Mesh-LLM/mesh-llm-runner-images")
            && is_str(at("workflow_path"), ".github/workflows/build-and-push.yml"),
        "unexpected producer",
    )?;
    let event = at("event");
    require(
        ["push", "schedule", "workflow_dispatch"]
            .iter()
            .any(|name| is_str(event, name)),
        "untrusted producer event",
    )?;
    for key in ["repository_id", "workflow_id", "run_id", "run_attempt"] {
        let bounded = at(key)
            .as_int()
            .is_some_and(|int| 0 < int && int <= MAX_SAFE);
        require(bounded, "invalid origin integer")?;
    }
    for key in ["runner_images_revision", "mesh_revision"] {
        revision(at(key))?;
    }
    let stamp = at("timestamp")
        .as_str()
        .filter(|stamp| stamp.len() == 14 && stamp.bytes().all(|b| b.is_ascii_digit()));
    require(stamp.is_some(), "invalid timestamp")?;
    check_timestamp(stamp.unwrap_or_default())
}

pub(crate) fn provenance(value: &Json) -> Outcome<()> {
    fields(
        value,
        "schema scope validation cohort_sha256 origin admission_validator_revision",
    )?;
    let at = |key: &str| value.get(key).unwrap_or(&Json::Null);
    require(is_int_one(at("schema")), "invalid provenance schema")?;
    require(
        is_str(at("scope"), "reviewed_producer_admission")
            && is_str(at("validation"), "offline_binding_only"),
        "unsupported trust claim",
    )?;
    digest(at("cohort_sha256"))?;
    origin(at("origin"))?;
    revision(at("admission_validator_revision"))
}

/// The architectures a validated cohort's index candidate declares.
pub(crate) fn candidate_architectures(image: &Json, cohort: &Json) -> Outcome<Vec<String>> {
    let key = item(item(image, "receipt")?, "index_candidate_key")?;
    let candidate = crate::ci_operations::python_access::item_by(item(cohort, "candidates")?, key)?;
    let mut architectures: Vec<String> = Vec::new();
    for child in item(candidate, "children")?.as_array().unwrap_or_default() {
        let architecture = display(Some(item(child, "architecture")?));
        if !architectures.contains(&architecture) {
            architectures.push(architecture);
        }
    }
    Ok(architectures)
}

/// `validate_binding(image, raw)`: returns the decoded cohort.
pub(crate) fn validate_binding(image: &Json, raw: &[u8]) -> Outcome<Json> {
    let receipt = item(image, "receipt")?;
    let proof = item(image, "provenance")?;
    fields(receipt, "schema cohort_sha256 index_candidate_key")?;
    require(
        is_int_one(item(receipt, "schema")?),
        "invalid receipt schema",
    )?;
    provenance(proof)?;
    let anchored = item(receipt, "cohort_sha256")?;
    digest(anchored)?;
    let proof_sha = item(proof, "cohort_sha256")?;
    require(
        eq(anchored, proof_sha) && eq(proof_sha, &string(&hash_bytes(raw))),
        "cohort bytes differ from reviewed anchor",
    )?;
    let key = item(receipt, "index_candidate_key")?;
    text(key)?;
    let cohort = decode(raw)?;
    fields(
        &cohort,
        "schema type image origin catalog_sha256 candidates platforms",
    )?;
    let at = |key: &str| cohort.get(key).unwrap_or(&Json::Null);
    require(
        is_int_one(at("schema"))
            && is_str(at("type"), "mesh-llm-runner-staged-cohort")
            && is_str(at("image"), IMAGE),
        "invalid cohort",
    )?;
    let proof_origin = item(proof, "origin")?;
    require(
        eq(at("origin"), proof_origin),
        "origin differs from reviewed anchor",
    )?;
    digest(at("catalog_sha256"))?;
    let bounded_map = |value: &Json| {
        value
            .as_object()
            .is_some_and(|map| !map.is_empty() && map.len() <= 256)
    };
    require(
        bounded_map(at("candidates")) && bounded_map(at("platforms")),
        "invalid cohort maps",
    )?;
    let candidate = at("candidates")
        .get(key.as_str().unwrap_or_default())
        .unwrap_or(&Json::Null);
    fields(
        candidate,
        "schema type image environment backend mesh_revision runner_images_revision digest children",
    )?;
    let binding = CandidateBinding {
        image,
        proof_origin,
        candidate,
        cohort: &cohort,
    };
    binding.check_candidate(key)?;
    binding.check_children()?;
    Ok(cohort)
}

pub(crate) struct CandidateBinding<'a> {
    pub(crate) image: &'a Json,
    pub(crate) proof_origin: &'a Json,
    pub(crate) candidate: &'a Json,
    pub(crate) cohort: &'a Json,
}

pub(crate) fn field<'a>(value: &'a Json, key: &str) -> &'a Json {
    value.get(key).unwrap_or(&Json::Null)
}

impl CandidateBinding<'_> {
    pub(crate) fn family(&self) -> &Json {
        field(self.candidate, "backend")
    }

    fn expected_backend(&self) -> Outcome<String> {
        let backend = item(self.image, "backend")?;
        let backend = display(Some(backend));
        Ok(if backend.starts_with("cuda") {
            "cuda".to_owned()
        } else {
            backend
        })
    }

    fn check_candidate(&self, key: &Json) -> Outcome<()> {
        let (candidate, image) = (self.candidate, self.image);
        require(
            is_int_one(field(candidate, "schema"))
                && is_str(field(candidate, "type"), "mesh-llm-runner-image-candidate"),
            "invalid index candidate",
        )?;
        let candidate_digest = field(candidate, "digest");
        let bound = is_str(field(candidate, "image"), IMAGE)
            && eq(field(candidate, "environment"), item(image, "environment")?)
            && {
                let Some(text) = candidate_digest.as_str() else {
                    let kind = type_name(candidate_digest);
                    return Err(format!("can only concatenate str (not \"{kind}\") to str"));
                };
                eq(
                    item(image, "reference")?,
                    &string(&format!("{IMAGE}@{text}")),
                )
            };
        require(bound, "image/index binding mismatch")?;
        digest(candidate_digest)?;
        let family = self.family();
        fields(family, "id name cuda_series rocm_version")?;
        text(field(family, "id"))?;
        text(field(family, "name"))?;
        let id = field(family, "id").as_str().unwrap_or_default();
        let backend = item(image, "backend")?;
        let rocm_id = id
            .strip_prefix("rocm")
            .is_some_and(|rest| !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()));
        require(
            eq(field(family, "id"), backend)
                || is_str(backend, "rocm") && is_str(field(family, "name"), "rocm") && rocm_id,
            "family binding mismatch",
        )?;
        let environment = display(Some(field(candidate, "environment")));
        require(
            eq(key, &string(&format!("candidate-index-{environment}-{id}"))),
            "index candidate key/family mismatch",
        )?;
        let expected = self.expected_backend()?;
        require(
            is_str(field(family, "name"), &expected),
            "normalized family mismatch",
        )?;
        for toolkit in ["cuda_series", "rocm_version"] {
            if field(family, toolkit) != &Json::Null {
                text(field(family, toolkit))?;
            }
        }
        require(
            (field(family, "cuda_series") != &Json::Null) == (expected == "cuda")
                && (field(family, "rocm_version") != &Json::Null) == (expected == "rocm"),
            "toolkit family mismatch",
        )?;
        for source in ["mesh_revision", "runner_images_revision"] {
            require(
                eq(field(candidate, source), item(self.proof_origin, source)?),
                "candidate source mismatch",
            )?;
        }
        Ok(())
    }
}
