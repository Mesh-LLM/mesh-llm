//! Profile admission shared by catalog display and scheduler replay callers.
use std::path::Path;

use serde_json::Value;

use super::fixture_catalog as catalog;
use crate::command::DynResult;

pub(crate) fn resolve<'a>(input: &'a Value, name: &str) -> DynResult<&'a Value> {
    catalog::validate(input)?;
    input["profiles"]
        .get(name)
        .ok_or_else(|| format!("unknown scheduler fixture profile {name}").into())
}

pub(super) fn validate_inputs(
    profile: &Value,
    model_id: &str,
    model_sha256: &str,
    prompt_manifest: Option<&Path>,
) -> DynResult<()> {
    let model = catalog::object(profile, "model")?;
    if model_id != catalog::text(model, "id")? {
        return Err("fixture model id differs from the pinned identity".into());
    }
    if model_sha256 != catalog::text(model, "sha256")? {
        return Err("fixture model SHA-256 differs from the pinned identity".into());
    }
    let corpus = catalog::object(profile, "corpus")?;
    match (catalog::text(corpus, "kind")?, prompt_manifest) {
        ("hf", None) => Err("HF fixture profiles require a prepared prompt manifest".into()),
        ("synthetic", Some(_)) => {
            Err("synthetic fixture profiles do not accept an external prompt manifest".into())
        }
        ("hf", Some(_)) | ("synthetic", None) => Ok(()),
        _ => Err("unsupported scheduler fixture corpus kind".into()),
    }
}

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let mut values = std::collections::BTreeMap::new();
    let mut remaining = args;
    while !remaining.is_empty() {
        let [flag, value, rest @ ..] = remaining else {
            return Err("fixture input option requires a value".into());
        };
        if ![
            "--catalog",
            "--profile",
            "--model-id",
            "--model-sha256",
            "--prompt-manifest",
        ]
        .contains(&flag.as_str())
        {
            return Err(format!("unknown fixture input option {flag}").into());
        }
        if value.trim().is_empty() || values.insert(flag.as_str(), value.as_str()).is_some() {
            return Err(format!("empty or duplicate fixture input option {flag}").into());
        }
        remaining = rest;
    }
    let required = |key| -> DynResult<&str> {
        values
            .get(key)
            .copied()
            .ok_or_else(|| format!("missing {key}").into())
    };
    let input: Value = serde_json::from_slice(&std::fs::read(required("--catalog")?)?)?;
    let selected = resolve(&input, required("--profile")?)?;
    validate_inputs(
        selected,
        required("--model-id")?,
        required("--model-sha256")?,
        values.get("--prompt-manifest").map(Path::new),
    )?;
    println!("fixture model identity and manifest mode admitted");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checked_in() -> Value {
        serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../skippy/evals/skippy-scheduler-fixtures.json"
        )))
        .unwrap()
    }

    #[test]
    fn resolved_fixture_profile_owns_workload_shape_and_refuses_unknown_or_invalid_catalogs() {
        let mut input = checked_in();
        let selected = resolve(&input, "agentic-eviction-pressure").unwrap();
        assert_eq!(selected["workload"]["rounds"], 4);
        assert_eq!(selected["workload"]["families"], 8);
        assert_eq!(selected["workload"]["admission_concurrency"], 16);
        assert_eq!(selected["corpus"]["kind"], "hf");
        assert!(resolve(&input, "unknown").is_err());
        input["profiles"]["agentic-eviction-pressure"]["workload"]["ctx_size"] = 1.into();
        assert!(resolve(&input, "warm-affinity").is_err());
    }

    #[test]
    fn fixture_inputs_require_pinned_model_identity_and_matching_manifest_mode() {
        let input = checked_in();
        for (name, manifest) in [
            ("warm-affinity", None),
            (
                "agentic-eviction-pressure",
                Some(Path::new("prepared.json")),
            ),
        ] {
            let profile = resolve(&input, name).unwrap();
            let model = &profile["model"];
            let id = catalog::text(model, "id").unwrap();
            let hash = catalog::text(model, "sha256").unwrap();
            validate_inputs(profile, id, hash, manifest).unwrap();
            assert!(validate_inputs(profile, "different", hash, manifest).is_err());
            assert!(validate_inputs(profile, id, &"0".repeat(64), manifest).is_err());
            let incompatible = if manifest.is_some() {
                None
            } else {
                Some(Path::new("external.json"))
            };
            assert!(validate_inputs(profile, id, hash, incompatible).is_err());
        }
    }
}
