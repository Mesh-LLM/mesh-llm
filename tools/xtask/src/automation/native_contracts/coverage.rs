use crate::command::DynResult;
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
struct Manifest {
    models: Vec<Model>,
}

#[derive(Deserialize)]
struct Model {
    family: String,
    class: Class,
    profile: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Class {
    CausalGeneration,
    Embedding,
    Rerank,
    EncoderDecoder,
    Ocr,
    SpeechSynthesis,
    SpeechRecognition,
}

#[derive(Deserialize)]
struct FamilyMap {
    families: BTreeMap<String, Vec<String>>,
}

#[derive(Deserialize)]
struct Report {
    builders: Vec<Builder>,
}

#[derive(Deserialize)]
struct Builder {
    file: String,
    verdict: String,
    #[serde(default)]
    proof: Proof,
}

#[derive(Default, Deserialize)]
struct Proof {
    #[serde(default)]
    execution_scope: String,
}

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let values = super::options(args, &["--manifest", "--family-map", "--report"])?;
    let manifest =
        serde_json::from_slice(&std::fs::read(super::required(&values, "--manifest")?)?)?;
    let map = serde_json::from_slice(&std::fs::read(super::required(&values, "--family-map")?)?)?;
    let report = serde_json::from_slice(&std::fs::read(super::required(&values, "--report")?)?)?;
    verify(&manifest, &map, &report)
}

fn verify(manifest: &Manifest, map: &FamilyMap, report: &Report) -> DynResult<()> {
    for model in &manifest.models {
        let workload = matches!(model.profile.as_str(), "workload-smoke" | "workload-oracle");
        match model.class {
            Class::CausalGeneration => {
                if workload {
                    return Err(format!(
                        "{}: causal split target cannot use a workload profile",
                        model.family
                    )
                    .into());
                }
                let sources = map
                    .families
                    .get(&model.family)
                    .filter(|sources| !sources.is_empty())
                    .ok_or_else(|| {
                        format!("{}: no generated family source mapping", model.family)
                    })?;
                if !report.builders.iter().any(|builder| {
                    builder.verdict == "transformable"
                        && builder.proof.execution_scope == "partitioned_decoder"
                        && sources.iter().any(|source| {
                            builder.file == *source || builder.file.ends_with(&format!("/{source}"))
                        })
                }) {
                    return Err(format!(
                        "{}: no mapped partitioned decoder was transformed",
                        model.family
                    )
                    .into());
                }
            }
            Class::Embedding
            | Class::Rerank
            | Class::EncoderDecoder
            | Class::Ocr
            | Class::SpeechSynthesis
            | Class::SpeechRecognition => {
                if !workload {
                    return Err(format!(
                        "{}: non-chat class requires a workload profile",
                        model.family
                    )
                    .into());
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auxiliary_transform_cannot_certify_causal_decoder() {
        let manifest: Manifest = serde_json::from_str(
            r#"{"models":[{"family":"dense","class":"causal_generation","profile":"split"}]}"#,
        )
        .unwrap();
        let map: FamilyMap =
            serde_json::from_str(r#"{"families":{"dense":["src/models/dense.cpp"]}}"#).unwrap();
        let report: Report = serde_json::from_str(r#"{"builders":[{"file":"src/models/dense.cpp","verdict":"transformable","proof":{"execution_scope":"final_stage_auxiliary"}}]}"#).unwrap();
        let result = verify(&manifest, &map, &report);
        assert!(result.is_err());
    }

    #[test]
    fn transformed_decoder_certifies_mapped_family() {
        let manifest: Manifest = serde_json::from_str(
            r#"{"models":[{"family":"dense","class":"causal_generation","profile":"split"}]}"#,
        )
        .unwrap();
        let map: FamilyMap =
            serde_json::from_str(r#"{"families":{"dense":["src/models/dense.cpp"]}}"#).unwrap();
        let report: Report = serde_json::from_str(r#"{"builders":[{"file":"/checkout/src/models/dense.cpp","verdict":"transformable","proof":{"execution_scope":"partitioned_decoder"}}]}"#).unwrap();
        let result = verify(&manifest, &map, &report);
        assert!(result.is_ok());
    }
}
