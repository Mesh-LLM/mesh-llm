//! Every frozen suite manifest x every artifact selection (including none) x
//! every registry cadence plus an unauthorized one x single-file mode, in
//! both stdout-JSON and GitHub-output modes.

use crate::support::{
    RESOLVER, Stage, TestResult, assert_same, assert_streams, code, field, fixture, legacy,
    run_legacy, xtask,
};
use serde_json::Value;

/// The legacy argv order: manifest, cadence, optional id, optional flag.
fn argv(case: &Value) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut args = vec![
        format!("manifests/{}", field(case, "manifest")?),
        "--cadence".to_owned(),
        field(case, "cadence")?.to_owned(),
    ];
    if let Some(id) = case["artifact_id"].as_str() {
        args.extend(["--artifact-id".to_owned(), id.to_owned()]);
    }
    if case["require_single_file"].as_bool() == Some(true) {
        args.push("--require-single-file".to_owned());
    }
    Ok(args)
}

fn label(case: &Value) -> String {
    format!(
        "{} id={} cadence={} single={}",
        case["manifest"], case["artifact_id"], case["cadence"], case["require_single_file"]
    )
}

#[test]
fn migration_models_resolver_matrix_matches_legacy_for_every_suite_and_cadence() -> TestResult {
    // Given: the frozen suite manifests and every captured legacy resolution.
    let stage = Stage::new("matrix")?;
    stage.frozen_manifests()?;
    let cases = fixture("resolver-matrix.json")?;
    let cases = cases.as_array().ok_or("matrix must be an array")?;
    let python = legacy("MIGRATION_MODELS_LEGACY_PYTHON");
    let script = python
        .as_ref()
        .map(|_| stage.legacy_script(RESOLVER))
        .transpose()?;
    let (mut accepted, mut rejected) = (0, 0);
    for case in cases {
        let name = label(case);
        let args = argv(case)?;
        let args = args.iter().map(String::as_str).collect::<Vec<_>>();
        // When: the JSON summary mode runs.
        let mut ported_args = vec!["models", "resolve"];
        ported_args.extend_from_slice(&args);
        let ported = xtask(stage.path(), &ported_args)?;
        // Then: status and both streams equal the legacy capture.
        let json = &case["json"];
        assert_streams(
            &name,
            &ported,
            code(json)?,
            field(json, "stdout")?,
            field(json, "stderr")?,
        );
        if code(json)? == 0 {
            accepted += 1;
        } else {
            rejected += 1;
        }
        // When: GitHub-output mode appends to a file with prior content.
        stage.write("gh.out", b"preexisting=1\n")?;
        let mut output_args = ported_args.clone();
        output_args.extend_from_slice(&["--github-output", "gh.out"]);
        let ported = xtask(stage.path(), &output_args)?;
        // Then: the appended bytes and streams equal the legacy capture.
        let gh = &case["github_output"];
        assert_streams(
            &name,
            &ported,
            code(gh)?,
            field(gh, "stdout")?,
            field(gh, "stderr")?,
        );
        assert_eq!(
            stage.read("gh.out")?,
            field(gh, "file")?,
            "{name}: github output"
        );
        if let (Some(python), Some(script)) = (&python, &script) {
            let original = run_legacy(python, script, stage.path(), &args)?;
            let ported = xtask(stage.path(), &ported_args)?;
            assert_same(&name, &original, &ported);
        }
    }
    // Then: the matrix exercised both authorized and rejected resolutions.
    assert_eq!(cases.len(), 648);
    assert_eq!((accepted, rejected), (173, 475));
    Ok(())
}

#[test]
fn migration_models_required_smoke_models_resolve_at_their_authorized_cadences() -> TestResult {
    // Given: the frozen product, HF-download, correctness and SafeTensors suites.
    let stage = Stage::new("authorized")?;
    stage.frozen_manifests()?;
    let required: &[(&str, &str, &[&str], bool)] = &[
        (
            "product-smoke",
            "smollm2-q8-inference",
            &["pull-request", "main", "release"],
            true,
        ),
        (
            "product-smoke",
            "family-granite-hybrid",
            &["pull-request", "main", "release"],
            true,
        ),
        (
            "scripted-binary-smoke",
            "family-granite-hybrid",
            &["pull-request", "main", "release"],
            true,
        ),
        (
            "sdk-smoke",
            "smollm2-q8-inference",
            &["pull-request", "main", "release"],
            true,
        ),
        (
            "hf-download-smoke",
            "smollm2-q4-download",
            &["pull-request", "main", "manual"],
            true,
        ),
        (
            "hf-download-smoke",
            "gemma3-bf16-metadata",
            &["pull-request", "main", "manual"],
            false,
        ),
        (
            "skippy-correctness",
            "qwen3-q8-correctness",
            &["pull-request", "main", "manual"],
            true,
        ),
        (
            "skippy-ci-smoke",
            "family-qwen3-dense",
            &["pull-request", "main", "manual"],
            true,
        ),
        (
            "safetensors-runtime-smoke",
            "smollm2-safetensors",
            &["pull-request"],
            false,
        ),
    ];
    for (suite, id, cadences, single) in required {
        for cadence in *cadences {
            // When: the consumer resolves its model at its cadence.
            let manifest = format!("manifests/{suite}.json");
            let mut args = vec![
                "models",
                "resolve",
                &manifest,
                "--artifact-id",
                id,
                "--cadence",
                cadence,
            ];
            if *single {
                args.push("--require-single-file");
            }
            let output = xtask(stage.path(), &args)?;
            // Then: it resolves the pinned identity.
            assert_eq!(output.status.code(), Some(0), "{suite}/{id}@{cadence}");
            let summary: Value = serde_json::from_slice(&output.stdout)?;
            assert_eq!(summary["artifact_id"], *id);
        }
    }
    Ok(())
}
