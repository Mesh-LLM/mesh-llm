use super::{catalog, document::Json, selection};
use crate::command::DynResult;
use serde_json::Value;

#[test]
fn relocated_paths_and_crates_preserve_semantic_domains() -> DynResult<()> {
    let root = crate::repo_consistency::repo_root()?;
    let bytes = std::fs::read(root.join("ci/ownership.yml"))?;
    let ownership = catalog::validate_ownership(&Json::parse(&bytes)?).map_err(|error| error.0)?;
    for (old, new) in [
        (
            "crates/mesh-llm-cli/src/parser.rs",
            "mesh/crates/mesh-llm-cli/src/parser.rs",
        ),
        (
            "crates/mesh-llm-ui/src/App.tsx",
            "mesh/crates/mesh-llm-ui/src/App.tsx",
        ),
        (
            "crates/mesh-llm-protocol/src/lib.rs",
            "mesh/crates/mesh-llm-protocol/src/lib.rs",
        ),
        (
            "crates/skippy-ffi/src/abi.rs",
            "skippy/crates/skippy-ffi/src/abi.rs",
        ),
        (
            "crates/skippy-server/src/lib.rs",
            "skippy/crates/skippy-serving/src/lib.rs",
        ),
        (
            "crates/model-hf/src/lib.rs",
            "skippy/crates/skippy-model-hf/src/lib.rs",
        ),
        (
            "third_party/llama.cpp/patches/0001.patch",
            "skippy/third_party/llama.cpp/patches/0001.patch",
        ),
        (
            "scripts/prepare-llama.sh",
            "skippy/scripts/prepare-llama.sh",
        ),
        ("scripts/build-host.sh", "mesh/scripts/build-host.sh"),
        (
            "sdk/kotlin/build.gradle.kts",
            "mesh/sdk/kotlin/build.gradle.kts",
        ),
        ("website/src/index.njk", "mesh/website/src/index.njk"),
        ("docs/skippy/CONFIG.md", "skippy/docs/CONFIG.md"),
        ("evals/parity.json", "skippy/evals/parity.json"),
    ] {
        assert_eq!(
            selection::matched_domains(&ownership, &[old.into()], &[]).map_err(|error| error.0)?,
            selection::matched_domains(&ownership, &[new.into()], &[]).map_err(|error| error.0)?
        );
    }
    for (old, new) in [
        ("openai-frontend", "skippy-openai-frontend"),
        ("skippy-server", "skippy-serving"),
        ("skippy-model-package", "skippy-package-builder"),
        ("model-hf", "skippy-model-hf"),
        ("model-artifact", "skippy-model-artifact"),
        ("model-ref", "skippy-model-ref"),
        ("model-resolver", "skippy-model-resolver"),
        ("mesh-llm-gpu-bench", "skippy-gpu-bench"),
    ] {
        assert_eq!(
            selection::matched_domains(&ownership, &[], &[old.into()]).map_err(|error| error.0)?,
            selection::matched_domains(&ownership, &[], &[new.into()]).map_err(|error| error.0)?
        );
    }
    for name in [
        "mesh-llm-skippy-adapter",
        "mesh-llm-membership",
        "mesh-llm-control-api",
        "mesh-llm-runtime",
    ] {
        for prefix in ["crates/", "mesh/crates/"] {
            let domains = selection::matched_domains(
                &ownership,
                &[format!("{prefix}{name}/src/lib.rs")],
                &[name.into()],
            )
            .map_err(|error| error.0)?;
            assert!(domains.contains(&"runtime-product".into()));
        }
    }
    for path in [
        "mesh/unowned/payload.bin",
        "skippy/unowned/payload.bin",
        "shared/crates/demo/src/lib.rs",
    ] {
        assert!(selection::matched_domains(&ownership, &[path.into()], &[]).is_err());
    }
    Ok(())
}

#[test]
fn relocation_preserves_complete_plan_and_reused_name_ownership() -> DynResult<()> {
    let root = crate::repo_consistency::repo_root()?;
    let mut input: Value = serde_json::from_slice(&std::fs::read(
        root.join("scripts/tests/fixtures/ci-plan/runtime.json"),
    )?)?;
    let old = super::build_for_validation(&root, &serde_json::to_vec(&input)?)?;
    for path in input["changed_files"].as_array_mut().ok_or("paths")? {
        *path = format!("mesh/{}", path.as_str().ok_or("path")?).into();
    }
    for package in input["workspace_packages"]
        .as_array_mut()
        .ok_or("packages")?
    {
        let owner = if package["name"]
            .as_str()
            .ok_or("name")?
            .starts_with("skippy-")
        {
            "skippy"
        } else {
            "mesh"
        };
        package["path"] = format!("{owner}/{}", package["path"].as_str().ok_or("path")?).into();
    }
    let new = super::build_for_validation(&root, &serde_json::to_vec(&input)?)?;
    for key in [
        "domains",
        "required_slices",
        "direct_crates",
        "affected_crates",
        "matrices",
        "signals",
    ] {
        assert_eq!(old[key], new[key]);
    }
    let ownership = catalog::validate_ownership(&Json::parse(&std::fs::read(
        root.join("ci/ownership.yml"),
    )?)?)
    .map_err(|error| error.0)?;
    assert_eq!(
        selection::matched_domains(&ownership, &[], &["skippy-model-package".into()])
            .map_err(|error| error.0)?,
        ["rust", "split-serving"]
    );
    assert_eq!(
        selection::matched_domains(
            &ownership,
            &["skippy/crates/skippy-model-package/src/lib.rs".into()],
            &["skippy-model-package".into()]
        )
        .map_err(|error| error.0)?,
        ["rust", "split-serving", "model-download"]
    );
    Ok(())
}

#[test]
fn successor_catalog_gaps_are_exact_and_never_gain_domains() -> DynResult<()> {
    use std::collections::{BTreeMap, BTreeSet};
    let root = crate::repo_consistency::repo_root()?;
    let ownership = catalog::validate_ownership(&Json::parse(&std::fs::read(
        root.join("ci/ownership.yml"),
    )?)?)
    .map_err(|error| error.0)?;
    let mut gaps = BTreeMap::new();
    for (predecessor, successors) in crate::repository::cargo_packages::successors::SUCCESSORS {
        let expected: BTreeSet<_> =
            selection::matched_domains(&ownership, &[], &[(*predecessor).into()])
                .map_err(|error| error.0)?
                .into_iter()
                .collect();
        for successor in *successors {
            if *predecessor == "model-package" && *successor == "skippy-model-package" {
                continue;
            }
            let actual: BTreeSet<_> =
                selection::matched_domains(&ownership, &[], &[(*successor).into()])
                    .map_err(|error| error.0)?
                    .into_iter()
                    .collect();
            if actual != expected {
                assert!(actual.is_subset(&expected), "{successor}");
                gaps.insert(
                    (*successor).to_owned(),
                    expected.difference(&actual).cloned().collect::<Vec<_>>(),
                );
            }
        }
    }
    assert_eq!(
        gaps,
        BTreeMap::from([
            (
                "mesh-llm-control-api".into(),
                vec!["platform-windows-cfg".into()]
            ),
            (
                "mesh-llm-membership".into(),
                vec!["platform-windows-cfg".into()]
            ),
            (
                "mesh-llm-skippy-adapter".into(),
                vec!["platform-windows-cfg".into()]
            ),
            (
                "mesh-llm-transport".into(),
                vec!["platform-windows-cfg".into()]
            ),
            ("skippy-api".into(), vec!["split-serving".into()]),
            ("skippy-events".into(), vec!["split-serving".into()]),
            ("skippy-hf-hub".into(), vec!["model-download".into()]),
        ])
    );
    Ok(())
}
