//! Existing warmer cache maintenance, with a retained plan before any deletion.

use super::{
    ci_metrics_github::{Gh, GhCli},
    windows_cache_consumers::{self, Context},
    windows_cache_retention,
};
use crate::{command::DynResult, repository::check_report::CheckReport};
use serde_json::Value;
use std::{collections::BTreeMap, fs, path::Path};

const REPOSITORY: &str = "Mesh-LLM/mesh-llm";
const ENDPOINT: &str = "repos/Mesh-LLM/mesh-llm/actions/caches";

pub(super) fn run(args: &[String], root: &Path) -> CheckReport {
    let result = (|| -> DynResult<()> {
        let [flag, directory, rest @ ..] = args else {
            return Err(
                "usage: ci-ops windows-cache-maintenance --output DIRECTORY [--execute]".into(),
            );
        };
        if flag != "--output" || !(rest.is_empty() || rest == ["--execute"]) {
            return Err(
                "usage: ci-ops windows-cache-maintenance --output DIRECTORY [--execute]".into(),
            );
        }
        let execute = !rest.is_empty();
        if std::env::var("GITHUB_REPOSITORY").as_deref() != Ok(REPOSITORY)
            || std::env::var("GITHUB_REF").as_deref() != Ok("refs/heads/main")
            || !matches!(
                std::env::var("GITHUB_EVENT_NAME").as_deref(),
                Ok("push" | "workflow_dispatch")
            )
        {
            return Err("Windows cache maintenance requires this repository's trusted main push or dispatch".into());
        }
        let directory = Path::new(directory);
        fs::create_dir(directory)?;
        let mut policy = windows_cache_consumers::policy(root, &context()?)?;
        let interrupt = crate::command_interrupt::Interrupt::install()?;
        let result = (|| {
            let mut gh = GhCli::new(&interrupt.cancellation());
            let observed = inventory(&mut gh)?;
            windows_cache_consumers::preserve_observed_images(&mut policy, &observed)?;
            let policy = serde_json::to_vec_pretty(&policy)?;
            fs::write(directory.join("policy.json"), &policy)?;
            apply(&mut gh, &policy, &observed, directory, execute)
        })();
        interrupt.finish()?;
        result
    })();
    match result {
        Ok(()) => CheckReport::success("Windows cache retention evidence written\n".into()),
        Err(error) => CheckReport::failure(
            String::new(),
            format!("Windows cache maintenance: {error}\n"),
        ),
    }
}

fn context() -> DynResult<Context> {
    let required = |name| std::env::var(name).map_err(|_| format!("{name} is required"));
    Ok(Context {
        epoch: required("WINDOWS_CACHE_TOOLCHAIN_EPOCH")?,
        input_hash: required("WINDOWS_CACHE_INPUT_HASH")?,
        variables: BTreeMap::from([
            (
                "CUDA_VERSION".into(),
                std::env::var("CACHE_CUDA_VERSION").unwrap_or_default(),
            ),
            (
                "VULKAN_SDK_VERSION".into(),
                std::env::var("CACHE_VULKAN_VERSION").unwrap_or_default(),
            ),
        ]),
        max_bytes: required("WINDOWS_CACHE_MAX_BYTES")?.parse()?,
        history: required("LLAMA_WINDOWS_CACHE_RETENTION")?.parse()?,
    })
}

fn request(gh: &mut dyn Gh, arguments: &[String]) -> DynResult<Value> {
    let result = gh.run(arguments)?;
    if !result.success {
        return Err(format!("GitHub cache request failed: {}", result.stderr).into());
    }
    if result.stdout.len() > 16 * 1024 * 1024 {
        return Err("GitHub cache response exceeds 16 MiB".into());
    }
    Ok(serde_json::from_str(&result.stdout)?)
}

fn inventory(gh: &mut dyn Gh) -> DynResult<Value> {
    let mut rows = Vec::new();
    let mut expected = None;
    for page in 1..=100 {
        let response = request(
            gh,
            &[
                "api".into(),
                format!(
                    "{ENDPOINT}?ref=refs%2Fheads%2Fmain&key=mesh-llm-windows-2022-skippy-abi-&per_page=100&page={page}&sort=created_at&direction=desc"
                ),
            ],
        )?;
        let count = response["total_count"]
            .as_u64()
            .ok_or("cache total_count missing")?;
        if expected.is_some_and(|expected| expected != count) {
            return Err(
                "cache inventory changed during pagination; no deletion plan produced".into(),
            );
        }
        expected = Some(count);
        let entries = response["actions_caches"]
            .as_array()
            .ok_or("cache entries missing")?;
        rows.extend(entries.iter().cloned());
        if entries.len() < 100 {
            return Ok(serde_json::json!({"total_count":count,"actions_caches":rows}));
        }
    }
    Err("cache inventory pagination limit reached; no deletion plan produced".into())
}

fn apply(
    gh: &mut dyn Gh,
    policy: &[u8],
    observed: &Value,
    directory: &Path,
    execute: bool,
) -> DynResult<()> {
    let inventory = serde_json::to_vec_pretty(observed)?;
    fs::write(directory.join("inventory.json"), &inventory)?;
    let plan = windows_cache_retention::plan_json(&inventory, policy)?;
    fs::write(
        directory.join("plan.json"),
        serde_json::to_vec_pretty(&plan)?,
    )?;
    let mut deleted = Vec::new();
    if execute {
        // Cache IDs are immutable; a newer publication has a different ID.
        for cache in &plan.delete {
            let id = cache.id;
            let result = gh.run(&[
                "api".into(),
                "--method".into(),
                "DELETE".into(),
                format!("{ENDPOINT}/{id}"),
            ])?;
            if !result.success {
                return Err(format!("cache deletion {id} failed: {}", result.stderr).into());
            }
            deleted.push(id);
            fs::write(
                directory.join("deleted.json"),
                serde_json::to_vec(&deleted)?,
            )?;
        }
    }
    fs::write(
        directory.join("result.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema_version":1,"executed":execute,"deleted_ids":deleted,
        }))?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ci_operations::ci_metrics_github::GhOutput;

    fn maintain(gh: &mut dyn Gh, policy: &[u8], directory: &Path, execute: bool) -> DynResult<()> {
        let observed = inventory(gh)?;
        apply(gh, policy, &observed, directory, execute)
    }

    struct Fixture {
        response: String,
        calls: Vec<Vec<String>>,
    }

    #[test]
    fn dry_run_preserves_cache_and_execute_deletes_only_the_planned_id() {
        let key = format!(
            "mesh-llm-windows-2022-skippy-abi-cpu--cpu-epoch-{}",
            "a".repeat(64)
        );
        let older = key.replace(&"a".repeat(64), &"b".repeat(64));
        let entry = |id, key: &str| {
            serde_json::json!({
            "id":id,"key":key,"version":"v1","ref":"refs/heads/main",
            "size_in_bytes":10,"created_at":format!("2026-11-0{id}T00:00:00Z"),
            "last_accessed_at":"2026-11-16T00:00:00Z"})
        };
        let response =
            serde_json::json!({"total_count":2,"actions_caches":[entry(1,&older),entry(2,&key)]})
                .to_string();
        let policy = serde_json::json!({"schema_version":1,"target_ref":"refs/heads/main",
            "current_keys":[key],"active_families_complete":true,"max_bytes":10,"history_per_family":1});
        for execute in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let mut gh = Fixture {
                response: response.clone(),
                calls: Vec::new(),
            };
            maintain(
                &mut gh,
                &serde_json::to_vec(&policy).unwrap(),
                directory.path(),
                execute,
            )
            .unwrap();
            let receipt: Value =
                serde_json::from_slice(&fs::read(directory.path().join("result.json")).unwrap())
                    .unwrap();
            assert_eq!(
                receipt["deleted_ids"],
                if execute {
                    serde_json::json!([1])
                } else {
                    serde_json::json!([])
                }
            );
            assert_eq!(gh.calls.len(), if execute { 2 } else { 1 });
            if execute {
                assert_eq!(gh.calls[1].last().unwrap(), &format!("{ENDPOINT}/1"));
            }
            assert!(directory.path().join("plan.json").is_file());
        }
    }
    impl Gh for Fixture {
        fn run(&mut self, arguments: &[String]) -> std::io::Result<GhOutput> {
            self.calls.push(arguments.to_vec());
            Ok(GhOutput {
                success: true,
                stdout: self.response.clone(),
                stderr: String::new(),
            })
        }
    }

    #[test]
    fn incomplete_inventory_cannot_reach_deletion() {
        let directory = tempfile::tempdir().unwrap();
        let mut gh = Fixture {
            response: r#"{"total_count":1,"actions_caches":[]}"#.into(),
            calls: Vec::new(),
        };
        let policy = serde_json::json!({"schema_version":1,"target_ref":"refs/heads/main",
            "current_keys":[format!("mesh-llm-windows-2022-skippy-abi-cpu--cpu-github-win22-20261115-X64-{}", "a".repeat(64))],
            "active_families_complete":true,"max_bytes":100,"history_per_family":2});
        assert!(
            maintain(
                &mut gh,
                &serde_json::to_vec(&policy).unwrap(),
                directory.path(),
                true
            )
            .is_err()
        );
        assert_eq!(gh.calls.len(), 1);
        assert!(!directory.path().join("plan.json").exists());
    }
}
