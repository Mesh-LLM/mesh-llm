//! Derive retained ABI identities from the actual CI, release and warmer rows.

use super::windows_cache_identity::{Inputs, identity};
use ci_controller::lane_results::workflow_yaml::{self, Node};
use serde::Serialize;
use std::{collections::BTreeMap, fs, path::Path};

const RESTORE: &str = "./.github/actions/restore-windows-abi-cache";
const WORKFLOWS: &[&str] = &[
    "ci-windows-runtime-slice.yml",
    "release.yml",
    "windows-warm-caches.yml",
];

#[derive(Debug, Serialize)]
pub(super) struct Policy {
    schema_version: u32,
    target_ref: &'static str,
    current_keys: Vec<String>,
    active_families_complete: bool,
    max_bytes: u64,
    history_per_family: usize,
}

pub(super) struct Context {
    pub epoch: String,
    pub input_hash: String,
    pub variables: BTreeMap<String, String>,
    pub max_bytes: u64,
    pub history: usize,
}

pub(super) fn preserve_observed_images(
    policy: &mut Policy,
    inventory: &serde_json::Value,
) -> Result<(), String> {
    use super::windows_cache_retention::compatibility;
    let active = policy
        .current_keys
        .iter()
        .map(|key| {
            compatibility(key)
                .map(str::to_owned)
                .map_err(|error| error.to_string())
        })
        .collect::<Result<std::collections::BTreeSet<_>, _>>()?;
    let entries = inventory["actions_caches"]
        .as_array()
        .ok_or("cache entries missing")?;
    for cache in entries {
        if cache["ref"] != policy.target_ref {
            continue;
        }
        let key = cache["key"].as_str().ok_or("cache key missing")?;
        if !key.starts_with(super::windows_cache_identity::PREFIX) {
            continue;
        }
        let prefix = compatibility(key).map_err(|error| error.to_string())?;
        if active.contains(prefix) {
            // Hosted image rollout is not atomic across runners. Observed image cohorts
            // remain active; the byte bound refuses unsafe pruning if they no longer fit.
            policy.current_keys.push(key.to_owned());
        }
    }
    policy.current_keys.sort();
    policy.current_keys.dedup();
    Ok(())
}

pub(super) fn policy(root: &Path, context: &Context) -> Result<Policy, String> {
    for entry in fs::read_dir(root.join(".github/workflows")).map_err(|error| error.to_string())? {
        let path = entry.map_err(|error| error.to_string())?.path();
        if !matches!(
            path.extension().and_then(|value| value.to_str()),
            Some("yml" | "yaml")
        ) {
            continue;
        }
        let source = fs::read_to_string(&path).map_err(|error| error.to_string())?;
        let (_, actions) = workflow_yaml::parse_with_actions(&source)?;
        if actions.iter().any(|action| action.reference == RESTORE)
            && !WORKFLOWS.contains(
                &path
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or(""),
            )
        {
            return Err(format!(
                "unregistered Windows ABI consumer: {}",
                path.display()
            ));
        }
    }
    let mut current_keys = Vec::new();
    for workflow in WORKFLOWS {
        let source = fs::read_to_string(root.join(".github/workflows").join(workflow))
            .map_err(|error| error.to_string())?;
        let document = workflow_yaml::parse_resolved_aliases(&source)?;
        let jobs = document.get("jobs").ok_or("workflow jobs missing")?;
        for (_, job) in jobs.entries() {
            let Some(Node::Seq(steps)) = job.get("steps") else {
                continue;
            };
            for step in steps {
                if step.get("uses").and_then(Node::text) != Some(RESTORE) {
                    continue;
                }
                let inputs = step.get("with").ok_or("ABI restore inputs missing")?;
                for row in rows(root, job)? {
                    let resolver = Resolver {
                        workflow: document.get("env"),
                        job: job.get("env"),
                        row: &row,
                        context,
                    };
                    let input = |name| {
                        resolver.resolve(inputs.get(name).and_then(Node::text).unwrap_or(""), 0)
                    };
                    current_keys.push(
                        identity(&Inputs {
                            backend: &input("backend")?,
                            architectures: &input("architecture_set")?,
                            epoch: &context.epoch,
                            input_hash: &context.input_hash,
                            cuda: &input("cuda_toolchain_version")?,
                            vulkan: &input("vulkan_toolchain_version")?,
                            rocm: &input("rocm_toolchain_version")?,
                        })?
                        .key,
                    );
                }
            }
        }
    }
    current_keys.sort();
    current_keys.dedup();
    if current_keys.is_empty() {
        return Err("no active Windows ABI consumers found".into());
    }
    Ok(Policy {
        schema_version: 1,
        target_ref: "refs/heads/main",
        current_keys,
        active_families_complete: true,
        max_bytes: context.max_bytes,
        history_per_family: context.history,
    })
}

fn rows(root: &Path, job: &Node) -> Result<Vec<BTreeMap<String, String>>, String> {
    let Some(matrix) = job.get("strategy").and_then(|node| node.get("matrix")) else {
        return Ok(vec![BTreeMap::new()]);
    };
    if matrix.get("runtime").and_then(Node::text) == Some("${{ fromJson(inputs.runtime_matrix) }}")
    {
        let bytes = fs::read(root.join("ci/slices.yml")).map_err(|error| error.to_string())?;
        let catalog: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
        let rows = catalog["runtime_rows"]
            .as_array()
            .ok_or("runtime rows missing")?;
        return rows
            .iter()
            .filter(|row| row["platform"] == "windows")
            .map(|row| {
                let row = row.as_object().ok_or("invalid runtime row")?;
                Ok(row
                    .iter()
                    .filter_map(|(key, value)| {
                        value
                            .as_str()
                            .map(|value| (format!("runtime.{key}"), value.to_owned()))
                    })
                    .collect())
            })
            .collect();
    }
    let Some(Node::Seq(rows)) = matrix.get("include") else {
        return Err("unsupported Windows ABI consumer matrix".into());
    };
    rows.iter()
        .map(|row| {
            row.entries()
                .iter()
                .map(|(key, value)| {
                    Ok((
                        key.clone(),
                        value
                            .text()
                            .ok_or("non-scalar ABI matrix value")?
                            .to_owned(),
                    ))
                })
                .collect()
        })
        .collect()
}

struct Resolver<'a> {
    workflow: Option<&'a Node>,
    job: Option<&'a Node>,
    row: &'a BTreeMap<String, String>,
    context: &'a Context,
}

impl Resolver<'_> {
    fn resolve(&self, value: &str, depth: usize) -> Result<String, String> {
        if depth > 8 {
            return Err("recursive Windows ABI environment".into());
        }
        let Some(expression) = value
            .strip_prefix("${{ ")
            .and_then(|s| s.strip_suffix(" }}"))
        else {
            if value.contains("${{") {
                return Err("unsupported Windows ABI expression".into());
            }
            return Ok(value.to_owned());
        };
        for term in expression.split(" || ") {
            let value = if let Some(name) = term.strip_prefix("matrix.") {
                self.row.get(name).cloned().unwrap_or_default()
            } else if let Some(name) = term.strip_prefix("vars.") {
                self.context
                    .variables
                    .get(name)
                    .cloned()
                    .unwrap_or_default()
            } else if let Some(name) = term.strip_prefix("env.") {
                let value = self
                    .job
                    .and_then(|node| node.get(name))
                    .or_else(|| self.workflow.and_then(|node| node.get(name)))
                    .and_then(Node::text)
                    .ok_or("missing Windows ABI environment value")?;
                self.resolve(value, depth + 1)?
            } else if let Some(literal) = term.strip_prefix('\'').and_then(|s| s.strip_suffix('\''))
            {
                literal.to_owned()
            } else {
                return Err(format!("unsupported Windows ABI selector: {term}"));
            };
            if !value.is_empty() {
                return Ok(value);
            }
        }
        Ok(String::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> Context {
        Context {
            epoch: "github-win22-20261115-X64".into(),
            input_hash: "a".repeat(64),
            variables: BTreeMap::new(),
            max_bytes: 4 * 1024 * 1024 * 1024,
            history: 2,
        }
    }

    fn fixture() -> tempfile::TempDir {
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let root = tempfile::tempdir().unwrap();
        fs::create_dir_all(root.path().join(".github/workflows")).unwrap();
        fs::create_dir_all(root.path().join("ci")).unwrap();
        for workflow in WORKFLOWS {
            let path = format!(".github/workflows/{workflow}");
            fs::copy(source.join(&path), root.path().join(&path)).unwrap();
        }
        fs::copy(
            source.join("ci/slices.yml"),
            root.path().join("ci/slices.yml"),
        )
        .unwrap();
        root
    }

    #[test]
    fn observed_image_rollout_preserves_both_active_toolkit_cohorts() {
        let root = fixture();
        let mut policy = policy(root.path(), &context()).unwrap();
        let current = policy
            .current_keys
            .iter()
            .find(|key| key.contains("-cpu--cpu-"))
            .unwrap()
            .clone();
        let earlier = current.replace("20261115", "20261114");
        let entries: Vec<_> = [&current, &earlier].into_iter().enumerate().map(|(index, key)| {
            serde_json::json!({"id":index + 1,"key":key,"version":"v1","ref":"refs/heads/main",
                "size_in_bytes":10,"created_at":"2026-11-16T00:00:00Z","last_accessed_at":"2026-11-16T01:00:00Z"})
        }).collect();
        let inventory = serde_json::json!({"total_count":2,"actions_caches":entries});
        preserve_observed_images(&mut policy, &inventory).unwrap();
        let plan = super::super::windows_cache_retention::plan_json(
            &serde_json::to_vec(&inventory).unwrap(),
            &serde_json::to_vec(&policy).unwrap(),
        )
        .unwrap();
        assert_eq!(plan.retain.len(), 2);
        assert!(plan.delete.is_empty());
        policy.max_bytes = 19;
        assert!(
            super::super::windows_cache_retention::plan_json(
                &serde_json::to_vec(&inventory).unwrap(),
                &serde_json::to_vec(&policy).unwrap()
            )
            .is_err()
        );
    }

    #[test]
    fn new_consumer_refuses_retention_until_roster_is_complete() {
        let root = fixture();
        fs::write(root.path().join(".github/workflows/extra.yml"),
            "jobs:\n  cache:\n    steps:\n      - uses: ./.github/actions/restore-windows-abi-cache\n").unwrap();
        assert!(
            policy(root.path(), &context())
                .unwrap_err()
                .contains("unregistered")
        );
    }

    #[test]
    fn changed_consumer_architecture_updates_retained_family() {
        let root = fixture();
        let path = root.path().join("ci/slices.yml");
        let mut catalog: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        for row in catalog["runtime_rows"].as_array_mut().unwrap() {
            if row["id"] == "windows-cuda" {
                row["cuda_architectures"] = "86;120".into();
            }
        }
        fs::write(path, serde_json::to_vec(&catalog).unwrap()).unwrap();
        let policy = policy(root.path(), &context()).unwrap();
        assert!(
            policy
                .current_keys
                .iter()
                .any(|key| key.contains("cuda-86_120-cuda-12.6.3-"))
        );
        // The warmup still consumes its own 75-only family.
        assert!(
            policy
                .current_keys
                .iter()
                .any(|key| key.contains("cuda-75-cuda-12.6.3-"))
        );
    }

    #[test]
    fn repository_consumers_preserve_ci_release_and_warmer_families() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let policy = policy(
            &root,
            &Context {
                epoch: "github-win22-20261115-X64".into(),
                input_hash: "a".repeat(64),
                variables: BTreeMap::new(),
                max_bytes: 20 * 1024 * 1024 * 1024,
                history: 2,
            },
        )
        .unwrap();
        let families: Vec<_> = policy
            .current_keys
            .iter()
            .map(|key| {
                key.strip_prefix(super::super::windows_cache_identity::PREFIX)
                    .unwrap()
                    .split("-github-")
                    .next()
                    .unwrap()
            })
            .collect();
        assert_eq!(
            families,
            [
                "cpu--cpu",
                "cuda-61_75_80_86_89_90-cuda-12.9.2-Jimver-v0.2.36",
                "cuda-75-cuda-12.6.3-Jimver-v0.2.36",
                "cuda-75_80_86_89_90_100_103_120-cuda-13.1.2-Jimver-v0.2.36",
                "rocm-gfx1100-rocm-AMD-Software-PRO-Edition-25.Q3-WinSvr2022-For-HIP.exe",
                "rocm-gfx90a_gfx942_gfx1100_gfx1101_gfx1102_gfx1103_gfx1151_gfx1200_gfx1201-rocm-AMD-Software-PRO-Edition-25.Q3-WinSvr2022-For-HIP.exe",
                "vulkan--vulkan-1.4.328.1-jakoch-v1.5.2",
            ]
        );
    }
}
