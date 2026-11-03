//! Conservative placement derived from protected governance scope and the
//! complete change list, independently of exhaustive coverage selection.

use super::request::FORCE_ALL;
use serde::Deserialize;
use std::{fs, path::Path};

#[derive(Deserialize)]
struct Policy {
    schema_version: u64,
    scope: Scope,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Scope {
    directory_prefixes: Vec<String>,
    exact_paths: Vec<String>,
    root_filename_prefixes: Vec<String>,
}

impl Scope {
    fn valid(&self) -> bool {
        let valid_path = |path: &str| {
            let components = path.trim_end_matches('/');
            !components.is_empty()
                && !path.starts_with('/')
                && !path.contains(['\\', '\0', '\n', '\r'])
                && !components
                    .split('/')
                    .any(|part| part.is_empty() || part == ".." || part == ".")
        };
        !self.directory_prefixes.is_empty()
            && !self.exact_paths.is_empty()
            && !self.root_filename_prefixes.is_empty()
            && self
                .directory_prefixes
                .iter()
                .all(|path| path.ends_with('/') && !path.ends_with("//") && valid_path(path))
            && self
                .exact_paths
                .iter()
                .all(|path| !path.ends_with('/') && valid_path(path))
            && self
                .root_filename_prefixes
                .iter()
                .all(|path| valid_path(path) && !path.contains('/'))
    }

    fn contains(&self, path: &str) -> bool {
        self.exact_paths.iter().any(|exact| path == exact)
            || self
                .directory_prefixes
                .iter()
                .any(|prefix| path.starts_with(prefix))
            || (!path.contains('/')
                && self
                    .root_filename_prefixes
                    .iter()
                    .any(|prefix| path.starts_with(prefix)))
    }
}

fn load_scope(root: &Path) -> Option<Scope> {
    let path = root.join("ci/governance/policy.json");
    let metadata = fs::symlink_metadata(&path).ok()?;
    if !metadata.is_file() || metadata.len() > 64 * 1024 {
        return None;
    }
    let policy: Policy = serde_json::from_slice(&fs::read(path).ok()?).ok()?;
    (policy.schema_version == 1 && policy.scope.valid()).then_some(policy.scope)
}

pub(super) fn force_hosted(
    root: &Path,
    base_sha: &str,
    files: &[String],
    domains: &[String],
) -> bool {
    if base_sha.is_empty()
        || files.iter().any(|path| path == FORCE_ALL)
        || domains
            .iter()
            .any(|domain| matches!(domain.as_str(), "ci-control" | "runner-infra" | "tooling"))
    {
        return true;
    }
    let Some(scope) = load_scope(root) else {
        return true;
    };
    files.iter().any(|path| scope.contains(path))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope() -> Scope {
        Scope {
            directory_prefixes: vec![".github/".into(), "tools/".into()],
            exact_paths: vec!["Cargo.toml".into()],
            root_filename_prefixes: vec!["Dockerfile.".into()],
        }
    }

    #[test]
    fn governed_paths_use_exact_and_bounded_prefix_matches() {
        let scope = scope();
        assert!(scope.valid());
        for path in [
            ".github/workflows/ci.yml",
            "tools/x.rs",
            "Cargo.toml",
            "Dockerfile.ci",
        ] {
            assert!(scope.contains(path), "{path}");
        }
        for path in [
            "mesh/src/lib.rs",
            "toolshed/x.rs",
            "nested/Dockerfile.ci",
            "Cargo.toml.bak",
        ] {
            assert!(!scope.contains(path), "{path}");
        }
    }

    #[test]
    fn missing_policy_or_incomplete_inputs_force_hosted() {
        let missing = Path::new("/this-policy-does-not-exist");
        assert!(force_hosted(
            missing,
            "base",
            &["mesh/src/lib.rs".into()],
            &[]
        ));
        assert!(force_hosted(missing, "", &[], &[]));
        assert!(force_hosted(missing, "base", &[FORCE_ALL.into()], &[]));
        assert!(force_hosted(missing, "base", &[], &["runner-infra".into()]));
        let mut invalid = scope();
        invalid.directory_prefixes[0] = "../.github/".into();
        assert!(!invalid.valid());
    }

    #[test]
    fn current_policy_allows_ordinary_changes_and_blocks_control_changes() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let base = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        for path in ["mesh/crates/mesh-llm/src/main.rs", "README.md"] {
            assert!(!force_hosted(root, base, &[path.into()], &[]), "{path}");
        }
        for path in [
            ".github/workflows/main_linux.yml",
            "tools/ci-controller/src/plan/signals.rs",
            "mesh/website/package.json",
            "Cargo.toml",
        ] {
            assert!(force_hosted(root, base, &[path.into()], &[]), "{path}");
        }
    }
}
