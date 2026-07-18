use crate::command::{
    DynResult, ensure_contains, ensure_contains_normalized, ensure_eq, ensure_nonempty_option,
    workflow_job_section,
};
use crate::repo_consistency::{CargoDependency, CargoMetadata, CargoPackage, workspace_metadata};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

pub(crate) fn check_publish_crates_consistency(repo_root: &Path) -> DynResult<()> {
    let metadata = workspace_metadata(repo_root, "publish crate consistency")?;
    let publish_crates = publish_script_crates(repo_root)?;
    let workspace_members = metadata
        .workspace_members
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let packages_by_name = workspace_packages_by_name(&metadata, &workspace_members);
    let packages_by_dir = workspace_packages_by_dir(&metadata, &workspace_members)?;
    let publish_order = publish_order(&publish_crates)?;

    check_publish_crate_metadata(repo_root, &publish_crates, &packages_by_name)?;
    check_publish_crate_dependencies(&publish_order, &packages_by_name, &packages_by_dir)?;
    check_publish_literal_includes(&publish_crates, &packages_by_name)?;
    check_publish_catalog_sync(repo_root)?;
    check_publish_workflow_invariants(repo_root)?;

    Ok(())
}

fn publish_script_crates(repo_root: &Path) -> DynResult<Vec<String>> {
    let relative_path = "scripts/publish-crates.sh";
    let contents = fs::read_to_string(repo_root.join(relative_path))?;
    let mut in_array = false;
    let mut crates = Vec::new();
    let mut seen = BTreeSet::new();

    for line in contents.lines() {
        let trimmed = line.trim();
        if !in_array {
            if trimmed == "publish_crates=(" {
                in_array = true;
            }
            continue;
        }

        if trimmed == ")" {
            if crates.is_empty() {
                return Err(format!("{relative_path}: publish_crates array is empty").into());
            }
            return Ok(crates);
        }

        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let crate_name = trimmed.trim_matches('"').to_string();
        if !seen.insert(crate_name.clone()) {
            return Err(
                format!("{relative_path}: duplicate publish_crates entry `{crate_name}`").into(),
            );
        }
        crates.push(crate_name);
    }

    Err(format!("{relative_path}: missing publish_crates array").into())
}

fn workspace_packages_by_name<'a>(
    metadata: &'a CargoMetadata,
    workspace_members: &BTreeSet<String>,
) -> BTreeMap<String, &'a CargoPackage> {
    metadata
        .packages
        .iter()
        .filter(|package| workspace_members.contains(&package.id))
        .map(|package| (package.name.clone(), package))
        .collect()
}

fn workspace_packages_by_dir<'a>(
    metadata: &'a CargoMetadata,
    workspace_members: &BTreeSet<String>,
) -> DynResult<BTreeMap<PathBuf, &'a CargoPackage>> {
    let mut packages = BTreeMap::new();
    for package in metadata
        .packages
        .iter()
        .filter(|package| workspace_members.contains(&package.id))
    {
        let dir = package
            .manifest_path
            .parent()
            .ok_or_else(|| format!("{}: manifest path has no parent", package.name))?
            .to_path_buf();
        packages.insert(dir, package);
    }
    Ok(packages)
}

fn publish_order(crates: &[String]) -> DynResult<BTreeMap<String, usize>> {
    let mut order = BTreeMap::new();
    for (index, crate_name) in crates.iter().enumerate() {
        if order.insert(crate_name.clone(), index).is_some() {
            return Err(format!("duplicate publish crate `{crate_name}`").into());
        }
    }
    Ok(order)
}

fn check_publish_crate_metadata(
    repo_root: &Path,
    publish_crates: &[String],
    packages_by_name: &BTreeMap<String, &CargoPackage>,
) -> DynResult<()> {
    for crate_name in publish_crates {
        let package = packages_by_name
            .get(crate_name)
            .ok_or_else(|| format!("publish crate `{crate_name}` is not a workspace package"))?;
        if !package_is_publishable(package) {
            return Err(format!("publish crate `{crate_name}` is marked publish=false").into());
        }
        ensure_nonempty_option(&package.description, &format!("{crate_name} description"))?;
        if package.license.as_deref().unwrap_or("").is_empty()
            && package.license_file.as_deref().unwrap_or("").is_empty()
        {
            return Err(format!("{crate_name}: missing license or license-file").into());
        }
        ensure_nonempty_option(&package.repository, &format!("{crate_name} repository"))?;
        check_publish_readme(repo_root, package)?;
    }

    Ok(())
}

fn check_publish_readme(repo_root: &Path, package: &CargoPackage) -> DynResult<()> {
    let manifest_dir = package
        .manifest_path
        .parent()
        .ok_or_else(|| format!("{}: manifest path has no parent", package.name))?;
    let readme = package.readme.as_deref().unwrap_or("README.md");
    let readme_path = manifest_dir.join(readme);
    if readme_path.exists() {
        return Ok(());
    }

    let relative = readme_path
        .strip_prefix(repo_root)
        .unwrap_or(readme_path.as_path())
        .display();
    Err(format!("{}: missing publish readme `{relative}`", package.name).into())
}

fn check_publish_crate_dependencies(
    publish_order: &BTreeMap<String, usize>,
    packages_by_name: &BTreeMap<String, &CargoPackage>,
    packages_by_dir: &BTreeMap<PathBuf, &CargoPackage>,
) -> DynResult<()> {
    for (crate_name, package) in packages_by_name {
        let Some(package_index) = publish_order.get(crate_name) else {
            continue;
        };
        for dependency in package
            .dependencies
            .iter()
            .filter(|dep| dep.kind.as_deref() != Some("dev"))
        {
            let Some(path) = dependency.path.as_ref() else {
                continue;
            };
            let Some(target) = packages_by_dir.get(path) else {
                return Err(format!(
                    "{crate_name}: workspace path dependency `{}` has no package at {}",
                    dependency.name,
                    path.display()
                )
                .into());
            };
            if !package_is_publishable(target) {
                return Err(format!(
                    "{crate_name}: publishable crate depends on non-publishable workspace crate `{}`",
                    target.name
                )
                .into());
            }
            check_publish_dependency_version(crate_name, target, dependency)?;
            let Some(dep_index) = publish_order.get(&target.name) else {
                return Err(format!(
                    "{crate_name}: publishable dependency `{}` is missing from scripts/publish-crates.sh",
                    target.name
                )
                .into());
            };
            if dep_index >= package_index {
                return Err(format!(
                    "{crate_name}: dependency `{}` must appear earlier in scripts/publish-crates.sh",
                    target.name
                )
                .into());
            }
        }
    }

    Ok(())
}

fn check_publish_dependency_version(
    crate_name: &str,
    target: &CargoPackage,
    dependency: &CargoDependency,
) -> DynResult<()> {
    let caret = format!("^{}", target.version);
    if dependency.req == target.version || dependency.req == caret {
        return Ok(());
    }
    Err(format!(
        "{crate_name}: dependency `{}` uses version requirement `{}`, expected `{}`",
        target.name, dependency.req, caret
    )
    .into())
}

fn package_is_publishable(package: &CargoPackage) -> bool {
    package
        .publish
        .as_ref()
        .map(|registries| !registries.is_empty())
        .unwrap_or(true)
}

fn check_publish_literal_includes(
    publish_crates: &[String],
    packages_by_name: &BTreeMap<String, &CargoPackage>,
) -> DynResult<()> {
    for crate_name in publish_crates {
        let package = packages_by_name
            .get(crate_name)
            .ok_or_else(|| format!("publish crate `{crate_name}` is not a workspace package"))?;
        check_package_literal_includes(package)?;
    }
    Ok(())
}

fn check_package_literal_includes(package: &CargoPackage) -> DynResult<()> {
    let package_dir = package
        .manifest_path
        .parent()
        .ok_or_else(|| format!("{}: manifest path has no parent", package.name))?;
    let src_dir = package_dir.join("src");
    if !src_dir.exists() {
        return Ok(());
    }

    let package_root = package_dir.canonicalize()?;
    for rust_file in rust_files_under(&src_dir)? {
        let source = fs::read_to_string(&rust_file)?;
        for include_path in literal_include_paths(&source) {
            let resolved = rust_file
                .parent()
                .ok_or_else(|| format!("{}: source path has no parent", rust_file.display()))?
                .join(&include_path);
            if !resolved.exists() {
                return Err(format!(
                    "{}: literal include `{}` does not exist",
                    rust_file.display(),
                    include_path
                )
                .into());
            }
            let resolved = resolved.canonicalize()?;
            if !resolved.starts_with(&package_root) {
                return Err(format!(
                    "{}: literal include `{}` points outside publish package root",
                    rust_file.display(),
                    include_path
                )
                .into());
            }
        }
    }

    Ok(())
}

fn rust_files_under(root: &Path) -> DynResult<Vec<PathBuf>> {
    let mut files = Vec::new();
    collect_rust_files(root, &mut files)?;
    Ok(files)
}

fn collect_rust_files(dir: &Path, files: &mut Vec<PathBuf>) -> DynResult<()> {
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_rust_files(&path, files)?;
        } else if path.extension().and_then(|value| value.to_str()) == Some("rs") {
            files.push(path);
        }
    }
    Ok(())
}

fn literal_include_paths(source: &str) -> Vec<String> {
    let mut paths = Vec::new();
    for line in source.lines() {
        for pattern in ["include_str!(\"", "include_bytes!(\""] {
            let Some(start) = line.find(pattern) else {
                continue;
            };
            let tail = &line[start + pattern.len()..];
            if let Some(end) = tail.find('"') {
                paths.push(tail[..end].to_string());
            }
        }
    }
    paths
}

fn check_publish_catalog_sync(repo_root: &Path) -> DynResult<()> {
    let client_catalog = fs::read_to_string(
        repo_root
            .join("crates")
            .join("mesh-client")
            .join("src")
            .join("models")
            .join("catalog.json"),
    )?;
    let node_catalog = fs::read_to_string(
        repo_root
            .join("crates")
            .join("mesh-llm-node")
            .join("src")
            .join("catalog.json"),
    )?;
    ensure_eq(
        &client_catalog,
        &node_catalog,
        "mesh-llm-node packaged catalog copy",
    )
}

fn check_publish_workflow_invariants(repo_root: &Path) -> DynResult<()> {
    let release = fs::read_to_string(repo_root.join("RELEASE.md"))?;
    let release_script = fs::read_to_string(repo_root.join("scripts/release.sh"))?;
    let release_workflow = fs::read_to_string(repo_root.join(".github/workflows/release.yml"))?;
    let pr_quality_workflow =
        fs::read_to_string(repo_root.join(".github/workflows/pr_quality.yml"))?;

    ensure_contains(
        &release,
        "cargo run -p xtask -- repo-consistency publish-crates",
        "RELEASE publish-chain consistency command",
    )?;
    ensure_contains(
        &release_workflow,
        "publish_crates_preflight:",
        "release workflow crates.io preflight job",
    )?;
    ensure_contains(
        &release_workflow,
        "cargo run -p xtask -- repo-consistency publish-crates",
        "release workflow publish-chain consistency check",
    )?;
    ensure_contains(
        &release_workflow,
        "scripts/publish-crates.sh --dry-run --allow-dirty --sleep-seconds 0",
        "release workflow publish-chain dry-run",
    )?;
    let publish_job = workflow_job_section(&release_workflow, "publish")
        .ok_or("release workflow: missing `publish` job for release tag staging check")?;
    ensure_contains(
        publish_job,
        "git add --update",
        "release workflow complete tracked release version staging",
    )?;
    ensure_contains_normalized(
        publish_job,
        "if ! git diff --quiet; then
            echo \"Release preparation left unstaged tracked changes:\" >&2
            git status --short >&2
            exit 1
          fi",
        "release workflow unstaged tracked release change guard",
    )?;
    ensure_contains(
        &release_script,
        "git add --update",
        "local release complete tracked release version staging",
    )?;
    ensure_contains_normalized(
        &release_script,
        "if ! git diff --quiet; then
        git status --short >&2
        die \"release preparation left unstaged tracked changes\"
    fi",
        "local release unstaged tracked release change guard",
    )?;
    ensure_contains_normalized(
        &release_workflow,
        "publish_crates_preflight:
          name: Preflight crates.io packages
          needs: [metadata, publish]
          if: ${{ needs.metadata.outputs.prerelease != 'true' && needs.metadata.outputs.canary != 'true' }}
          runs-on: ubuntu-24.04
          steps:
            - uses: actions/checkout@v5
              with:
                ref: ${{ needs.metadata.outputs.tag }}
                persist-credentials: false
            - uses: dtolnay/rust-toolchain@stable
            - name: Prepare dispatched release version
              if: github.event_name == 'workflow_dispatch'
              env:
                RELEASE_TAG: ${{ needs.metadata.outputs.tag }}
              run: scripts/release-version.sh \"$RELEASE_TAG\"",
        "release workflow publish preflight dispatched version preparation",
    )?;
    ensure_contains(
        &release_workflow,
        "needs: [metadata, publish, publish_crates_preflight]",
        "release workflow real publish preflight dependency",
    )?;
    ensure_contains(
        &pr_quality_workflow,
        "cargo run -p xtask -- repo-consistency publish-crates",
        "PR quality publish-chain drift check",
    )?;

    Ok(())
}
