use crate::command::{DynResult, ensure_contains, ensure_not_contains, workflow_job_section};

pub(super) fn check_release_container_contracts(
    release_workflow: &str,
    configure_sccache_action: &str,
) -> DynResult<()> {
    const REQUIRED_STEP: &str = "Trust checkout directory";
    const REQUIRED_COMMAND: &str = "git config --global --add safe.directory \"$GITHUB_WORKSPACE\"";
    const LOCAL_SCCACHE_ENV: &str = "      SCCACHE_GHA_ENABLED: \"false\"";
    const CONFIGURE_SCCACHE_ACTION: &str = "      - uses: ./.github/actions/configure-sccache-gha";
    const COMPOSE_PRODUCT_ACTION: &str = "uses: ./.github/actions/compose-product-input";
    const PREPARE_RUNTIME_ACTION: &str = "uses: ./.github/actions/prepare-native-runtime-input";
    const PINNED_GITHUB_SCRIPT: &str =
        "uses: actions/github-script@ed597411d8f924073f98dfc5c65a23a2325f34cd";

    for (required, context) in [
        (
            "  SCCACHE_DIR: ${{ github.workspace }}/../.sccache",
            "release workflow sccache disk cache",
        ),
        (
            "  SCCACHE_IGNORE_SERVER_IO_ERROR: \"1\"",
            "release workflow sccache compiler fallback",
        ),
        (
            "  SCCACHE_MULTILEVEL_CHAIN: disk,gha",
            "release workflow sccache cache chain",
        ),
        (
            "  SCCACHE_MULTILEVEL_WRITE_ERROR_POLICY: ignore",
            "release workflow sccache write fallback",
        ),
    ] {
        ensure_contains(release_workflow, required, context)?;
    }

    ensure_contains(
        configure_sccache_action,
        PINNED_GITHUB_SCRIPT,
        "sccache GHA action pinned credential exporter",
    )?;
    ensure_not_contains(
        configure_sccache_action,
        "mozilla-actions/sccache-action",
        "sccache GHA action must use the baked binary",
    )?;
    for (required, context) in [
        (
            "core.exportVariable('ACTIONS_RESULTS_URL'",
            "sccache GHA action cache URL export",
        ),
        (
            "core.exportVariable('ACTIONS_RUNTIME_TOKEN'",
            "sccache GHA action runtime token export",
        ),
        (
            "core.exportVariable('SCCACHE_GHA_ENABLED', 'true')",
            "sccache GHA action remote enable",
        ),
        (
            "core.exportVariable('SCCACHE_GHA_ENABLED', 'false')",
            "sccache GHA action job-local fallback",
        ),
        (
            "core.exportVariable('SCCACHE_IGNORE_SERVER_IO_ERROR', '1')",
            "sccache GHA action compiler fallback",
        ),
        (
            "core.exportVariable('SCCACHE_MULTILEVEL_CHAIN', 'disk,gha')",
            "sccache GHA action cache chain",
        ),
        (
            "process.env.SCCACHE_WEBDAV_ENDPOINT",
            "sccache Depot WebDAV endpoint",
        ),
        ("process.env.DEPOT_CACHE_TOKEN", "sccache Depot job token"),
        (
            "core.exportVariable('SCCACHE_MULTILEVEL_CHAIN', 'disk,webdav')",
            "sccache Depot cache chain",
        ),
        (
            "core.exportVariable('SCCACHE_MULTILEVEL_CHAIN', 'disk')",
            "sccache GHA action disk-only fallback",
        ),
        (
            "core.exportVariable('SCCACHE_MULTILEVEL_WRITE_ERROR_POLICY', 'all')",
            "sccache GHA action synchronous remote writes",
        ),
        ("['--start-server']", "sccache GHA action server start"),
        ("['--stop-server']", "sccache GHA action server stop"),
    ] {
        ensure_contains(configure_sccache_action, required, context)?;
    }

    let container_jobs = release_container_job_names(release_workflow);
    if container_jobs.is_empty() {
        return Err("release workflow: expected at least one container job".into());
    }

    for job_name in container_jobs {
        let job = workflow_job_section(release_workflow, job_name).ok_or_else(|| {
            format!("release workflow: missing `{job_name}` job for container contract check")
        })?;
        ensure_contains(
            job,
            REQUIRED_STEP,
            &format!("release workflow `{job_name}` safe-directory step"),
        )?;
        ensure_contains(
            job,
            REQUIRED_COMMAND,
            &format!("release workflow `{job_name}` safe-directory command"),
        )?;
        let composition_only =
            job.contains(COMPOSE_PRODUCT_ACTION) && !job.contains(PREPARE_RUNTIME_ACTION);
        if composition_only {
            ensure_not_contains(
                job,
                CONFIGURE_SCCACHE_ACTION.trim(),
                &format!(
                    "release workflow `{job_name}` composition must not configure a compiler cache"
                ),
            )?;
            ensure_not_contains(
                job,
                "uses: actions/cache@",
                &format!(
                    "release workflow `{job_name}` composition must not restore a compiler cache"
                ),
            )?;
            continue;
        }
        if !job.lines().any(|line| line == LOCAL_SCCACHE_ENV) {
            return Err(format!(
                "release workflow `{job_name}`: missing job-level `{}`",
                LOCAL_SCCACHE_ENV.trim()
            )
            .into());
        }
        if !job.lines().any(|line| line == CONFIGURE_SCCACHE_ACTION) {
            return Err(format!(
                "release workflow `{job_name}`: missing `{}`",
                CONFIGURE_SCCACHE_ACTION.trim()
            )
            .into());
        }
    }

    Ok(())
}

fn release_container_job_names(release_workflow: &str) -> Vec<&str> {
    release_workflow
        .lines()
        .filter_map(|line| {
            let job_name = line.strip_prefix("  ")?.strip_suffix(':')?;
            if job_name.is_empty() || job_name.starts_with(' ') || job_name.contains(' ') {
                return None;
            }
            let job = workflow_job_section(release_workflow, job_name)?;
            job.lines()
                .any(|job_line| job_line == "    container:")
                .then_some(job_name)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::check_release_container_contracts;

    const VALID_SCCACHE_ACTION: &str = r#"
uses: actions/github-script@ed597411d8f924073f98dfc5c65a23a2325f34cd
core.exportVariable('ACTIONS_RESULTS_URL'
core.exportVariable('ACTIONS_RUNTIME_TOKEN'
core.exportVariable('SCCACHE_GHA_ENABLED', 'true')
core.exportVariable('SCCACHE_GHA_ENABLED', 'false')
core.exportVariable('SCCACHE_IGNORE_SERVER_IO_ERROR', '1')
core.exportVariable('SCCACHE_MULTILEVEL_CHAIN', 'disk,gha')
process.env.SCCACHE_WEBDAV_ENDPOINT
process.env.DEPOT_CACHE_TOKEN
core.exportVariable('SCCACHE_MULTILEVEL_CHAIN', 'disk,webdav')
core.exportVariable('SCCACHE_MULTILEVEL_CHAIN', 'disk')
core.exportVariable('SCCACHE_MULTILEVEL_WRITE_ERROR_POLICY', 'all')
['--start-server']
['--stop-server']
"#;

    const VALID_CONTAINER_WORKFLOW: &str = r#"env:
  SCCACHE_DIR: ${{ github.workspace }}/../.sccache
  SCCACHE_IGNORE_SERVER_IO_ERROR: "1"
  SCCACHE_MULTILEVEL_CHAIN: disk,gha
  SCCACHE_MULTILEVEL_WRITE_ERROR_POLICY: ignore
jobs:
  build_linux_cuda:
    container:
      image: example.invalid/runner@sha256:digest
    env:
      SCCACHE_GHA_ENABLED: "false"
    steps:
      - uses: actions/checkout@v5
      - name: Trust checkout directory
        run: git config --global --add safe.directory "$GITHUB_WORKSPACE"
      - uses: ./.github/actions/configure-sccache-gha
  publish:
    runs-on: ubuntu-24.04
"#;

    const VALID_COMPOSITION_CONTAINER_WORKFLOW: &str = r#"env:
  SCCACHE_DIR: ${{ github.workspace }}/../.sccache
  SCCACHE_IGNORE_SERVER_IO_ERROR: "1"
  SCCACHE_MULTILEVEL_CHAIN: disk,gha
  SCCACHE_MULTILEVEL_WRITE_ERROR_POLICY: ignore
jobs:
  compose_linux_cuda:
    container:
      image: example.invalid/runner@sha256:digest
    steps:
      - uses: actions/checkout@v5
      - name: Trust checkout directory
        run: git config --global --add safe.directory "$GITHUB_WORKSPACE"
      - uses: ./.github/actions/compose-product-input
  publish:
    runs-on: ubuntu-24.04
"#;

    #[test]
    fn release_container_contract_accepts_remote_sccache_with_local_fallback() {
        check_release_container_contracts(VALID_CONTAINER_WORKFLOW, VALID_SCCACHE_ACTION).unwrap();
    }

    #[test]
    fn release_container_contract_accepts_cache_free_product_composition() {
        check_release_container_contracts(
            VALID_COMPOSITION_CONTAINER_WORKFLOW,
            VALID_SCCACHE_ACTION,
        )
        .unwrap();
    }

    #[test]
    fn release_container_contract_requires_safe_checkout() {
        let workflow = VALID_CONTAINER_WORKFLOW.replace(
            "      - name: Trust checkout directory\n        run: git config --global --add safe.directory \"$GITHUB_WORKSPACE\"\n",
            "",
        );

        let error = check_release_container_contracts(&workflow, VALID_SCCACHE_ACTION).unwrap_err();
        assert!(error.to_string().contains("safe-directory"));
    }

    #[test]
    fn release_container_contract_requires_job_local_sccache() {
        let workflow =
            VALID_CONTAINER_WORKFLOW.replace("      SCCACHE_GHA_ENABLED: \"false\"\n", "");

        let error = check_release_container_contracts(&workflow, VALID_SCCACHE_ACTION).unwrap_err();
        assert!(error.to_string().contains("SCCACHE_GHA_ENABLED"));
    }

    #[test]
    fn release_container_contract_requires_sccache_gha_configuration() {
        let workflow = VALID_CONTAINER_WORKFLOW.replace(
            "      - uses: ./.github/actions/configure-sccache-gha\n",
            "",
        );

        let error = check_release_container_contracts(&workflow, VALID_SCCACHE_ACTION).unwrap_err();
        assert!(error.to_string().contains("configure-sccache-gha"));
    }

    #[test]
    fn release_container_contract_requires_sccache_job_local_fallback() {
        let action =
            VALID_SCCACHE_ACTION.replace("core.exportVariable('SCCACHE_GHA_ENABLED', 'false')", "");

        let error =
            check_release_container_contracts(VALID_CONTAINER_WORKFLOW, &action).unwrap_err();
        assert!(error.to_string().contains("job-local fallback"));
    }

    #[test]
    fn release_container_contract_requires_fail_open_sccache_writes() {
        let workflow = VALID_CONTAINER_WORKFLOW
            .replace("  SCCACHE_MULTILEVEL_WRITE_ERROR_POLICY: ignore\n", "");

        let error = check_release_container_contracts(&workflow, VALID_SCCACHE_ACTION).unwrap_err();
        assert!(error.to_string().contains("write fallback"));
    }

    #[test]
    fn release_container_contract_requires_disk_first_sccache_chain() {
        let action = VALID_SCCACHE_ACTION.replace(
            "core.exportVariable('SCCACHE_MULTILEVEL_CHAIN', 'disk,gha')",
            "",
        );

        let error =
            check_release_container_contracts(VALID_CONTAINER_WORKFLOW, &action).unwrap_err();
        assert!(error.to_string().contains("cache chain"));
    }
}
