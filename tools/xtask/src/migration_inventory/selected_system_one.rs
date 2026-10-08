use crate::command::DynResult;

const CALLER: &str = "scripts/skippy-system-one-smoke.sh";

// These are caller admission boundaries, not a general Bash parser. Minimal
// mixed-driver fixtures remain independently admitted by the existing binding.
pub(super) fn check_shape(path: &str, lines: &[&str]) -> DynResult<()> {
    if (path != CALLER && path != "skippy/scripts/skippy-system-one-smoke.sh")
        || !lines
            .iter()
            .any(|line| line.trim().starts_with("SMOKE_MANIFEST="))
    {
        return Ok(());
    }
    let source = lines
        .iter()
        .map(|line| line.trim())
        .collect::<Vec<_>>()
        .join("\n");
    for required in [
        "pick_port() {\n\"${automation[@]}\" automation local-ports 1\n}",
        "write_stage_config() {\n\"${automation[@]}\" automation system-one-smoke stage \"$@\"\n}",
        "port=\"$(pick_port)\" || return 2",
        "write_stage_config \"$config\" \"$model_id\" \"$model_path\" \"$(jq -r '.sha256' <<<\"$summary\")\" \\\n\"$layer_end\" \"127.0.0.1:${port}\" 1 \"$CTX_SIZE\" \"$n_batch\" \"$gpu_layers\" || return 2",
        "if [[ \"$CASES_DRIVER\" != /* || ! -f \"$CASES_DRIVER\" || ! -x \"$CASES_DRIVER\" || -L \"$CASES_DRIVER\" ]]; then",
        "\"${automation[@]}\" automation system-one-smoke report \\\n\"$REPORT_PATH\" \"$status\" \"$contract_status\" \"$read_status\" \\\n\"$READ_ARTIFACT_ID\" \"$BUILD_BACKEND\" \"$READ_RESOLVED_ARTIFACT_PATH\" \\\n\"$READ_ARTIFACT_CACHE_CHECKED\" \"$CERTIFIED_BACKENDS\" \\\n\"$REQUIRE_QUALIFIED\" \"$SKIP_CONTRACT\" \"$REPORT_REASONS\"",
    ] {
        if source.matches(required).count() != 1 {
            return Err(
                "selected process: changed System One serializer admission boundary".into(),
            );
        }
    }
    for definition in ["pick_port() {", "write_stage_config() {"] {
        if lines
            .iter()
            .filter(|line| line.trim() == definition)
            .count()
            != 1
        {
            return Err("selected process: duplicate System One serializer helper".into());
        }
    }
    if lines
        .iter()
        .any(|line| line.trim() == "require_cmd python3 || exit 2")
    {
        return Err("selected process: obsolete System One interpreter admission".into());
    }
    Ok(())
}
