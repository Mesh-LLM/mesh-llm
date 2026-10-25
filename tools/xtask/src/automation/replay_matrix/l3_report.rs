use super::l3_contract::Run;
use crate::{
    command::DynResult,
    repository::{check_args::Grammar, check_report::CheckReport},
};
use std::path::{Path, PathBuf};
pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation replay-matrix l3-report --artifact PATH",
        values: &["--artifact"],
        flags: &["--help"],
    };
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage)).emit();
    }
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error("unexpected positional arguments").emit();
    }
    let output = Path::new(parsed.last("--artifact").ok_or("missing --artifact")?);
    let document: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output.join("run.json"))?)?;
    let path = if document.get("owner").is_some() || document["config"].get("prompt_min").is_some()
    {
        let mut run: Run = serde_json::from_value(document)?;
        run.gates = Some(super::l3_gates::evaluate(&run)?);
        write(output, &run)?
    } else {
        legacy(output, document)?
    };
    CheckReport::success(format!("{}\n", path.display())).emit()
}
fn text(value: &str) -> String {
    value.replace(['\n', '\r'], " ").replace('`', "'")
}
pub(super) fn write(output: &Path, run: &Run) -> DynResult<PathBuf> {
    let gates = run.gates.as_ref().ok_or("missing disk-L3 gates")?;
    render(
        output,
        &run.build,
        &run.config.model,
        &run.config.backend,
        gates,
    )
}
pub(super) fn legacy(output: &Path, document: serde_json::Value) -> DynResult<PathBuf> {
    if document["schema_version"] != 1 || document["kind"] != "disk-l3-lifecycle" {
        return Err("unsupported legacy disk-L3 artifact".into());
    }
    let gates: super::l3_contract::Gates = serde_json::from_value(document["gates"].clone())?;
    if !gates.evaluated
        || gates.checks.is_empty()
        || gates.passed != gates.checks.iter().all(|check| check.passed)
    {
        return Err("invalid retained legacy gate receipt".into());
    }
    render(
        output,
        &document["build"],
        document["config"]["model"]
            .as_str()
            .ok_or("missing legacy model")?,
        document["config"]["backend"]
            .as_str()
            .ok_or("missing legacy backend")?,
        &gates,
    )
}
fn render(
    output: &Path,
    build: &serde_json::Value,
    model: &str,
    backend: &str,
    gates: &super::l3_contract::Gates,
) -> DynResult<PathBuf> {
    let mut lines = vec![
        "# Disk L3 KV cache lifecycle certification".into(),
        String::new(),
        format!(
            "- Commit: `{}`",
            text(build["commit"].as_str().ok_or("missing build commit")?)
        ),
        format!("- Model: `{}`", text(model)),
        format!("- Backend: `{}`", text(backend)),
        format!(
            "- Result: **{}**",
            if gates.passed { "PASS" } else { "FAIL" }
        ),
        format!("- Cold TTFT p50: {:?}s", gates.cold_ttft_p50_seconds),
        format!(
            "- Restart L3 TTFT p50: {:?}s",
            gates.restart_l3_ttft_p50_seconds
        ),
        format!("- Restart/cold ratio: {:?}", gates.restart_l3_ttft_ratio),
        String::new(),
        "## Gates".into(),
        String::new(),
    ];
    for gate in &gates.checks {
        lines.push(format!(
            "- **{}** `{}` — {}",
            if gate.passed { "PASS" } else { "FAIL" },
            text(&gate.name),
            text(&gate.detail)
        ));
    }
    let path = output.join("REPORT.md");
    std::fs::write(&path, format!("{}\n", lines.join("\n")))?;
    super::report::inventory(output)?;
    Ok(path)
}
