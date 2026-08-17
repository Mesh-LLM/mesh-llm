use super::*;

pub(crate) struct TuneOutputRequest<'a> {
    pub(crate) command: &'static str,
    pub(crate) json_output: bool,
    pub(crate) launch_args: bool,
    pub(crate) config: &'a mesh_llm_config::MeshConfig,
    pub(crate) apply_mode: TuneApplyMode,
    pub(crate) prepared: &'a [crate::gpus::tune_apply::PreparedTunePlan],
    pub(crate) target_failures: &'a [TuneTargetFailure],
    pub(crate) global_blockers: &'a [String],
    pub(crate) benchmark_reports: &'a [TuneBenchmarkTargetReport],
}

pub(crate) fn emit_tune_output(
    writer: &mut impl std::io::Write,
    request: TuneOutputRequest<'_>,
) -> anyhow::Result<()> {
    let report = build_tune_run_report(
        request.command,
        request.config,
        request.apply_mode,
        request.prepared,
        request.target_failures,
        request.global_blockers,
        request.benchmark_reports,
    );
    if request.json_output {
        serde_json::to_writer_pretty(&mut *writer, &report)?;
        writeln!(writer)?;
        return Ok(());
    }
    let rendered = if request.launch_args {
        render_tune_launch_args_output(&report)
    } else {
        render_tune_human_output(&report)
    };
    write!(writer, "{rendered}")?;
    Ok(())
}
