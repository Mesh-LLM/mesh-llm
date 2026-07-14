use super::*;

mod mcp_atlas;
mod speed_bench;
mod swe_bench_pro;
mod terminal_bench;

pub(super) use self::{
    mcp_atlas::mcp_atlas_command, speed_bench::speed_bench_command,
    swe_bench_pro::swe_bench_pro_command, terminal_bench::terminal_bench_command,
};

pub(super) fn run_command(
    definition: EvalDefinition,
    args: &EvalRunArgs,
    root: &Path,
    run_dir: &Path,
) -> Result<CommandSpec> {
    Ok(match definition.id {
        EvalId::SpeedBench => speed_bench_command(definition, args, root, run_dir)?,
        EvalId::TerminalBench => terminal_bench_command(args, run_dir),
        EvalId::SweBenchPro => swe_bench_pro_command(args, root, run_dir)?,
        EvalId::McpAtlas => mcp_atlas_command(args, root, run_dir)?,
    })
}
