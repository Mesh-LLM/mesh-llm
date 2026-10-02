use super::recorded_requests::{Selection, Trajectory, build};
use crate::command::DynResult;
use crate::repository::check_args::Grammar;
use crate::repository::check_report::CheckReport;

pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation replay-matrix recorded-requests --input PATH --output PATH --model ID [--max-output-tokens N] [--turn-limit N] [--qualification-probe]",
        values: &[
            "--input",
            "--output",
            "--model",
            "--max-output-tokens",
            "--turn-limit",
        ],
        flags: &["--help", "--qualification-probe"],
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
    let input = parsed.last("--input").ok_or("missing --input")?;
    let output = parsed.last("--output").ok_or("missing --output")?;
    let model = parsed.last("--model").ok_or("missing --model")?;
    let trajectory: Trajectory = serde_json::from_slice(&std::fs::read(input)?)?;
    let selection = Selection {
        model,
        maximum_output_tokens: parsed
            .last("--max-output-tokens")
            .unwrap_or("2048")
            .parse()?,
        turn_limit: parsed.last("--turn-limit").map(str::parse).transpose()?,
        qualification_probe: parsed.flag("--qualification-probe"),
    };
    crate::command::write_json_file(
        std::path::Path::new(output),
        &build(&trajectory, &selection)?,
    )
}
