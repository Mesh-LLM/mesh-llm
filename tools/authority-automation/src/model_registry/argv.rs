//! argparse-compatible diagnostics for the two legacy entry points: the
//! `<prog>: error:` prefix and the quoted value of an ignored flag argument.

use ci_controller::repository::check_args::{Grammar, ParsedArgs};
use ci_controller::repository::check_report::CheckReport;
use ci_controller::repository::text::repr;

pub fn parse(grammar: &Grammar, program: &str, args: &[String]) -> Result<ParsedArgs, CheckReport> {
    let flag_value = args.iter().find_map(|arg| {
        let (name, value) = arg.split_once('=')?;
        grammar.flags.contains(&name).then_some((name, value))
    });
    if let Some((name, value)) = flag_value {
        let message = format!("argument {name}: ignored explicit argument {}", repr(value));
        return Err(usage(grammar, program, &message));
    }
    grammar
        .parse(args)
        .map_err(|report| with_program(report, program))
}

pub fn usage(grammar: &Grammar, program: &str, message: &str) -> CheckReport {
    with_program(grammar.error(message), program)
}

fn with_program(report: CheckReport, program: &str) -> CheckReport {
    CheckReport {
        stderr: report
            .stderr
            .replacen("\nerror: ", &format!("\n{program}: error: "), 1),
        ..report
    }
}
