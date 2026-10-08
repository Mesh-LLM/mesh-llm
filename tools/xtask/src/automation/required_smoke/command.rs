use super::{
    Attestation, Budget, Check, Response, Session, Transfer, args::Options, failure::Failure,
    output,
};
use crate::{automation::command_interrupt::Interrupt, command::DynResult, process::Cancellation};
use std::{
    io::{self, Write},
    path::Path,
    time::Instant,
};

pub(crate) const USAGE: &str = "automation required-smoke run --binary <absolute path> --model <id or path> --native-runtime-root <directory> [--model-class dense|recurrent] [--stack default|constrained] [--device CPU|MTL0|CUDA0] [--mmproj <path>] [--public-key-file <path>] [--expected-attestation valid|missing|invalid] [--ready-max-wait <1..3600>] [--shutdown-max-wait <1..3600>] [--state-parent <directory>]";

pub(crate) fn run(root: &Path, args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        writeln!(io::stdout().lock(), "{USAGE}")?;
        return Ok(());
    }
    let options = Options::parse(args)?;
    let interrupt = Interrupt::install()?;
    let result = execute(root, &options, &interrupt.cancellation());
    let result = finalize(result, interrupt.finish());
    match result {
        Ok(receipt) => {
            serde_json::to_writer(io::stdout().lock(), &receipt)?;
            writeln!(io::stdout().lock())?;
            Ok(())
        }
        Err(error) => {
            output::reject(error.as_ref())?;
            Err(error)
        }
    }
}

pub(super) fn finalize(
    result: DynResult<output::Receipt>,
    interruption: Result<(), crate::automation::command_interrupt::Reason>,
) -> DynResult<output::Receipt> {
    match interruption {
        Ok(()) => result,
        Err(reason) => Err(Failure::Interrupt {
            reason,
            preceding: result,
        }
        .into()),
    }
}

pub(super) fn execute(
    root: &Path,
    options: &Options,
    cancellation: &Cancellation,
) -> DynResult<output::Receipt> {
    let started = Instant::now();
    let attestation = match &options.key {
        None => Attestation::Disabled,
        Some(_) => Attestation::Required {
            expected: options.expected.clone(),
        },
    };
    let mut session = Session::new(
        options.variant,
        attestation,
        Budget::new(options.readiness)?,
    );
    if let Some(key) = &options.key {
        let summary = crate::attestation::inspect_release_attestation_summary(
            &crate::attestation::InspectArgs {
                binary: Some(options.binary.clone()),
                public_key_file: Some(key.clone()),
                json: true,
            },
        )?;
        let body = serde_json::to_vec(&summary)?;
        session.observe(
            (
                Check::InspectAttestation,
                Transfer::Complete(Response {
                    status: 200,
                    body: &body,
                }),
            ),
            started.elapsed(),
        )?;
    }
    if cancellation.is_cancelled() {
        session.cancel()?;
    }
    super::coordinated::execute((root, options, cancellation), session, started)
}
