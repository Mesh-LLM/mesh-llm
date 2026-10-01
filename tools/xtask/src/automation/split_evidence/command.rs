use super::{
    Error,
    args::{self, Mode, Request},
    boundary, evidence, reconcile, serialization, storage, verify,
};
use crate::automation::codepoint_json::parser;
use std::fs;
use std::io::Write;

pub(crate) const USAGE: &str = "cargo xtool automation split-evidence --seed-status <file> --seed-stages <file> --seed-models <file> --worker-status <file> --worker-stages <file> --worker-models <file> --model-label <label> {--output <file>|--verify <file>}";

pub(crate) fn run(args: &[String]) -> crate::command::DynResult<()> {
    let request = args::parse(args)?;
    let outcome = execute(&request)?;
    crate::cli_output::stdout().write_all(outcome.as_bytes())?;
    Ok(())
}

pub(super) fn execute(request: &Request) -> Result<String, Error> {
    std::thread::scope(|scope| {
        let worker = std::thread::Builder::new()
            .name("split-evidence".into())
            .stack_size(64 * 1024 * 1024)
            .spawn_scoped(scope, || execute_inner(request))?;
        match worker.join() {
            Ok(result) => result,
            Err(panic) => std::panic::resume_unwind(panic),
        }
    })
}

fn execute_inner(request: &Request) -> Result<String, Error> {
    let label = crate::repository::python_text::strip(&request.model_label);
    if label.is_empty() {
        return Err(Error::Contract("model label must be non-empty".into()));
    }
    let result = storage::load(&request.paths).and_then(|snapshots| {
        let ready = reconcile::reconcile(&snapshots)?;
        let evidence = evidence::ready(&ready, &snapshots, label);
        Ok((ready, evidence))
    });
    let (ready, evidence) = match result {
        Ok(result) => result,
        Err(error) => {
            if let Mode::Output(path) = &request.mode {
                storage::write(
                    path,
                    &serialization::render(&evidence::failed(label, &error.to_string())),
                )?;
            }
            return Err(Error::Contract(format!(
                "two-node split evidence reconciliation failed: {error}"
            )));
        }
    };
    match &request.mode {
        Mode::Output(path) => {
            storage::write(path, &serialization::render(&evidence))?;
            Ok(format!(
                "ready=true topology={} run={} model={} stages=2 observers=2\n",
                ready.topology.identity.topology_id.display(),
                ready.topology.identity.run_id.display(),
                ready.model.display()
            ))
        }
        Mode::Verify(path) => {
            let actual = fs::read(path)
                .map_err(|error| error.to_string())
                .and_then(|raw| parser::parse(&raw))
                .and_then(|actual| {
                    boundary::object(&actual, "split evidence")
                        .map_err(|error| error.to_string())?;
                    Ok(actual)
                })
                .map_err(|error| {
                    Error::Contract(format!(
                        "cannot verify two-node split evidence {}: {error}",
                        path.display()
                    ))
                })?;
            if !verify::equal(&actual, &evidence) {
                return Err(Error::Contract(format!(
                    "two-node split evidence does not match persisted snapshots: {}",
                    path.display()
                )));
            }
            Ok(format!(
                "Verified two-node split evidence: {}\n",
                path.display()
            ))
        }
    }
}
