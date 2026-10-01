mod class;
mod command;
mod document;
mod error;
mod metrics;
mod request;
pub(super) mod serialization;
mod verify;
mod write;

pub(crate) use class::WorkloadClass;
pub(crate) use error::Error;
pub(crate) use request::{VerifyRequest, WriteRequest};
pub(crate) use verify::verify_evidence;
pub(crate) use write::write_evidence;

pub(crate) fn run(args: &[String]) -> crate::command::DynResult<()> {
    command::run(args)
}

#[cfg(test)]
pub(crate) fn validate_tts_metrics(bytes: &[u8]) -> Result<(), Error> {
    with_worker(|| metrics::PcmMetrics::parse(bytes).map(|_| ()))
}

fn with_worker(operation: impl FnOnce() -> Result<(), Error> + Send) -> Result<(), Error> {
    std::thread::scope(|scope| {
        let worker = std::thread::Builder::new()
            .name("workload-oracle-evidence".into())
            .stack_size(64 * 1024 * 1024)
            .spawn_scoped(scope, operation)
            .map_err(Error::Worker)?;
        match worker.join() {
            Ok(result) => result,
            Err(panic) => std::panic::resume_unwind(panic),
        }
    })
}

#[cfg(test)]
#[path = "../../../tests/migration_workload_oracle_evidence/mod.rs"]
mod migration_workload_oracle_evidence;
