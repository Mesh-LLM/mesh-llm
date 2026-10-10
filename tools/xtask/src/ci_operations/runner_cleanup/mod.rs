mod admission;
mod boundary;
pub(super) mod command;
mod deletion;
mod error;
mod finalization;
mod git;
mod paths;
mod plan;
mod replay;

pub(crate) use boundary::{Options, Profile};
pub(crate) use error::Error;
pub(crate) use git::Git;
pub(crate) use plan::{Plan, Target};

#[cfg(test)]
mod finalization_tests;
#[cfg(all(test, unix))]
mod fixture_git_admission;
#[cfg(test)]
mod fixture_guard_tests;
#[cfg(test)]
mod fixtures;
#[cfg(test)]
mod plan_tests;
#[cfg(all(test, unix))]
mod replay_admission_tests;
#[cfg(all(test, unix))]
mod replay_tests;
#[cfg(test)]
mod safety_tests;

pub(crate) fn run(
    options: &Options,
    environment: &std::collections::BTreeMap<std::ffi::OsString, std::ffi::OsString>,
    adapter: (Option<&Git>, &mut impl std::io::Write),
) -> Result<(), error::Failure<()>> {
    let (git, output) = adapter;
    let plan = boundary::from_environment(options, environment)?;
    let admitted = admission::admit(&plan)?;
    let interrupt = crate::command_interrupt::Interrupt::install().map_err(Error::Interrupt)?;
    let result = deletion::execute(admitted, git, (&interrupt, output));
    finalization::finalize(result, interrupt.finish())
}
