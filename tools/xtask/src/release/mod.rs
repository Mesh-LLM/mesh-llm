//! `release`: Rust owners of the release-notes helper scripts. Each command
//! keeps its legacy script's argv, streams, exit statuses and written files.
//!
//! - `notes-base` replaces `scripts/select-release-notes-base.py`.
//! - `notes-link` replaces `scripts/release-notes-link.py`; its `git` and `gh`
//!   calls go through [`link_host::ReleaseHost`].
//! - `notes-classify` replaces `scripts/release-notes-classify.py`; its `git log`
//!   goes through the same host.

mod classify;
mod classify_argv;
mod classify_rules;
mod classify_subject;
mod link;
mod link_argv;
mod link_body;
mod link_commits;
mod link_gh;
mod link_host;
mod notes_base;
mod preflight;
mod python_failure;
mod regroup;
mod regroup_argv;
mod regroup_body;
mod regroup_date;
mod regroup_plan;
mod regroup_py;

use crate::command::DynResult;
use std::io::Read as _;

/// A `release` subcommand.
#[derive(Clone, Copy)]
pub(crate) enum ReleaseCommand {
    Base,
    Link,
    Classify,
    Regroup,
    VersionAtLeast,
}

impl ReleaseCommand {
    pub(crate) fn parse(name: &str) -> Option<Self> {
        match name {
            "notes-base" => Some(Self::Base),
            "notes-link" => Some(Self::Link),
            "notes-classify" => Some(Self::Classify),
            "notes-regroup" => Some(Self::Regroup),
            "version-at-least" => Some(Self::VersionAtLeast),
            _ => None,
        }
    }
}

pub(crate) fn run(command: ReleaseCommand, args: &[String]) -> DynResult<()> {
    let report = match command {
        ReleaseCommand::Base => notes_base::run(args, || {
            let mut bytes = Vec::new();
            std::io::stdin().read_to_end(&mut bytes).map(|_| bytes)
        }),
        ReleaseCommand::Link => link::run(args, &mut link_host::SystemHost),
        ReleaseCommand::Classify => classify::run(args, &mut link_host::SystemHost),
        ReleaseCommand::Regroup => regroup::run(args),
        ReleaseCommand::VersionAtLeast => preflight::run(args),
    };
    report.emit()
}
