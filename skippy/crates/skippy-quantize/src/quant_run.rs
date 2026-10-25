//! Resume a quantization manifest within an explicitly admitted optional split range.
use crate::{
    RunQuantWindowArgs,
    manifest::read_manifest,
    splits::{SplitWindow, validate_split_window},
};
use anyhow::{Result, ensure};
use clap::{Args as ClapArgs, Parser};

#[derive(Debug, Default, ClapArgs)]
pub(crate) struct RequestedRange {
    /// First split in the requested inclusive range; requires --last-split.
    #[arg(long, requires = "last_split", value_parser = clap::value_parser!(u32).range(1..))]
    first_split: Option<u32>,
    /// Last split in the requested inclusive range; requires --first-split.
    #[arg(long, requires = "first_split", value_parser = clap::value_parser!(u32).range(1..))]
    last_split: Option<u32>,
}

impl RequestedRange {
    fn selected(&self) -> bool {
        self.first_split.is_some() || self.last_split.is_some()
    }

    fn admit(&self, expected_splits: u32) -> Result<SplitWindow> {
        let (Some(first_split), Some(last_split)) = (self.first_split, self.last_split) else {
            anyhow::bail!("--first-split and --last-split must be supplied together");
        };
        let window = SplitWindow {
            first_split,
            last_split,
        };
        validate_split_window(window, expected_splits)?;
        Ok(window)
    }
}

#[derive(Debug, Parser)]
pub(crate) struct RunQuantArgs {
    #[command(flatten)]
    pub(crate) window: RunQuantWindowArgs,
    #[arg(skip)]
    pub(crate) window_override: Option<SplitWindow>,
    #[command(flatten)]
    pub(crate) requested_range: RequestedRange,
    #[arg(long)]
    pub(crate) max_windows: Option<u32>,
}

fn selected_window(args: &RunQuantArgs) -> Result<Option<SplitWindow>> {
    if !args.requested_range.selected() {
        // Preserve the existing internal override/default path without new manifest reads.
        return Ok(args.window_override);
    }
    ensure!(
        args.window_override.is_none(),
        "CLI split range conflicts with internal window override"
    );
    let manifest = read_manifest(&args.window.manifest)?;
    Ok(Some(args.requested_range.admit(manifest.expected_splits)?))
}

pub(crate) fn run_quant(args: RunQuantArgs) -> Result<()> {
    let manifest_path = args.window.manifest.clone();
    crate::with_manifest_lock(&manifest_path, || run_quant_unlocked(args))
}

pub(crate) fn run_quant_unlocked(args: RunQuantArgs) -> Result<()> {
    ensure!(
        !args.window.runner.print_only,
        "run-quant does not support --print-only; use run-quant-window"
    );
    let window_override = selected_window(&args)?;
    if args.window.runner.dry_run {
        return crate::run_quant_window_once(&args.window, window_override).map(|_| ());
    }
    crate::run_window_loop("quant", args.max_windows, || {
        crate::run_quant_window_once(&args.window, window_override)
    })
}

#[cfg(test)]
#[path = "quant_run_tests.rs"]
mod tests;
