use std::ffi::CString;
use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde::Serialize;

use crate::output::{print_json_pretty, print_success};

#[derive(Debug, Parser)]
pub(crate) struct ValidateProjectorArgs {
    #[arg(long)]
    projector: PathBuf,
    #[arg(long = "no-warmup", action = clap::ArgAction::SetFalse, default_value_t = true)]
    warmup: bool,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Serialize)]
struct ProjectorReport {
    projector: PathBuf,
    warmup: bool,
    loaded: bool,
}

pub(crate) fn run_validate_projector(args: ValidateProjectorArgs) -> Result<()> {
    ensure!(
        args.projector.is_file(),
        "projector does not exist: {}",
        args.projector.display()
    );
    ensure!(
        skippy_ffi::native_runtime_loaded(),
        "validate-projector requires a statically linked standalone build or a loaded native runtime"
    );
    let projector = CString::new(args.projector.to_string_lossy().as_bytes())
        .context("projector path contains an interior NUL byte")?;
    let mut params = unsafe { skippy_ffi::mtmd_context_params_default() };
    params.use_gpu = false;
    params.warmup = args.warmup;
    params.progress_callback = None;
    params.progress_callback_user_data = std::ptr::null_mut();
    let raw =
        unsafe { skippy_ffi::mtmd_init_from_file(projector.as_ptr(), std::ptr::null(), params) };
    ensure!(
        !raw.is_null(),
        "failed to load multimodal projector {}",
        args.projector.display()
    );
    unsafe { skippy_ffi::mtmd_free(raw) };

    let report = ProjectorReport {
        projector: args.projector,
        warmup: args.warmup,
        loaded: true,
    };
    if args.json {
        print_json_pretty(&report)?;
    } else {
        print_success(format!(
            "projector valid: path={} warmup={} loaded=true",
            report.projector.display(),
            report.warmup
        ));
    }
    Ok(())
}
