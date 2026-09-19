use crate::cli::NativeRuntimeArgs;
use anyhow::Result;
use skippy_api::native_runtime::NativeRuntimeOptions;
use skippy_commands::runtime::RuntimeRunOptions;

pub fn resolve_options(args: NativeRuntimeArgs) -> Result<NativeRuntimeOptions> {
    let mut options: NativeRuntimeOptions = args.into();
    if options.cache_dir.is_none() {
        options.cache_dir = Some(skippy_config::paths::native_runtime_cache_default()?);
    }
    options
        .bundle_dirs
        .extend(skippy_config::paths::native_runtime_bundle_dirs_from_env());
    Ok(options)
}

/// Explicit conversion from the resolved native runtime options; a free
/// function because both endpoint types are foreign to this crate (orphan
/// rule).
pub fn command_options(options: &NativeRuntimeOptions) -> RuntimeRunOptions {
    RuntimeRunOptions {
        cache_dir: options.cache_dir.clone(),
        release: options.release.clone(),
        bundle_dirs: options.bundle_dirs.clone(),
        selection: options.selection.clone(),
    }
}
