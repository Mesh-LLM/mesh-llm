//! Product console adapters for the shared model-command presentation.
use std::{future::Future, io::Write, path::PathBuf};

type ByteProgressWriter = fn(&str, u64, u64) -> std::io::Result<()>;

/// A product supplies output destinations and its detected local memory budget.
/// Model policy, result schemas, and command hints remain owned by Skippy.
#[derive(Clone)]
pub struct ModelCommandContext {
    pub program: &'static str,
    pub cache_root: PathBuf,
    pub fit_budget_bytes: u64,
    pub terminal_progress: bool,
    pub byte_progress: Option<ByteProgressWriter>,
    pub console_out: fn() -> Box<dyn Write + Send>,
    pub console_err: fn() -> Box<dyn Write + Send>,
    pub machine_out: fn() -> Box<dyn Write + Send>,
}

tokio::task_local! { static CONTEXT: ModelCommandContext; }

pub(super) async fn scope<T>(context: ModelCommandContext, future: impl Future<Output = T>) -> T {
    CONTEXT.scope(context, future).await
}
pub(super) fn sync_scope<T>(context: ModelCommandContext, operation: impl FnOnce() -> T) -> T {
    CONTEXT.sync_scope(context, operation)
}
pub(super) fn context() -> ModelCommandContext {
    CONTEXT
        .try_with(Clone::clone)
        .unwrap_or_else(|_| standalone_context())
}
pub(super) fn program() -> &'static str {
    CONTEXT
        .try_with(|context| context.program)
        .unwrap_or("skippy")
}
pub(super) fn fit_budget_bytes() -> u64 {
    context().fit_budget_bytes
}
pub(super) fn cache_root() -> PathBuf {
    context().cache_root
}

pub(super) fn standalone_context() -> ModelCommandContext {
    static FIT_BUDGET: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let fit_budget_bytes = *FIT_BUDGET
        .get_or_init(skippy_hardware_profile::model_capacity::local_model_fit_budget_bytes);
    ModelCommandContext {
        program: "skippy",
        cache_root: skippy_model_hf::application_cache_dir(),
        fit_budget_bytes,
        terminal_progress: crate::console::mode() == crate::console::OutputMode::Human
            && crate::console::stderr_is_terminal(),
        byte_progress: if crate::console::mode() == crate::console::OutputMode::Jsonl {
            Some(crate::console::progress)
        } else {
            None
        },
        console_out: crate::console::model_console_out,
        console_err: crate::console::model_console_err,
        machine_out: crate::console::model_machine_out,
    }
}

pub(super) fn console_out() -> Box<dyn Write + Send> {
    (context().console_out)()
}
pub(super) fn console_err() -> Box<dyn Write + Send> {
    (context().console_err)()
}
pub(super) fn machine_out() -> Box<dyn Write + Send> {
    (context().machine_out)()
}

pub(super) use super::progress::{DeterminateProgressLine, clear_stderr_line, start_spinner};

pub(super) fn progress(label: &str, current: u64, total: u64) -> std::io::Result<()> {
    if let Some(writer) = context().byte_progress {
        writer(label, current, total)
    } else {
        super::progress::draw_bytes(label, current, total)
    }
}

pub(super) fn print_or_page(text: &str) -> anyhow::Result<()> {
    if !crate::console::page_model_table(text)? {
        let mut output = console_out();
        output.write_all(text.as_bytes())?;
        output.flush()?;
    }
    Ok(())
}
