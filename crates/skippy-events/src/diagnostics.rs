//! Process-wide diagnostic output selected by the embedding product.
//!
//! Diagnostics are optional observations. The default sink is silent; product
//! entry points install their renderer before starting serving. This routing is
//! independent of tracing subscribers and their level filters.

use std::{
    io,
    sync::{Arc, OnceLock, RwLock},
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ServingDiagnostic {
    Warning {
        message: String,
        context: Option<String>,
    },
    Status {
        message: String,
    },
}

/// Render a serving diagnostic using the embedding product's output policy.
/// Implementations must be thread-safe and return promptly.
pub trait DiagnosticSink: Send + Sync {
    fn emit(&self, diagnostic: ServingDiagnostic) -> io::Result<()>;
}

fn sink() -> &'static RwLock<Option<Arc<dyn DiagnosticSink>>> {
    static SINK: OnceLock<RwLock<Option<Arc<dyn DiagnosticSink>>>> = OnceLock::new();
    SINK.get_or_init(|| RwLock::new(None))
}

/// Select the process-wide renderer. An embedded product owns this choice;
/// libraries must not install a terminal renderer as a side effect of loading.
pub fn set_diagnostic_sink(renderer: Arc<dyn DiagnosticSink>) {
    *sink()
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(renderer);
}

pub fn emit(diagnostic: ServingDiagnostic) -> io::Result<()> {
    let renderer = sink()
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();
    // Release the registry lock before calling user code.
    match renderer {
        Some(renderer) => renderer.emit(diagnostic),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ReplacingSink;
    impl DiagnosticSink for ReplacingSink {
        fn emit(&self, _: ServingDiagnostic) -> io::Result<()> {
            // An observer may replace itself without deadlocking the registry.
            set_diagnostic_sink(Arc::new(FailingSink));
            Ok(())
        }
    }

    struct FailingSink;
    impl DiagnosticSink for FailingSink {
        fn emit(&self, _: ServingDiagnostic) -> io::Result<()> {
            Err(io::Error::new(io::ErrorKind::BrokenPipe, "closed output"))
        }
    }

    #[test]
    fn callbacks_run_outside_registry_lock_and_preserve_output_errors() {
        let status = || ServingDiagnostic::Status {
            message: "listening".into(),
        };
        assert!(emit(status()).is_ok());
        set_diagnostic_sink(Arc::new(ReplacingSink));
        assert!(emit(status()).is_ok());
        assert_eq!(
            emit(status()).unwrap_err().kind(),
            io::ErrorKind::BrokenPipe
        );
        *sink().write().unwrap() = None;
    }
}
