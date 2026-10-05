#![forbid(unsafe_code)]

use std::{panic::PanicHookInfo, sync::Once};

pub mod output;
pub mod terminal_progress;

pub use output::*;

static PANIC_HOOK: Once = Once::new();

pub fn install_terminal_panic_hook() {
    PANIC_HOOK.call_once(|| {
        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            output::force_restore_tui_after_panic();
            let _ = output::emit_fatal_panic(panic_message(info), panic_context(info));
            previous_hook(info);
        }));
    });
}

fn panic_message(info: &PanicHookInfo<'_>) -> String {
    if let Some(message) = info.payload().downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = info.payload().downcast_ref::<String>() {
        message.clone()
    } else {
        "panic occurred".to_string()
    }
}

fn panic_context(info: &PanicHookInfo<'_>) -> Option<String> {
    info.location()
        .map(|location| format!("panic at {}:{}", location.file(), location.line()))
}

#[cfg(test)]
mod tests {
    use super::install_terminal_panic_hook;
    use std::{
        panic::{self, AssertUnwindSafe},
        sync::{
            Arc, Mutex,
            atomic::{AtomicUsize, Ordering},
        },
    };

    #[test]
    fn install_terminal_panic_hook_chains_previous_hook() {
        let previous_hook = panic::take_hook();
        let previous_hook_calls = Arc::new(AtomicUsize::new(0));
        let previous_payload = Arc::new(Mutex::new(None));
        let previous_location = Arc::new(Mutex::new(None));

        let hook_calls = Arc::clone(&previous_hook_calls);
        let hook_payload = Arc::clone(&previous_payload);
        let hook_location = Arc::clone(&previous_location);
        panic::set_hook(Box::new(move |info| {
            hook_calls.fetch_add(1, Ordering::SeqCst);
            let payload = info
                .payload()
                .downcast_ref::<&str>()
                .map(|message| (*message).to_string())
                .or_else(|| info.payload().downcast_ref::<String>().cloned());
            *hook_payload.lock().expect("payload lock") = payload;
            *hook_location.lock().expect("location lock") =
                info.location().map(|location| location.file().to_string());
        }));

        install_terminal_panic_hook();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            panic!("panic hook smoke test");
        }));

        panic::set_hook(previous_hook);

        assert!(result.is_err());
        assert_eq!(previous_hook_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            previous_payload.lock().expect("payload lock").as_deref(),
            Some("panic hook smoke test")
        );
        assert_eq!(
            previous_location.lock().expect("location lock").as_deref(),
            Some(file!())
        );
    }
}
