use super::{GLOBAL_OUTPUT_MANAGER, OutputEvent, emit_event, write_emergency_event};
use anyhow::Error as AnyhowError;
use std::io;

fn build_fatal_error_event(err: &AnyhowError) -> OutputEvent {
    let message = err.to_string();
    let context = err
        .chain()
        .skip(1)
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    OutputEvent::Fatal {
        message,
        context: (!context.is_empty()).then(|| context.join(": ")),
    }
}

pub fn emit_fatal_error(err: &AnyhowError) -> io::Result<()> {
    emit_event_or_write_emergency(
        build_fatal_error_event(err),
        emit_event,
        global_output_manager_initialized,
        write_emergency_event,
    )
}

pub fn emit_fatal_panic(message: impl Into<String>, context: Option<String>) -> io::Result<()> {
    let event = OutputEvent::Fatal {
        message: message.into(),
        context,
    };
    write_emergency_event(&event)
}

fn global_output_manager_initialized() -> bool {
    GLOBAL_OUTPUT_MANAGER.get().is_some()
}

fn emit_event_or_write_emergency(
    event: OutputEvent,
    emit: impl FnOnce(OutputEvent) -> io::Result<()>,
    output_manager_initialized: impl FnOnce() -> bool,
    write_emergency: impl FnOnce(&OutputEvent) -> io::Result<()>,
) -> io::Result<()> {
    let output_manager_was_initialized = output_manager_initialized();
    match emit(event.clone()) {
        Ok(()) if output_manager_was_initialized => Ok(()),
        Ok(()) | Err(_) => write_emergency(&event),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fatal_error_emits_emergency_event_when_output_worker_fails() {
        let event = OutputEvent::Fatal {
            message: "fatal startup failure".to_string(),
            context: Some("output manager worker unavailable".to_string()),
        };
        let mut emitted_event = None;
        let mut emergency_event = None;

        emit_event_or_write_emergency(
            event.clone(),
            |attempted_event| {
                emitted_event = Some(attempted_event);
                Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "output manager worker unavailable",
                ))
            },
            || true,
            |fallback_event| {
                emergency_event = Some(fallback_event.clone());
                Ok(())
            },
        )
        .expect("emergency fallback should handle failed output worker");

        assert_eq!(emitted_event, Some(event.clone()));
        assert_eq!(emergency_event, Some(event));
    }

    #[test]
    fn fatal_error_emits_emergency_event_when_output_manager_is_missing() {
        let event = OutputEvent::Fatal {
            message: "fatal startup failure".to_string(),
            context: Some("output manager unavailable".to_string()),
        };
        let mut emitted_event = None;
        let mut emergency_event = None;

        emit_event_or_write_emergency(
            event.clone(),
            |attempted_event| {
                emitted_event = Some(attempted_event);
                Ok(())
            },
            || false,
            |fallback_event| {
                emergency_event = Some(fallback_event.clone());
                Ok(())
            },
        )
        .expect("emergency fallback should handle missing output manager");

        assert_eq!(emitted_event, Some(event.clone()));
        assert_eq!(emergency_event, Some(event));
    }
}
