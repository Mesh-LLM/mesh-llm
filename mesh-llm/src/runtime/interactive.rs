use crate::api::{MeshApi, RuntimeControlRequest};
use crate::cli::output::{
    ConsoleSessionMode, OutputManager, TuiControlFlow, TuiEvent, TuiKeyEvent,
};
use crossterm::event::{
    self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent,
    MouseEventKind,
};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, size};
#[cfg(test)]
use std::io::Write;
use std::io::{BufRead, IsTerminal};
use std::time::Duration;

pub(crate) const HELP_TEXT: &str = "help: h=help, q=quit, i=info snapshot";
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) const READY_PROMPT: &str = "> ";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InitialPromptMode {
    Immediate,
    Deferred,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InteractiveCommand {
    Help,
    Quit,
    Info,
}

fn parse_command(line: &str) -> Option<InteractiveCommand> {
    match line.trim() {
        "h" => Some(InteractiveCommand::Help),
        "q" => Some(InteractiveCommand::Quit),
        "i" => Some(InteractiveCommand::Info),
        _ => None,
    }
}

pub(crate) fn console_session_mode(stdin_is_tty: bool, stderr_is_tty: bool) -> ConsoleSessionMode {
    if stdin_is_tty && stderr_is_tty {
        ConsoleSessionMode::InteractiveDashboard
    } else {
        ConsoleSessionMode::Fallback
    }
}

pub(crate) fn current_console_session_mode() -> ConsoleSessionMode {
    console_session_mode(
        std::io::stdin().is_terminal(),
        std::io::stderr().is_terminal(),
    )
}

#[cfg(test)]
fn write_ready_prompt<W: Write>(writer: &mut W) -> std::io::Result<()> {
    writer.write_all(READY_PROMPT.as_bytes())?;
    writer.flush()
}

#[cfg(test)]
fn maybe_write_initial_prompt<W: Write>(
    writer: &mut W,
    mode: InitialPromptMode,
) -> std::io::Result<()> {
    if matches!(mode, InitialPromptMode::Immediate) {
        write_ready_prompt(writer)?;
    }
    Ok(())
}

pub(crate) fn spawn_handler(
    control_tx: tokio::sync::mpsc::UnboundedSender<RuntimeControlRequest>,
    console_state: MeshApi,
    output_manager: &'static OutputManager,
    initial_prompt_mode: InitialPromptMode,
) {
    match output_manager.console_session_mode() {
        Some(ConsoleSessionMode::InteractiveDashboard) => {
            spawn_tui_handler(control_tx, output_manager)
        }
        _ => spawn_line_handler(
            control_tx,
            console_state,
            output_manager,
            initial_prompt_mode,
        ),
    }
}

fn spawn_line_handler(
    control_tx: tokio::sync::mpsc::UnboundedSender<RuntimeControlRequest>,
    console_state: MeshApi,
    output_manager: &'static OutputManager,
    initial_prompt_mode: InitialPromptMode,
) {
    if matches!(initial_prompt_mode, InitialPromptMode::Immediate) {
        let _ = output_manager.write_ready_prompt();
    }

    let (line_tx, mut line_rx) =
        tokio::sync::mpsc::unbounded_channel::<Result<String, std::io::Error>>();
    if let Err(err) = std::thread::Builder::new()
        .name("mesh-llm-interactive-stdin".to_string())
        .spawn(move || {
            let stdin = std::io::stdin();
            let mut locked = stdin.lock();
            loop {
                let mut line = String::new();
                match locked.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        if line_tx.send(Ok(line)).is_err() {
                            break;
                        }
                    }
                    Err(err) => {
                        let _ = line_tx.send(Err(err));
                        break;
                    }
                }
            }
        })
    {
        tracing::warn!("interactive stdin thread failed to start: {err}");
        return;
    }

    tokio::spawn(async move {
        loop {
            match line_rx.recv().await {
                Some(Ok(line)) => match parse_command(&line) {
                    Some(InteractiveCommand::Help) => {
                        eprintln!("{HELP_TEXT}");
                    }
                    Some(InteractiveCommand::Quit) => {
                        if control_tx.send(RuntimeControlRequest::Shutdown).is_err() {
                            tracing::warn!("interactive shutdown request dropped because runtime control is unavailable");
                        }
                        break;
                    }
                    Some(InteractiveCommand::Info) => {
                        eprintln!("{}", console_state.status_snapshot_string().await);
                    }
                    None => {}
                },
                None => break,
                Some(Err(err)) => {
                    tracing::warn!("interactive stdin read failed: {err}");
                    break;
                }
            }

            if output_manager.ready_prompt_active() {
                let _ = output_manager.write_ready_prompt();
            }
        }
    });
}

fn spawn_tui_handler(
    control_tx: tokio::sync::mpsc::UnboundedSender<RuntimeControlRequest>,
    output_manager: &'static OutputManager,
) {
    let runtime_handle = tokio::runtime::Handle::current();
    if let Err(err) = std::thread::Builder::new()
        .name("mesh-llm-interactive-tui".to_string())
        .spawn(move || {
            if let Err(err) = run_tui_loop(&runtime_handle, control_tx, output_manager) {
                tracing::warn!("interactive pretty loop failed: {err}");
            }
        })
    {
        tracing::warn!("interactive pretty stdin thread failed to start: {err}");
    }
}

fn run_tui_loop(
    runtime_handle: &tokio::runtime::Handle,
    control_tx: tokio::sync::mpsc::UnboundedSender<RuntimeControlRequest>,
    output_manager: &'static OutputManager,
) -> std::io::Result<()> {
    enable_raw_mode().map_err(std::io::Error::other)?;
    let mut shutdown_requested = false;
    let mut shutdown_sent = false;

    let result = (|| -> std::io::Result<()> {
        runtime_handle.block_on(output_manager.enter_tui())?;

        let mut last_size = None;
        if let Ok((columns, rows)) = size() {
            last_size = Some((columns, rows));
            let _ = runtime_handle
                .block_on(output_manager.dispatch_tui_event(TuiEvent::Resize { columns, rows }));
        }

        loop {
            if let Ok((columns, rows)) = size() {
                if Some((columns, rows)) != last_size {
                    last_size = Some((columns, rows));
                    runtime_handle.block_on(
                        output_manager.dispatch_tui_event(TuiEvent::Resize { columns, rows }),
                    )?;
                }
            }

            if !event::poll(Duration::from_millis(50)).map_err(std::io::Error::other)? {
                continue;
            }

            let Some(event) = read_tui_event(event::read().map_err(std::io::Error::other)?) else {
                continue;
            };
            match runtime_handle.block_on(output_manager.dispatch_tui_event(event))? {
                TuiControlFlow::Continue => {}
                TuiControlFlow::Quit => {
                    shutdown_requested = true;
                    if control_tx.send(RuntimeControlRequest::Shutdown).is_err() {
                        tracing::warn!(
                            "interactive shutdown request dropped because runtime control is unavailable"
                        );
                    }
                    shutdown_sent = true;
                    std::thread::sleep(Duration::from_millis(500));
                    if let Err(err) = runtime_handle.block_on(output_manager.render_tui_if_dirty())
                    {
                        tracing::warn!("interactive shutdown frame render failed: {err}");
                    }
                    break;
                }
            }
        }

        Ok(())
    })();

    let exit_result = runtime_handle
        .block_on(output_manager.exit_tui())
        .or_else(|err| {
            tracing::warn!("interactive pretty loop worker cleanup failed: {err}");
            crate::cli::output::force_restore_tui_terminal()
        });
    let raw_result = disable_raw_mode().map_err(std::io::Error::other);

    if shutdown_requested
        && !shutdown_sent
        && control_tx.send(RuntimeControlRequest::Shutdown).is_err()
    {
        tracing::warn!(
            "interactive shutdown request dropped because runtime control is unavailable"
        );
    }

    result?;
    exit_result?;
    raw_result
}

fn read_tui_event(event: Event) -> Option<TuiEvent> {
    match event {
        Event::Resize(columns, rows) => Some(TuiEvent::Resize { columns, rows }),
        Event::Key(KeyEvent {
            code,
            modifiers,
            kind: KeyEventKind::Press,
            ..
        }) => map_key_event(code, modifiers),
        Event::Mouse(MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column,
            row,
            ..
        }) => Some(TuiEvent::MouseDown { column, row }),
        _ => None,
    }
}

fn map_key_event(code: KeyCode, modifiers: KeyModifiers) -> Option<TuiEvent> {
    let key = match code {
        KeyCode::Tab => TuiKeyEvent::Tab,
        KeyCode::BackTab => TuiKeyEvent::BackTab,
        KeyCode::Backspace => TuiKeyEvent::Backspace,
        KeyCode::Enter => TuiKeyEvent::Enter,
        KeyCode::Esc => TuiKeyEvent::Escape,
        KeyCode::Left => TuiKeyEvent::Left,
        KeyCode::Right => TuiKeyEvent::Right,
        KeyCode::Up => TuiKeyEvent::Up,
        KeyCode::Down => TuiKeyEvent::Down,
        KeyCode::PageUp => TuiKeyEvent::PageUp,
        KeyCode::PageDown => TuiKeyEvent::PageDown,
        KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => TuiKeyEvent::Interrupt,
        KeyCode::Char(_ch) if modifiers.contains(KeyModifiers::CONTROL) => return None,
        KeyCode::Char(ch) => TuiKeyEvent::Char(ch),
        _ => return None,
    };
    Some(TuiEvent::Key(key))
}

#[cfg(test)]
mod tests {
    use super::{
        console_session_mode, map_key_event, maybe_write_initial_prompt, parse_command,
        read_tui_event, write_ready_prompt, InitialPromptMode, InteractiveCommand, HELP_TEXT,
        READY_PROMPT,
    };
    use crate::cli::output::{ConsoleSessionMode, TuiEvent, TuiKeyEvent};
    use crossterm::event::{Event, KeyCode, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};

    #[test]
    fn parse_command_accepts_supported_shortcuts() {
        assert_eq!(parse_command("h"), Some(InteractiveCommand::Help));
        assert_eq!(parse_command("q"), Some(InteractiveCommand::Quit));
        assert_eq!(parse_command("i"), Some(InteractiveCommand::Info));
    }

    #[test]
    fn parse_command_trims_surrounding_whitespace() {
        assert_eq!(parse_command("  h  \n"), Some(InteractiveCommand::Help));
        assert_eq!(parse_command("\ti\t"), Some(InteractiveCommand::Info));
    }

    #[test]
    fn parse_command_rejects_other_inputs() {
        assert_eq!(parse_command(""), None);
        assert_eq!(parse_command("help"), None);
        assert_eq!(parse_command("x"), None);
        assert_eq!(HELP_TEXT, "help: h=help, q=quit, i=info snapshot");
    }

    #[test]
    fn ready_prompt_is_exact_raw_prompt_bytes() {
        let mut output = Vec::new();
        write_ready_prompt(&mut output).expect("prompt write should succeed");
        assert_eq!(output, READY_PROMPT.as_bytes());
        assert_eq!(READY_PROMPT, "> ");
    }

    #[test]
    fn deferred_initial_prompt_does_not_write_immediately() {
        let mut output = Vec::new();
        maybe_write_initial_prompt(&mut output, InitialPromptMode::Deferred)
            .expect("deferred prompt should be a no-op");
        assert!(output.is_empty());
    }

    #[test]
    fn immediate_initial_prompt_writes_prompt_bytes() {
        let mut output = Vec::new();
        maybe_write_initial_prompt(&mut output, InitialPromptMode::Immediate)
            .expect("immediate prompt should write prompt bytes");
        assert_eq!(output, READY_PROMPT.as_bytes());
    }

    #[test]
    fn parse_command_quit_alias_still_supported_in_fallback_mode() {
        assert_eq!(parse_command("q\n"), Some(InteractiveCommand::Quit));
    }

    #[test]
    fn tui_uses_interactive_mode_only_when_stdin_and_stderr_are_ttys() {
        assert_eq!(
            console_session_mode(true, true),
            ConsoleSessionMode::InteractiveDashboard
        );
        assert_eq!(
            console_session_mode(true, false),
            ConsoleSessionMode::Fallback
        );
        assert_eq!(
            console_session_mode(false, true),
            ConsoleSessionMode::Fallback
        );
        assert_eq!(
            console_session_mode(false, false),
            ConsoleSessionMode::Fallback
        );
    }

    #[test]
    fn tui_maps_ctrl_c_to_interrupt_quit() {
        assert_eq!(
            map_key_event(KeyCode::Char('c'), KeyModifiers::CONTROL),
            Some(TuiEvent::Key(TuiKeyEvent::Interrupt))
        );
    }

    #[test]
    fn tui_ignores_other_control_chars() {
        assert_eq!(
            map_key_event(KeyCode::Char('l'), KeyModifiers::CONTROL),
            None
        );
    }

    #[test]
    fn tui_maps_left_mouse_down_to_click_coordinates() {
        assert_eq!(
            read_tui_event(Event::Mouse(MouseEvent {
                kind: MouseEventKind::Down(MouseButton::Left),
                column: 42,
                row: 7,
                modifiers: KeyModifiers::empty(),
            })),
            Some(TuiEvent::MouseDown { column: 42, row: 7 })
        );
    }
}
