use anyhow::{Context, Result};
use crossterm::terminal::size as terminal_size;
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    widgets::{LineGauge, Widget},
};
use std::io::Write;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::Duration;

const INLINE_GAUGE_WIDTH: u16 = 96;
const INLINE_GAUGE_MIN_BAR_WIDTH: usize = 24;
const INLINE_GAUGE_WRAP_GUARD_WIDTH: u16 = 1;

pub fn clear_stderr_line() -> Result<()> {
    if crate::json_mode_enabled() {
        return Ok(());
    }
    eprint!("\r\x1b[2K");
    std::io::stderr()
        .flush()
        .context("Flush terminal progress clear")?;
    Ok(())
}

pub struct SpinnerHandle {
    done: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl SpinnerHandle {
    pub fn finish(&mut self) {
        self.done.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        let _ = clear_stderr_line();
    }
}

impl Drop for SpinnerHandle {
    fn drop(&mut self) {
        self.finish();
    }
}

pub fn start_spinner(message: &str) -> SpinnerHandle {
    if crate::json_mode_enabled() {
        return SpinnerHandle {
            done: Arc::new(AtomicBool::new(true)),
            thread: None,
        };
    }
    let done = Arc::new(AtomicBool::new(false));
    let done_thread = Arc::clone(&done);
    let message = Arc::new(Mutex::new(message.to_string()));
    let message_thread = Arc::clone(&message);
    let thread = thread::spawn(move || {
        let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let mut index = 0usize;
        while !done_thread.load(Ordering::Relaxed) {
            let current = message_thread
                .lock()
                .map(|guard| guard.clone())
                .unwrap_or_else(|_| "Working".to_string());
            eprint!("\r\x1b[2K{} {}", frames[index % frames.len()], current);
            let _ = std::io::stderr().flush();
            index += 1;
            thread::sleep(Duration::from_millis(120));
        }
    });
    SpinnerHandle {
        done,
        thread: Some(thread),
    }
}

pub struct DeterminateProgressLine {
    prefix: String,
}

impl DeterminateProgressLine {
    pub fn new(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
        }
    }

    pub fn draw_counts(
        &self,
        label: &str,
        current: usize,
        total: usize,
        detail: Option<&str>,
    ) -> Result<()> {
        if crate::json_mode_enabled() {
            return Ok(());
        }
        let percent = if total > 0 {
            (current as f64 / total as f64) * 100.0
        } else {
            100.0
        };
        let detail = detail.unwrap_or("");
        let gauge = render_inline_gauge(
            ratio_complete(current, total),
            &format!(
                "{} {} {:>5.1}% [{}/{}]{}",
                self.prefix, label, percent, current, total, detail
            ),
        );
        eprint!("\r\x1b[2K{gauge}");
        std::io::stderr()
            .flush()
            .context("Flush determinate progress")?;
        Ok(())
    }
}

pub fn render_inline_gauge(ratio: f64, label: &str) -> String {
    render_inline_gauge_with_reserved_width(ratio, label, 0)
}

pub fn render_inline_gauge_with_reserved_width(
    ratio: f64,
    label: &str,
    reserved_columns: u16,
) -> String {
    let width = inline_gauge_width(reserved_columns);
    render_inline_gauge_in_width(ratio, label, width)
}

fn render_inline_gauge_in_width(ratio: f64, label: &str, width: u16) -> String {
    let area = Rect::new(0, 0, width, 1);
    let mut buffer = Buffer::empty(area);
    let label = fit_inline_gauge_label(label, width);
    LineGauge::default()
        .ratio(ratio.clamp(0.0, 1.0))
        .label(label)
        .style(Style::default().fg(Color::Gray))
        .filled_symbol("━")
        .unfilled_symbol("·")
        .filled_style(
            Style::default()
                .fg(Color::LightCyan)
                .add_modifier(Modifier::BOLD),
        )
        .unfilled_style(Style::default().fg(Color::DarkGray))
        .render(area, &mut buffer);
    styled_buffer_line(&buffer)
}

fn inline_gauge_width(reserved_columns: u16) -> u16 {
    let terminal_width = terminal_size()
        .map(|(width, _)| width)
        .unwrap_or(INLINE_GAUGE_WIDTH);
    available_inline_gauge_width(terminal_width, reserved_columns)
}

fn available_inline_gauge_width(terminal_width: u16, reserved_columns: u16) -> u16 {
    let available = terminal_width
        .saturating_sub(reserved_columns)
        .saturating_sub(INLINE_GAUGE_WRAP_GUARD_WIDTH);
    if available >= INLINE_GAUGE_MIN_BAR_WIDTH as u16 {
        available
    } else {
        available.max(1)
    }
}

fn fit_inline_gauge_label(label: &str, width: u16) -> String {
    let max_label_len = usize::from(width)
        .saturating_sub(INLINE_GAUGE_MIN_BAR_WIDTH)
        .saturating_sub(1);
    if label.chars().count() <= max_label_len {
        return label.to_string();
    }
    let keep_len = max_label_len.saturating_sub(3);
    format!("{}...", label.chars().take(keep_len).collect::<String>())
}

fn styled_buffer_line(buffer: &Buffer) -> String {
    let last_visible = buffer
        .content()
        .iter()
        .rposition(|cell| cell.symbol() != " ");
    let Some(last_visible) = last_visible else {
        return String::new();
    };
    let mut line = String::new();
    let mut active_style = InlineCellStyle::default();
    let mut used_style = false;
    for cell in &buffer.content()[..=last_visible] {
        let style = InlineCellStyle::from_cell(cell);
        if style != active_style {
            if let Some(sequence) = style.ansi_sequence() {
                line.push_str(&sequence);
                used_style = true;
            }
            active_style = style;
        }
        line.push_str(cell.symbol());
    }
    if used_style {
        line.push_str("\x1b[0m");
    }
    line
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct InlineCellStyle {
    fg: Color,
    bg: Color,
    modifier: Modifier,
}

impl InlineCellStyle {
    fn from_cell(cell: &ratatui::buffer::Cell) -> Self {
        Self {
            fg: cell.fg,
            bg: cell.bg,
            modifier: cell.modifier & (Modifier::BOLD | Modifier::DIM),
        }
    }

    fn ansi_sequence(self) -> Option<String> {
        if self == Self::default() {
            return Some("\x1b[0m".to_string());
        }
        let mut codes = Vec::new();
        if self.modifier.contains(Modifier::BOLD) {
            codes.push("1".to_string());
        }
        if self.modifier.contains(Modifier::DIM) {
            codes.push("2".to_string());
        }
        if let Some(code) = ansi_color_code(self.fg, false) {
            codes.push(code);
        }
        if let Some(code) = ansi_color_code(self.bg, true) {
            codes.push(code);
        }
        (!codes.is_empty()).then(|| format!("\x1b[0m\x1b[{}m", codes.join(";")))
    }
}

fn ansi_color_code(color: Color, background: bool) -> Option<String> {
    let base = if background { 10 } else { 0 };
    let code = match color {
        Color::Reset => return None,
        Color::Black => 30 + base,
        Color::Red => 31 + base,
        Color::Green => 32 + base,
        Color::Yellow => 33 + base,
        Color::Blue => 34 + base,
        Color::Magenta => 35 + base,
        Color::Cyan => 36 + base,
        Color::Gray => 37 + base,
        Color::DarkGray => 90 + base,
        Color::LightRed => 91 + base,
        Color::LightGreen => 92 + base,
        Color::LightYellow => 93 + base,
        Color::LightBlue => 94 + base,
        Color::LightMagenta => 95 + base,
        Color::LightCyan => 96 + base,
        Color::White => 97 + base,
        Color::Rgb(red, green, blue) => {
            let target = if background { 48 } else { 38 };
            return Some(format!("{target};2;{red};{green};{blue}"));
        }
        Color::Indexed(index) => {
            let target = if background { 48 } else { 38 };
            return Some(format!("{target};5;{index}"));
        }
    };
    Some(code.to_string())
}

pub fn ratio_complete(current: usize, total: usize) -> f64 {
    if total == 0 {
        1.0
    } else {
        (current as f64 / total as f64).clamp(0.0, 1.0)
    }
}

pub fn ratio_complete_u64(current: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        (current as f64 / total as f64).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        available_inline_gauge_width, ratio_complete_u64, render_inline_gauge,
        render_inline_gauge_in_width,
    };

    #[test]
    fn inline_gauge_renders_styled_progress_label_and_bar() {
        let line = render_inline_gauge(0.5, "downloaded 50MB / 100MB (50%)");
        let visible = strip_ansi(&line);

        assert!(line.contains("\x1b["));
        assert!(visible.contains('━'));
        assert!(visible.contains('·'));
        assert!(visible.len() > 10);
    }

    #[test]
    fn inline_gauge_keeps_bar_visible_for_long_labels() {
        let line = render_inline_gauge(
            0.25,
            "download very-long-model-name-with-many-segments-and-a-large-quantized-artifact.gguf 25%",
        );
        let visible = strip_ansi(&line);

        assert!(visible.contains('━'));
        assert!(visible.contains('·'));
    }

    #[test]
    fn byte_ratio_clamps_to_valid_ratatui_range() {
        assert_eq!(ratio_complete_u64(0, 0), 0.0);
        assert_eq!(ratio_complete_u64(150, 100), 1.0);
    }

    #[test]
    fn available_width_reserves_prefix_columns() {
        assert_eq!(available_inline_gauge_width(80, 3), 76);
    }

    #[test]
    fn available_width_leaves_one_column_wrap_guard() {
        let terminal_width = 80;
        let prefix_width = 3;
        let gauge_width = available_inline_gauge_width(terminal_width, prefix_width);

        assert!(prefix_width + gauge_width < terminal_width);
    }

    #[test]
    fn available_width_shrinks_below_minimum_on_tiny_terminals() {
        assert_eq!(available_inline_gauge_width(20, 3), 16);
    }

    #[test]
    fn explicit_width_gauge_matches_available_columns() {
        let line = render_inline_gauge_in_width(0.5, "downloaded 50MB / 100MB", 93);
        let visible = strip_ansi(&line);

        assert_eq!(visible.chars().count(), 93);
    }

    fn strip_ansi(line: &str) -> String {
        let mut stripped = String::new();
        let mut chars = line.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '\x1b' && chars.peek() == Some(&'[') {
                chars.next();
                for code_ch in chars.by_ref() {
                    if code_ch == 'm' {
                        break;
                    }
                }
                continue;
            }
            stripped.push(ch);
        }
        stripped
    }
}
