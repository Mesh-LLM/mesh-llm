use std::env;
use std::io::{IsTerminal, Write};

use anyhow::{Context, Result};
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    widgets::{LineGauge, Widget},
};

const GAUGE_WIDTH: u16 = 72;

pub(crate) struct PackageProgress {
    total: usize,
    completed: usize,
    interactive: bool,
    enabled: bool,
}

impl PackageProgress {
    pub(crate) fn new(total: usize) -> Self {
        Self {
            total,
            completed: 0,
            interactive: std::io::stderr().is_terminal(),
            enabled: progress_enabled(),
        }
    }

    pub(crate) fn start_step(&mut self, detail: &str) -> Result<()> {
        if self.enabled && self.interactive {
            self.draw("writing", detail)?;
        }
        Ok(())
    }

    pub(crate) fn finish_step(&mut self, detail: &str) -> Result<()> {
        if self.completed < self.total {
            self.completed += 1;
        }
        if !self.enabled {
            return Ok(());
        }
        if self.interactive {
            self.draw("wrote", detail)?;
        } else {
            eprintln!(
                "package progress: {}/{} wrote {}",
                self.completed, self.total, detail
            );
        }
        Ok(())
    }

    pub(crate) fn finish(&self) -> Result<()> {
        if self.enabled && self.interactive {
            eprintln!();
            std::io::stderr()
                .flush()
                .context("flush package progress finish")?;
        }
        Ok(())
    }

    fn draw(&self, verb: &str, detail: &str) -> Result<()> {
        eprint!("\r\x1b[2K{} {}", self.render_gauge(verb), detail);
        std::io::stderr()
            .flush()
            .context("flush package progress")?;
        Ok(())
    }

    fn render_gauge(&self, verb: &str) -> String {
        render_inline_gauge(
            ratio_complete(self.completed, self.total),
            &format!(
                "package {:>5.1}% [{}/{}] {verb}",
                percent_complete(self.completed, self.total),
                self.completed,
                self.total
            ),
        )
    }
}

pub(crate) fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0usize;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} {}", UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

fn progress_enabled() -> bool {
    env::var("SKIPPY_MODEL_PACKAGE_PROGRESS")
        .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
        .unwrap_or(true)
}

fn percent_complete(completed: usize, total: usize) -> f64 {
    ratio_complete(completed, total) * 100.0
}

fn ratio_complete(completed: usize, total: usize) -> f64 {
    if total == 0 {
        1.0
    } else {
        (completed as f64 / total as f64).clamp(0.0, 1.0)
    }
}

fn render_inline_gauge(ratio: f64, label: &str) -> String {
    let area = Rect::new(0, 0, GAUGE_WIDTH, 1);
    let mut buffer = Buffer::empty(area);
    LineGauge::default()
        .ratio(ratio.clamp(0.0, 1.0))
        .label(label)
        .filled_symbol("=")
        .unfilled_symbol("-")
        .render(area, &mut buffer);
    buffer
        .content()
        .iter()
        .map(|cell| cell.symbol())
        .collect::<String>()
        .trim_end()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::{format_bytes, render_inline_gauge};

    #[test]
    fn inline_gauge_renders_label_and_progress() {
        let line = render_inline_gauge(0.5, "package 50.0% [5/10] wrote");

        assert!(line.starts_with("package 50.0% [5/10] wrote "));
        assert!(line.contains('='));
        assert!(line.contains('-'));
    }

    #[test]
    fn inline_gauge_clamps_ratio() {
        let line = render_inline_gauge(2.0, "done");

        assert!(line.starts_with("done "));
        assert!(line.contains('='));
        assert!(!line.contains('-'));
    }

    #[test]
    fn format_bytes_uses_binary_units() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1536), "1.5 KiB");
        assert_eq!(format_bytes(5 * 1024 * 1024), "5.0 MiB");
    }
}
