use std::path::Path;

use anyhow::Result;
use serde::Serialize;

use crate::splits::{ShardRange, SplitWindow};

const BAR_WIDTH: usize = 24;

pub(crate) fn print_json_pretty(value: &impl Serialize) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

pub(crate) fn progress_bar(completed: usize, expected: u32) -> String {
    let expected = expected as usize;
    let filled = completed
        .saturating_mul(BAR_WIDTH)
        .checked_div(expected)
        .unwrap_or_default()
        .min(BAR_WIDTH);
    format!(
        "[{}{}]",
        "█".repeat(filled),
        "░".repeat(BAR_WIDTH.saturating_sub(filled))
    )
}

pub(crate) fn percent(completed: usize, expected: u32) -> f64 {
    if expected == 0 {
        0.0
    } else {
        (completed as f64 / f64::from(expected)) * 100.0
    }
}

pub(crate) fn print_progress_line(label: &str, completed: usize, expected: u32) {
    println!(
        "📊 {label}: {} {completed}/{expected} shards ({:.2}%)",
        progress_bar(completed, expected),
        percent(completed, expected)
    );
}

pub(crate) fn print_success(message: impl AsRef<str>) {
    println!("✅ {}", message.as_ref());
}

pub(crate) fn print_info(message: impl AsRef<str>) {
    println!("ℹ️  {}", message.as_ref());
}

pub(crate) fn print_warn(message: impl AsRef<str>) {
    println!("⚠️  {}", message.as_ref());
}

pub(crate) fn print_copy(source: &Path, target: &Path, size_bytes: Option<u64>) {
    match size_bytes {
        Some(size_bytes) => println!(
            "📤 Copying {} -> {} ({})",
            source.display(),
            target.display(),
            format_bytes(size_bytes)
        ),
        None => println!("📤 Copying {} -> {}", source.display(), target.display()),
    }
}

pub(crate) fn print_path_event(emoji: &str, label: &str, path: &Path) {
    println!("{emoji} {label}: {}", path.display());
}

pub(crate) fn print_window(label: &str, window: SplitWindow) {
    println!("🪟 {label}: {}", format_window(window));
}

pub(crate) fn format_window(window: SplitWindow) -> String {
    if window.first_split == window.last_split {
        window.first_split.to_string()
    } else {
        format!("{}..{}", window.first_split, window.last_split)
    }
}

pub(crate) fn format_shard_ranges(ranges: &[ShardRange]) -> String {
    ranges
        .iter()
        .map(|range| {
            if range.first_split == range.last_split {
                range.first_split.to_string()
            } else {
                format!("{}..{}", range.first_split, range.last_split)
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

pub(crate) fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let bytes_f = bytes as f64;
    if bytes_f >= GIB {
        format!("{:.2} GiB", bytes_f / GIB)
    } else if bytes_f >= MIB {
        format!("{:.2} MiB", bytes_f / MIB)
    } else if bytes_f >= KIB {
        format!("{:.2} KiB", bytes_f / KIB)
    } else {
        format!("{bytes} B")
    }
}
