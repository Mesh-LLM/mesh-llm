use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

use crate::output::{print_copy, print_info};
use crate::residency::{copy_file_bounded_with_label, remove_dir_if_exists, symlink_file};

#[derive(Debug, Serialize)]
pub struct Progress {
    pub expected_splits: u32,
    pub completed_count: usize,
    pub missing_count: usize,
    pub missing_ranges: Vec<ShardRange>,
    pub completed_percent: f64,
    pub complete: bool,
    pub first_missing: Option<u32>,
    pub last_present: Option<u32>,
    pub next_window: Option<SplitWindow>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SplitWindow {
    pub first_split: u32,
    pub last_split: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ShardRange {
    pub first_split: u32,
    pub last_split: u32,
}

pub fn split_status(root: &Path, prefix: &str, expected_splits: Option<u32>) -> Result<Progress> {
    let scan_root = root.join(prefix);
    let mut seen = BTreeSet::new();
    if scan_root.exists() {
        for entry in fs::read_dir(&scan_root)
            .with_context(|| format!("read directory {}", scan_root.display()))?
        {
            let entry = entry?;
            let file_name = entry.file_name();
            let file_name = file_name.to_string_lossy();
            if let Some((index, total)) = parse_split_file_name(&file_name)
                && expected_splits.is_none_or(|expected| expected == total)
            {
                seen.insert(index);
            }
        }
        if seen.is_empty() && expected_splits.is_none() && has_single_unsplit_gguf(&scan_root)? {
            seen.insert(1);
        }
    }

    let expected = expected_splits.unwrap_or_else(|| {
        seen.iter()
            .next_back()
            .copied()
            .unwrap_or_default()
            .max(parse_expected_total_from_dir(root, prefix).unwrap_or_default())
    });
    let first_missing = (1..=expected).find(|index| !seen.contains(index));
    Ok(progress_from_seen(expected, seen, first_missing))
}

pub fn split_status_for_basename(
    root: &Path,
    prefix: &str,
    basename: &str,
    expected_splits: u32,
) -> Result<Progress> {
    let scan_root = root.join(prefix);
    let mut seen = BTreeSet::new();
    if scan_root.exists() {
        for index in 1..=expected_splits {
            let name = format!("{basename}-{index:05}-of-{expected_splits:05}.gguf");
            if scan_root.join(name).is_file() {
                seen.insert(index);
            }
        }
        if expected_splits == 1 && scan_root.join(format!("{basename}.gguf")).is_file() {
            seen.insert(1);
        }
    }
    let first_missing = (1..=expected_splits).find(|index| !seen.contains(index));
    Ok(progress_from_seen(expected_splits, seen, first_missing))
}

fn progress_from_seen(
    expected_splits: u32,
    seen: BTreeSet<u32>,
    first_missing: Option<u32>,
) -> Progress {
    let completed_count = seen.len();
    let expected_count = expected_splits as usize;
    let missing_count = expected_count.saturating_sub(completed_count);
    let complete = expected_count > 0 && missing_count == 0 && first_missing.is_none();
    let completed_percent = if expected_splits == 0 {
        0.0
    } else {
        (completed_count as f64 / f64::from(expected_splits)) * 100.0
    };
    let missing_ranges = missing_ranges(expected_splits, &seen);
    Progress {
        expected_splits,
        completed_count,
        missing_count,
        missing_ranges,
        completed_percent,
        complete,
        first_missing,
        last_present: seen.iter().next_back().copied(),
        next_window: None,
    }
}

fn missing_ranges(expected_splits: u32, seen: &BTreeSet<u32>) -> Vec<ShardRange> {
    let mut ranges = Vec::new();
    let mut current_start = None;
    let mut previous_missing = None;

    for index in 1..=expected_splits {
        if seen.contains(&index) {
            if let Some(start) = current_start.take() {
                ranges.push(ShardRange {
                    first_split: start,
                    last_split: previous_missing.expect("missing range has previous index"),
                });
            }
            previous_missing = None;
            continue;
        }

        current_start.get_or_insert(index);
        previous_missing = Some(index);
    }

    if let Some(start) = current_start {
        ranges.push(ShardRange {
            first_split: start,
            last_split: previous_missing.expect("missing range has previous index"),
        });
    }

    ranges
}

pub fn parse_split_file_name(file_name: &str) -> Option<(u32, u32)> {
    let stem = file_name.strip_suffix(".gguf")?;
    let (before_total, total) = stem.rsplit_once("-of-")?;
    let (_, index) = before_total.rsplit_once('-')?;
    Some((index.parse().ok()?, total.parse().ok()?))
}

pub fn shard_name_for(first_shard: &Path, index: u32, total: u32) -> Result<String> {
    let first_name = first_shard
        .file_name()
        .and_then(|name| name.to_str())
        .with_context(|| format!("invalid first shard path {}", first_shard.display()))?;
    let stem = first_name
        .strip_suffix(".gguf")
        .with_context(|| format!("first shard is not a GGUF file: {}", first_shard.display()))?;
    if total == 1 && index == 1 && parse_split_file_name(first_name).is_none() {
        return Ok(first_name.to_string());
    }
    let (before_total, _) = stem
        .rsplit_once("-of-")
        .with_context(|| format!("invalid GGUF split shard name: {}", first_shard.display()))?;
    let (base, _) = before_total
        .rsplit_once('-')
        .with_context(|| format!("invalid GGUF split shard name: {}", first_shard.display()))?;
    Ok(format!("{base}-{index:05}-of-{total:05}.gguf"))
}

pub fn find_first_shard(source: &Path, source_prefix: &str) -> Result<PathBuf> {
    let source_root = prefixed_path(source, source_prefix);
    let mut candidates = Vec::new();
    let mut unsplit_candidates = Vec::new();
    for entry in fs::read_dir(&source_root)
        .with_context(|| format!("read directory {}", source_root.display()))?
    {
        let entry = entry?;
        let file_name = entry.file_name();
        let file_name = file_name.to_string_lossy();
        if let Some((index, _)) = parse_split_file_name(&file_name)
            && index == 1
        {
            candidates.push(entry.path());
        } else if is_unsplit_gguf_name(&file_name) {
            unsplit_candidates.push(entry.path());
        }
    }
    if candidates.is_empty() && unsplit_candidates.len() == 1 {
        return Ok(unsplit_candidates.remove(0));
    }
    ensure!(
        !candidates.is_empty(),
        "no first GGUF split shard found under {}",
        source_root.display()
    );
    ensure!(
        candidates.len() == 1,
        "multiple first GGUF split shards found under {}",
        source_root.display()
    );
    Ok(candidates.remove(0))
}

pub fn stage_source_window(
    source: &Path,
    source_prefix: &str,
    first_source_shard: &Path,
    stage_path: &Path,
    window: SplitWindow,
    total: u32,
) -> Result<PathBuf> {
    remove_dir_if_exists(stage_path)?;
    let stage_root = prefixed_path(stage_path, source_prefix);
    fs::create_dir_all(&stage_root).with_context(|| format!("create {}", stage_root.display()))?;
    let source_root = prefixed_path(source, source_prefix);

    for index in 1..=total {
        let name = shard_name_for(first_source_shard, index, total)?;
        let source_shard = source_root.join(&name);
        ensure!(
            source_shard.is_file(),
            "source shard does not exist: {}",
            source_shard.display()
        );
        let staged_shard = stage_root.join(name);
        if window.first_split <= index && index <= window.last_split {
            print_copy(
                &source_shard,
                &staged_shard,
                source_shard.metadata().ok().map(|m| m.len()),
            );
            copy_file_bounded_with_label("stage_source_copy", &source_shard, &staged_shard)?;
        } else {
            symlink_file(&source_shard, &staged_shard)?;
        }
    }

    let staged_first = stage_root.join(shard_name_for(first_source_shard, 1, total)?);
    print_info(format!(
        "Staged source window {}..{} at {}",
        window.first_split,
        window.last_split,
        stage_path.display()
    ));
    Ok(staged_first)
}

pub fn next_missing_window(missing_ranges: &[ShardRange], window_size: u32) -> Option<SplitWindow> {
    let first_range = missing_ranges.first()?;
    let capped_last = first_range
        .first_split
        .saturating_add(window_size.max(1).saturating_sub(1))
        .min(first_range.last_split);
    Some(SplitWindow {
        first_split: first_range.first_split,
        last_split: capped_last,
    })
}

pub fn next_missing_window_in_range(
    missing_ranges: &[ShardRange],
    requested: SplitWindow,
) -> Option<SplitWindow> {
    missing_ranges.iter().find_map(|range| {
        let first_split = range.first_split.max(requested.first_split);
        let last_split = range.last_split.min(requested.last_split);
        (first_split <= last_split).then_some(SplitWindow {
            first_split,
            last_split,
        })
    })
}

pub fn validate_split_window(window: SplitWindow, expected_splits: u32) -> Result<()> {
    ensure!(
        window.first_split > 0,
        "first split must be greater than zero"
    );
    ensure!(
        window.first_split <= window.last_split,
        "first split {} must be <= last split {}",
        window.first_split,
        window.last_split
    );
    ensure!(
        window.last_split <= expected_splits,
        "last split {} exceeds expected split count {}",
        window.last_split,
        expected_splits
    );
    Ok(())
}

fn parse_expected_total_from_dir(root: &Path, prefix: &str) -> Option<u32> {
    let scan_root = root.join(prefix);
    let entries = fs::read_dir(scan_root).ok()?;
    entries.filter_map(Result::ok).find_map(|entry| {
        parse_split_file_name(&entry.file_name().to_string_lossy()).map(|(_, total)| total)
    })
}

fn has_single_unsplit_gguf(root: &Path) -> Result<bool> {
    let mut count = 0_u32;
    for entry in fs::read_dir(root).with_context(|| format!("read directory {}", root.display()))? {
        let entry = entry?;
        if is_unsplit_gguf_name(&entry.file_name().to_string_lossy()) {
            count += 1;
        }
    }
    Ok(count == 1)
}

fn is_unsplit_gguf_name(file_name: &str) -> bool {
    file_name.ends_with(".gguf") && parse_split_file_name(file_name).is_none()
}

fn prefixed_path(root: &Path, prefix: &str) -> PathBuf {
    if prefix.is_empty() {
        root.to_path_buf()
    } else {
        root.join(prefix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_split_file_names() {
        assert_eq!(
            parse_split_file_name("GLM-5.2-Q2_K-MTP-Q8-00174-of-00306.gguf"),
            Some((174, 306))
        );
        assert_eq!(parse_split_file_name("not-a-shard.gguf"), None);
    }

    #[test]
    fn plans_next_window_from_first_missing_range() {
        let ranges = vec![
            ShardRange {
                first_split: 2,
                last_split: 2,
            },
            ShardRange {
                first_split: 5,
                last_split: 9,
            },
        ];
        assert_eq!(
            next_missing_window(&ranges, 4).map(|w| (w.first_split, w.last_split)),
            Some((2, 2))
        );
        assert_eq!(
            next_missing_window(&ranges[1..], 3).map(|w| (w.first_split, w.last_split)),
            Some((5, 7))
        );
        assert_eq!(
            next_missing_window(&ranges[1..], 0).map(|w| (w.first_split, w.last_split)),
            Some((5, 5))
        );
        assert!(next_missing_window(&[], 4).is_none());
    }

    #[test]
    fn plans_next_missing_window_inside_requested_range() {
        let ranges = vec![
            ShardRange {
                first_split: 2,
                last_split: 3,
            },
            ShardRange {
                first_split: 7,
                last_split: 9,
            },
        ];

        assert_eq!(
            next_missing_window_in_range(
                &ranges,
                SplitWindow {
                    first_split: 3,
                    last_split: 8
                }
            )
            .map(|w| (w.first_split, w.last_split)),
            Some((3, 3))
        );
        assert_eq!(
            next_missing_window_in_range(
                &ranges[1..],
                SplitWindow {
                    first_split: 3,
                    last_split: 8
                }
            )
            .map(|w| (w.first_split, w.last_split)),
            Some((7, 8))
        );
        assert!(
            next_missing_window_in_range(
                &ranges,
                SplitWindow {
                    first_split: 4,
                    last_split: 6
                }
            )
            .is_none()
        );
    }

    #[test]
    fn validates_requested_split_window_bounds() {
        assert!(
            validate_split_window(
                SplitWindow {
                    first_split: 1,
                    last_split: 3
                },
                3
            )
            .is_ok()
        );
        assert!(
            validate_split_window(
                SplitWindow {
                    first_split: 0,
                    last_split: 1
                },
                3
            )
            .is_err()
        );
        assert!(
            validate_split_window(
                SplitWindow {
                    first_split: 3,
                    last_split: 2
                },
                3
            )
            .is_err()
        );
        assert!(
            validate_split_window(
                SplitWindow {
                    first_split: 1,
                    last_split: 4
                },
                3
            )
            .is_err()
        );
    }

    #[test]
    fn derives_matching_shard_names_from_first_shard() {
        let first = Path::new("/repo/BF16/GLM-5.2-BF16-00001-of-00306.gguf");
        assert_eq!(
            shard_name_for(first, 174, 306).unwrap(),
            "GLM-5.2-BF16-00174-of-00306.gguf"
        );
    }

    #[test]
    fn derives_single_shard_name_from_unsplit_gguf() {
        let first = Path::new("/repo/BF16/model.gguf");
        assert_eq!(shard_name_for(first, 1, 1).unwrap(), "model.gguf");
    }

    #[test]
    fn validates_exact_basename_splits() {
        let root = std::env::temp_dir().join(format!(
            "skippy-quantize-test-{}",
            crate::unix_timestamp_ms()
        ));
        let prefix_root = root.join("Q2_K");
        fs::create_dir_all(&prefix_root).unwrap();
        fs::write(prefix_root.join("out-00001-of-00003.gguf"), b"1").unwrap();
        fs::write(prefix_root.join("out-00003-of-00003.gguf"), b"3").unwrap();

        let progress = split_status_for_basename(&root, "Q2_K", "out", 3).unwrap();
        assert_eq!(progress.expected_splits, 3);
        assert_eq!(progress.completed_count, 2);
        assert_eq!(progress.missing_count, 1);
        assert_eq!(
            progress.missing_ranges,
            vec![ShardRange {
                first_split: 2,
                last_split: 2,
            }]
        );
        assert!((progress.completed_percent - (200.0 / 3.0)).abs() < 1e-9);
        assert!(!progress.complete);
        assert_eq!(progress.first_missing, Some(2));
        assert_eq!(progress.last_present, Some(3));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn validates_exact_unsplit_single_file() {
        let root = std::env::temp_dir().join(format!(
            "skippy-quantize-unsplit-status-test-{}",
            crate::unix_timestamp_ms()
        ));
        let prefix_root = root.join("Q2_K");
        fs::create_dir_all(&prefix_root).unwrap();
        fs::write(prefix_root.join("out.gguf"), b"one").unwrap();

        let progress = split_status_for_basename(&root, "Q2_K", "out", 1).unwrap();
        assert_eq!(progress.completed_count, 1);
        assert_eq!(progress.missing_count, 0);
        assert!(progress.missing_ranges.is_empty());
        assert_eq!(progress.completed_percent, 100.0);
        assert!(progress.complete);
        assert_eq!(progress.first_missing, None);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn reports_compact_missing_ranges() {
        let mut seen = BTreeSet::new();
        seen.insert(1);
        seen.insert(4);
        seen.insert(8);

        let progress = progress_from_seen(9, seen, Some(2));

        assert_eq!(progress.missing_count, 6);
        assert_eq!(
            progress.missing_ranges,
            vec![
                ShardRange {
                    first_split: 2,
                    last_split: 3,
                },
                ShardRange {
                    first_split: 5,
                    last_split: 7,
                },
                ShardRange {
                    first_split: 9,
                    last_split: 9,
                },
            ]
        );
    }
}
