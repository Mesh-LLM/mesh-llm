use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};

use crate::output::{format_bytes, print_copy, print_info, print_path_event, print_success};
use crate::residency::copy_file_bounded_with_label;
use crate::splits::SplitWindow;

pub fn execution_root(target: &Path, target_prefix: &str, spool_dir: Option<&Path>) -> PathBuf {
    spool_dir.unwrap_or(target).join(target_prefix)
}

pub fn publish_spooled_window(
    spool_dir: Option<&Path>,
    target: &Path,
    target_prefix: &str,
    output_basename: &str,
    expected_splits: u32,
    window: SplitWindow,
    keep_spool: bool,
) -> Result<()> {
    let Some(spool_dir) = spool_dir else {
        return Ok(());
    };

    let source_root = spool_dir.join(target_prefix);
    let target_root = target.join(target_prefix);
    fs::create_dir_all(&target_root)
        .with_context(|| format!("create {}", target_root.display()))?;

    for index in window.first_split..=window.last_split {
        publish_one_shard(
            &source_root,
            &target_root,
            output_basename,
            index,
            expected_splits,
            keep_spool,
        )?;
    }
    Ok(())
}

pub fn clean_spooled_window(
    spool_dir: Option<&Path>,
    target_prefix: &str,
    output_basename: &str,
    expected_splits: u32,
    window: SplitWindow,
) -> Result<()> {
    let Some(spool_dir) = spool_dir else {
        return Ok(());
    };
    let source_root = spool_dir.join(target_prefix);
    for index in window.first_split..=window.last_split {
        for path in stale_spool_candidates(&source_root, output_basename, index, expected_splits) {
            if path.exists() {
                fs::remove_file(&path).with_context(|| format!("remove {}", path.display()))?;
                print_path_event("🧹", "Removed stale spool shard", &path);
            }
        }
    }
    Ok(())
}

fn publish_one_shard(
    source_root: &Path,
    target_root: &Path,
    output_basename: &str,
    index: u32,
    expected_splits: u32,
    keep_spool: bool,
) -> Result<()> {
    let source = source_candidate(source_root, output_basename, index, expected_splits);
    let target = target_root.join(
        source
            .file_name()
            .context("spooled source shard has invalid file name")?,
    );
    ensure!(
        source.is_file(),
        "spooled output shard does not exist: {}",
        source.display()
    );
    publish_file(&source, &target)?;
    if !keep_spool {
        fs::remove_file(&source).with_context(|| format!("remove {}", source.display()))?;
        print_path_event("🧹", "Removed spooled shard", &source);
    }
    Ok(())
}

fn publish_file(source: &Path, target: &Path) -> Result<()> {
    let source_len = source
        .metadata()
        .with_context(|| format!("stat {}", source.display()))?
        .len();
    if target.exists() {
        let target_len = target
            .metadata()
            .with_context(|| format!("stat {}", target.display()))?
            .len();
        ensure!(
            source_len == target_len,
            "target shard already exists with different size: {} source_bytes={} target_bytes={}",
            target.display(),
            source_len,
            target_len
        );
        print_info(format!(
            "Publish target already exists: {} ({})",
            target.display(),
            format_bytes(source_len)
        ));
        return Ok(());
    }

    let temp_name = format!(
        "{}.part",
        target
            .file_name()
            .and_then(|name| name.to_str())
            .context("target shard has invalid file name")?
    );
    let temp = target.with_file_name(temp_name);
    if temp.exists() {
        fs::remove_file(&temp).with_context(|| format!("remove {}", temp.display()))?;
    }
    print_copy(source, target, Some(source_len));
    copy_file_bounded_with_label("publish_copy", source, &temp)?;
    fs::rename(&temp, target)
        .with_context(|| format!("rename {} -> {}", temp.display(), target.display()))?;
    print_success(format!(
        "Published {} ({})",
        target.display(),
        format_bytes(source_len)
    ));
    Ok(())
}

fn split_shard_name(output_basename: &str, index: u32, expected_splits: u32) -> String {
    format!("{output_basename}-{index:05}-of-{expected_splits:05}.gguf")
}

fn source_candidate(
    source_root: &Path,
    output_basename: &str,
    index: u32,
    expected_splits: u32,
) -> PathBuf {
    let split = source_root.join(split_shard_name(output_basename, index, expected_splits));
    if split.is_file() || expected_splits != 1 || index != 1 {
        return split;
    }
    let unsplit = source_root.join(format!("{output_basename}.gguf"));
    if unsplit.is_file() {
        return unsplit;
    }
    split
}

fn stale_spool_candidates(
    source_root: &Path,
    output_basename: &str,
    index: u32,
    expected_splits: u32,
) -> Vec<PathBuf> {
    let mut candidates =
        vec![source_root.join(split_shard_name(output_basename, index, expected_splits))];
    if expected_splits == 1 && index == 1 {
        candidates.push(source_root.join(format!("{output_basename}.gguf")));
    }
    candidates
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::records::unix_timestamp_ms;

    #[test]
    fn publishes_spooled_window_and_cleans_source() {
        let root = std::env::temp_dir().join(format!(
            "skippy-quantize-publish-test-{}",
            unix_timestamp_ms()
        ));
        let spool = root.join("spool");
        let target = root.join("target");
        let spool_root = spool.join("Q2_K");
        fs::create_dir_all(&spool_root).unwrap();
        fs::write(spool_root.join("model-00002-of-00003.gguf"), b"two").unwrap();

        publish_spooled_window(
            Some(&spool),
            &target,
            "Q2_K",
            "model",
            3,
            SplitWindow {
                first_split: 2,
                last_split: 2,
            },
            false,
        )
        .unwrap();

        assert_eq!(
            fs::read(target.join("Q2_K/model-00002-of-00003.gguf")).unwrap(),
            b"two"
        );
        assert!(!spool_root.join("model-00002-of-00003.gguf").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cleans_only_current_spooled_window() {
        let root = std::env::temp_dir().join(format!(
            "skippy-quantize-spool-clean-test-{}",
            unix_timestamp_ms()
        ));
        let spool = root.join("spool");
        let spool_root = spool.join("Q2_K");
        fs::create_dir_all(&spool_root).unwrap();
        fs::write(spool_root.join("model-00001-of-00003.gguf"), b"one").unwrap();
        fs::write(spool_root.join("model-00002-of-00003.gguf"), b"two").unwrap();
        fs::write(spool_root.join("model-00003-of-00003.gguf"), b"three").unwrap();

        clean_spooled_window(
            Some(&spool),
            "Q2_K",
            "model",
            3,
            SplitWindow {
                first_split: 2,
                last_split: 2,
            },
        )
        .unwrap();

        assert!(spool_root.join("model-00001-of-00003.gguf").exists());
        assert!(!spool_root.join("model-00002-of-00003.gguf").exists());
        assert!(spool_root.join("model-00003-of-00003.gguf").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn publishes_unsplit_single_file_output() {
        let root = std::env::temp_dir().join(format!(
            "skippy-quantize-unsplit-publish-test-{}",
            unix_timestamp_ms()
        ));
        let spool = root.join("spool");
        let target = root.join("target");
        let spool_root = spool.join("Q2_K");
        fs::create_dir_all(&spool_root).unwrap();
        fs::write(spool_root.join("model.gguf"), b"one").unwrap();

        publish_spooled_window(
            Some(&spool),
            &target,
            "Q2_K",
            "model",
            1,
            SplitWindow {
                first_split: 1,
                last_split: 1,
            },
            false,
        )
        .unwrap();

        assert_eq!(fs::read(target.join("Q2_K/model.gguf")).unwrap(), b"one");
        assert!(!spool_root.join("model.gguf").exists());
        fs::remove_dir_all(root).unwrap();
    }
}
