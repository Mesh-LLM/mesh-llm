//! Parsed representations of the Hub's raw diff format.
//!
//! [`crate::repository::HFRepository::get_raw_diff`] returns the raw compare payload as a
//! string, while [`crate::repository::HFRepository::get_raw_diff_stream`] uses the parser
//! in this module to yield one [`HFFileDiff`] per diff entry.
//!
//! The format is a line-oriented variant of git's raw diff output: each line
//! encodes the old/new blob ids, the file status, whether the payload is text
//! or binary, the destination-side file size, and the relevant paths. This
//! module does not parse patch hunks; it focuses on the metadata needed to
//! inspect revision-to-revision file changes efficiently.

use bytes::Bytes;
use futures::stream::{self, Stream, StreamExt};
use thiserror::Error;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio_util::io::StreamReader;

/// Errors produced while parsing raw diff lines into [`HFFileDiff`] values.
///
/// These errors surface on the stream returned by
/// [`crate::repository::HFRepository::get_raw_diff_stream`] when the Hub returns a
/// line that does not match the expected raw-diff shape, or when the response
/// stream itself fails while being read.
#[derive(Debug, Error)]
pub enum HFDiffParseError {
    /// The parser encountered an empty line, but every diff entry is expected
    /// to occupy exactly one non-empty line.
    #[error("diff line is empty")]
    EmptyLine,
    /// The size prefix in the raw diff line could not be parsed as a `u64`.
    #[error("failed to parse file size from {value:?} in line {line:?}: {source}")]
    InvalidFileSize {
        /// The substring that was expected to contain the file size.
        value: String,
        /// The full raw diff line being parsed.
        line: String,
        /// The underlying integer parse failure.
        source: std::num::ParseIntError,
    },
    /// The line was structurally different from the raw diff format this
    /// parser expects.
    #[error("incorrect diff line format: {line:?}")]
    InvalidFormat {
        /// The full raw diff line that could not be parsed.
        line: String,
    },
    /// Reading the response stream failed before the next diff line could be
    /// parsed.
    #[error("I/O error while reading diff stream: {0}")]
    Io(#[from] std::io::Error),
}

/// Parsed metadata for a single file entry in the Hub's raw diff output.
///
/// One `HFFileDiff` corresponds to one line in the payload returned by
/// [`crate::repository::HFRepository::get_raw_diff`] or one item yielded by
/// [`crate::repository::HFRepository::get_raw_diff_stream`]. For rename and copy
/// entries, [`new_file_path`](Self::new_file_path) contains the destination
/// path while [`file_path`](Self::file_path) remains the source path.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct HFFileDiff {
    /// Blob id for the file before the change.
    ///
    /// For additions this is typically all zeroes because there is no previous
    /// blob.
    pub old_blob_id: String,
    /// Blob id for the file after the change.
    ///
    /// For deletions this is typically all zeroes because there is no new
    /// blob.
    pub new_blob_id: String,
    /// High-level git status for the change.
    pub status: GitStatus,
    /// Repository-relative path of the original/source file.
    pub file_path: String,
    /// Repository-relative destination path for rename/copy entries.
    ///
    /// This is `None` for additions, deletions, and in-place modifications.
    pub new_file_path: Option<String>,
    /// Whether the Hub marked this entry as binary.
    ///
    /// `true` corresponds to a `B` prefix in the raw diff line; `false`
    /// corresponds to `T` (text).
    pub is_binary: bool,
    /// Destination-side file size in bytes, as reported by the raw diff line.
    pub new_file_size: u64,
}

/// File-level status code parsed from a raw diff entry.
///
/// The variants correspond to the single-letter status code in git raw diff
/// output (`A`, `C`, `D`, `M`, `R`, `T`, `U`, `X`).
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum GitStatus {
    /// A new file was added.
    Addition,
    /// The file was copied to a new path.
    Copy,
    /// The file was deleted.
    Deletion,
    /// The file contents changed in place.
    Modification,
    /// The file's mode/type changed without necessarily changing its path.
    FileTypeChange,
    /// The file was renamed.
    Rename,
    /// Git reported an unrecognized status code.
    Unknown,
    /// The entry is in an unmerged/conflicted state.
    Unmerged,
}

impl From<char> for GitStatus {
    fn from(value: char) -> Self {
        match value {
            'A' => Self::Addition,
            'C' => Self::Copy,
            'D' => Self::Deletion,
            'M' => Self::Modification,
            'R' => Self::Rename,
            'T' => Self::FileTypeChange,
            'U' => Self::Unmerged,
            'X' => Self::Unknown,
            _ => Self::Unknown,
        }
    }
}

/// Parse a single line of HF raw diff output into an `HFFileDiff`.
///
/// Format reference: <https://git-scm.com/docs/diff-format#_raw_output_format>
fn parse_hf_diff_line(line: &str) -> Result<HFFileDiff, HFDiffParseError> {
    let original_line = line;
    let fmt_err = || HFDiffParseError::InvalidFormat {
        line: original_line.to_owned(),
    };
    let bin_or_text = line.chars().next().ok_or(HFDiffParseError::EmptyLine)?;
    let is_binary = bin_or_text != 'T';
    // skip {B|T} and space
    let line = line.get(2..).ok_or_else(&fmt_err)?;
    let mut i = 0;
    for char in line.chars() {
        if !char.is_ascii_digit() {
            break;
        }
        i += 1;
    }
    let size_str = &line[..i];
    let new_file_size = size_str
        .parse()
        .map_err(|e| HFDiffParseError::InvalidFileSize {
            value: size_str.to_owned(),
            line: original_line.to_owned(),
            source: e,
        })?;
    // skip file size + \t
    let line = line.get(i + 1..).ok_or_else(&fmt_err)?;
    // skip :000000 000000 & space
    let line = line.get(15..).ok_or_else(&fmt_err)?;
    let old_blob_id = line.get(..40).ok_or_else(&fmt_err)?.to_owned();
    // skip sha1;0{40}, ... & space
    let line = line.get(44..).ok_or_else(&fmt_err)?;
    let new_blob_id = line.get(..40).ok_or_else(&fmt_err)?.to_owned();
    // skip sha1;0{40}, ... & space
    let line = line.get(44..).ok_or_else(&fmt_err)?;
    let status = line.chars().next().ok_or_else(&fmt_err)?.into();
    let line = line.get(1..).ok_or_else(&fmt_err)?;
    // skip optional score digits 1-3 chars & \t
    let mut i = 0;
    for char in line.chars() {
        if !char.is_ascii_digit() {
            break;
        }
        i += char.len_utf8();
    }
    let line = line.get(i + 1..).ok_or_else(&fmt_err)?;
    let separator_is_tab = line.contains('\t');
    // read up to next space or newline
    let i = if matches!(status, GitStatus::Copy | GitStatus::Rename) {
        let mut i = 0;
        for char in line.chars() {
            match (separator_is_tab, char) {
                (true, '\t') => break,
                (false, ' ') => break,
                _ => (),
            }
            i += char.len_utf8();
        }
        i
    } else {
        line.len()
    };
    let file_path = line[..i].to_owned();
    let line = &line[i..];
    let new_file_path = if !line.is_empty() {
        // skip separator
        let line = &line[1..];
        Some(line.to_owned())
    } else {
        None
    };

    Ok(HFFileDiff {
        old_blob_id,
        new_blob_id,
        status,
        file_path,
        new_file_path,
        is_binary,
        new_file_size,
    })
}

/// Stream-parse raw HF diff output line by line.
///
/// Takes a byte stream (e.g., from `HFRepository::get_raw_diff_stream`) and returns
/// a stream of `HFFileDiff` items, parsing each line as it arrives without buffering
/// the entire response.
pub(crate) fn stream_raw_diff<S, E>(
    byte_stream: S,
) -> impl Stream<Item = Result<HFFileDiff, HFDiffParseError>> + Unpin
where
    S: Stream<Item = Result<Bytes, E>>,
    E: Into<std::io::Error>,
{
    let mapped_stream = Box::pin(byte_stream).map(|r| r.map_err(Into::into));
    let reader = StreamReader::new(mapped_stream);
    let buf_reader = BufReader::new(reader);
    let lines = buf_reader.lines();

    Box::pin(stream::unfold(lines, |mut lines| async move {
        match lines.next_line().await {
            Ok(Some(line)) => {
                let result = parse_hf_diff_line(&line);
                if let Err(ref err) = result {
                    tracing::warn!(
                        line = %line,
                        error = %err,
                        "failed to parse diff line"
                    );
                }
                Some((result, lines))
            }
            Ok(None) => None,
            Err(e) => Some((Err(HFDiffParseError::Io(e)), lines)),
        }
    }))
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::{GitStatus, HFFileDiff, parse_hf_diff_line, stream_raw_diff};

    #[test]
    fn modified_hf_diff() {
        assert_eq!(
            parse_hf_diff_line("T 2305\t:100644 100644 97e7432a448baa9e97ec5e4f03c57b09b8e116ed... 0000000000000000000000000000000000000000... M\tapps/scan_orchestrator/src/dispatcher.rs").unwrap(),
            HFFileDiff {
                old_blob_id: "97e7432a448baa9e97ec5e4f03c57b09b8e116ed".to_owned(),
                new_blob_id: "0000000000000000000000000000000000000000".to_owned(),
                status: GitStatus::Modification,
                file_path: "apps/scan_orchestrator/src/dispatcher.rs".to_owned(),
                new_file_path: None,
                is_binary: false,
                new_file_size: 2305,
            }
        );
    }

    #[test]
    fn binary_copy_with_score() {
        assert_eq!(
            parse_hf_diff_line("B 421211\t:100644 100644 97e7432a448baa9e97ec5e4f03c57b09b8e116ed... 0000000000000000000000000000000000000000... C68\tapps/scan_orchestrator/src/dispatcher.rs apps/scan_orchestrator/src/blob").unwrap(),
            HFFileDiff {
                old_blob_id: "97e7432a448baa9e97ec5e4f03c57b09b8e116ed".to_owned(),
                new_blob_id: "0000000000000000000000000000000000000000".to_owned(),
                status: GitStatus::Copy,
                file_path: "apps/scan_orchestrator/src/dispatcher.rs".to_owned(),
                new_file_path: Some("apps/scan_orchestrator/src/blob".to_owned()),
                is_binary: true,
                new_file_size: 421211,
            }
        );
    }

    #[test]
    fn rename_without_score() {
        assert_eq!(
            parse_hf_diff_line("T 1679\t:100644 100644 f7b95e09e0573a829c338fe46e451b5609424a70... 0000000000000000000000000000000000000000... R\tapps/shared/src/scanner/file.rs apps/shared/src/scanner/file3.rs").unwrap(),
            HFFileDiff {
                old_blob_id: "f7b95e09e0573a829c338fe46e451b5609424a70".to_owned(),
                new_blob_id: "0000000000000000000000000000000000000000".to_owned(),
                status: GitStatus::Rename,
                file_path: "apps/shared/src/scanner/file.rs".to_owned(),
                new_file_path: Some("apps/shared/src/scanner/file3.rs".to_owned()),
                is_binary: false,
                new_file_size: 1679,
            }
        );
    }

    #[test]
    fn unicode_and_emoji_paths() {
        let d = parse_hf_diff_line("T 37861440\t:000000 100644 0000000000000000000000000000000000000000... 30a03d21620ebc6167e350aef9e2ac2774cf372d... A\tAI_popai/エイミ Eimi-ブルーアーカイブ Blue Archive (230270)/259889/eimi_(blue_archive).safetensors").unwrap();
        assert_eq!(d.status, GitStatus::Addition);
        assert_eq!(
            d.file_path,
            "AI_popai/エイミ Eimi-ブルーアーカイブ Blue Archive (230270)/259889/eimi_(blue_archive).safetensors"
        );
        assert_eq!(d.new_file_size, 37861440);

        let d = parse_hf_diff_line("T 228455604\t:000000 100644 0000000000000000000000000000000000000000... 77367f06242f620081e0103c599818bfde8d4c75... D\tFaeia/💀SDXL Antler Pagan💀 (236040)/266140/SDXLAntlerPagan.safetensors").unwrap();
        assert_eq!(d.status, GitStatus::Deletion);
        assert!(d.file_path.contains("💀"));
    }

    #[test]
    fn rename_with_unicode_paths() {
        let d = parse_hf_diff_line("T 1679\t:100644 100644 f7b95e09e0573a829c338fe46e451b5609424a70... 0000000000000000000000000000000000000000... R\tエイミ/old_file.rs\tエイミ/new_file.rs").unwrap();
        assert_eq!(d.status, GitStatus::Rename);
        assert_eq!(d.file_path, "エイミ/old_file.rs");
        assert_eq!(d.new_file_path, Some("エイミ/new_file.rs".to_owned()));
    }

    // A valid line used as a base for truncation tests.
    const VALID_LINE: &str = "T 2305\t:100644 100644 97e7432a448baa9e97ec5e4f03c57b09b8e116ed... 0000000000000000000000000000000000000000... M\tapps/scan_orchestrator/src/dispatcher.rs";

    fn assert_invalid_format(input: &str) {
        match parse_hf_diff_line(input) {
            Err(super::HFDiffParseError::InvalidFormat { .. }) => {}
            other => panic!("expected InvalidFormat for {input:?}, got {other:?}"),
        }
    }

    #[test]
    fn empty_line_error() {
        assert!(matches!(
            parse_hf_diff_line(""),
            Err(super::HFDiffParseError::EmptyLine)
        ));
    }

    #[test]
    fn single_char_truncated() {
        assert_invalid_format("T");
    }

    #[test]
    fn truncated_after_size() {
        assert_invalid_format("T 2305");
    }

    #[test]
    fn truncated_before_mode_pair() {
        assert_invalid_format("T 2305\t:100644");
    }

    #[test]
    fn truncated_before_old_blob_id() {
        assert_invalid_format("T 2305\t:100644 100644 abcdef");
    }

    #[test]
    fn truncated_before_new_blob_id() {
        assert_invalid_format(
            "T 2305\t:100644 100644 97e7432a448baa9e97ec5e4f03c57b09b8e116ed... ",
        );
    }

    #[test]
    fn truncated_before_status() {
        assert_invalid_format(
            "T 2305\t:100644 100644 97e7432a448baa9e97ec5e4f03c57b09b8e116ed... 0000000000000000000000000000000000000000...",
        );
    }

    #[test]
    fn truncated_after_status() {
        assert_invalid_format(
            "T 2305\t:100644 100644 97e7432a448baa9e97ec5e4f03c57b09b8e116ed... 0000000000000000000000000000000000000000... M",
        );
    }

    #[test]
    fn valid_line_parses_ok() {
        parse_hf_diff_line(VALID_LINE).unwrap();
    }

    #[tokio::test]
    async fn stream_modified_hf_diff() {
        let input = b"T 2305\t:100644 100644 97e7432a448baa9e97ec5e4f03c57b09b8e116ed... 0000000000000000000000000000000000000000... M\tapps/scan_orchestrator/src/dispatcher.rs\nT 4422\t:100644 100644 c417bf5a3fbec60b22aeab13cfa4d9439155303b... 0000000000000000000000000000000000000000... M\tapps/scan_orchestrator/src/git.rs\n";

        let byte_stream = futures::stream::once(async {
            Ok::<_, std::io::Error>(bytes::Bytes::from(&input[..]))
        });

        let diffs: Vec<HFFileDiff> = stream_raw_diff(byte_stream)
            .map(|r| r.unwrap())
            .collect()
            .await;

        assert_eq!(diffs.len(), 2);
        assert_eq!(
            diffs[0].file_path,
            "apps/scan_orchestrator/src/dispatcher.rs"
        );
        assert_eq!(diffs[0].status, GitStatus::Modification);
        assert_eq!(diffs[0].new_file_size, 2305);
        assert_eq!(diffs[1].file_path, "apps/scan_orchestrator/src/git.rs");
        assert_eq!(diffs[1].status, GitStatus::Modification);
        assert_eq!(diffs[1].new_file_size, 4422);
    }

    #[tokio::test]
    async fn stream_mixed_statuses() {
        let input = b"T 37861440\t:000000 100644 0000000000000000000000000000000000000000... 30a03d21620ebc6167e350aef9e2ac2774cf372d... A\tAI_popai/\xe3\x82\xa8\xe3\x82\xa4\xe3\x83\x9f Eimi (230270)/259889/eimi.safetensors\nT 228455604\t:100644 000000 77367f06242f620081e0103c599818bfde8d4c75... 0000000000000000000000000000000000000000... D\tFaeia/\xf0\x9f\x98\xa1SDXL Rage Style\xf0\x9f\x98\xa1 (234815)/264786/SDXLRageStyle.safetensors\nT 2305\t:100644 100644 97e7432a448baa9e97ec5e4f03c57b09b8e116ed... 0000000000000000000000000000000000000000... M\tapps/scan_orchestrator/src/dispatcher.rs\nB 421211\t:100644 100644 97e7432a448baa9e97ec5e4f03c57b09b8e116ed... 0000000000000000000000000000000000000000... C68\tapps/scan_orchestrator/src/dispatcher.rs apps/scan_orchestrator/src/blob\nT 1679\t:100644 100644 f7b95e09e0573a829c338fe46e451b5609424a70... 0000000000000000000000000000000000000000... R\tapps/shared/src/scanner/file.rs apps/shared/src/scanner/file3.rs\n";

        let byte_stream = futures::stream::once(async {
            Ok::<_, std::io::Error>(bytes::Bytes::from(&input[..]))
        });

        let diffs: Vec<HFFileDiff> = stream_raw_diff(byte_stream)
            .map(|r| r.unwrap())
            .collect()
            .await;

        assert_eq!(diffs.len(), 5);
        assert_eq!(diffs[0].status, GitStatus::Addition);
        assert!(diffs[0].file_path.contains("Eimi"));
        assert_eq!(diffs[1].status, GitStatus::Deletion);
        assert!(diffs[1].file_path.contains("😡"));
        assert_eq!(diffs[2].status, GitStatus::Modification);
        assert_eq!(diffs[3].status, GitStatus::Copy);
        assert_eq!(
            diffs[3].new_file_path.as_deref(),
            Some("apps/scan_orchestrator/src/blob")
        );
        assert!(diffs[3].is_binary);
        assert_eq!(diffs[4].status, GitStatus::Rename);
        assert_eq!(
            diffs[4].new_file_path.as_deref(),
            Some("apps/shared/src/scanner/file3.rs")
        );
    }

    #[tokio::test]
    async fn stream_across_chunk_boundaries() {
        let chunk1 = b"T 2305\t:100644 100644 97e7432a448baa9e97ec5e4f03c57b09b8e116ed... 0000000000000000000000000000000000000000... M\tapps/scan_orchestrator/src/dispatcher.rs\nT 44";
        let chunk2 = b"22\t:100644 100644 c417bf5a3fbec60b22aeab13cfa4d9439155303b... 0000000000000000000000000000000000000000... M\tapps/scan_orchestrator/src/git.rs\n";

        let byte_stream = futures::stream::iter(vec![
            Ok::<_, std::io::Error>(bytes::Bytes::from(&chunk1[..])),
            Ok(bytes::Bytes::from(&chunk2[..])),
        ]);

        let diffs: Vec<HFFileDiff> = stream_raw_diff(byte_stream)
            .map(|r| r.unwrap())
            .collect()
            .await;

        assert_eq!(diffs.len(), 2);
        assert_eq!(
            diffs[0].file_path,
            "apps/scan_orchestrator/src/dispatcher.rs"
        );
        assert_eq!(diffs[1].file_path, "apps/scan_orchestrator/src/git.rs");
    }
}
