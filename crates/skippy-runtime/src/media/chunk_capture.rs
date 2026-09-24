use std::{ffi::c_void, panic::AssertUnwindSafe};

use anyhow::{Result, anyhow, bail};

use crate::{ActivationFrame, MediaPrefillChunkFrame, StageSession};

/// Capture each live graph before the native helper decodes another microbatch.
pub(super) struct ChunkCapture<'a> {
    pub session: &'a mut StageSession,
    frames: Vec<(usize, ActivationFrame)>,
    error: Option<anyhow::Error>,
}

impl<'a> ChunkCapture<'a> {
    pub fn new(session: &'a mut StageSession) -> Self {
        Self {
            session,
            frames: Vec::new(),
            error: None,
        }
    }

    pub fn finish(self, status: i32) -> Result<Vec<(usize, ActivationFrame)>> {
        if let Some(error) = self.error {
            return Err(error);
        }
        if status != 0 {
            bail!("multimodal chunk evaluation failed with status {status}");
        }
        Ok(self.frames)
    }
}

/// The synchronous native helper borrows this capture only until it returns.
/// Contain panics and ordinary errors rather than unwinding through C++.
pub(super) unsafe extern "C" fn capture_microbatch(count: i32, opaque: *mut c_void) -> i32 {
    // SAFETY: eval_media_frame passes its live, exclusively borrowed ChunkCapture.
    let capture = unsafe { &mut *opaque.cast::<ChunkCapture<'_>>() };
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<()> {
        let count = usize::try_from(count)?;
        if count == 0 {
            bail!("native media helper returned an empty microbatch");
        }
        let frame = capture.session.copy_output_activation_frame(count, 0)?;
        capture.frames.push((count, frame));
        Ok(())
    }));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(error)) => {
            capture.error = Some(error);
            -1
        }
        Err(_) => {
            capture.error = Some(anyhow!("panic while capturing media activation microbatch"));
            -1
        }
    }
}

/// Keep text IDs and dimension-major M-RoPE positions aligned with each export.
pub(super) fn split_chunk_frames(
    frames: Vec<(usize, ActivationFrame)>,
    tokens: &[i32],
    positions: &[i32],
    total: usize,
) -> Result<Vec<MediaPrefillChunkFrame>> {
    if (!tokens.is_empty() && tokens.len() != total)
        || (!positions.is_empty() && positions.len() != total.checked_mul(4).unwrap_or(0))
    {
        bail!("media chunk metadata does not match its token count");
    }
    let mut offset = 0usize;
    let mut result = Vec::with_capacity(frames.len());
    for (count, output) in frames {
        let end = offset
            .checked_add(count)
            .filter(|end| *end <= total)
            .ok_or_else(|| anyhow!("media microbatch exceeds its chunk"))?;
        if count == 0 || output.desc.token_count as usize != count {
            bail!("media microbatch export has the wrong token count");
        }
        let batch_positions = if positions.is_empty() {
            Vec::new()
        } else {
            (0..4)
                .flat_map(|dimension| {
                    positions[dimension * total + offset..dimension * total + end]
                        .iter()
                        .copied()
                })
                .collect()
        };
        result.push(MediaPrefillChunkFrame {
            token_count: count,
            tokens: if tokens.is_empty() {
                Vec::new()
            } else {
                tokens[offset..end].to_vec()
            },
            positions: batch_positions,
            output,
        });
        offset = end;
    }
    if offset != total {
        bail!("media microbatches captured {offset} of {total} tokens");
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(count: u32) -> ActivationFrame {
        let mut desc = crate::ActivationDesc::from(crate::types::empty_raw_activation_desc());
        desc.token_count = count;
        ActivationFrame {
            desc,
            payload: Vec::new(),
        }
    }

    #[test]
    fn microbatch_exports_preserve_text_and_dimension_major_positions() {
        let positions = [10, 11, 12, 20, 21, 22, 30, 31, 32, 0, 0, 0];
        let chunks = split_chunk_frames(
            vec![(2, frame(2)), (1, frame(1))],
            &[4, 5, 6],
            &positions,
            3,
        )
        .unwrap();
        assert_eq!(chunks[0].tokens, [4, 5]);
        assert_eq!(chunks[1].tokens, [6]);
        assert_eq!(chunks[0].positions, [10, 11, 20, 21, 30, 31, 0, 0]);
        assert_eq!(chunks[1].positions, [12, 22, 32, 0]);
    }

    #[test]
    fn incomplete_or_stale_microbatch_exports_are_rejected() {
        assert!(split_chunk_frames(vec![(2, frame(2))], &[], &[], 3).is_err());
        assert!(split_chunk_frames(vec![(4, frame(4))], &[], &[], 3).is_err());
        assert!(split_chunk_frames(vec![(3, frame(1))], &[], &[], 3).is_err());
        assert!(split_chunk_frames(vec![(3, frame(3))], &[], &[0; 3], 3).is_err());
    }
}
