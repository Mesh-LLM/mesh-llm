//! ANE DFlash attachment and package selection.

use std::path::Path;

use anyhow::{Context, Result, bail};
use skippy_ane::dflash::{AneDraft, PairedGguf};

use super::{DFlashDraft, DFlashDraftOptions, DFlashVariant, RuntimeConfig};

pub(super) fn attach(
    draft: &mut DFlashDraft,
    options: &DFlashDraftOptions,
    config: &RuntimeConfig,
    draft_path: &Path,
) -> Result<()> {
    if options.device.as_deref() == Some("ANE") {
        if draft.info.variant != DFlashVariant::DFlash {
            bail!("ANE conversion is available only for plain DFlash drafts");
        }
        let target = options
            .ane_target_path
            .as_deref()
            .context("ANE DFlash requires a local target GGUF")?;
        let rows = draft.info.max_draft_tokens + 1;
        let context = usize::try_from(config.ctx_size)?.min(512).max(rows);
        draft.ane = Some(AneDraft::from_pair(
            usize::try_from(draft.lanes).context("negative ANE lane count")?,
            draft.n_vocab,
            draft.mask_token,
            PairedGguf {
                draft: draft_path,
                target,
                context,
                rows,
            },
        )?);
    } else if std::env::var_os("SKIPPY_DFLASH_ANE_MODEL").is_some() {
        if draft.info.variant != DFlashVariant::DFlash {
            bail!("ANE conversion is available only for plain DFlash drafts");
        }
        draft.ane = AneDraft::from_env(
            usize::try_from(draft.lanes).context("negative ANE lane count")?,
            draft.n_vocab,
            draft.mask_token,
            draft_path,
        )?;
        if draft
            .ane
            .as_ref()
            .is_some_and(|ane| draft.info.max_draft_tokens > ane.max_proposals())
        {
            bail!("ANE DFlash package has fewer proposal rows than the configured draft limit");
        }
    }
    Ok(())
}
