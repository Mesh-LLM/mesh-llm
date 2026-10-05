//! Host download/progress boundary for configured projectors.
use anyhow::{Context, Result};

pub(crate) async fn materialize_projector_url(
    resolved: &mut super::ResolvedSkippyConfig,
) -> Result<()> {
    if resolved.hardware.projector_path.is_some() {
        return Ok(());
    }
    let Some(projector_url) = resolved.multimodal.projector_url.as_deref() else {
        return Ok(());
    };
    resolved.hardware.projector_path = Some(
        crate::models::resolve::download_direct_ref_with_progress(projector_url, true)
            .await
            .with_context(|| format!("download multimodal.mmproj_url {projector_url}"))?,
    );
    Ok(())
}
