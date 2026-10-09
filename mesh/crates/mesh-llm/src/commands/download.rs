use std::io::Write;

use anyhow::Result;
use skippy_commands::models::lifecycle::{
    acquire_model_ref, canonical_catalog_ref, catalog_summaries,
};

pub(crate) async fn dispatch_download_command(name: Option<&str>, draft: bool) -> Result<()> {
    match name {
        Some(query) => {
            let model_ref = canonical_catalog_ref(query);
            let download = acquire_model_ref(
                &model_ref,
                mesh_llm_commands::model_output::model_output_context(),
            )
            .await;
            // Reported before `?` so an attempt that fails still counts: what
            // people try and cannot get is the more useful half of this.
            mesh_llm_commands::usage_reporting::record_model_download(&model_ref, download.is_ok());
            let download = download?;
            if draft {
                if let Some(draft_name) = download.draft_ref.as_deref() {
                    let draft_ref = canonical_catalog_ref(draft_name);
                    let draft_download = acquire_model_ref(
                        &draft_ref,
                        mesh_llm_commands::model_output::model_output_context(),
                    )
                    .await;
                    // The draft is a second model this command fetches, so it
                    // is a second attempt to count. Reported before `?` for the
                    // same reason as the primary.
                    mesh_llm_commands::usage_reporting::record_model_download(
                        &draft_ref,
                        draft_download.is_ok(),
                    );
                    draft_download?;
                } else {
                    let mut err = mesh_llm_events::console_err();
                    writeln!(err, "⚠ No draft model available for {}", query)?;
                }
            }
        }
        None => {
            let models = catalog_summaries()?;
            let mut err = mesh_llm_events::console_err();
            writeln!(err, "Available models:")?;
            writeln!(err)?;
            for model in models {
                let size = model.size.as_deref().unwrap_or("?");
                let description = model.description.as_deref().unwrap_or("");
                writeln!(err, "  {:40} {:>6}  {}", model.name, size, description)?;
            }
        }
    }
    Ok(())
}
