use anyhow::Result;

pub(crate) async fn dispatch_download_command(name: Option<&str>, draft: bool) -> Result<()> {
    match name {
        Some(query) => {
            let model_ref =
                mesh_llm_host_runtime::command_support::models::find_remote_catalog_model_exact(
                    query,
                )
                .map(|model| {
                    mesh_llm_host_runtime::command_support::models::remote_catalog_model_ref(&model)
                })
                .unwrap_or_else(|| query.to_string());
            let (_path, details) =
                mesh_llm_host_runtime::command_support::models::download_model_ref_with_progress_details(&model_ref, true).await?;
            if draft {
                if let Some(draft_name) = details
                    .as_ref()
                    .and_then(|details| details.draft.as_deref())
                {
                    let draft_ref = mesh_llm_host_runtime::command_support::models::find_remote_catalog_model_exact(draft_name)
                        .map(|model| mesh_llm_host_runtime::command_support::models::remote_catalog_model_ref(&model))
                        .unwrap_or_else(|| draft_name.to_string());
                    mesh_llm_host_runtime::command_support::models::download_model_ref_with_progress_details(&draft_ref, true)
                        .await?;
                } else {
                    eprintln!("⚠ No draft model available for {}", query);
                }
            }
        }
        None => {
            mesh_llm_host_runtime::command_support::models::remote_catalog::ensure_catalog()?;
            eprintln!("Available models:");
            eprintln!();
            for model in
                mesh_llm_host_runtime::command_support::models::remote_catalog::loaded_models()?
            {
                let size = model.size.as_deref().unwrap_or("?");
                let description = model.description.as_deref().unwrap_or("");
                eprintln!("  {:40} {:>6}  {}", model.name, size, description);
            }
        }
    }
    Ok(())
}
