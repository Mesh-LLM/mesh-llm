use super::{
    cli::ModelsCommand, formatters::models_formatter, handlers::*, installed::run_model_installed,
};
use anyhow::Result;
pub(super) async fn dispatch_models_command(command: &ModelsCommand) -> Result<()> {
    match command {
        ModelsCommand::Package {
            source_repo,
            quant,
            target,
            model_id,
            generation_defaults,
            flavor,
            timeout,
            mesh_llm_ref,
            experimental,
            dry_run,
            confirm,
            follow,
            status,
            logs,
            cancel,
            list,
            update_script,
            json,
        } => {
            crate::models::package::dispatch_model_package(
                crate::models::package::ModelPrepareArgs {
                    source_repo: source_repo.as_deref(),
                    quant: quant.as_deref(),
                    target: target.as_deref(),
                    model_id: model_id.as_deref(),
                    generation_defaults: generation_defaults.as_deref(),
                    flavor,
                    timeout,
                    mesh_llm_ref,
                    experimental: *experimental,
                    dry_run: *dry_run,
                    confirm: *confirm,
                    follow: *follow,
                    json: *json,
                    status: status.as_deref(),
                    logs: logs.as_deref(),
                    cancel: cancel.as_deref(),
                    list: *list,
                    update_script: *update_script,
                },
            )
            .await?;
        }
        ModelsCommand::Recommended { json } | ModelsCommand::List { json } => {
            run_model_recommended(*json)?
        }
        ModelsCommand::Installed { json } => run_model_installed(*json)?,
        ModelsCommand::Cleanup {
            unused_since,
            yes,
            json,
        } => run_model_cleanup(unused_since.as_deref(), *yes, *json)?,
        ModelsCommand::Prune { yes, json } => run_model_prune(*yes, *json)?,
        ModelsCommand::Certify {
            model,
            report_out,
            json,
            package_only,
            api_base,
            prompt,
            max_tokens,
        } => {
            run_model_certify(
                model,
                report_out.as_deref(),
                *json,
                *package_only,
                api_base.as_deref(),
                prompt,
                *max_tokens,
            )
            .await?
        }
        ModelsCommand::Search {
            query,
            gguf,
            mlx,
            catalog,
            limit,
            sort,
            json,
        } => run_model_search(query, *gguf, *mlx, *catalog, *limit, *sort, *json).await?,
        ModelsCommand::Show { model, json } => run_model_show(model, *json).await?,
        ModelsCommand::Download {
            model,
            draft,
            direct,
            json,
        } => run_model_download(model, *draft, *direct, *json).await?,
        ModelsCommand::Updates {
            repo,
            all,
            check,
            json,
        } => {
            let repo_for_update = repo.clone();
            let repo_for_render = repo.clone();
            let all = *all;
            let check = *check;
            let context = super::output::context();
            tokio::task::spawn_blocking(move || {
                super::output::sync_scope(context, || {
                    crate::models::updates::run_update(repo_for_update.as_deref(), all, check)
                })
            })
            .await
            .map_err(anyhow::Error::from)??;
            if *json {
                let formatter = models_formatter(*json);
                formatter.render_updates_status(repo_for_render.as_deref(), all, check)?;
            }
        }
        ModelsCommand::Delete { model, yes, json } => {
            run_model_delete(model.as_str(), *yes, *json).await?
        }
    }
    Ok(())
}
