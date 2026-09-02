use super::{ModelPrepareSummary, SummaryAssembly};

pub(super) fn format_models(
    command: &mesh_llm_cli::models::ModelsCommand,
    assembly: &mut SummaryAssembly,
) {
    use mesh_llm_cli::models::ModelsCommand;
    match command {
        ModelsCommand::Package {
            source_repo,
            quant,
            target,
            model_id,
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
            assembly.command.push_str(" models package");
            assembly.redact("source_repo", source_repo.is_some());
            assembly.redact("--quant", quant.is_some());
            assembly.redact("--target", target.is_some());
            assembly.redact("--model-id", model_id.is_some());
            assembly.redact("--flavor", flavor != "auto");
            assembly.redact("--timeout", timeout != "1h");
            assembly.redact("--mesh-llm-ref", mesh_llm_ref != "main");
            assembly.flag("experimental", *experimental);
            assembly.flag("dry-run", *dry_run);
            assembly.flag("confirm", *confirm);
            assembly.flag("follow", *follow);
            assembly.redact("--status", status.is_some());
            assembly.redact("--logs", logs.is_some());
            assembly.redact("--cancel", cancel.is_some());
            assembly.flag("list", *list);
            assembly.flag("update-script", *update_script);
            assembly.flag("json", *json);
        }
        ModelsCommand::Recommended { json } => format_model_json("recommended", *json, assembly),
        ModelsCommand::Installed { json } => format_model_json("installed", *json, assembly),
        ModelsCommand::Cleanup {
            unused_since,
            yes,
            json,
        } => {
            assembly.command.push_str(" models cleanup");
            assembly.redact("--unused-since", unused_since.is_some());
            assembly.flag("yes", *yes);
            assembly.flag("json", *json);
        }
        ModelsCommand::Prune { yes, json } => {
            assembly.command.push_str(" models prune");
            assembly.flag("yes", *yes);
            assembly.flag("json", *json);
        }
        ModelsCommand::Certify {
            model,
            report_out,
            json,
            package_only,
            api_base,
            prompt,
            max_tokens,
        } => {
            assembly.command.push_str(" models certify");
            assembly.redact("model", !model.is_empty());
            assembly.redact("--report-out", report_out.is_some());
            assembly.flag("json", *json);
            assembly.flag("package-only", *package_only);
            assembly.redact("--api-base", api_base.is_some());
            assembly.redact("--prompt", prompt != "Say ok.");
            assembly.redact("--max-tokens", *max_tokens != 2);
        }
        ModelsCommand::List { json } => format_model_json("list", *json, assembly),
        ModelsCommand::Search {
            query,
            gguf,
            mlx,
            catalog,
            limit,
            sort,
            json,
        } => {
            assembly.command.push_str(" models search");
            assembly.redact("query", !query.is_empty());
            assembly.flag("gguf", *gguf);
            assembly.flag("mlx", *mlx);
            assembly.flag("catalog", *catalog);
            assembly.redact("--limit", *limit != 20);
            assembly.redact(
                "--sort",
                *sort != mesh_llm_cli::models::ModelSearchSort::Trending,
            );
            assembly.flag("json", *json);
        }
        ModelsCommand::Show { model, json } => {
            assembly.command.push_str(" models show");
            assembly.redact("model", !model.is_empty());
            assembly.flag("json", *json);
        }
        ModelsCommand::Download {
            model,
            draft,
            direct,
            json,
        } => {
            assembly.command.push_str(" models download");
            assembly.redact("model", !model.is_empty());
            assembly.flag("draft", *draft);
            assembly.flag("direct", *direct);
            assembly.flag("json", *json);
        }
        ModelsCommand::Updates {
            repo,
            all,
            check,
            json,
        } => {
            assembly.command.push_str(" models updates");
            assembly.redact("repo", repo.is_some());
            assembly.flag("all", *all);
            assembly.flag("check", *check);
            assembly.flag("json", *json);
        }
        ModelsCommand::Delete { model, yes, json } => {
            assembly.command.push_str(" models delete");
            assembly.redact("model", !model.is_empty());
            assembly.flag("yes", *yes);
            assembly.flag("json", *json);
        }
    }
}

fn format_model_json(name: &str, json: bool, assembly: &mut SummaryAssembly) {
    assembly.command.push_str(" models ");
    assembly.command.push_str(name);
    assembly.flag("json", json);
}

pub(super) fn format_model_prepare(
    options: ModelPrepareSummary<'_>,
    assembly: &mut SummaryAssembly,
) {
    assembly.command.push_str(" model-prepare");
    assembly.redact("source_repo", options.source_repo.is_some());
    assembly.redact("--quant", options.quant.is_some());
    assembly.redact("--target", options.target.is_some());
    assembly.redact("--model-id", options.model_id.is_some());
    assembly.redact("--flavor", options.flavor != "auto");
    assembly.redact("--timeout", options.timeout != "1h");
    assembly.redact("--mesh-llm-ref", options.mesh_llm_ref != "main");
    assembly.flag("dry-run", options.dry_run);
    assembly.flag("confirm", options.confirm);
    assembly.flag("follow", options.follow);
    assembly.flag("json", options.json);
    assembly.redact("--status", options.status.is_some());
    assembly.redact("--logs", options.logs.is_some());
    assembly.redact("--cancel", options.cancel.is_some());
    assembly.flag("list", options.list);
    assembly.flag("update-script", options.update_script);
}
