use anyhow::Result;
use mesh_llm_cli::benchmark::{BenchmarkCommand, PromptImportSource};
use mesh_llm_system::benchmark_prompts::{self, ImportPromptsArgs};
use std::path::Path;

pub async fn dispatch_benchmark_command(
    config_path: Option<&Path>,
    command: &BenchmarkCommand,
) -> Result<()> {
    match command {
        BenchmarkCommand::Tune(_) => {
            crate::gpus::tune_runner::run_benchmark_tune_command(config_path, command)
        }
        BenchmarkCommand::ImportPrompts {
            source,
            limit,
            max_tokens,
            output,
        } => {
            let args = ImportPromptsArgs {
                source: map_prompt_source(*source),
                limit: *limit,
                max_tokens: *max_tokens,
                output: output.clone(),
                user_agent_version: mesh_llm_build_info::BUILD_VERSION,
            };
            benchmark_prompts::import_prompt_corpus(args).await
        }
    }
}

fn map_prompt_source(source: PromptImportSource) -> benchmark_prompts::PromptImportSource {
    match source {
        PromptImportSource::MtBench => benchmark_prompts::PromptImportSource::MtBench,
        PromptImportSource::Gsm8k => benchmark_prompts::PromptImportSource::Gsm8k,
        PromptImportSource::Humaneval => benchmark_prompts::PromptImportSource::Humaneval,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_import_source_mapping_covers_all_cli_variants() {
        assert_eq!(
            map_prompt_source(PromptImportSource::MtBench),
            benchmark_prompts::PromptImportSource::MtBench
        );
        assert_eq!(
            map_prompt_source(PromptImportSource::Gsm8k),
            benchmark_prompts::PromptImportSource::Gsm8k
        );
        assert_eq!(
            map_prompt_source(PromptImportSource::Humaneval),
            benchmark_prompts::PromptImportSource::Humaneval
        );
    }
}
