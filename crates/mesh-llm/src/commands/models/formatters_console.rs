use super::formatters::{
    ConsoleFormatter, DownloadRenderInput, DownloadStats, InstalledRow, ModelsFormatter,
    SearchFormatter, catalog_model_capabilities, filter_label, fit_hint_for_size_label,
    format_count, format_installed_size, format_relative_timestamp, format_source_label,
    huggingface_cache_dir, huggingface_repo_url, installed_model_kind, model_kind_code, sort_label,
    variant_selector_label,
};
use anyhow::Result;
use mesh_llm_host_runtime::command_support::models::{
    DeleteResult as CliDeleteResult, ModelDetails, ResolvedModel as CliResolvedModel,
    SearchArtifactFilter, SearchHit, SearchSort, remote_catalog, remote_catalog_model_draft_ref,
    remote_catalog_model_ref,
};
use std::fmt::Write as FmtWrite;
use std::io::{IsTerminal, Write};
use std::time::Duration;
use tabwriter::TabWriter;

const ANSI_RESET: &str = "\x1b[0m";
const ANSI_BOLD_GREEN: &str = "\x1b[1;32m";
const ANSI_BRIGHT_CYAN: &str = "\x1b[96m";
const ANSI_DIM: &str = "\x1b[2m";
const ANSI_MUTED: &str = "\x1b[90m";

fn downloaded_model_headline(stats: Option<&DownloadStats>, colors: bool) -> String {
    let Some(stats) = stats else {
        return styled_download_success("✓ Downloaded model", colors);
    };

    let title = styled_download_success("✓ Downloaded model", colors);
    let divider = styled_muted(" ─ ", colors);
    let Some(bytes) = stats.bytes else {
        return format!(
            "{title}{divider}{}",
            styled_stat(&format!("in {}", format_elapsed(stats.elapsed)), colors)
        );
    };

    let mut headline = format!(
        "{title}{divider}{}",
        styled_stat(
            &format!(
                "{} in {}",
                format_download_stat_bytes(bytes),
                format_elapsed(stats.elapsed)
            ),
            colors
        )
    );
    if let Some((bytes_per_sec, label)) = download_bytes_per_sec(stats) {
        let _ = write!(
            headline,
            " {} {}",
            styled_muted(&format!("· {label}"), colors),
            styled_stat(
                &format!("{}/s", format_download_stat_bytes(bytes_per_sec)),
                colors
            )
        );
    }
    headline
}

fn styled_download_success(value: &str, colors: bool) -> String {
    if colors {
        format!("{ANSI_BOLD_GREEN}{value}{ANSI_RESET}")
    } else {
        value.to_string()
    }
}

fn styled_stat(value: &str, colors: bool) -> String {
    if colors {
        format!("{ANSI_BRIGHT_CYAN}{value}{ANSI_RESET}")
    } else {
        value.to_string()
    }
}

fn styled_muted(value: &str, colors: bool) -> String {
    if colors {
        format!("{ANSI_MUTED}{value}{ANSI_RESET}")
    } else {
        value.to_string()
    }
}

fn styled_label(value: &str, colors: bool) -> String {
    if colors {
        format!("{ANSI_DIM}{value}{ANSI_RESET}")
    } else {
        value.to_string()
    }
}

fn plain_model_kind(kind: &str) -> &'static str {
    match model_kind_code(kind) {
        "gguf" => "GGUF",
        "mlx" => "MLX",
        _ => "unknown",
    }
}

fn average_bytes_per_sec(stats: &DownloadStats) -> Option<u64> {
    let bytes = stats.bytes?;
    let elapsed = stats.elapsed.as_secs_f64();
    (elapsed > 0.0).then(|| (bytes as f64 / elapsed).round() as u64)
}

fn download_bytes_per_sec(stats: &DownloadStats) -> Option<(u64, &'static str)> {
    if let Some(bytes_per_second) = stats.bytes_per_second {
        return Some((bytes_per_second, "speed"));
    }
    average_bytes_per_sec(stats).map(|bytes_per_second| (bytes_per_second, "avg"))
}

fn format_elapsed(duration: Duration) -> String {
    let seconds = duration.as_secs_f64();
    if seconds < 60.0 {
        format!("{seconds:.1}s")
    } else {
        format!("{seconds:.0}s")
    }
}

fn format_download_stat_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{:.1} MB", bytes as f64 / 1e6)
    } else if bytes >= 1_000 {
        format!("{:.1} KB", bytes as f64 / 1e3)
    } else {
        format!("{bytes}B")
    }
}

fn download_summary_lines(input: &DownloadRenderInput<'_>, colors: bool) -> Vec<String> {
    let model_name = input
        .path
        .file_name()
        .and_then(|value| value.to_str())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| input.path.to_string_lossy().to_string());
    let mut lines = Vec::new();
    lines.push(format!(
        "   {}  {model_name}",
        styled_label("model", colors)
    ));
    if let Some(details) = input.details {
        lines.push(format!(
            "   {}   {}",
            styled_label("type", colors),
            plain_model_kind(details.kind)
        ));
    }
    let paths = input.all_paths();
    let part_count = input.part_count();
    if part_count > 1 {
        lines.push(format!(
            "   {}  {part_count}",
            styled_label("parts", colors)
        ));
        lines.push(format!("   {}", styled_label("paths", colors)));
        for path in paths {
            lines.push(format!("      {}", path.display()));
        }
    } else {
        lines.push(format!(
            "   {}   {}",
            styled_label("path", colors),
            input.path.display()
        ));
    }
    lines
}

impl SearchFormatter for ConsoleFormatter {
    fn is_json(&self) -> bool {
        false
    }

    fn render_catalog_empty(
        &self,
        query: &str,
        filter: SearchArtifactFilter,
        sort: SearchSort,
    ) -> Result<()> {
        eprintln!(
            "🔎 No {} catalog models matched '{}' (sorted by {}).",
            filter_label(filter),
            query,
            sort_label(sort)
        );
        Ok(())
    }

    fn render_catalog_results(
        &self,
        query: &str,
        filter: SearchArtifactFilter,
        results: &[remote_catalog::RemoteCatalogModel],
        limit: usize,
        sort: SearchSort,
    ) -> Result<()> {
        let mut output = String::new();
        writeln!(
            &mut output,
            "📚 {} catalog matches for '{}' ({})",
            filter_label(filter),
            query,
            sort_label(sort)
        )?;
        if let Some(summary) = super::formatters::local_capacity_summary() {
            writeln!(&mut output, "{}", summary)?;
        }
        writeln!(&mut output)?;
        for model in results.iter().take(limit) {
            let model_ref = remote_catalog_model_ref(model);
            let size = model.size.as_deref().unwrap_or("unknown size");
            writeln!(&mut output, "• {}  {}", model.name, size)?;
            writeln!(&mut output, "  ref: {}", model_ref)?;
            if let Some(description) = model.description.as_deref() {
                writeln!(&mut output, "  {}", description)?;
            }
            if let Some(size) = model.size.as_deref()
                && let Some(fit) = fit_hint_for_size_label(size)
            {
                writeln!(&mut output, "  {}", fit)?;
            }
            writeln!(&mut output)?;
        }
        mesh_llm_cli::pager::print_or_page(&output)
    }

    fn render_hf_empty(
        &self,
        query: &str,
        filter: SearchArtifactFilter,
        sort: SearchSort,
    ) -> Result<()> {
        eprintln!(
            "🔎 No Hugging Face {} matches for '{}' (sorted by {}).",
            filter_label(filter),
            query,
            sort_label(sort)
        );
        Ok(())
    }

    fn render_hf_results(
        &self,
        query: &str,
        filter: SearchArtifactFilter,
        sort: SearchSort,
        results: &[SearchHit],
    ) -> Result<()> {
        let mut output = String::new();
        writeln!(
            &mut output,
            "🔎 Hugging Face {} matches for '{}' ({})",
            filter_label(filter),
            query,
            sort_label(sort)
        )?;
        if let Some(summary) = super::formatters::local_capacity_summary() {
            writeln!(&mut output, "{}", summary)?;
        }
        writeln!(&mut output)?;
        for (index, result) in results.iter().enumerate() {
            writeln!(&mut output, "{}. 📦 {}", index + 1, result.repo_id)?;
            writeln!(&mut output, "   type: {}", result.kind)?;
            if let Some(variant_count) = result.variant_count {
                writeln!(&mut output, "   🧬 variants: {} available", variant_count)?;
            }
            let mut stats = Vec::new();
            if let Some(size) = &result.size_label {
                stats.push(format!("size: {} 📏", size));
            }
            if let Some(downloads) = result.downloads {
                stats.push(format!("⬇️ {}", format_count(downloads)));
            }
            if let Some(likes) = result.likes {
                stats.push(format!("❤️ {}", format_count(likes)));
            }
            if !stats.is_empty() {
                writeln!(&mut output, "   {}", stats.join("  "))?;
            }
            let mut caps = vec!["💬 text".to_string()];
            if result.capabilities.multimodal_label().is_some() {
                caps.push("🎛️ multimodal".to_string());
            }
            if let Some(label) = result.capabilities.vision_label() {
                caps.push(format!("👁️ vision ({label})"));
            }
            if let Some(label) = result.capabilities.audio_label() {
                caps.push(format!("🔊 audio ({label})"));
            }
            if let Some(label) = result.capabilities.reasoning_label() {
                caps.push(format!("🧠 reasoning ({label})"));
            }
            if let Some(label) = result.capabilities.tool_use_label() {
                caps.push(format!("🛠️ tool use ({label})"));
            }
            writeln!(&mut output, "   capabilities: {}", caps.join("  "))?;
            writeln!(
                &mut output,
                "   repo: {}",
                huggingface_repo_url(&result.repo_id)
            )?;
            writeln!(&mut output, "   ref: {}", result.exact_ref)?;
            writeln!(
                &mut output,
                "   show: mesh-llm models show {}",
                result.exact_ref
            )?;
            writeln!(
                &mut output,
                "   download: mesh-llm models download {}",
                result.exact_ref
            )?;
            if let Some(size) = &result.size_label
                && let Some(fit) = fit_hint_for_size_label(size)
            {
                writeln!(&mut output, "   {}", fit)?;
            }
            if let Some(model) = result.catalog.as_ref() {
                match model.size.as_deref() {
                    Some(size) => {
                        writeln!(&mut output, "   ⭐ Recommended: {} ({})", model.name, size)?
                    }
                    None => writeln!(&mut output, "   ⭐ Recommended: {}", model.name)?,
                }
                if let Some(description) = model.description.as_deref() {
                    writeln!(&mut output, "   {}", description)?;
                }
            }
            writeln!(&mut output)?;
        }
        mesh_llm_cli::pager::print_or_page(&output)
    }
}

impl ModelsFormatter for ConsoleFormatter {
    fn render_recommended(&self, models: &[remote_catalog::RemoteCatalogModel]) -> Result<()> {
        let mut output = String::new();
        writeln!(&mut output, "📚 Recommended models")?;
        writeln!(&mut output)?;
        for model in models {
            let model_capabilities = catalog_model_capabilities(model);
            let model_ref = remote_catalog_model_ref(model);
            let size = model.size.as_deref().unwrap_or("unknown size");
            writeln!(&mut output, "• {}  {}", model.name, size)?;
            writeln!(&mut output, "  ref: {}", model_ref)?;
            if let Some(description) = model.description.as_deref() {
                writeln!(&mut output, "  {}", description)?;
            }
            if let Some(draft) = remote_catalog_model_draft_ref(model) {
                writeln!(&mut output, "  🧠 Draft: {}", draft)?;
            }
            if let Some(label) = model_capabilities.vision_label() {
                writeln!(&mut output, "  👁️ Vision: {}", label)?;
            }
            if let Some(label) = model_capabilities.audio_label() {
                writeln!(&mut output, "  🔊 Audio: {}", label)?;
            }
            if let Some(label) = model_capabilities.reasoning_label() {
                writeln!(&mut output, "  🧠 Reasoning: {}", label)?;
            }
            writeln!(&mut output)?;
        }
        mesh_llm_cli::pager::print_or_page(&output)
    }

    fn render_installed(&self, rows: &[InstalledRow]) -> Result<()> {
        if rows.is_empty() {
            println!("📦 No installed models found");
            println!("   HF cache: {}", huggingface_cache_dir().display());
            return Ok(());
        }

        let mut output = String::new();
        writeln!(&mut output, "💾 Installed models")?;
        writeln!(
            &mut output,
            "📁 HF cache: {}",
            huggingface_cache_dir().display()
        )?;
        writeln!(&mut output)?;
        writeln!(&mut output, "🗑️ Delete example: {}", rows[0].delete_command)?;
        writeln!(&mut output)?;
        for row in rows {
            writeln!(&mut output, "📦 {}", row.name)?;
            writeln!(&mut output, "   type: {}", installed_model_kind(&row.path))?;
            if let Some(layer_count) = row.layer_count {
                writeln!(&mut output, "   layers: {} 🧩", layer_count)?;
            }
            if let Some(bytes) = row.size {
                writeln!(&mut output, "   size: {} 📏", format_installed_size(bytes))?;
            }
            writeln!(
                &mut output,
                "   owner: {}",
                if row.managed_by_mesh {
                    "mesh-managed"
                } else {
                    "external"
                }
            )?;
            if let Some(last_used_at) = row.last_used_at.as_deref()
                && let Some(label) = format_relative_timestamp(last_used_at)
            {
                writeln!(&mut output, "   last used: {}", label)?;
            }
            let mut caps = vec!["💬 text".to_string()];
            if row.capabilities.multimodal_label().is_some() {
                caps.push("🎛️ multimodal".to_string());
            }
            if let Some(label) = row.capabilities.vision_label() {
                caps.push(format!("👁️ vision ({label})"));
            }
            if let Some(label) = row.capabilities.audio_label() {
                caps.push(format!("🔊 audio ({label})"));
            }
            if let Some(label) = row.capabilities.reasoning_label() {
                caps.push(format!("🧠 reasoning ({label})"));
            }
            if let Some(label) = row.capabilities.tool_use_label() {
                caps.push(format!("🛠️ tool use ({label})"));
            }
            writeln!(&mut output, "   capabilities: {}", caps.join("  "))?;
            writeln!(&mut output, "   ref: {}", row.model_ref)?;
            if let Some(command) = row.show_command.as_deref() {
                writeln!(&mut output, "   show: {command}")?;
            } else {
                writeln!(&mut output, "   show: not available for layered packages")?;
            }
            if let Some(command) = row.download_command.as_deref() {
                writeln!(&mut output, "   download: {command}")?;
            } else {
                writeln!(
                    &mut output,
                    "   download: not available for layered packages"
                )?;
            }
            writeln!(&mut output, "   delete: {}", row.delete_command)?;
            writeln!(&mut output, "   path: {}", row.path.display())?;
            if let Some(model) = row.catalog_model.as_ref() {
                if let Some(description) = model.description.as_deref() {
                    writeln!(&mut output, "   about: {}", description)?;
                }
                if let Some(draft) = model.draft.as_deref() {
                    writeln!(&mut output, "   🧠 draft: {}", draft)?;
                }
            }
            writeln!(&mut output)?;
        }
        mesh_llm_cli::pager::print_or_page(&output)
    }

    fn render_show(&self, details: &ModelDetails, variants: Option<&[ModelDetails]>) -> Result<()> {
        if model_kind_code(details.kind) == "mlx" {
            println!("🔎 {}", details.exact_ref);
        } else {
            println!("🔎 {}", details.display_name);
        }
        if let Some(summary) = super::formatters::local_capacity_summary() {
            println!("{}", summary);
        }
        println!();
        println!("Ref: {}", details.exact_ref);
        println!("Type: {}", details.kind);
        println!("Source: {}", format_source_label(details.source));
        if let Some(size) = details.size_label.as_deref() {
            println!("Size: {size}");
            if let Some(fit) = fit_hint_for_size_label(size) {
                println!("Fit: {}", fit);
            }
        }
        if let Some(description) = details.description.as_deref() {
            println!("About: {description}");
        }
        if let Some(draft) = details.draft.as_deref() {
            println!("🧠 Draft: {draft}");
        }
        println!("Capabilities:");
        println!("  💬 text");
        if details.capabilities.multimodal_label().is_some() {
            println!("  🎛️ multimodal");
        }
        if let Some(label) = details.capabilities.vision_label() {
            println!("  👁️ vision ({label})");
        }
        if let Some(label) = details.capabilities.audio_label() {
            println!("  🔊 audio ({label})");
        }
        if let Some(label) = details.capabilities.reasoning_label() {
            println!("  🧠 reasoning ({label})");
        }
        println!("📥 Download:");
        if model_kind_code(details.kind) == "mlx" {
            println!("   mesh-llm models download {}", details.exact_ref);
        } else {
            println!("   {}", details.download_url);
        }

        if let Some(variants) = variants
            && !variants.is_empty()
        {
            println!();
            println!("Variants:");
            let mut rows = Vec::new();
            for variant in variants {
                let size = variant.size_label.as_deref().unwrap_or("-");
                let fit = variant
                    .size_label
                    .as_deref()
                    .and_then(fit_hint_for_size_label)
                    .unwrap_or_else(|| "-".to_string());
                let selected = variant.exact_ref == details.exact_ref;
                rows.push((
                    variant_selector_label(&variant.exact_ref),
                    size.to_string(),
                    fit,
                    variant.exact_ref.clone(),
                    selected,
                ));
            }
            let mut table = TabWriter::new(Vec::new()).padding(2);
            writeln!(&mut table, "sel\tquant\tsize\tfit\tref")?;
            writeln!(&mut table, "---\t-----\t----\t---\t---")?;
            for (quant, size, fit, r#ref, selected) in rows {
                writeln!(
                    &mut table,
                    "{}\t{}\t{}\t{}\t{}",
                    if selected { "*" } else { " " },
                    quant,
                    size,
                    fit,
                    r#ref
                )?;
            }
            table.flush()?;
            print!("{}", String::from_utf8_lossy(&table.into_inner()?));
        }
        Ok(())
    }

    fn render_download(&self, input: DownloadRenderInput<'_>) -> Result<()> {
        let colors = std::io::stdout().is_terminal();
        println!("{}", downloaded_model_headline(input.stats, colors));
        println!();
        for line in download_summary_lines(&input, colors) {
            println!("{line}");
        }
        if let Some((_draft_name, draft_path)) = input.draft {
            println!();
            println!("{}", styled_download_success("✓ Downloaded draft", colors));
            println!(
                "   {}   {}",
                styled_label("path", colors),
                draft_path.display()
            );
        }
        Ok(())
    }

    fn render_layer_package_download(
        &self,
        model_ref: &str,
        package_ref: &str,
        path: &std::path::Path,
    ) -> Result<()> {
        println!("✅ Downloaded layer package");
        println!("   requested: {model_ref}");
        println!("   package: {package_ref}");
        println!("   {}", path.display());
        Ok(())
    }

    fn render_updates_status(&self, _repo: Option<&str>, _all: bool, _check: bool) -> Result<()> {
        Ok(())
    }

    fn render_delete_preview(&self, resolved: &CliResolvedModel) -> Result<()> {
        println!("🗑️ Model delete preview");
        println!();
        println!("Name: {}", resolved.display_name);
        if resolved.paths.len() > 1 {
            println!("Paths ({}):", resolved.paths.len());
            for path in &resolved.paths {
                println!("  {}", path.display());
            }
        } else {
            println!("Path: {}", resolved.path.display());
        }
        println!("Mode: installed model ref resolution");
        let file_size = resolved
            .paths
            .iter()
            .map(|path| std::fs::metadata(path).map(|m| m.len()).unwrap_or(0))
            .sum();
        println!("Size: {}", format_installed_size(file_size));
        if resolved.derived_stage_paths.is_empty() {
            println!("Derived stage cache files: 0");
        } else {
            println!(
                "Derived stage cache files ({}):",
                resolved.derived_stage_paths.len()
            );
            for path in &resolved.derived_stage_paths {
                println!("  {}", path.display());
            }
        }
        if !resolved.matched_records.is_empty() {
            println!();
            println!("{} usage record(s) found:", resolved.matched_records.len());
            for record in &resolved.matched_records {
                println!(
                    "  - {} (last used: {})",
                    record.lookup_key, record.last_used_at
                );
            }
        }
        println!();
        println!("To confirm deletion, run with --yes flag.");
        Ok(())
    }

    fn render_delete_result(&self, result: &CliDeleteResult) -> Result<()> {
        println!("✅ Model deleted successfully");
        println!();
        println!("Deleted paths:");
        for p in &result.deleted_paths {
            println!("  {}", p.display());
        }
        println!();
        println!(
            "Reclaimed: {}",
            format_installed_size(result.reclaimed_bytes)
        );
        println!("Metadata files removed: {}", result.removed_metadata_files);
        println!("Usage records purged: {}", result.removed_usage_records);
        println!(
            "Derived stage cache files removed: {}",
            result.removed_derived_cache_files
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{DownloadStats, download_summary_lines, downloaded_model_headline};
    use crate::commands::models::formatters::DownloadRenderInput;
    use std::path::Path;
    use std::time::Duration;

    #[test]
    fn downloaded_model_headline_includes_inline_stats() {
        let stats = DownloadStats {
            bytes: Some(105_000_000),
            elapsed: Duration::from_millis(14_800),
            bytes_per_second: None,
        };

        assert_eq!(
            downloaded_model_headline(Some(&stats), false),
            "✓ Downloaded model ─ 105.0 MB in 14.8s · avg 7.1 MB/s"
        );
    }

    #[test]
    fn downloaded_model_headline_styles_terminal_output() {
        let stats = DownloadStats {
            bytes: Some(105_000_000),
            elapsed: Duration::from_millis(14_800),
            bytes_per_second: None,
        };

        assert_eq!(
            downloaded_model_headline(Some(&stats), true),
            "\u{1b}[1;32m✓ Downloaded model\u{1b}[0m\u{1b}[90m ─ \u{1b}[0m\u{1b}[96m105.0 MB in 14.8s\u{1b}[0m \u{1b}[90m· avg\u{1b}[0m \u{1b}[96m7.1 MB/s\u{1b}[0m"
        );
    }

    #[test]
    fn multipart_download_summary_lists_all_parts() {
        let paths = vec![
            Path::new("/cache/model-00001-of-00003.gguf"),
            Path::new("/cache/model-00002-of-00003.gguf"),
            Path::new("/cache/model-00003-of-00003.gguf"),
        ];
        let input = DownloadRenderInput {
            model_ref: "org/repo:model",
            path: paths[0],
            paths: &paths,
            details: None,
            stats: None,
            include_draft: false,
            draft: None,
        };

        let lines = download_summary_lines(&input, false);

        assert!(
            lines
                .iter()
                .any(|line| line.contains("parts") && line.contains("3"))
        );
        assert!(
            !lines
                .iter()
                .any(|line| line.split_whitespace().next() == Some("path"))
        );
        assert!(lines.iter().any(|line| line.trim_start() == "paths"));
        assert!(
            lines
                .iter()
                .any(|line| line.contains("model-00001-of-00003.gguf"))
        );
        assert!(
            lines
                .iter()
                .any(|line| line.contains("model-00002-of-00003.gguf"))
        );
        assert!(
            lines
                .iter()
                .any(|line| line.contains("model-00003-of-00003.gguf"))
        );
    }
}
