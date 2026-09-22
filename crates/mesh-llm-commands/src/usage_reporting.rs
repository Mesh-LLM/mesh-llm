//! Bridge between CLI dispatch and [`mesh_llm_analytics`].
//!
//! Kept out of `mesh_llm_analytics` so that crate stays a leaf with no CLI or
//! hardware dependencies, and out of the shipped binary crate because these
//! are reporting policy decisions, not dispatch wiring.

use mesh_llm_analytics::{Event, Properties};
use mesh_llm_cli::{Cli, Command};
use mesh_llm_events::{CliCommandFamily, CliCommandOutcome};
use std::ffi::OsString;
use std::path::Path;
use std::time::Instant;

/// Start reporting for this process, unless the command is itself an
/// analytics command.
///
/// Running `mesh-llm analytics disable` must not report anything: a command
/// whose purpose is to stop reporting is the worst possible moment to report.
/// `mesh-llm analytics status` is excluded for the same reason — asking what
/// is collected should not itself be collected.
pub fn init_for_cli(cli: &Cli) {
    if matches!(cli.command, Some(Command::Analytics { .. })) {
        return;
    }
    mesh_llm_analytics::init(crate::analytics::config_preference(cli.config.as_deref()));
}

/// Start reporting when the CLI failed to parse, so install and version
/// counts are not silently biased toward well-formed invocations.
///
/// `raw_args` is argv. A malformed analytics invocation
/// (`mesh-llm analytics --typo`, `mesh-llm analytics --help`) never reaches
/// the parsed exclusion in [`init_for_cli`], so it is excluded here too —
/// otherwise the one command family promised not to report would report.
pub fn init_for_unparsed(raw_args: &[OsString], config_path: Option<&Path>) {
    if mesh_llm_cli::raw_args_invoke_analytics(raw_args) {
        return;
    }
    mesh_llm_analytics::init(crate::analytics::config_preference(config_path));
}

/// Record the outcome of a one-shot command.
///
/// Only the family and the outcome are reported. Both are closed enums, so no
/// argument, path, model name, or error text can ride along.
pub fn record_cli_command(family: CliCommandFamily, outcome: CliCommandOutcome) {
    mesh_llm_analytics::capture(
        Event::CliCommand,
        Properties::new()
            .with("family", family.as_str())
            .with("outcome", outcome.as_str()),
    );
}

/// A `serve` session being measured from start to shutdown.
pub struct ServeSession {
    started: Instant,
}

impl ServeSession {
    /// Record that a runtime surface started, and begin timing it.
    pub fn start(cli: &Cli) -> Self {
        mesh_llm_analytics::capture(Event::ServeStarted, serve_properties(cli));
        report_hardware_profile();
        Self {
            started: Instant::now(),
        }
    }

    /// Record the session ending, bucketing how long it lasted.
    ///
    /// Session length is the signal that separates a node someone actually
    /// runs from one that crashed or was tried once.
    pub fn finish(self, succeeded: bool) {
        mesh_llm_analytics::capture(
            Event::ServeStopped,
            Properties::new()
                .with(
                    "session_length",
                    mesh_llm_analytics::bucket_duration_secs(self.started.elapsed().as_secs()),
                )
                .with("succeeded", succeeded),
        );
    }
}

/// Shape of a runtime surface invocation, from flags only.
fn serve_properties(cli: &Cli) -> Properties {
    Properties::new()
        .with("surface", if cli.client { "client" } else { "serve" })
        .with("auto", cli.auto)
        // Whether peers were named, never which peers.
        .with("joined_explicitly", !cli.join.is_empty())
        // Whether discovery was requested, never the mesh name given to it.
        .with("discover", cli.discover.is_some())
        .with("publish", cli.publish)
        .with("headless", cli.headless)
        // Whether a model was requested, never which one: both `--model` and
        // `--gguf` take filesystem paths.
        .with(
            "model_requested",
            !cli.model.is_empty() || !cli.gguf.is_empty(),
        )
}

/// Report the shape of this machine, once per serving process.
///
/// Hardware probes shell out to platform tools, so this runs on a blocking
/// thread and reports whenever it finishes. Startup never waits for it.
pub fn report_hardware_profile() {
    tokio::task::spawn_blocking(|| {
        use mesh_llm_system::hardware::Metric;

        // Note the metric *not* requested: `Metric::Hostname` would name the
        // machine, so it is never collected rather than collected and then
        // dropped.
        let survey = mesh_llm_system::hardware::query(&[
            Metric::GpuName,
            Metric::VramBytes,
            Metric::GpuCount,
            Metric::IsSoc,
        ]);
        let flavors = mesh_llm_hardware_profile::host_runtime_profile().available_flavors;
        mesh_llm_analytics::capture(
            Event::HardwareProfile,
            hardware_properties(&survey, &flavors),
        );
    });
}

/// Build hardware properties from a survey and the detected backends.
///
/// Device names are slugged, counts and sizes are bucketed, and the available
/// backends become one boolean each so they stay queryable without needing an
/// array property.
///
/// `HardwareSurvey` also carries a hostname, and its `GpuFacts` carry
/// `stable_id`, `pci_bdf`, `vendor_uuid`, `metal_registry_id`, `dxgi_luid`,
/// and `pnp_instance_id`. Those identify a machine rather than describe it,
/// and none of them are read here.
fn hardware_properties(
    survey: &mesh_llm_system::hardware::HardwareSurvey,
    flavors: &std::collections::BTreeSet<mesh_llm_native_runtime::NativeRuntimeBackendKind>,
) -> Properties {
    use mesh_llm_native_runtime::NativeRuntimeBackendKind as Backend;

    let mut properties = Properties::new()
        .with(
            "gpu_count",
            mesh_llm_analytics::bucket_count(u64::from(survey.gpu_count)),
        )
        .with("unified_memory", survey.is_soc)
        .with("backend_metal", flavors.contains(&Backend::Metal))
        .with("backend_cuda", flavors.contains(&Backend::Cuda))
        .with("backend_rocm", flavors.contains(&Backend::Rocm))
        .with("backend_vulkan", flavors.contains(&Backend::Vulkan));

    if survey.vram_bytes > 0 {
        properties = properties.with(
            "vram_total",
            mesh_llm_analytics::bucket_gigabytes(survey.vram_bytes),
        );
    }
    if let Some(system_ram) = survey.system_ram_bytes.filter(|bytes| *bytes > 0) {
        properties = properties.with(
            "system_ram",
            mesh_llm_analytics::bucket_gigabytes(system_ram),
        );
    }
    if let Some(name) = survey.gpu_name.as_deref() {
        properties = properties.with("gpu_model", mesh_llm_analytics::Label::slug_or_redact(name));
    }
    properties
}

/// Record a model download attempt, which is the clearest signal of what
/// people are *trying* — a failed download still answers the question.
pub fn record_model_download(model_ref: &str, succeeded: bool) {
    mesh_llm_analytics::capture(
        Event::ModelDownload,
        Properties::new()
            .with(
                "model",
                mesh_llm_analytics::Label::sanitize_or_redact(model_ref),
            )
            .with("succeeded", succeeded),
    );
}

/// Deliver anything still queued, within the crate's shutdown budget.
pub async fn shutdown() {
    mesh_llm_analytics::shutdown().await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn serve_properties_describe_shape_without_naming_anything() {
        let cli = Cli::parse_from([
            "mesh-llm",
            "--auto",
            "--join",
            "some-secret-peer-token",
            "--model",
            "Qwen2.5-32B-Instruct",
        ]);
        let properties = serve_properties(&cli);
        let rendered = format!("{properties:?}");

        assert!(rendered.contains("joined_explicitly"));
        assert!(!rendered.contains("some-secret-peer-token"), "{rendered}");
        assert!(!rendered.contains("Qwen2.5-32B-Instruct"), "{rendered}");
    }

    #[test]
    fn gguf_paths_never_reach_the_properties() {
        let cli = Cli::parse_from(["mesh-llm", "--gguf", "/Users/dan/models/private.gguf"]);
        let rendered = format!("{:?}", serve_properties(&cli));
        assert!(!rendered.contains("/Users/dan"), "{rendered}");
        assert!(rendered.contains("model_requested"));
    }

    fn survey(
        gpu_name: Option<&str>,
        gpu_count: u8,
        vram_bytes: u64,
        is_soc: bool,
    ) -> mesh_llm_system::hardware::HardwareSurvey {
        mesh_llm_system::hardware::HardwareSurvey {
            vram_bytes,
            gpu_name: gpu_name.map(str::to_owned),
            gpu_count,
            // A real survey can carry this. The assertions below check it
            // never reaches the properties.
            hostname: Some("dans-macbook-pro.local".to_owned()),
            is_soc,
            gpu_vram: Vec::new(),
            gpu_reserved: Vec::new(),
            gpus: Vec::new(),
            gpu_name_source: None,
            system_ram_bytes: None,
            ram_offload_bytes: 0,
        }
    }

    fn metal_only() -> std::collections::BTreeSet<mesh_llm_native_runtime::NativeRuntimeBackendKind>
    {
        std::collections::BTreeSet::from([mesh_llm_native_runtime::NativeRuntimeBackendKind::Metal])
    }

    #[test]
    fn hardware_properties_slug_the_device_and_bucket_its_memory() {
        const GB: u64 = 1024 * 1024 * 1024;
        let rendered = format!(
            "{:?}",
            hardware_properties(
                &survey(Some("Apple M1 Pro"), 1, 36 * GB, true),
                &metal_only()
            )
        );

        assert!(rendered.contains("apple-m1-pro"), "{rendered}");
        assert!(rendered.contains("32-64"), "vram not bucketed: {rendered}");
        assert!(rendered.contains("backend_metal"), "{rendered}");
        // Exact VRAM is a fingerprint; only the bucket should survive.
        assert!(
            !rendered.contains("38654705664"),
            "exact vram leaked: {rendered}"
        );
    }

    #[test]
    fn hardware_properties_never_carry_the_hostname() {
        let rendered = format!(
            "{:?}",
            hardware_properties(
                &survey(Some("NVIDIA GeForce RTX 4090"), 1, 24 << 30, false),
                &metal_only(),
            )
        );
        assert!(
            !rendered.contains("dans-macbook"),
            "hostname leaked: {rendered}"
        );
        assert!(rendered.contains("nvidia-geforce-rtx-4090"), "{rendered}");
    }

    #[test]
    fn hardware_properties_bucket_multi_gpu_counts() {
        let rendered = format!(
            "{:?}",
            hardware_properties(
                &survey(Some("NVIDIA GeForce RTX 4090"), 6, 144 << 30, false),
                &metal_only(),
            )
        );
        // Six GPUs land in the 5-8 bucket, not as an exact count.
        assert!(rendered.contains("5-8"), "{rendered}");
    }

    #[test]
    fn hardware_properties_tolerate_a_machine_with_no_gpu() {
        let rendered = format!(
            "{:?}",
            hardware_properties(&survey(None, 0, 0, false), &metal_only())
        );
        assert!(rendered.contains("gpu_count"), "{rendered}");
        assert!(!rendered.contains("gpu_model"), "{rendered}");
        assert!(!rendered.contains("vram_total"), "{rendered}");
    }

    #[test]
    fn analytics_commands_do_not_start_reporting() {
        // `analytics disable` must be inert, so the opt-out is not itself an
        // event. This asserts the guard, not the global reporter.
        let cli = Cli::parse_from(["mesh-llm", "analytics", "disable"]);
        assert!(matches!(cli.command, Some(Command::Analytics { .. })));
        init_for_cli(&cli);
    }

    #[test]
    fn malformed_analytics_invocations_do_not_start_reporting() {
        // These never produce a parsed `Cli`, so they bypass the check above
        // and land on the parse-exit path instead.
        for raw in [
            &["mesh-llm", "analytics", "--typo"][..],
            &["mesh-llm", "analytics", "--help"][..],
            &["mesh-llm", "--debug", "analytics", "bogus"][..],
        ] {
            let args: Vec<OsString> = raw.iter().map(OsString::from).collect();
            assert!(
                mesh_llm_cli::raw_args_invoke_analytics(&args),
                "{raw:?} must be recognized as an analytics invocation",
            );
            init_for_unparsed(&args, None);
        }
    }
}
