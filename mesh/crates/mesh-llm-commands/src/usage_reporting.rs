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

/// Whether this process is an internal plugin service rather than something a
/// person started.
///
/// `serve` spawns its own executable back with the hidden `--plugin <id>`
/// flag (see `blobstore_plugin_spec`). That child reaches the same CLI entry
/// point, so without this it initializes reporting too and the node is
/// counted twice: two `serve_started` — the child's carrying
/// `model_requested: false`, because it was handed no model — and two
/// `hardware_profile` for one machine, against a single `serve_stopped` from
/// the parent. A node with N plugins would report N+1 of each.
///
/// The parent has already reported. The child is an implementation detail of
/// that same session, so it reports nothing.
fn is_internal_plugin_service(cli: &Cli) -> bool {
    cli.plugin.is_some()
}

/// Whether mesh-llm spawned this process as an internal helper.
///
/// The updater runs the freshly extracted binary with `--version` to verify a
/// bundle before installing it. That child reaches the same entry point, so
/// without this it reports a command, records the new build as the current
/// version, and thereby steals the version transition from the restart that
/// actually applies the update -- which then reads as `Unchanged` and gets
/// classified `external`. Worse, an install that fails after verification
/// leaves a recorded upgrade that never happened.
///
/// Same reasoning as [`is_internal_plugin_service`]: the process exists to
/// serve a user action already being reported, so it reports nothing itself.
fn is_internal_helper() -> bool {
    std::env::var_os(mesh_llm_analytics::ENV_INTERNAL_HELPER).is_some()
}

/// Start reporting for this process, unless the command is itself an
/// analytics command or an internal plugin service.
///
/// Running `mesh-llm analytics disable` must not report anything: a command
/// whose purpose is to stop reporting is the worst possible moment to report.
/// `mesh-llm analytics status` is excluded for the same reason — asking what
/// is collected should not itself be collected.
pub fn init_for_cli(cli: &Cli) {
    if matches!(cli.command, Some(Command::Analytics { .. }))
        || is_internal_plugin_service(cli)
        || is_internal_helper()
    {
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
    if mesh_llm_cli::raw_args_invoke_analytics(raw_args) || is_internal_helper() {
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
        // Whether this node manages its own upgrades. Without it an
        // auto-updating fleet and a hand-run node are indistinguishable.
        .with("auto_update", cli.auto_update)
        // Whether this process is the second half of a self-update.
        //
        // A self-update `exec`s a new binary, so one user session produces
        // two `serve_started` and at most one `serve_stopped`. The first
        // process may or may not have flushed before it was replaced, which
        // makes the duplicate nondeterministic. Marking the continuation is
        // what lets it be excluded instead of quietly inflating starts.
        .with("post_update_restart", is_post_update_restart())
}

/// Whether the self-updater `exec`d this process.
fn is_post_update_restart() -> bool {
    std::env::var_os(mesh_llm_analytics::ENV_SELF_UPDATE_MARKER).is_some()
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
        let flavors = skippy_hardware_profile::host_runtime_profile().available_flavors;
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
    flavors: &std::collections::BTreeSet<skippy_native_runtime::NativeRuntimeBackendKind>,
) -> Properties {
    use skippy_native_runtime::NativeRuntimeBackendKind as Backend;

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
    // Always set. Leaving the property absent when the probe returns no name
    // collapses every such install into one unlabelled bucket, which reads as
    // "unknown" when it usually means "no GPU at all" -- a different and more
    // interesting answer. The two reasons it can be missing are split apart
    // rather than merged.
    properties = properties.with("gpu_model", gpu_model(survey));
    properties
}

/// The `gpu_model` value for a survey.
///
/// A name is only reported when a naming probe actually stands behind it.
/// `HardwareSurvey` documents that `hydrate_gpu_facts_with_identities`
/// backfills placeholder `"GPU N"` names tagged [`GpuNameSource::Unknown`],
/// and `CpuBrandString` is a CPU string kept only so older surveys still
/// deserialize. Slugging either would invent a convincing-looking device --
/// `gpu-0`, or a CPU model reported as a GPU -- so both are `unreported`.
///
/// "No GPU" is only claimed when nothing in the survey evidences one. The
/// naming and counting probes fail independently of each other, so the
/// per-device facts are consulted rather than trusting `gpu_count` alone.
///
/// `vram_bytes` is deliberately *not* evidence: it is a budget, not a device
/// fact, and on a machine with no accelerator it is the RAM-offload credit,
/// which is routinely non-zero. Treating it as a GPU would report `unreported`
/// for every GPU-less Linux host.
fn gpu_model(survey: &mesh_llm_system::hardware::HardwareSurvey) -> mesh_llm_analytics::Value {
    use mesh_llm_system::hardware::GpuNameSource;

    let probed_name = survey.gpu_name.as_deref().filter(|_| {
        !matches!(
            survey.gpu_name_source,
            None | Some(GpuNameSource::Unknown) | Some(GpuNameSource::CpuBrandString)
        )
    });
    if let Some(name) = probed_name {
        return mesh_llm_analytics::Value::from(mesh_llm_analytics::Label::slug_or_redact(name));
    }

    let has_a_gpu = survey.gpu_count > 0 || !survey.gpus.is_empty() || !survey.gpu_vram.is_empty();
    if has_a_gpu {
        // A device is there; nothing trustworthy named it.
        mesh_llm_analytics::Value::from("unreported")
    } else {
        mesh_llm_analytics::Value::from("none")
    }
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
    fn an_internal_plugin_service_is_not_a_started_node() {
        // `serve` spawns `mesh-llm --log-format json --plugin blobstore`.
        // Observed against a local sink before this guard: one user `serve`
        // produced two `serve_started` (the second `model_requested: false`)
        // and two `hardware_profile`, against one `serve_stopped`.
        let plugin = Cli::parse_from(["mesh-llm", "--log-format", "json", "--plugin", "blobstore"]);
        assert!(is_internal_plugin_service(&plugin));

        let user_serve = Cli::parse_from(["mesh-llm", "--gguf", "/tmp/model.gguf"]);
        assert!(!is_internal_plugin_service(&user_serve));
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
        // A real naming probe stands behind a name unless a test says otherwise.
        let source = gpu_name
            .is_some()
            .then_some(mesh_llm_system::hardware::GpuNameSource::NativeRuntimeDevice);
        survey_named_by(gpu_name, source, gpu_count, vram_bytes, is_soc)
    }

    fn survey_named_by(
        gpu_name: Option<&str>,
        gpu_name_source: Option<mesh_llm_system::hardware::GpuNameSource>,
        gpu_count: u8,
        vram_bytes: u64,
        is_soc: bool,
    ) -> mesh_llm_system::hardware::HardwareSurvey {
        mesh_llm_system::hardware::HardwareSurvey {
            gpu_name_source,
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
            system_ram_bytes: None,
            ram_offload_bytes: 0,
        }
    }

    fn metal_only() -> std::collections::BTreeSet<skippy_native_runtime::NativeRuntimeBackendKind> {
        std::collections::BTreeSet::from([skippy_native_runtime::NativeRuntimeBackendKind::Metal])
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

    /// A GPU-less machine reports `gpu_model: none` rather than omitting the
    /// property. An absent property groups every such install under one
    /// unlabelled bucket, which is indistinguishable from a probe failure.
    #[test]
    fn hardware_properties_tolerate_a_machine_with_no_gpu() {
        let rendered = format!(
            "{:?}",
            hardware_properties(&survey(None, 0, 0, false), &metal_only())
        );
        assert!(rendered.contains("gpu_count"), "{rendered}");
        assert!(
            rendered.contains("none"),
            "no explicit no-GPU value: {rendered}"
        );
        assert!(!rendered.contains("vram_total"), "{rendered}");
    }

    /// A GPU that exists but was not named is a probe gap, not an absence of
    /// hardware, and the two must not collapse into the same bucket.
    #[test]
    fn an_unnamed_gpu_is_distinguished_from_having_no_gpu() {
        let unnamed = format!(
            "{:?}",
            hardware_properties(&survey(None, 2, 48 << 30, false), &metal_only())
        );
        assert!(unnamed.contains("unreported"), "{unnamed}");

        let absent = format!(
            "{:?}",
            hardware_properties(&survey(None, 0, 0, false), &metal_only())
        );
        assert!(!absent.contains("unreported"), "{absent}");
    }

    /// `hydrate_gpu_facts_with_identities` backfills placeholder `"GPU N"`
    /// names with no probe behind them. Slugging one would publish `gpu-0` as
    /// though it were a real device.
    #[test]
    fn a_placeholder_gpu_name_is_not_reported_as_a_device() {
        let rendered = format!(
            "{:?}",
            hardware_properties(
                &survey_named_by(
                    Some("GPU 0"),
                    Some(mesh_llm_system::hardware::GpuNameSource::Unknown),
                    1,
                    8 << 30,
                    false,
                ),
                &metal_only(),
            )
        );
        assert!(
            !rendered.contains("gpu-0"),
            "placeholder published: {rendered}"
        );
        assert!(rendered.contains("unreported"), "{rendered}");
    }

    /// `CpuBrandString` is a CPU string retained only so older surveys still
    /// deserialize. Reporting it would file a CPU model as a GPU.
    #[test]
    fn a_cpu_brand_string_is_not_reported_as_a_gpu() {
        let rendered = format!(
            "{:?}",
            hardware_properties(
                &survey_named_by(
                    Some("Apple M1 Pro"),
                    Some(mesh_llm_system::hardware::GpuNameSource::CpuBrandString),
                    1,
                    16 << 30,
                    true,
                ),
                &metal_only(),
            )
        );
        assert!(!rendered.contains("apple-m1-pro"), "{rendered}");
        assert!(rendered.contains("unreported"), "{rendered}");
    }

    /// The counting probe can fail while the per-device facts survive, so a
    /// zero count alone is not enough to claim there is no GPU.
    #[test]
    fn per_device_facts_outvote_a_zero_gpu_count() {
        let mut survey = survey(None, 0, 24 << 30, false);
        survey.gpu_vram = vec![24 << 30];
        let rendered = format!("{:?}", hardware_properties(&survey, &metal_only()));
        assert!(rendered.contains("unreported"), "{rendered}");
    }

    /// `vram_bytes` is a budget, not a device: on a GPU-less host it carries
    /// the RAM-offload credit. Treating it as evidence would report
    /// `unreported` for every CPU-only Linux box.
    #[test]
    fn a_ram_offload_budget_is_not_evidence_of_a_gpu() {
        let rendered = format!(
            "{:?}",
            hardware_properties(&survey(None, 0, 12 << 30, false), &metal_only())
        );
        assert!(rendered.contains("none"), "{rendered}");
        assert!(!rendered.contains("unreported"), "{rendered}");
    }

    /// The flag that decides whether a node upgrades itself has to be on the
    /// event, or an auto-updating fleet cannot be told from hand-run nodes.
    #[test]
    fn serve_properties_report_whether_the_node_self_updates() {
        let updating = format!(
            "{:?}",
            serve_properties(&Cli::parse_from(["mesh-llm", "--auto-update"]))
        );
        assert!(updating.contains("auto_update"), "{updating}");
        assert!(updating.contains("post_update_restart"), "{updating}");
    }

    /// `mesh-llm-analytics` reads the self-update marker by name rather than
    /// depending on `mesh-llm-system`, which would drag the release-fetch and
    /// hardware tree into a deliberately leaf crate. This crate sees both
    /// definitions, so it is where the two are held together.
    #[test]
    fn the_self_update_marker_name_agrees_across_crates() {
        assert_eq!(
            mesh_llm_analytics::ENV_SELF_UPDATE_MARKER,
            mesh_llm_system::autoupdate::SELF_UPDATE_ATTEMPTED_ENV,
        );
        assert_eq!(
            mesh_llm_analytics::ENV_INTERNAL_HELPER,
            mesh_llm_system::autoupdate::INTERNAL_HELPER_ENV,
        );
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
