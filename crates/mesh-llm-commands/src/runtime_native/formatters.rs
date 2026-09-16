use anyhow::{Error, Result};
use mesh_llm_native_runtime::{
    CachePrunePlan, CandidateRejection, HostRuntimeProfile, InstalledNativeRuntime,
};
use mesh_llm_runtime_install::{
    NativeRuntimeCatalogSources, NativeRuntimeInstallOutcome, NativeRuntimeInstallStatus,
    NativeRuntimeResolutionError,
};
use serde::Serialize;
use serde_json::json;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize)]
pub(crate) struct AvailableRuntimeRow {
    pub(crate) id: String,
    pub(crate) mesh_version: Option<String>,
    pub(crate) skippy_abi: String,
    pub(crate) backend: String,
    pub(crate) os: String,
    pub(crate) arch: String,
    pub(crate) supported: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) rejection_reasons: Vec<CandidateRejection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) url: Option<String>,
}

#[derive(Serialize)]
pub(crate) struct NativeRuntimeDoctorReport {
    pub(crate) healthy: bool,
    pub(crate) status: String,
    pub(crate) blockers: Vec<String>,
    pub(crate) recommendations: Vec<String>,
    pub(crate) running_mesh_version: String,
    pub(crate) selected_mesh_version: String,
    pub(crate) configured_skippy_abi: Option<String>,
    /// The raw `runtime.native_runtime.selection` value from config.
    pub(crate) configured_selection: Option<String>,
    /// The selection the doctor actually resolved against: a `--llama-flavor`
    /// CLI override wins over the configured value.
    pub(crate) effective_selection: Option<String>,
    pub(crate) host: HostRuntimeProfile,
    pub(crate) cache_path: PathBuf,
    pub(crate) selected_runtime_id: Option<String>,
    pub(crate) selected_runtime_flavor: Option<String>,
    pub(crate) selected_runtime_path: Option<PathBuf>,
    pub(crate) installed_count: usize,
    pub(crate) selected_version_installed_count: usize,
}

pub(crate) trait RuntimeNativeFormatter {
    /// Renders the available runtimes together with the catalogs that were
    /// consulted to list them, so both output modes explain where the rows
    /// came from.
    fn render_available(
        &self,
        rows: &[AvailableRuntimeRow],
        sources: &NativeRuntimeCatalogSources,
    ) -> Result<()>;
    fn render_installed(
        &self,
        installed: &[InstalledNativeRuntime],
        cache_root: &Path,
    ) -> Result<()>;
    fn render_install(&self, outcome: &NativeRuntimeInstallOutcome) -> Result<()>;
    fn render_install_error(&self, error: &Error) -> Result<()>;
    fn render_remove(
        &self,
        native_runtime_id: &str,
        mesh_version: &str,
        removed: bool,
    ) -> Result<()>;
    fn render_prune(&self, plan: &CachePrunePlan) -> Result<()>;
    fn render_doctor(&self, report: &NativeRuntimeDoctorReport) -> Result<()>;
}

pub(crate) struct HumanFormatter;
pub(crate) struct JsonFormatter;

pub(crate) fn runtime_native_formatter(json_output: bool) -> Box<dyn RuntimeNativeFormatter> {
    if json_output {
        Box::new(JsonFormatter)
    } else {
        Box::new(HumanFormatter)
    }
}

impl RuntimeNativeFormatter for HumanFormatter {
    fn render_available(
        &self,
        rows: &[AvailableRuntimeRow],
        sources: &NativeRuntimeCatalogSources,
    ) -> Result<()> {
        let mut err = mesh_llm_events::console_err();
        writeln!(err, "🔎 Catalogs consulted")?;
        for line in sources.describe() {
            writeln!(err, "   {line}")?;
        }
        print_available_human(rows);
        Ok(())
    }

    fn render_installed(
        &self,
        installed: &[InstalledNativeRuntime],
        cache_root: &Path,
    ) -> Result<()> {
        print_installed_human(installed, cache_root);
        Ok(())
    }

    fn render_install(&self, outcome: &NativeRuntimeInstallOutcome) -> Result<()> {
        print_install_human(outcome);
        Ok(())
    }

    fn render_install_error(&self, error: &Error) -> Result<()> {
        let mut err = mesh_llm_events::console_err();
        writeln!(err, "❌ Native runtime install failed")?;
        writeln!(err, "   Reason: {error}")?;
        // A resolution failure carries its explanation (catalogs consulted,
        // rejected candidates) as structure; the causes underneath, such as
        // the resolver's own verdict or a manifest that failed to parse, stay
        // one per line.
        if let Some(resolution) = error.downcast_ref::<NativeRuntimeResolutionError>() {
            for line in resolution.explanation_lines() {
                writeln!(err, "   {line}")?;
            }
        }
        for cause in error.chain().skip(1) {
            writeln!(err, "   cause: {cause}")?;
        }
        writeln!(err, "   Try: mesh-llm runtime list --available")?;
        Ok(())
    }

    fn render_remove(
        &self,
        native_runtime_id: &str,
        mesh_version: &str,
        removed: bool,
    ) -> Result<()> {
        let mut err = mesh_llm_events::console_err();
        if removed {
            writeln!(
                err,
                "✅ Removed native runtime {native_runtime_id} for MeshLLM {mesh_version}"
            )?;
        } else {
            writeln!(
                err,
                "🔎 Native runtime {native_runtime_id} for MeshLLM {mesh_version} was not installed"
            )?;
        }
        Ok(())
    }

    fn render_prune(&self, plan: &CachePrunePlan) -> Result<()> {
        let mut err = mesh_llm_events::console_err();
        if plan.remove_dirs.is_empty() {
            writeln!(err, "✅ Native runtime cache already pruned")?;
        } else {
            writeln!(
                err,
                "✅ Pruned {} native runtime cache version(s)",
                plan.remove_dirs.len()
            )?;
            for dir in &plan.remove_dirs {
                writeln!(err, "   removed: {}", dir.display())?;
            }
        }
        Ok(())
    }

    fn render_doctor(&self, report: &NativeRuntimeDoctorReport) -> Result<()> {
        print_doctor_human(report);
        Ok(())
    }
}

impl RuntimeNativeFormatter for JsonFormatter {
    fn render_available(
        &self,
        rows: &[AvailableRuntimeRow],
        sources: &NativeRuntimeCatalogSources,
    ) -> Result<()> {
        print_json(&json!({
            "catalogs": sources,
            "runtimes": rows,
        }))
    }

    fn render_installed(
        &self,
        installed: &[InstalledNativeRuntime],
        _cache_root: &Path,
    ) -> Result<()> {
        print_json(installed)
    }

    fn render_install(&self, outcome: &NativeRuntimeInstallOutcome) -> Result<()> {
        print_json(&json!({
            "status": install_status_label(outcome.status.clone()),
            "runtime": outcome.runtime,
            "resolution": outcome.resolution,
            "catalogs": outcome.sources,
        }))
    }

    fn render_install_error(&self, error: &Error) -> Result<()> {
        // `resolution` is the structured explanation of a failed selection
        // (catalogs, plausible candidates and their rejection reasons); it is
        // null for every other failure, and `context` keeps the cause chain.
        print_json(&json!({
            "status": "error",
            "error": {
                "type": "native_runtime_install_failed",
                "message": error.to_string(),
                "context": error.chain().skip(1).map(ToString::to_string).collect::<Vec<_>>(),
                "resolution": error.downcast_ref::<NativeRuntimeResolutionError>(),
            },
        }))
    }

    fn render_remove(
        &self,
        native_runtime_id: &str,
        mesh_version: &str,
        removed: bool,
    ) -> Result<()> {
        print_json(&json!({
            "mesh_version": mesh_version,
            "native_runtime_id": native_runtime_id,
            "removed": removed,
        }))
    }

    fn render_prune(&self, plan: &CachePrunePlan) -> Result<()> {
        print_json(plan)
    }

    fn render_doctor(&self, report: &NativeRuntimeDoctorReport) -> Result<()> {
        print_json(report)
    }
}

fn print_json(value: &(impl Serialize + ?Sized)) -> Result<()> {
    let mut out = mesh_llm_events::machine_out();
    writeln!(out, "{}", serde_json::to_string_pretty(value)?)?;
    Ok(())
}

fn install_status_label(status: NativeRuntimeInstallStatus) -> &'static str {
    match status {
        NativeRuntimeInstallStatus::AlreadyInstalled => "already_installed",
        NativeRuntimeInstallStatus::Installed => "installed",
    }
}

fn print_available_human(rows: &[AvailableRuntimeRow]) {
    let mut out = mesh_llm_events::console_out();
    if rows.is_empty() {
        let _ = writeln!(out, "📦 No native runtime manifest entries found");
        let _ = writeln!(
            out,
            "   Pass --manifest or --bundle-dir to inspect available runtimes."
        );
        return;
    }
    let _ = writeln!(out, "📦 Available native runtimes");
    for row in rows {
        let marker = if row.supported { "✅" } else { "⚠️" };
        let status = if row.supported {
            "compatible"
        } else {
            "not compatible"
        };
        let _ = writeln!(
            out,
            "  - {marker} {} {status} ({}, {}/{})",
            row.id, row.backend, row.os, row.arch
        );
        if let Some(mesh_version) = row.mesh_version.as_deref() {
            let _ = writeln!(
                out,
                "    MeshLLM: {mesh_version}; Skippy ABI: {}",
                row.skippy_abi
            );
        } else {
            let _ = writeln!(
                out,
                "    MeshLLM: unspecified; Skippy ABI: {}",
                row.skippy_abi
            );
        }
        for reason in &row.rejection_reasons {
            let _ = writeln!(out, "    reason: {}", format_rejection(reason));
        }
    }
}

fn print_installed_human(installed: &[InstalledNativeRuntime], cache_root: &Path) {
    let mut out = mesh_llm_events::console_out();
    if installed.is_empty() {
        let _ = writeln!(out, "📦 No local native runtimes found");
        let _ = writeln!(out, "   cache: {}", cache_root.display());
        return;
    }
    let _ = writeln!(out, "📦 Local native runtimes");
    let _ = writeln!(out, "   cache: {}", cache_root.display());
    for runtime in installed {
        let _ = writeln!(
            out,
            "  - ✅ {} {} ({})",
            runtime.native_runtime_id, runtime.mesh_version, runtime.flavor
        );
        let _ = writeln!(out, "    path: {}", runtime.path.display());
    }
}

fn print_install_human(outcome: &NativeRuntimeInstallOutcome) {
    let mut err = mesh_llm_events::console_err();
    match outcome.status {
        NativeRuntimeInstallStatus::AlreadyInstalled => {
            let _ = writeln!(
                err,
                "✅ Native runtime already installed: {}",
                outcome.runtime.native_runtime_id
            );
            let _ = writeln!(err, "   version: {}", outcome.runtime.mesh_version);
            let _ = writeln!(err, "   flavor: {}", outcome.runtime.flavor);
            let _ = writeln!(err, "   path: {}", outcome.runtime.path.display());
        }
        NativeRuntimeInstallStatus::Installed => {
            let _ = writeln!(err, "✅ Installed {}", outcome.runtime.native_runtime_id);
            let _ = writeln!(err, "   version: {}", outcome.runtime.mesh_version);
            let _ = writeln!(err, "   flavor: {}", outcome.runtime.flavor);
            let _ = writeln!(err, "   path: {}", outcome.runtime.path.display());
        }
    }
    for line in outcome.sources.describe() {
        let _ = writeln!(err, "   catalog: {line}");
    }
}

fn print_doctor_human(report: &NativeRuntimeDoctorReport) {
    let mut out = mesh_llm_events::console_out();
    let _ = writeln!(out, "🩺 MeshLLM doctor");
    let _ = writeln!(out);
    let _ = writeln!(out, "Native runtime:");
    let _ = writeln!(out, "  status: {}", report.status);
    let _ = writeln!(
        out,
        "  running MeshLLM version: {}",
        report.running_mesh_version
    );
    let _ = writeln!(
        out,
        "  selected runtime version: {}",
        report.selected_mesh_version
    );
    if report.selected_mesh_version != report.running_mesh_version {
        let _ = writeln!(
            out,
            "  version pin: native runtime version is pinned by config"
        );
    }
    if let Some(skippy_abi) = &report.configured_skippy_abi {
        let _ = writeln!(out, "  configured Skippy ABI: {skippy_abi}");
    }
    if let Some(selection) = &report.configured_selection {
        let _ = writeln!(out, "  configured selection: {selection}");
    }
    if let Some(selection) = &report.effective_selection
        && report.configured_selection.as_deref() != Some(selection.as_str())
    {
        let _ = writeln!(
            out,
            "  effective selection: {selection} (from --llama-flavor)"
        );
    }
    let _ = writeln!(out, "  cache: {}", report.cache_path.display());
    let _ = writeln!(out, "  host: {}/{}", report.host.os, report.host.arch);
    let flavors = report
        .host
        .available_flavors
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let _ = writeln!(out, "  detected flavors: {flavors}");
    match &report.selected_runtime_id {
        Some(id) => {
            let _ = writeln!(out, "  selected: {id}");
            if let Some(flavor) = &report.selected_runtime_flavor {
                let _ = writeln!(out, "  flavor: {flavor}");
            }
            if let Some(path) = &report.selected_runtime_path {
                let _ = writeln!(out, "  path: {}", path.display());
            }
        }
        None => {
            let _ = writeln!(out, "  selected: none");
        }
    }
    let _ = writeln!(out, "  installed: {}", report.installed_count);
    let _ = writeln!(
        out,
        "  installed for selected version: {}",
        report.selected_version_installed_count
    );
    if !report.blockers.is_empty() {
        let _ = writeln!(out);
        let _ = writeln!(out, "Blockers:");
        for blocker in &report.blockers {
            let _ = writeln!(out, "  - {blocker}");
        }
    }
    if !report.recommendations.is_empty() {
        let _ = writeln!(out);
        let _ = writeln!(out, "Recommended next steps:");
        for recommendation in &report.recommendations {
            let _ = writeln!(out, "  - {recommendation}");
        }
    }
}

fn format_rejection(reason: &CandidateRejection) -> String {
    // The wording lives on `CandidateRejection` itself so the install
    // diagnostics and this listing describe a rejection the same way.
    reason.to_string()
}
