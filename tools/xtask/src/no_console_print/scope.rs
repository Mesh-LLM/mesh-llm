//! Product ownership for the console-print gate. Exemptions are category rules,
//! never approvals for individual print locations.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use syn::{spanned::Spanned, visit::Visit};

use crate::command::DynResult;
use crate::repo_consistency::{CargoMetadata, workspace_metadata};

/// Crate directories outside `cargo tree -p mesh-llm --edges normal`, checked
/// against Cargo metadata below. The guard includes optional dependencies and
/// all target platforms, so a default-feature or host-only tree cannot hide a
/// future product dependency. `mesh-client` is deliberately NOT exempt: the
/// shipping host now depends on the `mesh-llm-client` package.
const NON_PRODUCT_CRATES: &[&str] = &[
    "skippy-prompt",
    "skippy-bench",
    "skippy-model-package",
    "llama-spec-bench",
    "skippy-quantize",
    "skippy-correctness",
    "mesh-llm-test-harness",
    "metrics-server",
];

/// Files that implement the console output facility, and therefore legitimately
/// hold a terminal handle. Every other product file must route output through
/// that facility. These are whole-file category rules: a surface either owns
/// terminal access or it does not, so this list can never grow to excuse one
/// convenient write inside an ordinary module.
///
/// - the sink-aware console writer and the pre-sink CLI lifecycle fallback,
/// - the inline progress renderers, which paint a transient cursor-addressed
///   redraw that has no structured representation,
/// - the TUI's own output manager, fd capture, and terminal backend,
/// - the runtime's tracing writer, the last-resort path used when event
///   emission itself fails,
/// - skippy-server's stderr telemetry sink, whose entire purpose is writing
///   newline-delimited events to stderr,
/// - CLI presentation surfaces that render to the user's terminal by design.
pub(super) const CONSOLE_OUTPUT_OWNERS: &[&str] = &[
    "crates/mesh-llm-events/src/console.rs",
    "crates/mesh-llm-events/src/command_lifecycle.rs",
    "crates/mesh-llm-events/src/terminal_progress.rs",
    "crates/mesh-llm-tui/src/terminal_progress.rs",
    "crates/mesh-llm-tui/src/output/console_capture.rs",
    "crates/mesh-llm-tui/src/output/formatting.rs",
    "crates/mesh-llm-tui/src/output/terminal_out.rs",
    "crates/mesh-llm-host-runtime/src/runtime/tracing_writer.rs",
    "crates/skippy-server/src/telemetry.rs",
    "crates/mesh-llm-cli/src/pager.rs",
    "crates/mesh-llm-commands/src/gpus/tune_runner.rs",
];

pub(super) fn owns_console_output(path: &str) -> bool {
    CONSOLE_OUTPUT_OWNERS.contains(&path)
}

pub(super) fn is_product_source(path: &str) -> bool {
    let parts: Vec<_> = path.split('/').collect();
    if parts.len() < 3 || parts[0] != "crates" {
        return false;
    }
    let name = parts.last().expect("nonempty path");
    !NON_PRODUCT_CRATES.contains(&parts[1])
        && !parts[2..].iter().any(|part| matches!(*part, "tests" | "examples" | "benches"))
        && *name != "tests.rs"
        && !name.ends_with("_tests.rs")
        && *name != "build.rs"
        && name.ends_with(".rs")
        // release_targets.rs ships only mesh-llm's src/main.rs. Other src/bin
        // targets are auxiliary tools; src/main.rs remains checked everywhere.
        && !(parts.get(2) == Some(&"src") && parts.get(3) == Some(&"bin"))
}

pub(super) fn check_exempt_crates(repo_root: &Path) -> DynResult<()> {
    let metadata = workspace_metadata(repo_root, "console print product scope")?;
    check_metadata(&metadata)
}

fn check_metadata(metadata: &CargoMetadata) -> DynResult<()> {
    let packages: BTreeMap<_, _> = metadata
        .packages
        .iter()
        .map(|package| (package.name.as_str(), package))
        .collect();
    if !packages.contains_key("mesh-llm") {
        return Err("console print scope: missing mesh-llm package".into());
    }
    let mut visited = BTreeSet::new();
    let mut pending = vec!["mesh-llm"];
    while let Some(name) = pending.pop() {
        if !visited.insert(name) {
            continue;
        }
        let package = packages[name];
        let directory = package
            .manifest_path
            .parent()
            .and_then(Path::file_name)
            .and_then(|name| name.to_str())
            .ok_or("invalid package directory")?;
        if NON_PRODUCT_CRATES.contains(&directory) {
            return Err(format!("console print scope: exempt crate {directory} is a normal dependency of mesh-llm; remove its exemption").into());
        }
        pending.extend(
            package
                .dependencies
                .iter()
                .filter(|dep| dep.kind.is_none() && packages.contains_key(dep.name.as_str()))
                .map(|dep| dep.name.as_str()),
        );
    }
    Ok(())
}

/// Blank only parsed, explicitly test-only modules, retaining byte positions
/// and newlines for the exact occurrence ratchet. Comments, strings and braces
/// inside raw strings cannot change a module's extent. Unsupported fragments
/// remain fully in scope rather than accidentally suppressing product code.
pub(super) fn without_test_modules(source: &str) -> String {
    let Ok(file) = syn::parse_file(source) else {
        return source.to_owned();
    };
    let mut visitor = TestModules { spans: Vec::new() };
    visitor.visit_file(&file);
    let mut bytes = source.as_bytes().to_vec();
    for span in visitor.spans {
        for byte in &mut bytes[span.byte_range()] {
            if *byte != b'\n' && *byte != b'\r' {
                *byte = b' ';
            }
        }
    }
    String::from_utf8(bytes).expect("replacing complete syntax spans preserves UTF-8")
}

struct TestModules {
    spans: Vec<proc_macro2::Span>,
}

impl<'ast> Visit<'ast> for TestModules {
    fn visit_item_mod(&mut self, module: &'ast syn::ItemMod) {
        let test_only = module.attrs.iter().any(|attr| {
            attr.path().is_ident("cfg")
                && attr
                    .parse_args::<syn::Path>()
                    .is_ok_and(|path| path.is_ident("test"))
        });
        if test_only {
            self.spans.push(module.span());
        } else {
            syn::visit::visit_item_mod(self, module);
        }
    }
}

#[cfg(test)]
mod tests;
