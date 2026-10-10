//! Planner-owned boolean signals consumed by reusable workflows. They sit
//! beside the ownership decision so workflow YAML never re-implements path
//! matching in expressions.

use crate::ci_plan::request::{FORCE_ALL, Profile};
use crate::ci_plan::selection::documentation_only;
use serde_json::{Map, Value};

#[derive(Debug, Clone, Copy)]
enum Signal {
    RustChanged,
    UiChanged,
    WebsiteChanged,
    WebsiteDocsChanged,
    PluginExemplarsChanged,
    CliSurfaceChanged,
    DocsOnly,
    BackendChanged,
    RunnerContractRequired,
}

impl Signal {
    const ALL: [Signal; 9] = [
        Signal::RustChanged,
        Signal::UiChanged,
        Signal::WebsiteChanged,
        Signal::WebsiteDocsChanged,
        Signal::PluginExemplarsChanged,
        Signal::CliSurfaceChanged,
        Signal::DocsOnly,
        Signal::BackendChanged,
        Signal::RunnerContractRequired,
    ];

    fn name(self) -> &'static str {
        match self {
            Signal::RustChanged => "rust_changed",
            Signal::UiChanged => "ui_changed",
            Signal::WebsiteChanged => "website_changed",
            Signal::WebsiteDocsChanged => "website_docs_changed",
            Signal::PluginExemplarsChanged => "plugin_exemplars_changed",
            Signal::CliSurfaceChanged => "cli_surface_changed",
            Signal::DocsOnly => "docs_only",
            Signal::BackendChanged => "backend_changed",
            Signal::RunnerContractRequired => "runner_contract_required",
        }
    }
}

const RUST_DOMAINS: [&str; 6] = [
    "rust",
    "native-abi",
    "protocol",
    "split-serving",
    "model-download",
    "cli",
];
const BACKEND_DOMAINS: [&str; 5] = [
    "native-abi",
    "runtime-product",
    "backend-cuda",
    "backend-rocm",
    "backend-vulkan",
];

struct Change<'a> {
    files: &'a [String],
    domains: &'a [String],
}

impl Change<'_> {
    fn has_domain(&self, name: &str) -> bool {
        self.domains.iter().any(|domain| domain == name)
    }

    fn any_domain(&self, names: &[&str]) -> bool {
        names.iter().any(|name| self.has_domain(name))
    }

    fn any_path(&self, prefixes: &[&str]) -> bool {
        self.files
            .iter()
            .any(|path| prefixes.iter().any(|prefix| path.starts_with(prefix)))
    }

    fn value(&self, signal: Signal) -> bool {
        match signal {
            Signal::RustChanged => self.any_domain(&RUST_DOMAINS),
            Signal::UiChanged => self.has_domain("ui"),
            Signal::WebsiteChanged => self.has_domain("website"),
            Signal::WebsiteDocsChanged => {
                self.any_path(&["website/src/docs/pages/", "website/src/_includes/"])
            }
            Signal::PluginExemplarsChanged => self.any_path(&["docs/plugins/exemplars/"]),
            Signal::CliSurfaceChanged => self.has_domain("cli"),
            Signal::DocsOnly => documentation_only(self.domains),
            Signal::BackendChanged => self.any_domain(&BACKEND_DOMAINS),
            Signal::RunnerContractRequired => {
                self.has_domain("ci-control") || self.has_domain("runner-infra")
            }
        }
    }
}

/// Every signal; exhaustive runs set all but `docs_only`.
pub(super) fn signals(
    files: &[String],
    domains: &[String],
    profile: Profile,
) -> Map<String, Value> {
    let exhaustive = profile.exhaustive() || files.iter().any(|path| path == FORCE_ALL);
    let change = Change { files, domains };
    Signal::ALL
        .into_iter()
        .map(|signal| {
            let value = match signal {
                Signal::DocsOnly if exhaustive => false,
                _ if exhaustive => true,
                _ => change.value(signal),
            };
            (signal.name().to_owned(), Value::Bool(value))
        })
        .collect()
}
