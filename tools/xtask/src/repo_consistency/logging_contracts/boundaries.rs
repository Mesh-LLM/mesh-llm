use super::source;
use std::collections::BTreeSet;
use syn::visit::{self, Visit};

const HOST: &str = "mesh/crates/mesh-llm-host-runtime/src";
const STORE: &str = "mesh/crates/mesh-llm-log-store/src";
const MAX_LINES: usize = 999;

#[derive(Default)]
struct Declarations {
    external_modules: BTreeSet<String>,
    inline_modules: BTreeSet<String>,
    symbols: BTreeSet<String>,
    implementations: BTreeSet<String>,
    includes: BTreeSet<String>,
    has_tests: bool,
}
impl<'ast> Visit<'ast> for Declarations {
    fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
        let modules = if item.content.is_none() {
            &mut self.external_modules
        } else {
            &mut self.inline_modules
        };
        modules.insert(item.ident.to_string());
        visit::visit_item_mod(self, item);
    }
    fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
        self.symbols.insert(item.sig.ident.to_string());
        visit::visit_item_fn(self, item);
    }
    fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
        self.symbols.insert(item.sig.ident.to_string());
        visit::visit_impl_item_fn(self, item);
    }
    fn visit_item_struct(&mut self, item: &'ast syn::ItemStruct) {
        self.symbols.insert(item.ident.to_string());
        visit::visit_item_struct(self, item);
    }
    fn visit_item_enum(&mut self, item: &'ast syn::ItemEnum) {
        self.symbols.insert(item.ident.to_string());
        visit::visit_item_enum(self, item);
    }
    fn visit_item_const(&mut self, item: &'ast syn::ItemConst) {
        self.symbols.insert(item.ident.to_string());
        visit::visit_item_const(self, item);
    }
    fn visit_item_impl(&mut self, item: &'ast syn::ItemImpl) {
        let segment = match item.self_ty.as_ref() {
            syn::Type::Path(path) => path.path.segments.last(),
            _ => None,
        };
        if let Some(segment) = segment {
            self.implementations.insert(segment.ident.to_string());
        }
        visit::visit_item_impl(self, item);
    }
    fn visit_attribute(&mut self, attr: &'ast syn::Attribute) {
        self.has_tests |= attr
            .path()
            .segments
            .last()
            .is_some_and(|segment| segment.ident == "test");
        visit::visit_attribute(self, attr);
    }
    fn visit_item_macro(&mut self, item: &'ast syn::ItemMacro) {
        if item.mac.path.is_ident("include") {
            let path: syn::LitStr =
                syn::parse2(item.mac.tokens.clone()).expect("literal include path");
            self.includes.insert(path.value());
        }
        visit::visit_item_macro(self, item);
    }
}
fn parsed(text: &str) -> Declarations {
    let file = syn::parse_file(text).expect("valid owned Rust source");
    let mut declarations = Declarations::default();
    declarations.visit_file(&file);
    declarations
}

#[test]
fn logging_boundaries_imports_and_calls_do_not_claim_declaration_ownership() {
    let source = r#"
        use crate::ArtifactFileStore;
        pub use child::LoggingQueryFacade;
        mod child;
        fn caller() { child::start_persistence_worker(); }
        impl ArtifactFileStore { fn execute(&self) {} }
        #[cfg(test)] mod tests { #[tokio::test] async fn checks() {} }
    "#;
    let declarations = parsed(source);
    assert!(!declarations.symbols.contains("LoggingQueryFacade"));
    assert!(!declarations.symbols.contains("start_persistence_worker"));
    assert!(declarations.symbols.contains("caller"));
    assert!(declarations.symbols.contains("execute"));
    assert!(declarations.implementations.contains("ArtifactFileStore"));
    assert!(declarations.external_modules.contains("child"));
    assert!(declarations.inline_modules.contains("tests"));
    assert!(declarations.has_tests);
}
fn bounded(path: &str) -> String {
    let text = source(path);
    assert!(!text.trim().is_empty(), "empty owner {path}");
    assert!(text.lines().count() <= MAX_LINES, "oversized owner {path}");
    text
}
fn owner(parent: &str, module: &str, child: &str, moved: &[&str]) {
    let parent = parsed(&source(parent));
    assert!(
        parent.external_modules.contains(module),
        "missing external module {module}"
    );
    assert!(
        !parent.inline_modules.contains(module),
        "inline owner {module}"
    );
    bounded(child);
    for symbol in moved {
        assert!(
            !parent.symbols.contains(*symbol),
            "symbol remains in parent: {symbol}"
        );
    }
}
fn semantic_owner(parent: &str, module: &str, child: &str, moved: &[&str]) {
    owner(parent, module, child, moved);
    let text = source(child);
    let declarations = parsed(&text);
    for symbol in moved {
        assert!(
            declarations.symbols.contains(*symbol),
            "symbol missing in semantic child: {symbol}"
        );
    }
    let pure = text
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.trim_start().starts_with("//"))
        .count();
    assert!(
        pure <= 250,
        "semantic owner exceeds 250 nonblank noncomment lines: {child}"
    );
}
fn external_tests(parent: &str, child: &str) {
    owner(parent, "tests", child, &[]);
    assert!(
        !parsed(&source(parent)).has_tests,
        "tests remain in production parent"
    );
}

#[test]
fn logging_boundaries_production_responsibilities_have_named_owners() {
    for relative in [
        "logging/webhook_delivery",
        "logging/cleanup",
        "logging/raw_mesh_lifecycle",
        "runtime/operational_logging",
        "api/routes/logs/events/session",
    ] {
        let parent = format!("{HOST}/{relative}.rs");
        external_tests(&parent, &format!("{HOST}/{relative}/tests.rs"));
    }
    external_tests(
        &format!("{HOST}/api/routes/logs/mod.rs"),
        &format!("{HOST}/api/routes/logs/tests.rs"),
    );
    let state = format!("{HOST}/logging/runtime_state.rs");
    owner(
        &state,
        "query_facade",
        &format!("{HOST}/logging/runtime_state/query_facade.rs"),
        &["LoggingQueryFacade"],
    );
    owner(
        &state,
        "workers",
        &format!("{HOST}/logging/runtime_state/workers.rs"),
        &["start_persistence_worker"],
    );
    external_tests(&state, &format!("{HOST}/logging/runtime_state/tests.rs"));
    owner(
        &format!("{STORE}/maintenance.rs"),
        "execution",
        &format!("{STORE}/maintenance/execution.rs"),
        &[],
    );
    assert!(
        !parsed(&source(&format!("{STORE}/maintenance.rs")))
            .implementations
            .contains("ArtifactFileStore")
    );
    semantic_owner(
        &format!("{HOST}/logging/service/operational_audit.rs"),
        "context",
        &format!("{HOST}/logging/service/operational_audit/context.rs"),
        &[
            "OPERATIONAL_AUDIT_CONTEXT_VERSION",
            "MAX_CONTEXT_VALUE_CHARS",
            "MAX_NUMERIC_SUMMARIES",
            "OperationalAuditSubjectKind",
            "OperationalAuditPathType",
            "OperationalAuditContext",
            "insert_optional_string",
            "valid_static_code",
            "bounded_context_value",
        ],
    );
    for parent in [
        "logging/webhook_delivery.rs",
        "logging/cleanup.rs",
        "logging/raw_mesh_lifecycle.rs",
        "runtime/operational_logging.rs",
        "api/routes/logs/mod.rs",
        "api/routes/logs/events/session.rs",
        "logging/runtime_state.rs",
    ] {
        bounded(&format!("{HOST}/{parent}"));
    }
    bounded(&format!("{STORE}/maintenance.rs"));
}

#[test]
fn logging_boundaries_store_repositories_keep_audit_detail_and_metadata_owners() {
    let repositories = format!("{STORE}/repositories.rs");
    let audit = format!("{STORE}/repositories/audit.rs");
    owner(
        &repositories,
        "audit",
        &audit,
        &[
            "AuditEntryRow",
            "StoredAuditDetail",
            "audit_entry_query_parts",
            "audit_entry_row",
            "insert_audit_entry",
            "list_audit_entries",
            "list_audit_entries_after_sequence",
        ],
    );
    owner(
        &audit,
        "detail",
        &format!("{STORE}/repositories/audit/detail.rs"),
        &[
            "StoredAuditDetail",
            "bounded_audit_value",
            "bounded_command_summary",
            "bounded_audit_code",
            "bounded_code",
        ],
    );
    owner(
        &repositories,
        "caller_metadata",
        &format!("{STORE}/repositories/caller_metadata.rs"),
        &["upsert_summary_metadata"],
    );
}

#[test]
fn logging_boundaries_mesh_connection_responsibilities_have_semantic_owners() {
    let connections = format!("{HOST}/mesh/connections.rs");
    let inbound = format!("{HOST}/mesh/connections/inbound.rs");
    semantic_owner(
        &connections,
        "inbound",
        &inbound,
        &[
            "handle_incoming",
            "handle_control_incoming",
            "accept_mesh_stream",
            "admitted_mesh_stream",
        ],
    );
    semantic_owner(
        &connections,
        "tunnel",
        &format!("{HOST}/mesh/connections/tunnel.rs"),
        &[
            "dispatch_mesh_stream",
            "forward_tunnel_stream",
            "forward_tunnel_http_stream",
            "_dispatch_streams",
            "authenticated_peer_path",
            "remove_connection_if_stable_id",
        ],
    );
    semantic_owner(
        &inbound,
        "stage",
        &format!("{HOST}/mesh/connections/inbound/stage.rs"),
        &["handle_stage_alpn"],
    );
}

fn test_children(parent: &str, directory: &str, names: &[&str]) {
    let text = bounded(parent);
    assert!(
        !parsed(&text).has_tests,
        "test bodies remain in suite parent {parent}"
    );
    for name in names {
        owner(parent, name, &format!("{directory}/{name}.rs"), &[]);
    }
}

#[test]
fn logging_boundaries_characterization_suites_are_split_by_concern() {
    test_children(
        &format!("{HOST}/api/tests/logs_api_routes.rs"),
        &format!("{HOST}/api/tests/logs_api_routes"),
        &["access_and_mutation", "read_and_export", "event_stream"],
    );
    test_children(
        &format!("{STORE}/maintenance/tests.rs"),
        &format!("{STORE}/maintenance/tests"),
        &["cleanup", "delete_one"],
    );
    test_children(
        &format!("{HOST}/network/openai/transport_tests.rs"),
        &format!("{HOST}/network/openai/transport_tests"),
        &["lifecycle", "routing"],
    );
}

#[test]
fn logging_boundaries_audit_and_gossip_suites_keep_semantic_children() {
    test_children(
        &format!("{STORE}/api_acceptance_tests/summary_audit.rs"),
        &format!("{STORE}/api_acceptance_tests/summary_audit"),
        &["basic", "sanitization", "query"],
    );
    let parent = format!("{HOST}/mesh/tests/gossip.rs");
    let declarations = parsed(&bounded(&parent));
    assert!(
        !declarations.has_tests,
        "gossip parent still contains tests"
    );
    for name in ["merge_and_refresh", "admission", "discovery"] {
        assert!(
            declarations.includes.contains(&format!("gossip/{name}.rs")),
            "missing gossip include {name}"
        );
        bounded(&format!("{HOST}/mesh/tests/gossip/{name}.rs"));
    }
}
