use mesh_llm_plugin::{
    PluginMetadata, SimplePlugin, capability, config_integer, config_schema, config_setting,
    constraint_range, mcp, plugin, plugin_server_info, proto, web_ui, web_ui_bundle,
    web_ui_config_section, web_ui_page,
};

pub fn exemplar_plugin() -> SimplePlugin {
    plugin! {
        metadata: PluginMetadata::new(
            "web-ui-exemplar",
            "0.1.0",
            plugin_server_info(
                "web-ui-exemplar",
                "0.1.0",
                "Web UI exemplar",
                "Buildable reference plugin for host-projected web UI",
                None::<String>,
            ),
        ),
        provides: [capability("exemplar.notes.v1")],
        config: [config_schema("web-ui-exemplar")
            .setting(
                config_setting("retention_days", config_integer())
                    .default_value(&14)
                    .constraint(constraint_range(Some("1"), Some("365")))
                    .apply_mode(proto::PluginConfigApplyMode::DynamicValidationOnly)
                    .restart_scope(proto::PluginConfigRestartScope::PluginProcess)
                    .description("How long exemplar notes stay available.")
                    .label("Retention days")
                    .help("Persisted through host-owned plugin config, not bundle-local storage.")
                    .category("exemplar-retention", "Retention", "Exemplar retention settings", 10)
                    .order(20)
                    .unit("days")
                    .control_hint("number"),
            )],
        web_ui: [web_ui()
            .bundle(web_ui_bundle("main", "bundle"))
            .page(
                web_ui_page("overview", "Exemplar Notes", "overview", "register-mesh-plugin-ui.js")
                    .bundle_id("main"),
            )
            .config_section(
                web_ui_config_section("page-actions", "Exemplar page", "register-mesh-plugin-ui.js")
                    .parent_tab("integrations")
                    .bundle_id("main"),
            ),
        ],
        mcp: [
            mcp::tool("status")
                .description("Show that the exemplar's non-UI capability remains available.")
                .handle(|_args, _context| Box::pin(async {
                    Ok(serde_json::json!({
                        "capability": "exemplar.notes.v1",
                        "status": "available"
                    }))
                })),
        ],
    }
}
