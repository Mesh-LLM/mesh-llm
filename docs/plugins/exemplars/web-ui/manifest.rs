use mesh_llm_plugin::{
    capability, config_integer, config_schema, config_setting, constraint_range,
    plugin_manifest, proto, web_ui, web_ui_bundle, web_ui_config_section, web_ui_page,
};

pub fn exemplar_manifest() -> proto::PluginManifest {
    plugin_manifest! {
        capability("exemplar.notes.v1"),
        config_schema("web-ui-exemplar")
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
            ),
        web_ui()
            .bundle(web_ui_bundle("main", "bundle"))
            .page(
                web_ui_page("overview", "Exemplar Overview", "overview", "register-mesh-plugin-ui.js")
                    .bundle_id("main"),
            )
            .config_section(
                web_ui_config_section("retention", "Exemplar Retention", "register-mesh-plugin-ui.js")
                    .parent_tab("integrations")
                    .bundle_id("main"),
            ),
    }
}
