use super::*;

pub(super) fn native_runtime_presentation(rendered: &str) -> Option<SettingPresentation> {
    match rendered {
        "runtime.native_runtime.selection" => Some(
            sp(
                "Native runtime backend",
                "Pin the native runtime backend loaded on startup. Recommended auto-detects from host hardware; cpu, metal, cuda (or cudaNN), rocm, and vulkan force a backend, and exact:<id> or meshllm-<id> pin a specific installed runtime.",
                RUNTIME_CATEGORY,
                100,
            )
            .hint("select")
            .choices(&[(
                "recommended",
                "Recommended (auto-detect)",
                "Auto-detect the best backend for this host.",
            )]),
        ),
        "runtime.native_runtime.mesh_version" => Some(
            sp(
                "Native runtime Mesh version",
                "Pin the Mesh release whose native runtime bundle is loaded. Leave unset to track the current release.",
                RUNTIME_CATEGORY,
                110,
            )
            .placeholder("track current release")
            .hint("text"),
        ),
        "runtime.native_runtime.skippy_abi" => Some(
            sp(
                "Native runtime Skippy ABI",
                "Pin the Skippy ABI string of the native runtime bundle. Only meaningful alongside a pinned Mesh version.",
                RUNTIME_CATEGORY,
                120,
            )
            .hint("text"),
        ),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_runtime_settings_append_to_runtime_category() {
        for (path, order) in [
            ("runtime.native_runtime.selection", 100),
            ("runtime.native_runtime.mesh_version", 110),
            ("runtime.native_runtime.skippy_abi", 120),
        ] {
            let descriptor = built_in_config_schema_descriptor(
                &ConfigPath::parse_rendered(path).expect("valid config path"),
            )
            .expect("native runtime descriptor");
            let presentation = descriptor.presentation.expect("presentation metadata");
            assert_eq!(presentation.category_id.as_deref(), Some("runtime"));
            assert_eq!(presentation.category_label.as_deref(), Some("Runtime"));
            assert_eq!(presentation.setting_order, Some(order));
        }
        for (path, order) in [
            ("defaults.throughput.threads", 10),
            ("defaults.hardware.device", 90),
        ] {
            let setting =
                built_in_config_schema_descriptor(&ConfigPath::parse_rendered(path).unwrap())
                    .unwrap();
            assert_eq!(setting.presentation.unwrap().setting_order, Some(order));
        }
        let debug =
            built_in_config_schema_descriptor(&ConfigPath::from_fields(["runtime", "debug"]))
                .expect("debug descriptor");
        assert_eq!(
            debug
                .presentation
                .expect("debug presentation")
                .category_id
                .as_deref(),
            Some("meshllm")
        );
    }

    #[test]
    fn native_runtime_fallback_precedes_runtime_policy() {
        assert_eq!(
            fallback_category_for_path("runtime.native_runtime.future_option")
                .expect("native runtime category")
                .id,
            "runtime"
        );
        assert_eq!(
            fallback_category_for_path("runtime.activity.enabled")
                .expect("runtime policy category")
                .id,
            "runtime-policy"
        );
    }
}
