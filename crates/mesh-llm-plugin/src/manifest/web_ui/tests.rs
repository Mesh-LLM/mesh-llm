use super::*;

fn manifest_with_bundle(root_path: &str) -> proto::PluginWebUiManifest {
    proto::PluginWebUiManifest {
        bundles: vec![proto::PluginWebUiBundleManifest {
            id: "main".into(),
            root_path: root_path.into(),
        }],
        ..Default::default()
    }
}

#[test]
fn web_ui_packaging_rejects_remote_urls() {
    let manifest = proto::PluginWebUiManifest {
        pages: vec![proto::PluginWebUiPageManifest {
            route: "https://example.test/plugin.js".into(),
            ..Default::default()
        }],
        ..Default::default()
    };

    let error = PackagedPluginWebUi::try_from(&manifest).expect_err("remote route should fail");

    assert!(error.to_string().contains("remote URL"), "{error}");
}

#[test]
fn web_ui_packaging_rejects_absolute_paths() {
    let manifest = manifest_with_bundle("/var/lib/plugin-ui");

    let error = PackagedPluginWebUi::try_from(&manifest).expect_err("absolute root should fail");

    assert!(error.to_string().contains("absolute path"), "{error}");
}

#[test]
fn web_ui_packaging_rejects_traversal_paths() {
    let manifest = proto::PluginWebUiManifest {
        config_sections: vec![proto::PluginWebUiConfigSectionManifest {
            entry_script: "../escape.js".into(),
            ..Default::default()
        }],
        ..Default::default()
    };

    let error = PackagedPluginWebUi::try_from(&manifest).expect_err("traversal should fail");

    assert!(error.to_string().contains("traversal"), "{error}");
}

#[test]
fn web_ui_packaging_rejects_multiple_bundle_roots() {
    let manifest = proto::PluginWebUiManifest {
        bundles: vec![
            proto::PluginWebUiBundleManifest {
                id: "main".into(),
                root_path: "dist".into(),
            },
            proto::PluginWebUiBundleManifest {
                id: "admin".into(),
                root_path: "admin".into(),
            },
        ],
        ..Default::default()
    };

    let error = PackagedPluginWebUi::try_from(&manifest).expect_err("multiple roots should fail");

    assert!(error.to_string().contains("one bundle root"), "{error}");
}

#[test]
fn web_ui_packaging_rejects_invalid_config_parent_tab() {
    let manifest = proto::PluginWebUiManifest {
        config_sections: vec![proto::PluginWebUiConfigSectionManifest {
            entry_script: "settings.js".into(),
            parent_tab: Some("advanced".into()),
            ..Default::default()
        }],
        ..Default::default()
    };

    let error =
        PackagedPluginWebUi::try_from(&manifest).expect_err("invalid parent_tab should fail");

    assert!(error.to_string().contains("integrations"), "{error}");
}
