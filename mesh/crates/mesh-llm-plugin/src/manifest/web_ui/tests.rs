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
            id: "home".into(),
            label: "Home".into(),
            route: "https://example.test/plugin.js".into(),
            bundle_id: "main".into(),
            ..Default::default()
        }],
        bundles: vec![proto::PluginWebUiBundleManifest {
            id: "main".into(),
            root_path: "dist".into(),
        }],
        ..Default::default()
    };

    let error = PackagedPluginWebUi::try_from(&manifest).expect_err("remote route should fail");

    assert!(error.to_string().contains("URL-like"), "{error}");
}

#[test]
fn web_ui_packaging_rejects_absolute_paths() {
    let manifest = manifest_with_bundle("/var/lib/plugin-ui");

    let error = PackagedPluginWebUi::try_from(&manifest).expect_err("absolute root should fail");

    assert!(error.to_string().contains("absolute path"), "{error}");
}

/// `/var/lib/plugin-ui` is not `Path::is_absolute` on Windows, so a manifest
/// carrying a rooted bundle path used to be accepted there and refused
/// everywhere else. These are the shapes the rooted-path check must reject on
/// Windows; on Unix `\x`, `C:\x` and `\\server\share` are legal single
/// filenames, so the cases are Windows-only by construction.
#[cfg(windows)]
#[test]
fn web_ui_packaging_rejects_windows_rooted_paths() {
    for root_path in [
        "/plugin-ui",
        "\\plugin-ui",
        "C:\\plugin-ui",
        "\\\\server\\share\\plugin-ui",
    ] {
        let manifest = manifest_with_bundle(root_path);

        let error = match PackagedPluginWebUi::try_from(&manifest) {
            Ok(_) => panic!("rooted root should be rejected on Windows: {root_path}"),
            Err(error) => error,
        };

        assert!(
            error.to_string().contains("absolute path"),
            "{root_path}: {error}"
        );
    }
}

#[test]
fn web_ui_packaging_rejects_traversal_paths() {
    let manifest = proto::PluginWebUiManifest {
        config_sections: vec![proto::PluginWebUiConfigSectionManifest {
            id: "settings".into(),
            title: "Settings".into(),
            entry_script: "../escape.js".into(),
            bundle_id: "main".into(),
            ..Default::default()
        }],
        bundles: vec![proto::PluginWebUiBundleManifest {
            id: "main".into(),
            root_path: "dist".into(),
        }],
        ..Default::default()
    };

    let error = PackagedPluginWebUi::try_from(&manifest).expect_err("traversal should fail");

    assert!(error.to_string().contains("traversal"), "{error}");
}

#[test]
fn web_ui_packaging_rejects_hidden_path_segments() {
    let manifest = proto::PluginWebUiManifest {
        pages: vec![proto::PluginWebUiPageManifest {
            id: "home".into(),
            label: "Home".into(),
            route: "home".into(),
            bundle_id: "main".into(),
            entry_script: ".vite/app.js".into(),
            ..Default::default()
        }],
        bundles: vec![proto::PluginWebUiBundleManifest {
            id: "main".into(),
            root_path: "dist".into(),
        }],
        ..Default::default()
    };

    let error = PackagedPluginWebUi::try_from(&manifest).expect_err("hidden path should fail");

    assert!(error.to_string().contains("hidden path"), "{error}");
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
            id: "settings".into(),
            title: "Settings".into(),
            entry_script: "settings.js".into(),
            parent_tab: Some("advanced".into()),
            bundle_id: "main".into(),
        }],
        bundles: vec![proto::PluginWebUiBundleManifest {
            id: "main".into(),
            root_path: "dist".into(),
        }],
        ..Default::default()
    };

    let error =
        PackagedPluginWebUi::try_from(&manifest).expect_err("invalid parent_tab should fail");

    assert!(error.to_string().contains("integrations"), "{error}");
}

#[test]
fn web_ui_packaging_requires_bundle_for_declared_pages() {
    let manifest = proto::PluginWebUiManifest {
        pages: vec![proto::PluginWebUiPageManifest {
            id: "home".into(),
            label: "Home".into(),
            route: "home".into(),
            bundle_id: "main".into(),
            entry_script: "app.js".into(),
            ..Default::default()
        }],
        ..Default::default()
    };

    let error = PackagedPluginWebUi::try_from(&manifest).expect_err("missing bundle should fail");

    assert!(error.to_string().contains("exactly one bundle"), "{error}");
}

#[test]
fn web_ui_packaging_rejects_empty_bundle_id() {
    let manifest = proto::PluginWebUiManifest {
        pages: vec![proto::PluginWebUiPageManifest {
            id: "home".into(),
            label: "Home".into(),
            route: "home".into(),
            bundle_id: "".into(),
            entry_script: "app.js".into(),
            ..Default::default()
        }],
        bundles: vec![proto::PluginWebUiBundleManifest {
            id: "".into(),
            root_path: "dist".into(),
        }],
        ..Default::default()
    };

    let error = PackagedPluginWebUi::try_from(&manifest).expect_err("empty bundle id should fail");

    assert!(error.to_string().contains("bundle id"), "{error}");
}

#[test]
fn web_ui_packaging_rejects_empty_page_or_section_identity() {
    let manifest = proto::PluginWebUiManifest {
        pages: vec![proto::PluginWebUiPageManifest {
            label: "Home".into(),
            route: "home".into(),
            bundle_id: "main".into(),
            entry_script: "app.js".into(),
            ..Default::default()
        }],
        bundles: vec![proto::PluginWebUiBundleManifest {
            id: "main".into(),
            root_path: "dist".into(),
        }],
        ..Default::default()
    };

    let error = PackagedPluginWebUi::try_from(&manifest).expect_err("empty page id should fail");

    assert!(error.to_string().contains("page id"), "{error}");
}

#[test]
fn web_ui_packaging_rejects_empty_or_package_root_paths() {
    let empty_entry = proto::PluginWebUiManifest {
        pages: vec![proto::PluginWebUiPageManifest {
            id: "home".into(),
            label: "Home".into(),
            route: "home".into(),
            bundle_id: "main".into(),
            ..Default::default()
        }],
        bundles: vec![proto::PluginWebUiBundleManifest {
            id: "main".into(),
            root_path: "dist".into(),
        }],
        ..Default::default()
    };
    let root_package = proto::PluginWebUiManifest {
        bundles: vec![proto::PluginWebUiBundleManifest {
            id: "main".into(),
            root_path: ".".into(),
        }],
        ..Default::default()
    };

    assert!(
        PackagedPluginWebUi::try_from(&empty_entry)
            .expect_err("empty entry script should fail")
            .to_string()
            .contains("entry_script")
    );
    assert!(
        PackagedPluginWebUi::try_from(&root_package)
            .expect_err("package root should not be an asset root")
            .to_string()
            .contains("below the package root")
    );
}

#[test]
fn web_ui_packaging_rejects_unknown_bundle_id() {
    let manifest = proto::PluginWebUiManifest {
        pages: vec![proto::PluginWebUiPageManifest {
            id: "home".into(),
            label: "Home".into(),
            route: "home".into(),
            bundle_id: "admin".into(),
            entry_script: "app.js".into(),
            ..Default::default()
        }],
        bundles: vec![proto::PluginWebUiBundleManifest {
            id: "main".into(),
            root_path: "dist".into(),
        }],
        ..Default::default()
    };

    let error =
        PackagedPluginWebUi::try_from(&manifest).expect_err("unknown bundle id should fail");

    assert!(
        error.to_string().contains("declared web UI bundle"),
        "{error}"
    );
}

#[test]
fn web_ui_packaging_rejects_path_shaped_route_slug() {
    let manifest = proto::PluginWebUiManifest {
        pages: vec![proto::PluginWebUiPageManifest {
            id: "home".into(),
            label: "Home".into(),
            route: "admin/home".into(),
            bundle_id: "main".into(),
            entry_script: "app.js".into(),
            ..Default::default()
        }],
        bundles: vec![proto::PluginWebUiBundleManifest {
            id: "main".into(),
            root_path: "dist".into(),
        }],
        ..Default::default()
    };

    let error = PackagedPluginWebUi::try_from(&manifest).expect_err("path route should fail");

    assert!(error.to_string().contains("slug"), "{error}");
}

#[test]
fn web_ui_packaging_rejects_url_shaped_route_slug() {
    let manifest = proto::PluginWebUiManifest {
        pages: vec![proto::PluginWebUiPageManifest {
            id: "home".into(),
            label: "Home".into(),
            route: "plugin://home".into(),
            bundle_id: "main".into(),
            entry_script: "app.js".into(),
            ..Default::default()
        }],
        bundles: vec![proto::PluginWebUiBundleManifest {
            id: "main".into(),
            root_path: "dist".into(),
        }],
        ..Default::default()
    };

    let error = PackagedPluginWebUi::try_from(&manifest).expect_err("URL route should fail");

    assert!(error.to_string().contains("URL-like"), "{error}");
}

#[test]
fn web_ui_packaging_accepts_valid_slug_and_single_bundle_reference() {
    let manifest = proto::PluginWebUiManifest {
        pages: vec![proto::PluginWebUiPageManifest {
            id: "home".into(),
            label: "Home".into(),
            route: "home".into(),
            bundle_id: "main".into(),
            entry_script: "app.js".into(),
            ..Default::default()
        }],
        config_sections: vec![proto::PluginWebUiConfigSectionManifest {
            id: "settings".into(),
            title: "Settings".into(),
            entry_script: "settings.js".into(),
            parent_tab: Some("integrations".into()),
            bundle_id: "main".into(),
        }],
        bundles: vec![proto::PluginWebUiBundleManifest {
            id: "main".into(),
            root_path: "dist".into(),
        }],
        contributions: Vec::new(),
    };

    let packaged = PackagedPluginWebUi::try_from(&manifest).expect("valid web UI should pass");

    assert_eq!(packaged.pages[0].route, "home");
    assert_eq!(packaged.pages[0].bundle_id, "main");
    assert_eq!(packaged.config_sections[0].bundle_id, "main");
}

#[test]
fn web_ui_packaging_defaults_unset_placement_to_auxiliary() {
    let manifest = proto::PluginWebUiManifest {
        pages: vec![proto::PluginWebUiPageManifest {
            id: "home".into(),
            label: "Home".into(),
            route: "home".into(),
            bundle_id: "main".into(),
            entry_script: "app.js".into(),
            ..Default::default()
        }],
        bundles: vec![proto::PluginWebUiBundleManifest {
            id: "main".into(),
            root_path: "dist".into(),
        }],
        ..Default::default()
    };

    let packaged = PackagedPluginWebUi::try_from(&manifest).expect("valid web UI should pass");

    assert_eq!(
        packaged.pages[0].placement,
        PackagedPluginWebUiPagePlacement::Auxiliary
    );
}

#[test]
fn web_ui_packaging_carries_primary_placement_hint() {
    let manifest = proto::PluginWebUiManifest {
        pages: vec![proto::PluginWebUiPageManifest {
            id: "home".into(),
            label: "Home".into(),
            route: "home".into(),
            bundle_id: "main".into(),
            entry_script: "app.js".into(),
            placement: proto::PluginWebUiPagePlacement::Primary as i32,
            ..Default::default()
        }],
        bundles: vec![proto::PluginWebUiBundleManifest {
            id: "main".into(),
            root_path: "dist".into(),
        }],
        ..Default::default()
    };

    let packaged = PackagedPluginWebUi::try_from(&manifest).expect("valid web UI should pass");

    assert_eq!(
        packaged.pages[0].placement,
        PackagedPluginWebUiPagePlacement::Primary
    );
}

#[test]
fn web_ui_packaging_rejects_unknown_placement_value() {
    let manifest = proto::PluginWebUiManifest {
        pages: vec![proto::PluginWebUiPageManifest {
            id: "home".into(),
            label: "Home".into(),
            route: "home".into(),
            bundle_id: "main".into(),
            entry_script: "app.js".into(),
            placement: 99,
            ..Default::default()
        }],
        bundles: vec![proto::PluginWebUiBundleManifest {
            id: "main".into(),
            root_path: "dist".into(),
        }],
        ..Default::default()
    };

    let error =
        PackagedPluginWebUi::try_from(&manifest).expect_err("unknown placement should fail");

    assert!(error.to_string().contains("placement"), "{error}");
}

#[test]
fn web_ui_page_builder_primary_placement_sets_proto_field() {
    let page = web_ui_page("home", "Home", "home", "app.js").primary_placement();

    assert_eq!(
        proto::PluginWebUiPageManifest::from(page).placement,
        proto::PluginWebUiPagePlacement::Primary as i32
    );
}

#[test]
fn web_ui_packaging_carries_the_page_host_header_choice() {
    let manifest: proto::PluginWebUiManifest = web_ui()
        .bundle(web_ui_bundle("main", "dist"))
        .page(
            web_ui_page("home", "Home", "home", "app.js")
                .bundle_id("main")
                .host_header(false),
        )
        .page(web_ui_page("other", "Other", "other", "app.js").bundle_id("main"))
        .into();

    let packaged = PackagedPluginWebUi::try_from(&manifest).expect("valid web UI should pass");

    assert_eq!(packaged.pages[0].host_header, Some(false));
    assert_eq!(packaged.pages[1].host_header, None);
    let json = serde_json::to_value(&packaged).unwrap();
    assert!(
        json["pages"][1].get("host_header").is_none(),
        "absent stays absent"
    );
}

fn manifest_with_contribution(slot: &str) -> proto::PluginWebUiManifest {
    proto::PluginWebUiManifest {
        contributions: vec![proto::PluginWebUiContributionManifest {
            id: "note".into(),
            slot: slot.into(),
            label: "Note".into(),
            bundle_id: "main".into(),
            entry_script: "note.js".into(),
        }],
        ..manifest_with_bundle("dist")
    }
}

#[test]
fn web_ui_packaging_accepts_contributions_in_known_slots() {
    for slot in WEB_UI_CONTRIBUTION_SLOTS {
        let packaged = PackagedPluginWebUi::try_from(&manifest_with_contribution(slot))
            .expect("known slot should package");

        assert_eq!(packaged.contributions[0].slot, slot);
        assert_eq!(packaged.contributions[0].entry_script, "note.js");
    }
}

#[test]
fn web_ui_packaging_rejects_unknown_contribution_slot() {
    let error = PackagedPluginWebUi::try_from(&manifest_with_contribution("toolbar"))
        .expect_err("unknown slot should fail");

    assert!(error.to_string().contains("slot must be one of"), "{error}");
}

#[test]
fn web_ui_packaging_requires_bundle_for_declared_contributions() {
    let manifest = proto::PluginWebUiManifest {
        bundles: Vec::new(),
        ..manifest_with_contribution("chat_message")
    };

    let error = PackagedPluginWebUi::try_from(&manifest).expect_err("missing bundle should fail");

    assert!(error.to_string().contains("exactly one bundle"), "{error}");
}

#[test]
fn web_ui_packaging_rejects_contribution_entry_script_traversal() {
    let mut manifest = manifest_with_contribution("logs_request");
    manifest.contributions[0].entry_script = "../note.js".into();

    let error = PackagedPluginWebUi::try_from(&manifest).expect_err("traversal should fail");

    assert!(error.to_string().contains("traversal"), "{error}");
}
