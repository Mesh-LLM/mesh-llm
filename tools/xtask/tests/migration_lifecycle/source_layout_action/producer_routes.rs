//! Extra owning classifier cases after the frozen routingv5 module is integrated.
#[test]
fn immutable_sdk_producers_and_typed_archive_adapters_route_sdk_validation() {
    for workflow in [
        "native-sdk-artifact",
        "sdk-smoke",
        "static-abi-artifact",
        "swift-sdk-artifact",
    ] {
        let changed = format!(".github/workflows/{workflow}.yml");
        assert_eq!(
            super::classify(&changed, "false", "push")
                .split_whitespace()
                .last(),
            Some("true"),
            "{changed}"
        );
    }
    for action in ["prepare-native-sdk-input", "prepare-static-abi-input"] {
        let changed = format!(".github/actions/{action}/action.yml");
        assert_eq!(
            super::classify(&changed, "false", "push")
                .split_whitespace()
                .last(),
            Some("true"),
            "{changed}"
        );
    }
    for prefix in ["", "mesh/", "skippy/"] {
        for adapter in [
            "restore-native-sdk-input",
            "restore-static-abi-input",
            "safe-extract-tar",
            "safe-extract-zip",
            "verify-swift-xcframework",
        ] {
            let changed = format!("{prefix}scripts/{adapter}.sh");
            assert_eq!(
                super::classify(&changed, "false", "push")
                    .split_whitespace()
                    .last(),
                Some("true"),
                "{changed}"
            );
        }
    }
}
