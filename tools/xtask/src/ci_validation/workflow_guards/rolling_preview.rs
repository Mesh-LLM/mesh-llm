//! Preview publication must not enter stable channels or move main.
#[cfg(test)]
mod tests {
    const RELEASE: &str = include_str!("../../../../../.github/workflows/release.yml");
    const DISPATCH: &str = include_str!("../../../../../.github/workflows/rolling-preview.yml");

    #[test]
    fn preview_preserves_main_and_stages_complete_downloads() {
        for required in [
            "\"$CANARY\" == \"true\" || \"$PREVIEW\" == \"true\"",
            "\"$version\" != *-preview.*",
            "draft: ${{ inputs.preview == true }}",
            "--draft=false --prerelease --latest=false",
            "needs.metadata.outputs.prerelease != 'true'",
        ] {
            assert!(RELEASE.contains(required), "missing {required}");
        }
        assert!(DISPATCH.contains("gh workflow run release.yml --ref main"));
        assert!(DISPATCH.contains("-f preview=true"));
        assert!(DISPATCH.contains("cancel-in-progress: false"));
        assert!(!DISPATCH.contains("git push"));
    }
}

#[cfg(test)]
mod execution_tests {
    use std::process::Command;

    #[test]
    fn actual_metadata_script_refuses_stable_preview_and_classifies_valid_snapshot() {
        let workflow = include_str!("../../../../../.github/workflows/release.yml");
        let meta = workflow.split("      - id: meta\n").nth(1).unwrap();
        let body = meta
            .split("        run: |\n")
            .nth(1)
            .unwrap()
            .split("      - name:")
            .next()
            .unwrap();
        let script = body
            .lines()
            .map(|line| line.strip_prefix("          ").unwrap_or(line))
            .collect::<Vec<_>>()
            .join("\n");
        for (preview, version, accepted) in [
            ("true", "v0.78.0", false),
            ("true", "v0.78.0-rc.1", false),
            ("true", "v0.78.0-preview.123.1", true),
            ("false", "v0.78.0", true),
        ] {
            let output = tempfile::NamedTempFile::new().unwrap();
            let result = Command::new("bash")
                .args(["-c", &script])
                .env("GITHUB_EVENT_NAME", "workflow_dispatch")
                .env("GITHUB_REF_NAME", "main")
                .env("GITHUB_OUTPUT", output.path())
                .env("INPUT_VERSION", version)
                .env("INPUT_PREVIEW", preview)
                .env("INPUT_SKIP_GPU_BUNDLES", "false")
                .env("INPUT_CANARY", "false")
                .env("INPUT_FORCE_HOSTED_RUNNERS", "true")
                .output()
                .unwrap();
            assert_eq!(result.status.success(), accepted);
            if accepted {
                let values = std::fs::read_to_string(output.path()).unwrap();
                assert!(values.contains(&format!("tag={version}\n")));
                assert!(values.contains(if preview == "true" {
                    "prerelease=true\n"
                } else {
                    "prerelease=false\n"
                }));
            }
        }
    }
}
