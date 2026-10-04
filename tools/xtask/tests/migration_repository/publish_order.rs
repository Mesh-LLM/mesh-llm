use crate::support::{Invocation, Scratch, TestResult, assert_output};

fn metadata(root: &std::path::Path) -> String {
    serde_json::json!({
        "workspace_members": ["provider", "consumer", "controller-only"],
        "packages": [
            {"id":"provider", "name":"provider", "manifest_path":root.join("provider/Cargo.toml"), "publish":null, "dependencies":[]},
            {"id":"consumer", "name":"consumer", "manifest_path":root.join("consumer/Cargo.toml"), "publish":null, "dependencies":[{"kind":"build", "path":root.join("provider")}]},
            {"id":"controller-only", "name":"controller-only", "manifest_path":root.join("controller-only/Cargo.toml"), "publish":null, "dependencies":[]}
        ]
    }).to_string()
}

#[test]
fn migration_repository_selected_publish_roster_uses_historical_data_without_executing_it()
-> TestResult {
    let scratch = Scratch::new("historical-publish")?;
    // No Cargo alias, Rust sources or repository markers exist in this checkout.
    let script = scratch.write(
        "release-source/scripts/publish-crates.sh",
        "touch executed\nexit 99\npublish_crates=(\n provider\n consumer\n)\n",
    )?;
    let source = metadata(scratch.path());
    let args = [
        "repository",
        "publish-order",
        "--selected-script",
        script.to_str().ok_or("script path")?,
    ];
    let output = Invocation {
        cwd: scratch.path(),
        args: &args,
        stdin: Some(&source),
        env: &[],
    }
    .run()?;
    assert_output(&output, 0, "provider\nconsumer\n", "");
    assert!(!scratch.path().join("executed").exists());
    for roster in [
        "publish_crates=(\n consumer\n provider\n)",
        "publish_crates=(\n unknown\n)",
        "publish_crates=(\n $(touch executed)\n)",
    ] {
        std::fs::write(&script, roster)?;
        let failed = Invocation {
            cwd: scratch.path(),
            args: &args,
            stdin: Some(&source),
            env: &[],
        }
        .run()?;
        assert_output(&failed, 1, "", "diagnostic");
        assert!(!scratch.path().join("executed").exists());
    }
    Ok(())
}

#[test]
fn migration_repository_selected_publish_roster_rejects_oversized_script_before_output()
-> TestResult {
    let scratch = Scratch::new("oversized-publish")?;
    let script = scratch.write("publish-crates.sh", &"#".repeat(1_048_577))?;
    let args = [
        "repository",
        "publish-order",
        "--selected-script",
        script.to_str().ok_or("script path")?,
    ];
    let output = Invocation {
        cwd: scratch.path(),
        args: &args,
        stdin: Some("{}"),
        env: &[],
    }
    .run()?;
    assert_output(&output, 1, "", "diagnostic");
    assert!(crate::support::text(&output.stderr).contains("exceeds 1 MiB"));
    Ok(())
}
