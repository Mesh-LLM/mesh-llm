use crate::support::{Invocation, Scratch, TestResult, assert_output, text};

struct Source {
    scratch: Scratch,
    metadata: serde_json::Value,
}

impl Source {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let scratch = Scratch::new("release-source-invariants")?;
        // These markers deliberately cannot build a historical automation tool.
        scratch.write("Cargo.toml", "historical source marker\n")?;
        scratch.write(
            "tools/xtask/Cargo.toml",
            "do not build historical automation\n",
        )?;
        scratch.write(
            "scripts/publish-crates.sh",
            "touch executed\nexit 99\npublish_crates=(\n provider\n consumer\n)\n",
        )?;
        for name in ["provider", "consumer"] {
            scratch.write(&format!("{name}/Cargo.toml"), "[package]\n")?;
            scratch.write(&format!("{name}/README.md"), "package documentation\n")?;
            scratch.write(
                &format!("{name}/src/lib.rs"),
                "pub const TEXT: &str = include_str!(\"fixture.txt\");\n",
            )?;
            scratch.write(&format!("{name}/src/fixture.txt"), "owned include\n")?;
        }
        for path in [
            "crates/mesh-client/src/models/catalog.json",
            "crates/mesh-llm-node/src/catalog.json",
        ] {
            scratch.write(path, "{}\n")?;
        }
        let root = scratch.path();
        let packages: Vec<_> = ["provider", "consumer"].into_iter().map(|name| {
            let dependencies = if name == "consumer" {
                serde_json::json!([{"name":"provider","req":"^0.76.1","kind":"build","path":root.join("provider")}])
            } else { serde_json::json!([]) };
            serde_json::json!({"id":name,"name":name,"version":"0.76.1","manifest_path":root.join(name).join("Cargo.toml"),
                "publish":null,"description":"fixture package","license":"MIT","license_file":null,
                "repository":"https://example.invalid/repository","readme":"README.md","dependencies":dependencies})
        }).collect();
        let mut packages = packages;
        for (name, directory) in [
            ("mesh-llm-client", "mesh-client"),
            ("mesh-llm-node", "mesh-llm-node"),
        ] {
            scratch.write(&format!("crates/{directory}/Cargo.toml"), "[package]\n")?;
            packages.push(serde_json::json!({
                "id": name, "name": name, "version": "0.76.1",
                "manifest_path": root.join("crates").join(directory).join("Cargo.toml"),
                "dependencies": []
            }));
        }
        let metadata = serde_json::json!({"workspace_members":["provider","consumer","mesh-llm-client","mesh-llm-node"],"packages":packages});
        Ok(Self { scratch, metadata })
    }

    fn run(&self) -> Result<std::process::Output, Box<dyn std::error::Error>> {
        let script = self.scratch.path().join("scripts/publish-crates.sh");
        let source = self.metadata.to_string();
        Invocation {
            cwd: self.scratch.path(),
            args: &[
                "repository",
                "publish-order",
                "--selected-script",
                script.to_str().ok_or("script path")?,
                "--source-root",
                self.scratch.path().to_str().ok_or("source root")?,
            ],
            stdin: Some(&source),
            env: &[],
        }
        .run()
    }
}

#[test]
fn migration_repository_release_source_retains_package_checks_without_controller_layout()
-> TestResult {
    let source = Source::new()?;
    let output = source.run()?;
    assert_output(&output, 0, "provider\nconsumer\n", "");
    assert!(!source.scratch.path().join("executed").exists());
    assert!(!source.scratch.path().join(".github").exists());
    Ok(())
}

#[test]
fn migration_repository_release_source_rejects_metadata_and_dependency_version_before_output()
-> TestResult {
    for (field, diagnostic) in [
        ("description", "description"),
        ("license", "license"),
        ("repository", "repository"),
        ("version", "version requirement"),
    ] {
        let mut source = Source::new()?;
        if field == "version" {
            source.metadata["packages"][1]["dependencies"][0]["req"] = serde_json::json!("^0.75.0");
        } else {
            source.metadata["packages"][0][field] = serde_json::Value::Null;
        }
        let output = source.run()?;
        assert_output(&output, 1, "", "diagnostic");
        assert!(
            text(&output.stderr).contains(diagnostic),
            "{}",
            text(&output.stderr)
        );
    }
    Ok(())
}

#[test]
fn migration_repository_release_source_rejects_missing_and_escaping_packaged_files_and_catalog_drift()
-> TestResult {
    for case in ["readme", "missing-include", "escaped-include", "catalog"] {
        let source = Source::new()?;
        let diagnostic = match case {
            "readme" => {
                std::fs::remove_file(source.scratch.path().join("provider/README.md"))?;
                "readme"
            }
            "missing-include" => {
                std::fs::remove_file(source.scratch.path().join("provider/src/fixture.txt"))?;
                "does not exist"
            }
            "escaped-include" => {
                source
                    .scratch
                    .write("outside.txt", "outside packaged crate\n")?;
                source.scratch.write(
                    "provider/src/lib.rs",
                    "const TEXT: &str = include_str!(\"../../outside.txt\");\n",
                )?;
                "outside publish package root"
            }
            "catalog" => {
                source.scratch.write(
                    "crates/mesh-llm-node/src/catalog.json",
                    "{\"drift\":true}\n",
                )?;
                "packaged catalog copy"
            }
            _ => unreachable!(),
        };
        let output = source.run()?;
        assert_output(&output, 1, "", "diagnostic");
        assert!(
            text(&output.stderr).contains(diagnostic),
            "{}",
            text(&output.stderr)
        );
    }
    Ok(())
}

#[test]
fn migration_repository_release_source_rejects_foreign_script_and_manifest_bindings() -> TestResult
{
    let mut source = Source::new()?;
    let outside = Scratch::new("outside-selected-release")?;
    let script = outside.write(
        "publish-crates.sh",
        "publish_crates=(\n provider\n consumer\n)\n",
    )?;
    let metadata = source.metadata.to_string();
    let output = Invocation {
        cwd: source.scratch.path(),
        args: &[
            "repository",
            "publish-order",
            "--selected-script",
            script.to_str().ok_or("script path")?,
            "--source-root",
            source.scratch.path().to_str().ok_or("root path")?,
        ],
        stdin: Some(&metadata),
        env: &[],
    }
    .run()?;
    assert_output(&output, 1, "", "diagnostic");
    assert!(text(&output.stderr).contains("not bound to the selected source"));

    let manifest = outside.write("consumer/Cargo.toml", "[package]\n")?;
    source.metadata["packages"][1]["manifest_path"] = serde_json::json!(manifest);
    let output = source.run()?;
    assert_output(&output, 1, "", "diagnostic");
    assert!(text(&output.stderr).contains("manifest is outside the selected source"));
    Ok(())
}

#[test]
fn migration_repository_release_source_rejects_directory_scripts_and_invalid_options() -> TestResult
{
    let source = Source::new()?;
    let script = source.scratch.path().join("scripts/publish-crates.sh");
    std::fs::remove_file(&script)?;
    std::fs::create_dir(&script)?;
    let output = source.run()?;
    assert_output(&output, 1, "", "diagnostic");
    assert!(text(&output.stderr).contains("must be a regular file"));

    for args in [
        vec!["repository", "publish-order", "--source-root", "."],
        vec![
            "repository",
            "publish-order",
            "--selected-script",
            "missing",
            "--unknown",
            ".",
        ],
    ] {
        let output = Invocation {
            cwd: source.scratch.path(),
            args: &args,
            stdin: None,
            env: &[],
        }
        .run()?;
        assert_output(&output, 1, "", "diagnostic");
        assert!(text(&output.stderr).contains("usage:"));
    }
    Ok(())
}
