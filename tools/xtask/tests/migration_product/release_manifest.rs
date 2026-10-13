//! Current schema-two runtime publication preserves independent release identity.
//! Frozen schema-one catalog receipts remain unchanged historical evidence.
use crate::packaging_cases::{LINUX, MAC, manifest, runtime, tar_gz};
use crate::support::{Scratch, TestResult};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::Path,
    process::{Command, Output},
};

fn current_manifest(id: &str, platform: &str) -> Value {
    json!({"schema_version":2,"runtime":{"id":id,"release_version":"9.0","skippy_abi":"7","platform":serde_json::from_str::<Value>(platform).unwrap(),"backend":{"kind":"cpu"},"libraries":["libskippy.so"],"files":["lib/libskippy.so"]}})
}
fn archive(root: &Path, id: &str, value: &Value, library: &str) -> TestResult {
    runtime(root, id, &serde_json::to_string(value)?, library)
}
fn invoke(
    root: &Path,
    requested: &str,
    archives: &[&str],
) -> Result<Output, Box<dyn std::error::Error>> {
    Ok(Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(root)
        .args([
            "product",
            "runtime-release-manifest",
            "out/release.json",
            "Mesh-LLM/mesh-llm",
            "v1.2.3",
            requested,
            "tmp",
        ])
        .args(archives)
        .output()?)
}
fn expected_artifact(
    root: &Path,
    id: &str,
    value: &Value,
) -> Result<Value, Box<dyn std::error::Error>> {
    let name = format!("{id}.tar.gz");
    let mut artifact = value["runtime"].clone();
    artifact["sha256"] = json!(hex::encode(Sha256::digest(fs::read(
        root.join("dist").join(&name)
    )?)));
    artifact["url"] = json!(format!(
        "https://github.com/Mesh-LLM/mesh-llm/releases/download/v1.2.3/{name}"
    ));
    Ok(artifact)
}
#[test]
fn migration_product_release_manifest_writes_sorted_digests() -> TestResult {
    for (two, library) in [
        (true, "linux bytes"),
        (false, "linux bytes"),
        (false, "rebuilt bytes"),
    ] {
        let scratch = Scratch::new()?;
        let root = scratch.path();
        let linux = current_manifest("rt-linux", LINUX);
        let mac = current_manifest("rt-a-mac", MAC);
        archive(root, "rt-linux", &linux, library)?;
        let archives = if two {
            archive(root, "rt-a-mac", &mac, "mac bytes")?;
            vec!["dist/rt-linux.tar.gz", "dist/rt-a-mac.tar.gz"]
        } else {
            vec!["dist/rt-linux.tar.gz"]
        };
        let output = invoke(root, "v9.0", &archives)?;
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stdout.is_empty() && output.stderr.is_empty());
        let artifacts = if two {
            vec![
                expected_artifact(root, "rt-a-mac", &mac)?,
                expected_artifact(root, "rt-linux", &linux)?,
            ]
        } else {
            vec![expected_artifact(root, "rt-linux", &linux)?]
        };
        let bytes = fs::read(root.join("out/release.json"))?;
        assert_eq!(
            serde_json::from_slice::<Value>(&bytes)?,
            json!({"schema_version":2,"release_version":"9.0","skippy_abi":"7","artifacts":artifacts})
        );
        assert!(bytes.ends_with(b"\n"));
        // Publication tag1.2.3 and selected runtime9.0 remain distinct.
        assert!(
            fs::read(root.join("tmp/archive-0/rt-linux/lib/libskippy.so"))? == library.as_bytes()
        );
        assert!(!root.join("escape.txt").exists());
    }
    Ok(())
}
fn refused_case(
    root: &Path,
    case: &str,
) -> Result<(String, Vec<&'static str>, &'static str), Box<dyn std::error::Error>> {
    let mut linux = current_manifest("rt-linux", LINUX);
    let mut requested = "9.0".to_owned();
    let mut archives = vec!["dist/rt-linux.tar.gz"];
    let diagnostic = match case {
        "version" => {
            linux["runtime"]["release_version"] = json!("9.1");
            "does not match requested runtime release"
        }
        "legacy" => "schema_version 2",
        "missing-platform" => {
            linux["runtime"].as_object_mut().unwrap().remove("platform");
            "missing native runtime field"
        }
        "runtime-object" => {
            linux["runtime"] = json!([1]);
            "missing runtime manifest"
        }
        "root-object" => {
            linux = json!([1]);
            "manifest must be a JSON object"
        }
        "missing-release" => {
            linux["runtime"]
                .as_object_mut()
                .unwrap()
                .remove("release_version");
            "missing native runtime field"
        }
        "numeric-release" => {
            linux["runtime"]["release_version"] = json!(9);
            "release_version must be a string"
        }
        "mixed-abi" | "mixed-prefix" => {
            let mut mac = current_manifest("rt-a-mac", MAC);
            if case == "mixed-abi" {
                mac["runtime"]["skippy_abi"] = json!("8");
            } else {
                mac["runtime"]["release_version"] = json!("v9.0");
            }
            archive(root, "rt-a-mac", &mac, "mac bytes")?;
            archives.push("dist/rt-a-mac.tar.gz");
            if case == "mixed-abi" {
                "mixed Skippy ABI"
            } else {
                "mixed runtime releases"
            }
        }
        "duplicate" => "expected exactly one manifest.json",
        "traversal" => "unsafe or invalid native runtime archive",
        "missing-archive" => "No such file",
        "empty-release" => {
            requested = "v".into();
            "requested runtime release must contain a version"
        }
        "no-archives" => {
            archives.clear();
            "no native runtime artifacts supplied"
        }
        _ => unreachable!(),
    };
    match case {
        "legacy" => runtime(
            root,
            "rt-linux",
            &manifest("rt-linux", "v1.2.3", "7", LINUX),
            "linux bytes",
        )?,
        "duplicate" => {
            let text = serde_json::to_string(&linux)?;
            tar_gz(
                &root.join("dist/rt-linux.tar.gz"),
                &[("a/manifest.json", &text), ("b/manifest.json", &text)],
            )?;
        }
        "traversal" => tar_gz(
            &root.join("dist/rt-linux.tar.gz"),
            &[("../escape.txt", "escaped")],
        )?,
        "missing-archive" => {}
        _ => archive(root, "rt-linux", &linux, "linux bytes")?,
    }
    Ok((requested, archives, diagnostic))
}
#[test]
fn migration_product_release_manifest_rejects_mismatched_archives() -> TestResult {
    for case in [
        "version",
        "legacy",
        "missing-platform",
        "runtime-object",
        "root-object",
        "missing-release",
        "numeric-release",
        "mixed-abi",
        "mixed-prefix",
        "duplicate",
        "traversal",
        "missing-archive",
        "empty-release",
        "no-archives",
    ] {
        let scratch = Scratch::new()?;
        let root = scratch.path();
        let (requested, archives, diagnostic) = refused_case(root, case)?;
        fs::create_dir_all(root.join("out"))?;
        fs::write(root.join("out/release.json"), b"prior publication\n")?;
        let output = invoke(root, &requested, &archives)?;
        assert_eq!(output.status.code(), Some(1), "{case}");
        assert!(output.stdout.is_empty(), "{case}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(diagnostic),
            "{case}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            fs::read(root.join("out/release.json"))?,
            b"prior publication\n",
            "{case}"
        );
        assert!(!root.join("escape.txt").exists(), "{case}");
    }
    Ok(())
}
