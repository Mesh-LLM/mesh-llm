use super::fixture::{Fixture, executable, stderr, stdout};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
};

fn runtime(root: &Path, id: &str, declared: &str) -> PathBuf {
    let runtime = root.join(id);
    let tool = runtime.join("tools/skippy-package-builder");
    executable(&tool, "exit 0");
    let actual = fs::read(&tool).unwrap();
    fs::write(
        runtime.join("manifest.json"),
        serde_json::to_vec(
            &json!({"runtime":{"tools":{declared:hex::encode(Sha256::digest(&actual))}}}),
        )
        .unwrap(),
    )
    .unwrap();
    tool
}
#[test]
fn actual_split_package_resolver_binds_exact_verified_tool_and_refuses_unsafe_inputs_without_building()
 {
    for case in [
        "valid",
        "missing-checksum",
        "tampered",
        "ambiguous",
        "missing-bundle",
        "nonexecutable",
    ] {
        let fixture = Fixture::new();
        let bundle = fixture.root.join("runtime");
        fs::create_dir(&bundle).unwrap();
        let declared = if case == "missing-checksum" {
            "other/skippy-package-builder"
        } else {
            "tools/skippy-package-builder"
        };
        let tool = runtime(&bundle, "one", declared);
        match case {
            "tampered" => fs::write(&tool, b"tampered runtime tool").unwrap(),
            "ambiguous" => {
                runtime(&bundle, "two", declared);
            }
            "missing-bundle" => fs::remove_dir_all(&bundle).unwrap(),
            "nonexecutable" => {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&tool, fs::Permissions::from_mode(0o600)).unwrap();
            }
            _ => {}
        }
        let script = fixture.functions(&[("resolve_package_tool", "prepare_split_package")])
            + "resolve_package_tool\n";
        let result = fixture.run(script, &[("RUNTIME_BUNDLE", bundle.display().to_string())]);
        assert_eq!(
            result.process.success(),
            case == "valid",
            "{case}: {}",
            stderr(&result)
        );
        if case == "valid" {
            assert_eq!(
                stdout(&result),
                format!("{}\n", tool.canonicalize().unwrap().display())
            );
        } else {
            assert!(stdout(&result).is_empty());
            assert!(!stderr(&result).is_empty());
        }
    }
}
fn preparation(fixture: &Fixture) -> String {
    fixture.functions(&[
        ("sha256_file", "quant_selector_from_gguf_file"),
        ("quant_selector_from_gguf_file", "resolve_package_tool"),
        ("prepare_split_package", "prepare_split_inputs"),
    ]) + "prepare_split_package dense \"$SOURCE\" \"$PACKAGE_TOOL\"\n"
}
#[test]
fn actual_split_package_preparation_binds_digest_quant_identity_and_verification_order() {
    let fixture = Fixture::new();
    let source = fixture.root.join("Fixture-Q4_K_M.gguf");
    let bytes = b"immutable-gguf-fixture";
    fs::write(&source, bytes).unwrap();
    let package = fixture.root.join("work/prepared-packages/dense");
    let tool = fixture.root.join("package-tool");
    let digest = hex::encode(Sha256::digest(bytes));
    executable(
        &tool,
        r#"if [[ "$1" == write-package ]]; then
  [[ "$#" == 12 && "$2" == "$SOURCE" && "$3" == --model-id && "$4" == "$MODEL_ID" && "$5" == --out-dir && "$6" == "$PACKAGE" ]]
  [[ "$7" == --source-repo && "$8" == ci/two-node-split-smoke && "$9" == --source-revision && "${10}" == "$DIGEST" && "${11}" == --source-file && "${12}" == Fixture-Q4_K_M.gguf ]]
  printf 'write\n' >> "$EVENTS"
  mkdir -p "$PACKAGE"
  printf '{"schema_version":2}\n' > "$PACKAGE/model-package.json"
elif [[ "$1" == verify-package-v2 ]]; then
  [[ "$#" == 6 && "$2" == "$PACKAGE" && "$3" == --source && "$4" == "$SOURCE" && "$5" == --source-file && "$6" == Fixture-Q4_K_M.gguf ]]
  [[ -s "$PACKAGE/model-package.json" ]]
  printf 'verify\n' >> "$EVENTS"
else exit 97; fi
"#,
    );
    let result = fixture.run(
        preparation(&fixture),
        &[
            ("SOURCE", source.display().to_string()),
            ("PACKAGE_TOOL", tool.display().to_string()),
            ("PACKAGE", package.display().to_string()),
            ("DIGEST", digest.clone()),
            ("MODEL_ID", format!("ci/dense-{}:Q4_K_M", &digest[..16])),
            ("EVENTS", fixture.root.join("events").display().to_string()),
        ],
    );
    assert!(result.process.success(), "{}", stderr(&result));
    assert_eq!(stdout(&result), format!("{}\n", package.display()));
    assert_eq!(
        fs::read_to_string(fixture.root.join("events")).unwrap(),
        "write\nverify\n"
    );
}
#[test]
fn actual_split_existing_package_is_passed_through_without_writer_or_verifier() {
    let fixture = Fixture::new();
    let source = fixture.root.join("package");
    fs::create_dir(&source).unwrap();
    fs::write(source.join("model-package.json"), b"{\"schema_version\":2}").unwrap();
    let result = fixture.run(
        preparation(&fixture),
        &[
            ("SOURCE", source.display().to_string()),
            (
                "PACKAGE_TOOL",
                fixture.root.join("missing-tool").display().to_string(),
            ),
        ],
    );
    assert!(result.process.success(), "{}", stderr(&result));
    assert_eq!(stdout(&result), format!("{}\n", source.display()));
    assert!(!fixture.root.join("work").exists());
}
#[test]
fn actual_split_admission_creates_external_evidence_before_binary_check_and_bounds_override() {
    for (kind, explicit, expected) in [
        ("raw", None, "1"),
        ("package", None, "0"),
        ("raw", Some("0"), "0"),
        ("package", Some("1"), "1"),
        ("raw", Some("invalid"), "invalid"),
    ] {
        let fixture = Fixture::new();
        let source = fixture.root.join(kind);
        if kind == "raw" {
            fs::write(&source, b"fixture").unwrap();
        } else {
            fs::create_dir(&source).unwrap();
            fs::write(source.join("model-package.json"), b"{\"schema_version\":2}").unwrap();
        }
        let evidence = fixture.root.join("nested/evidence/new-work");
        let mut extra = vec![(
            "MESH_TWO_NODE_SPLIT_WORK_DIR",
            evidence.display().to_string(),
        )];
        if let Some(value) = explicit {
            extra.push(("MESH_TWO_NODE_SPLIT_ALLOW_UNCERTIFIED", value.into()));
        }
        let result = fixture.whole_adapter(&source, &extra);
        assert!(!result.process.success());
        assert!(evidence.is_dir());
        assert!(stdout(&result).contains(&format!("uncertified split override: {expected}")));
        let expected_error = if explicit == Some("invalid") {
            "must be 0 or 1"
        } else {
            "Missing executable mesh-llm binary"
        };
        assert!(
            stderr(&result).contains(expected_error),
            "{}",
            stderr(&result)
        );
    }
}
#[test]
fn actual_split_preparation_failures_never_publish_a_package_path_or_accept_stale_output() {
    for failure in ["write", "verify", "missing-manifest"] {
        let fixture = Fixture::new();
        let source = fixture.root.join("Fixture-Q4_K_M.gguf");
        fs::write(&source, b"immutable fixture").unwrap();
        let package = fixture.root.join("work/prepared-packages/dense");
        fs::create_dir_all(&package).unwrap();
        fs::write(package.join("model-package.json"), b"stale package").unwrap();
        let tool = fixture.root.join("package-tool");
        executable(
            &tool,
            r#"if [[ "$1" == write-package ]]; then
  [[ ! -e "$PACKAGE/model-package.json" ]]
  printf 'write\n' >> "$EVENTS"
  [[ "$FAILURE" != write ]] || exit 73
  mkdir -p "$PACKAGE"
  if [[ "$FAILURE" != missing-manifest ]]; then printf '{"schema_version":2}\n' > "$PACKAGE/model-package.json"; fi
elif [[ "$1" == verify-package-v2 ]]; then
  printf 'verify\n' >> "$EVENTS"
  [[ "$FAILURE" != verify ]] || exit 74
else exit 97; fi
"#,
        );
        let result = fixture.run(
            preparation(&fixture),
            &[
                ("SOURCE", source.display().to_string()),
                ("PACKAGE_TOOL", tool.display().to_string()),
                ("PACKAGE", package.display().to_string()),
                ("FAILURE", failure.into()),
                ("EVENTS", fixture.root.join("events").display().to_string()),
            ],
        );
        assert!(!result.process.success());
        assert!(stdout(&result).is_empty());
        let message = match failure {
            "write" => "Package-v2 preparation failed",
            "verify" => "Package-v2 verification failed",
            _ => "did not emit",
        };
        assert!(stderr(&result).contains(message), "{}", stderr(&result));
        let expected = if failure == "write" {
            "write\n"
        } else {
            "write\nverify\n"
        };
        assert_eq!(
            fs::read_to_string(fixture.root.join("events")).unwrap(),
            expected
        );
    }
}
#[test]
fn structured_split_workflow_owns_the_only_explicit_uncertified_override() {
    use crate::workflow_yaml::{self, Node};
    let source = fs::read_to_string(
        super::fixture::repository().join(".github/workflows/ci-linux-product-smoke-slice.yml"),
    )
    .unwrap();
    let document = workflow_yaml::parse(&source).unwrap();
    let Node::Map(jobs) = document.get("jobs").unwrap() else {
        panic!("jobs must be a mapping");
    };
    let overrides = jobs
        .iter()
        .filter_map(|(name, job)| {
            let command = job.get("with")?.get("smoke_script")?.text()?;
            command
                .contains("MESH_TWO_NODE_SPLIT_ALLOW_UNCERTIFIED")
                .then_some((name.as_str(), command))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        overrides,
        vec![(
            "two_node_split",
            "MESH_TWO_NODE_SPLIT_DURABLE_L3=1 MESH_TWO_NODE_SPLIT_ALLOW_UNCERTIFIED=1 scripts/ci-two-node-split-smoke.sh"
        )]
    );
}
