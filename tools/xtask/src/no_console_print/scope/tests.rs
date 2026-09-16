use super::*;
use crate::no_console_print::find_console_prints;

#[test]
fn exempts_only_named_source_categories() {
    for path in [
        "crates/demo/tests/integration.rs",
        "crates/demo/src/tests.rs",
        "crates/demo/src/router_tests.rs",
        "crates/demo/examples/demo.rs",
        "crates/demo/benches/speed.rs",
        "crates/demo/src/bin/helper.rs",
        "crates/demo/src/bin/helper/main.rs",
        "crates/demo/build.rs",
    ] {
        assert!(!is_product_source(path), "{path}");
    }
    for directory in NON_PRODUCT_CRATES {
        assert!(!is_product_source(&format!(
            "crates/{directory}/src/lib.rs"
        )));
    }
    for path in [
        "crates/mesh-llm/src/main.rs",
        "crates/mesh-client/src/lib.rs",
        "crates/new-product/src/lib.rs",
        "crates/demo/src/test_support.rs",
        "crates/demo/src/testing.rs",
        "crates/demo/src/binary_transport.rs",
        "crates/demo/src/latest.rs",
    ] {
        assert!(is_product_source(path), "{path}");
    }
}

#[test]
fn masks_test_modules_without_moving_product_hits() {
    let source = r###"fn before() { println!("before"); }
#[cfg ( test )]
mod arbitrary_name {
    const RAW: &str = r#"} println!(\"raw\") {"#;
    // } braces and Unicode é must not affect the span
    mod nested { fn f() { eprintln!("test"); } }
}
mod production {
    #[cfg(test)] mod nested_tests { fn f() { print!("test"); } }
    fn after() { eprintln!("after"); }
}
#[cfg(any(test, feature = "runtime"))]
mod also_product { fn f() { print!("product"); } }
"###;
    let masked = without_test_modules(source);
    assert_eq!(masked.len(), source.len());
    let hits = find_console_prints(&masked);
    assert_eq!(
        hits.iter().map(|hit| hit.line).collect::<Vec<_>>(),
        [1, 10, 13]
    );
}

#[test]
fn same_line_product_code_survives_and_string_attributes_are_not_rules() {
    let source = r##"#[cfg(test)] mod t { fn f() { print!("test"); } } fn f() { println!("product"); }
const TEXT: &str = "#[cfg(test)] mod fake {";
fn g() { eprintln!("product"); }
"##;
    let hits = find_console_prints(&without_test_modules(source));
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].macro_name, "println!");
    assert_eq!(hits[1].line, 3);
}

#[test]
fn malformed_sources_fail_closed() {
    let source = "#[cfg(test)] mod t { println!(\"still checked\");";
    assert_eq!(without_test_modules(source), source);
}

#[test]
fn exempt_crates_are_not_in_shipping_dependency_graph() {
    check_exempt_crates(&crate::repo_consistency::repo_root().unwrap()).unwrap();
}

fn metadata_with_dependency(kind: Option<&str>) -> CargoMetadata {
    let package = |name: &str, dependencies: serde_json::Value| {
        serde_json::json!({
            "id": name, "name": name, "version": "1.0.0",
            "manifest_path": format!("/repo/crates/{name}/Cargo.toml"),
            "dependencies": dependencies,
        })
    };
    serde_json::from_value(serde_json::json!({
        "workspace_members": ["mesh-llm", "middle", "skippy-bench"],
        "packages": [
            package("mesh-llm", serde_json::json!([
                {"name": "middle", "req": "*", "kind": null}
            ])),
            package("middle", serde_json::json!([
                {"name": "skippy-bench", "req": "*", "kind": kind,
                 "optional": true, "target": "cfg(windows)", "rename": "bench_alias"}
            ])),
            package("skippy-bench", serde_json::json!([]))
        ]
    }))
    .unwrap()
}

#[test]
fn dependency_guard_catches_transitive_optional_platform_and_renamed_dependencies() {
    let error = check_metadata(&metadata_with_dependency(None))
        .unwrap_err()
        .to_string();
    assert!(error.contains("skippy-bench"), "{error}");
    for kind in ["dev", "build"] {
        check_metadata(&metadata_with_dependency(Some(kind))).unwrap();
    }
}
