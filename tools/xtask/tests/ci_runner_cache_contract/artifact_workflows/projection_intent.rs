//! Execute the maintained jq programs through the existing bounded plan owner.
use super::{fixture, project};
use serde_json::{Value, json};
use std::fs;
fn replace_plan(fixture: &super::support::Fixture, plan: &Value) {
    fs::write(
        fixture.path().join("golden.json"),
        serde_json::to_vec(plan).unwrap(),
    )
    .unwrap();
}
fn golden(fixture: &super::support::Fixture) -> Value {
    serde_json::from_slice(&fs::read(fixture.path().join("golden.json")).unwrap()).unwrap()
}
#[test]
fn graph_windows_projection_requires_windows_product_before_core_smoke() {
    for selected in [false, true] {
        let fixture = fixture();
        let mut plan = golden(&fixture);
        for name in [
            "hosts",
            "runtime_products",
            "platform_checks",
            "rust_tests",
            "sdk",
        ] {
            plan["matrices"][name] = json!([]);
        }
        plan["matrices"]["smoke"] = json!([{"id":"core"},{"id":"metal-model-load"}]);
        plan["required_slices"] = json!([]);
        if selected {
            plan["matrices"]["runtime_products"] =
                json!([{"id":"windows-cpu","platform":"windows"}]);
        }
        replace_plan(&fixture, &plan);
        let output = project(&fixture, "[]");
        let windows: Value =
            serde_json::from_str(output["windows_lane_plan"].as_str().unwrap()).unwrap();
        assert_eq!(windows["required"], selected);
        assert_eq!(
            windows["matrices"]["smoke"],
            if selected {
                json!([{"id":"core"}])
            } else {
                json!([])
            }
        );
        fixture.0.close().unwrap();
    }
}
#[test]
fn graph_topic_projection_preserves_shared_fields_and_optional_noop() {
    for required in [false, true] {
        let fixture = fixture();
        let mut plan = golden(&fixture);
        plan["profile"] = json!("pr-ready");
        plan["domains"] = if required {
            json!(["ui"])
        } else {
            json!(["docs"])
        };
        plan["required_slices"] = if required {
            json!(["quality", "web"])
        } else {
            json!([])
        };
        replace_plan(&fixture, &plan);
        let output = project(&fixture, "[]");
        for lane in ["quality", "website"] {
            let projected: Value =
                serde_json::from_str(output[format!("{lane}_lane_plan")].as_str().unwrap())
                    .unwrap();
            assert_eq!(projected["lane"], lane);
            assert_eq!(projected["required"], required);
            for key in [
                "profile",
                "domains",
                "required_slices",
                "signals",
                "budgets",
            ] {
                assert_eq!(projected[key], plan[key]);
            }
            assert_eq!(
                projected["matrices"],
                if lane == "quality" {
                    json!({"clippy":plan["matrices"]["clippy"]})
                } else {
                    json!({})
                }
            );
        }
        fixture.0.close().unwrap();
    }
}
#[test]
fn graph_platform_projection_preserves_exact_smoke_sdk_and_platform_rows() {
    let fixture = fixture();
    let mut plan = golden(&fixture);
    for matrix in ["hosts", "runtime_products", "sdk", "platform_checks"] {
        plan["matrices"][matrix] = json!([{"id":"linux","platform":"linux"},{"id":"macos","platform":"macos"},{"id":"windows","platform":"windows"}]);
    }
    plan["matrices"]["smoke"] =
        json!([{"id":"core"},{"id":"metal-model-load"},{"id":"distributed"}]);
    plan["matrices"]["rust_tests"] = json!([{"id":"rust"}]);
    replace_plan(&fixture, &plan);
    let output = project(&fixture, "[]");
    for lane in ["linux", "macos", "windows"] {
        let projected: Value =
            serde_json::from_str(output[format!("{lane}_lane_plan")].as_str().unwrap()).unwrap();
        assert_eq!(projected["lane"], lane);
        assert_eq!(projected["required"], true);
        for key in [
            "profile",
            "domains",
            "required_slices",
            "signals",
            "budgets",
        ] {
            assert_eq!(projected[key], plan[key]);
        }
        let mut expected = json!({"hosts":[{"id":lane,"platform":lane}],"runtime_products":[{"id":lane,"platform":lane}]});
        if lane == "linux" {
            expected["rust_tests"] = plan["matrices"]["rust_tests"].clone();
            expected["smoke"] = json!([{"id":"core"},{"id":"distributed"}]);
            expected["sdk"] = json!([{"id":lane,"platform":lane}]);
        } else {
            expected["platform_checks"] = json!([{"id":lane,"platform":lane}]);
            expected["smoke"] = if lane == "macos" {
                json!([{"id":"metal-model-load"}])
            } else {
                json!([{"id":"core"}])
            };
            if lane == "macos" {
                expected["sdk"] = json!([{"id":lane,"platform":lane}]);
            }
        }
        assert_eq!(projected["matrices"], expected);
    }
    fixture.0.close().unwrap();
}
