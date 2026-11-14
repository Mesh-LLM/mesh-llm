use super::{Json, validate};
use serde_json::{Value, json};

const PLANNED: &[&str] = &[
    "ui_artifact",
    "static_abi",
    "rust_tests",
    "hosts",
    "native_runtimes",
    "runtime_product",
    "product_smoke",
    "authority_source",
    "authority_linux_x64",
    "product_automation_linux_x64",
];

fn plan() -> Json {
    Json::parse(
        &serde_json::to_vec(&json!({
            "lane": "linux",
            "required": true,
            "required_slices": ["static-abi"],
            "matrices": {
                "hosts": [{"id": "linux-amd64-host"}],
                "runtime_products": [{"id": "linux-cpu"}],
                "rust_tests": [{"id": "batch-0"}],
                "smoke": [{"id": "core"}],
                "sdk": []
            }
        }))
        .unwrap(),
    )
    .unwrap()
}

fn needs() -> Value {
    let mut value = json!({
        "sdk": {"result": "skipped"},
        "kotlin_sdk_input": {"result": "skipped"}
    });
    for job in PLANNED {
        value[*job] = json!({"result": "success"});
    }
    value
}

fn check(needs: &Value) -> Result<(), String> {
    let needs = Json::parse(&serde_json::to_vec(needs).unwrap()).unwrap();
    validate(&plan(), &needs).map(|_| ())
}

#[test]
fn grouped_linux_complete_results_survive_controller_upgrade() {
    assert_eq!(check(&needs()), Ok(()));
}

#[test]
fn grouped_linux_still_requires_every_planned_success() {
    for job in PLANNED {
        for result in ["failure", "cancelled", "skipped", "neutral"] {
            let mut value = needs();
            value[*job]["result"] = result.into();
            let error = check(&value).unwrap_err();
            assert!(error.contains(job), "{job}: {error}");
        }
        let mut value = needs();
        value.as_object_mut().unwrap().remove(*job);
        assert!(check(&value).is_err(), "missing {job}");
    }
}

#[test]
fn grouped_linux_rejects_mixed_layouts_and_unplanned_execution() {
    for job in [
        "native_runtime_cpu",
        "product_cpu",
        "product_smoke_cpu",
        "sdk_rust",
    ] {
        for result in ["success", "skipped"] {
            let mut value = needs();
            value[job] = json!({"result": result});
            assert!(check(&value).is_err(), "mixed job {job}: {result}");
        }
    }
    let mut value = needs();
    value["unexpected"] = json!({"result": "success"});
    assert!(check(&value).is_err());
    value["unexpected"]["result"] = "skipped".into();
    assert_eq!(check(&value), Ok(()));
}

#[test]
fn grouped_linux_requires_selected_sdk_consumers() {
    for sdk in ["rust", "kotlin"] {
        let plan = Json::parse(
            &serde_json::to_vec(&json!({
                "lane": "linux", "required": true, "required_slices": [],
                "matrices": {"sdk": [{"id": sdk}]}
            }))
            .unwrap(),
        )
        .unwrap();
        let mut value = json!({
            "native_runtimes": {"result": "skipped"},
            "runtime_product": {"result": "skipped"},
            "product_smoke": {"result": "skipped"},
            "sdk": {"result": "success"}
        });
        let consumers: &[&str] = if sdk == "kotlin" {
            &[
                "kotlin_sdk_input",
                "authority_source",
                "authority_linux_x64",
            ]
        } else {
            &[]
        };
        for job in consumers {
            value[*job] = json!({"result": "success"});
        }
        let parsed = Json::parse(&serde_json::to_vec(&value).unwrap()).unwrap();
        assert!(validate(&plan, &parsed).is_ok(), "{sdk}");
        for job in ["sdk"].iter().chain(consumers.iter()) {
            let mut failed = value.clone();
            failed[*job]["result"] = "failure".into();
            let parsed = Json::parse(&serde_json::to_vec(&failed).unwrap()).unwrap();
            assert!(validate(&plan, &parsed).is_err(), "{sdk}: {job}");
        }
    }
}
