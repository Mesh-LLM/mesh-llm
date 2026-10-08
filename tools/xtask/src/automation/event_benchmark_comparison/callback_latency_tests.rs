use super::*;

fn mac() -> Host {
    serde_json::from_value(json!({"system":"Darwin","machine":"arm64","certification_host":"macos-arm64-metal","p99_gate":"enforced"})).unwrap()
}

#[test]
fn certification_callback_p99_requires_an_actual_nonnegative_measurement() {
    for value in [None, Some(-1.0), Some(f64::NAN), Some(f64::INFINITY)] {
        let (result, blocking) = evaluate(&mac(), value, 100.0).unwrap();
        assert!(blocking);
        assert_eq!(result["status"], "blocked");
    }
}

#[test]
fn exact_callback_budget_passes_and_any_measured_excess_blocks() {
    assert!(!evaluate(&mac(), Some(100.0), 100.0).unwrap().1);
    assert!(evaluate(&mac(), Some(100.001), 100.0).unwrap().1);
    assert!(evaluate(&mac(), Some(50.0), f64::NAN).is_err());
}

#[test]
fn windows_cpu_and_other_unlisted_hosts_remain_informational() {
    for (system, machine) in [("Windows", "AMD64"), ("Linux", "aarch64")] {
        let host = Host {
            system: system.into(),
            machine: machine.into(),
            certification_host: None,
            p99_gate: Gate::Informational,
        };
        let (result, blocking) = evaluate(&host, None, 100.0).unwrap();
        assert!(!blocking);
        assert_eq!(result["status"], "informational");
        assert!(result["value"].is_null());
    }
}

#[test]
fn metadata_cannot_waive_a_frozen_certification_host_gate() {
    let mut host = mac();
    host.p99_gate = Gate::Informational;
    assert!(host.validate().is_err());
    host = mac();
    host.certification_host = None;
    assert!(host.validate().is_err());
    let linux = Host {
        system: "Linux".into(),
        machine: "x86_64".into(),
        certification_host: Some("linux-x86_64-cuda".into()),
        p99_gate: Gate::Enforced,
    };
    assert!(linux.validate().is_ok());
}

fn declared_runtime_budget() -> u64 {
    let syntax = syn::parse_file(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../mesh/crates/mesh-llm-host-runtime/src/runtime_events/config.rs"
    )))
    .unwrap();
    let budgets = syntax
        .items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Const(value) if value.ident == "CALLBACK_INGRESS_P99_BUDGET" => Some(value),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(budgets.len(), 1);
    let syn::Expr::Call(call) = budgets[0].expr.as_ref() else {
        panic!("runtime callback budget must be a duration constructor")
    };
    let syn::Expr::Path(function) = call.func.as_ref() else {
        panic!("runtime callback budget must name its duration constructor")
    };
    assert_eq!(function.path.segments.last().unwrap().ident, "from_micros");
    assert_eq!(call.args.len(), 1);
    let syn::Expr::Lit(value) = &call.args[0] else {
        panic!("runtime callback budget must be literal")
    };
    let syn::Lit::Int(value) = &value.lit else {
        panic!("runtime callback budget must be an integer")
    };
    value.base10_parse().unwrap()
}

#[test]
fn callback_comparator_budget_matches_the_component_owned_rust_constant() {
    assert_eq!(declared_runtime_budget() as f64, BUDGET_US);
}
