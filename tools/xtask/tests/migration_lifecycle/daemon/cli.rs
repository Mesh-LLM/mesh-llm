use super::protocol::{Behavior, Plan};
use crate::support::{Case, Sentinel};

pub(super) fn case(plan: &Plan) -> Case {
    let case = Case::new(crate::protocol::Behavior::Clean, vec![]);
    let fixture = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/migration_daemon_fixture");
    std::fs::copy(fixture, &case.binary).unwrap();
    std::fs::write(
        case.native.join("daemon.json"),
        serde_json::to_vec(plan).unwrap(),
    )
    .unwrap();
    case
}

pub(super) fn command(case: &Case) -> std::process::Command {
    let mut arguments = case.arguments();
    arguments[5] = "3".into();
    let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_xtask"));
    command
        .current_dir(crate::support::repository())
        .args(["automation", "daemon-readiness"])
        .args(arguments);
    for key in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "MESH_LLM_CONFIG",
        "MESH_LLM_JOIN",
        "MESH_LLM_MODEL",
        "MESH_LLM_NATIVE_RUNTIME_CACHE_DIR",
    ] {
        command.env(key, "http://ambient-private-value.invalid");
    }
    command
}

pub(super) fn run(plan: Plan, failure: Option<&str>) {
    let case = case(&plan);
    let mut sentinel = Sentinel::new(&case);
    let output = command(&case).output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.success(), failure.is_none(), "{stderr}");
    match failure {
        Some(cause) => {
            assert!(stderr.contains(cause), "{stderr}");
            assert!(output.stdout.is_empty());
        }
        None => assert!(
            String::from_utf8_lossy(&output.stdout)
                .contains("zero_model_serve_ready: bounded subset")
        ),
    }
    assert!(!stderr.contains("status-secret-never-print"));
    assert!(sentinel.0.try_wait().unwrap().is_none());
    let audit: super::protocol::Audit =
        serde_json::from_slice(&std::fs::read(case.native.join("audit.json")).unwrap()).unwrap();
    assert_eq!(
        audit
            .environment
            .get("MESH_LLM_EPHEMERAL_KEY")
            .map(String::as_str),
        Some("1")
    );
    for key in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "MESH_LLM_JOIN",
        "MESH_LLM_MODEL",
    ] {
        assert!(!audit.environment.contains_key(key));
    }
    if let Ok(pid) = std::fs::read_to_string(case.native.join("leaf.pid")) {
        crate::support::assert_absent(pid.parse().unwrap());
    }
    case.assert_removed();
}

#[test]
fn d24_invalid_direct_call_has_no_effects() {
    let result = crate::automation::daemon_readiness::run(
        &crate::support::repository(),
        &["--unknown".into(), "value".into()],
    );
    assert!(matches!(
        result,
        Err(crate::automation::daemon_readiness::Error::Invalid(
            "unknown option"
        ))
    ));
}

macro_rules! scenario {
    ($name:ident, $change:expr, $failure:expr) => {
        #[test]
        fn $name() {
            let mut plan = Plan::default();
            ($change)(&mut plan);
            run(plan, $failure);
        }
    };
}

scenario!(d01_empty, |_: &mut Plan| {}, None);
scenario!(
    d01_nonempty,
    |plan: &mut Plan| plan.models_body = br#"{"data":[{"id":"mesh"}]}"#.to_vec(),
    None
);
scenario!(
    d02_missing_correlation,
    |plan: &mut Plan| plan.record.clear(),
    Some("attribution_unavailable")
);
scenario!(
    d03_other_self_with_owned_sibling,
    |plan: &mut Plan| {
        plan.status = r#"{"api_port":API,"local_instances":[{"pid":1,"is_self":true},{"pid":PID,"is_self":false}]}"#.into()
    },
    Some("ownership_mismatch")
);
scenario!(
    d05_malformed,
    |plan: &mut Plan| plan.status = "{".into(),
    Some("malformed_status")
);
scenario!(
    d06_ignored_metadata,
    |plan: &mut Plan| {
        plan.status = r#"{"api_port":API,"local_instances":[{"pid":PID,"is_self":true,"api_port":null,"started_at_unix":0,"runtime_dir":""}]}"#.into()
    },
    None
);
scenario!(
    d07_retry_503,
    |plan: &mut Plan| plan.status_failures = 1,
    None
);
scenario!(
    d07_persistent_503,
    |plan: &mut Plan| plan.status_failures = 60,
    Some("deadline expired")
);
scenario!(
    d08_204,
    |plan: &mut Plan| {
        plan.models_code = 204;
        plan.models_body.clear();
    },
    None
);
scenario!(d08_302, |plan: &mut Plan| plan.models_code = 302, None);
scenario!(
    d08_400,
    |plan: &mut Plan| plan.models_code = 400,
    Some("models_transfer_failed")
);
scenario!(
    d08_503,
    |plan: &mut Plan| plan.models_code = 503,
    Some("models_transfer_failed")
);
scenario!(
    d09_opaque,
    |plan: &mut Plan| plan.models_body = vec![0xff, 0, b'{'],
    None
);
scenario!(
    d11_after_body,
    |plan: &mut Plan| plan.before_body = false,
    None
);
scenario!(
    d12_eof_fragment,
    |plan: &mut Plan| plan.record = format!("{}EOF", plan.record.trim_end()),
    Some("attribution_unavailable")
);
scenario!(
    d12_oversized,
    |plan: &mut Plan| plan.record = format!("{}{}\n", " ".repeat(8193), plan.record.trim_end()),
    Some("attribution_unavailable")
);
scenario!(
    d14_body_exact,
    |plan: &mut Plan| plan.models_body = vec![b'x'; 1048576],
    None
);
scenario!(
    d14_body_excess,
    |plan: &mut Plan| plan.models_body = vec![b'x'; 1048577],
    Some("response_limit")
);
scenario!(
    d14_slow_headers,
    |plan: &mut Plan| plan.behavior = Behavior::SlowHeaders,
    Some("deadline expired")
);
scenario!(
    d14_slow_body,
    |plan: &mut Plan| plan.behavior = Behavior::SlowBody,
    Some("deadline expired")
);
scenario!(
    d14_flood,
    |plan: &mut Plan| plan.behavior = Behavior::Flood,
    None
);
scenario!(
    d15_early_zero,
    |plan: &mut Plan| plan.behavior = Behavior::EarlyZero,
    Some("EarlyExit")
);
scenario!(
    d15_early_nonzero,
    |plan: &mut Plan| plan.behavior = Behavior::EarlyNonzero,
    Some("EarlyExit")
);
scenario!(
    d15_exit_models,
    |plan: &mut Plan| plan.behavior = Behavior::ExitModels,
    Some("EarlyExit")
);
scenario!(
    d18_nonzero,
    |plan: &mut Plan| plan.behavior = Behavior::Nonzero,
    Some("clean attributed unforced")
);
scenario!(
    d18_stubborn,
    |plan: &mut Plan| plan.behavior = Behavior::Stubborn,
    Some("clean attributed unforced")
);
scenario!(
    d10_cleanup_only,
    |plan: &mut Plan| plan.behavior = Behavior::CleanupRecord,
    Some("attribution_unavailable")
);
