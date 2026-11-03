//! Synthetic event/trust cases through actual direct selector inputs.

use super::super as workflows;
use super::{caller_census, plan, support};

#[derive(Clone, Copy, Debug)]
enum Class {
    MainAndManual,
    MainDisabled,
    CpuForced,
    Sentinel,
    ReleaseForced,
}

impl Class {
    fn index(self) -> usize {
        match self {
            Self::MainAndManual => 0,
            Self::MainDisabled => 1,
            Self::CpuForced => 2,
            Self::Sentinel => 3,
            Self::ReleaseForced => 4,
        }
    }
}

const CALLERS: &[(&str, &str, &str, Class)] = &[
    (
        "ci-linux-host-slice.yml",
        "runner_policy",
        "policy",
        Class::MainAndManual,
    ),
    (
        "ci-linux-product-slice.yml",
        "runner_policy",
        "policy",
        Class::MainDisabled,
    ),
    (
        "ci-linux-runtime-slice.yml",
        "runner_policy",
        "policy",
        Class::MainAndManual,
    ),
    (
        "ci-linux-runtime-slice.yml",
        "runner_policy",
        "cpu_policy",
        Class::CpuForced,
    ),
    (
        "ci-macos-host-slice.yml",
        "runner_policy",
        "policy",
        Class::MainDisabled,
    ),
    (
        "ci-macos-product-slice.yml",
        "runner_policy",
        "policy",
        Class::MainDisabled,
    ),
    (
        "ci-macos-runtime-slice.yml",
        "runner_policy",
        "policy",
        Class::MainDisabled,
    ),
    (
        "ci-platform-checks-slice.yml",
        "runner_policy",
        "policy",
        Class::MainDisabled,
    ),
    (
        "ci-quality-slice.yml",
        "runner_policy",
        "policy",
        Class::MainAndManual,
    ),
    (
        "ci-quality-slice.yml",
        "runner_policy",
        "sentinel_policy",
        Class::Sentinel,
    ),
    (
        "ci-rust-tests-slice.yml",
        "runner_policy",
        "policy",
        Class::MainDisabled,
    ),
    (
        "ci-ui-artifact-slice.yml",
        "runner_policy",
        "policy",
        Class::MainDisabled,
    ),
    (
        "ci-web-slice.yml",
        "runner_policy",
        "policy",
        Class::MainDisabled,
    ),
    (
        "ci-windows-host-slice.yml",
        "runner_policy",
        "policy",
        Class::MainDisabled,
    ),
    (
        "ci-windows-product-slice.yml",
        "runner_policy",
        "policy",
        Class::MainDisabled,
    ),
    (
        "ci-windows-runtime-slice.yml",
        "runner_policy",
        "policy",
        Class::MainDisabled,
    ),
    (
        "native-sdk-artifact.yml",
        "runner_policy",
        "policy",
        Class::MainAndManual,
    ),
    ("release.yml", "metadata", "runners", Class::ReleaseForced),
    (
        "static-abi-artifact.yml",
        "runner_policy",
        "policy",
        Class::MainAndManual,
    ),
    (
        "swift-sdk-artifact.yml",
        "runner_policy",
        "policy",
        Class::MainDisabled,
    ),
];

#[derive(Clone, Copy)]
struct Scenario {
    name: &'static str,
    event: &'static str,
    origin: &'static str,
    repository: &'static str,
    head_repository: &'static str,
    reference: &'static str,
    date: &'static str,
    main_gate: bool,
    pr_gate: bool,
    manual: bool,
    force_hosted: bool,
    normal_canary: &'static str,
    sentinel_canary: &'static str,
    expected_depot: [bool; 5],
    depot_native_cache: bool,
}

fn bit(value: bool) -> &'static str {
    if value { "true" } else { "false" }
}

fn assert_identity_inputs(step: &workflows::Node, class: Class, identity: &str) {
    let expected_head = if matches!(class, Class::ReleaseForced) {
        "${{ github.sha }}"
    } else {
        "${{ github.event.pull_request.head.sha || github.sha }}"
    };
    assert_eq!(
        workflows::input(step, "head_sha"),
        Some(expected_head),
        "{identity}/head_sha"
    );
}

fn assert_caller_inputs(step: &workflows::Node, class: Class, identity: &str) {
    for (key, expected) in [
        ("event_name", "${{ github.event_name }}"),
        ("repository", "${{ github.repository }}"),
        (
            "head_repository",
            "${{ github.event.pull_request.head.repo.full_name }}",
        ),
        ("ref", "${{ github.ref }}"),
    ] {
        assert_eq!(
            workflows::input(step, key),
            Some(expected),
            "{identity}/{key}"
        );
    }
    assert_identity_inputs(step, class, identity);
    let (main_gate, pr_gate, force, manual, origin, canary) = match class {
        Class::MainAndManual => (
            "${{ vars.DEPOT_RUNNERS_ENABLED == 'true' }}",
            "${{ vars.DEPOT_PR_RUNNERS_ENABLED == 'true' }}",
            "${{ inputs.force_hosted }}",
            Some("${{ inputs.use_depot }}"),
            "${{ inputs.original_event_name }}",
            Some("${{ vars.DEPOT_PR_CANARY_REF }}"),
        ),
        Class::MainDisabled => (
            "false",
            "${{ vars.DEPOT_PR_RUNNERS_ENABLED == 'true' }}",
            "${{ inputs.force_hosted }}",
            None,
            "${{ inputs.original_event_name }}",
            Some("${{ vars.DEPOT_PR_CANARY_REF }}"),
        ),
        Class::CpuForced => (
            "${{ vars.DEPOT_RUNNERS_ENABLED == 'true' }}",
            "${{ vars.DEPOT_PR_RUNNERS_ENABLED == 'true' }}",
            "true",
            None,
            "${{ inputs.original_event_name }}",
            Some("${{ vars.DEPOT_PR_CANARY_REF }}"),
        ),
        Class::Sentinel => (
            "false",
            "false",
            "${{ inputs.force_hosted }}",
            Some("false"),
            "${{ inputs.original_event_name }}",
            Some("${{ vars.DEPOT_PR_SENTINEL_REF }}"),
        ),
        Class::ReleaseForced => (
            "${{ vars.DEPOT_RUNNERS_ENABLED == 'true' }}",
            "false",
            "true",
            Some("${{ inputs.use_depot == true }}"),
            "",
            None,
        ),
    };
    for (key, expected) in [
        ("depot_main_enabled", main_gate),
        ("depot_pr_enabled", pr_gate),
        ("force_hosted", force),
        ("original_event_name", origin),
    ] {
        assert_eq!(
            workflows::input(step, key),
            Some(expected),
            "{identity}/{key}"
        );
    }
    assert_eq!(
        workflows::input(step, "manual_use_depot"),
        manual,
        "{identity}/manual"
    );
    assert_eq!(
        workflows::input(step, "pr_canary_ref"),
        canary,
        "{identity}/canary"
    );
}

#[test]
fn caller_identity_contract_rejects_wrong_pr_head_for_selection_trace() {
    let file = "ci-linux-host-slice.yml";
    let source = workflows::source(file);
    let original = "head_sha: ${{ github.event.pull_request.head.sha || github.sha }}";
    assert_eq!(source.matches(original).count(), 1);
    let unchanged = workflows::workflow_yaml::parse(&source).unwrap();
    let unchanged_step = workflows::steps(workflows::job(&unchanged, "runner_policy"))
        .iter()
        .find(|step| workflows::text(step, "id") == Some("policy"))
        .unwrap();
    assert_identity_inputs(unchanged_step, Class::MainAndManual, file);
    let changed = source.replacen(original, "head_sha: ${{ github.sha }}", 1);
    let document = workflows::workflow_yaml::parse(&changed).unwrap();
    let step = workflows::steps(workflows::job(&document, "runner_policy"))
        .iter()
        .find(|step| workflows::text(step, "id") == Some("policy"))
        .unwrap();
    assert!(
        std::panic::catch_unwind(|| assert_identity_inputs(step, Class::MainAndManual, file))
            .is_err()
    );
}

#[test]
fn selection_trace_records_only_a_well_formed_supplied_head() {
    for (supplied, expected) in [
        (
            "0123456789abcdef0123456789abcdef01234567",
            Some("0123456789abcdef0123456789abcdef01234567"),
        ),
        (
            "fedcba9876543210fedcba9876543210fedcba98",
            Some("fedcba9876543210fedcba9876543210fedcba98"),
        ),
        ("not-a-commit", None),
    ] {
        let fixture = support::Fixture::new();
        let output = fixture.selector(&[("INPUT_HEAD_SHA", supplied)]);
        assert!(output.status.success(), "{output:?}");
        let selected = fixture.outputs();
        let trace: serde_json::Value =
            serde_json::from_str(selected["selection_trace"].as_str().unwrap()).unwrap();
        assert_eq!(trace["declared_head_sha"].as_str(), expected);
        assert!(trace["actual_runner"].is_null());
    }
}

fn exercise(step: &workflows::Node, class: Class, scenario: &Scenario, identity: &str) {
    assert_caller_inputs(step, class, identity);
    let main_gate = matches!(
        class,
        Class::MainAndManual | Class::CpuForced | Class::ReleaseForced
    ) && scenario.main_gate;
    let pr_gate = !matches!(class, Class::Sentinel | Class::ReleaseForced) && scenario.pr_gate;
    let force = matches!(class, Class::CpuForced | Class::ReleaseForced) || scenario.force_hosted;
    let manual = matches!(class, Class::MainAndManual | Class::ReleaseForced) && scenario.manual;
    let origin = if matches!(class, Class::ReleaseForced) {
        ""
    } else {
        scenario.origin
    };
    let canary = match class {
        Class::Sentinel => scenario.sentinel_canary,
        Class::ReleaseForced => "",
        _ => scenario.normal_canary,
    };
    let fixture = support::Fixture::new();
    let result = fixture.selector(&[
        ("INPUT_EVENT_NAME", scenario.event),
        ("INPUT_ORIGINAL_EVENT_NAME", origin),
        ("INPUT_REPOSITORY", scenario.repository),
        ("INPUT_HEAD_REPOSITORY", scenario.head_repository),
        ("INPUT_REF", scenario.reference),
        ("FIXTURE_DATE", scenario.date),
        ("INPUT_DEPOT_MAIN_ENABLED", bit(main_gate)),
        ("INPUT_DEPOT_PR_ENABLED", bit(pr_gate)),
        ("INPUT_PR_CANARY_REF", canary),
        ("INPUT_MANUAL_USE_DEPOT", bit(manual)),
        ("INPUT_FORCE_HOSTED", bit(force)),
    ]);
    assert!(
        result.status.success(),
        "{identity}/{}: {result:?}",
        scenario.name
    );
    let selected = fixture.outputs();
    let depot = scenario.expected_depot[class.index()];
    let request = serde_json::json!({ "selection_trace": selected["selection_trace"] });
    let record =
        ci_controller::plan::decision_evidence::render(&serde_json::to_vec(&request).unwrap())
            .unwrap_or_else(|error| panic!("{identity}/{}: {error}", scenario.name));
    let record: serde_json::Value = serde_json::from_str(&record).unwrap();
    assert_eq!(record["authenticity"], "unverified");
    assert_eq!(record["authorizes_execution"], false);
    assert!(record["actual_runner"].is_null());
    assert_eq!(
        selected["depot_enabled"],
        bit(depot),
        "{identity}/{}",
        scenario.name
    );
    assert_eq!(
        selected["allow_depot_remote_cache"], "false",
        "{identity}/{}",
        scenario.name
    );
    assert_eq!(
        selected["allow_native_github_cache"],
        bit(!depot || scenario.depot_native_cache),
        "{identity}/{}",
        scenario.name
    );
    assert_eq!(
        selected["allow_trusted_sccache_seed"],
        bit(!depot),
        "{identity}/{}",
        scenario.name
    );
    for (key, expected) in [
        (
            "runner",
            if depot {
                "depot-ubuntu-24.04"
            } else {
                "ubuntu-24.04"
            },
        ),
        (
            "runner_4",
            if depot {
                "depot-ubuntu-24.04-4"
            } else {
                "ubuntu-24.04"
            },
        ),
        (
            "runner_8",
            if depot {
                "depot-ubuntu-24.04-8"
            } else {
                "ubuntu-24.04"
            },
        ),
        (
            "runner_16",
            if depot {
                "depot-ubuntu-24.04-16"
            } else {
                "ubuntu-24.04"
            },
        ),
        (
            "runner_arm",
            if depot {
                "depot-ubuntu-24.04-arm"
            } else {
                "ubuntu-24.04-arm"
            },
        ),
        (
            "runner_arm_4",
            if depot {
                "depot-ubuntu-24.04-arm-4"
            } else {
                "ubuntu-24.04-arm"
            },
        ),
        (
            "runner_arm_8",
            if depot {
                "depot-ubuntu-24.04-arm-8"
            } else {
                "ubuntu-24.04-arm"
            },
        ),
        (
            "runner_arm_16",
            if depot {
                "depot-ubuntu-24.04-arm-16"
            } else {
                "ubuntu-24.04-arm"
            },
        ),
        (
            "runner_macos",
            if depot { "depot-macos-15" } else { "macos-15" },
        ),
        (
            "runner_windows",
            if depot {
                "depot-windows-2022"
            } else {
                "windows-2022"
            },
        ),
    ] {
        assert_eq!(
            selected[key], expected,
            "{identity}/{}/{key}",
            scenario.name
        );
    }
    fixture.0.close().unwrap();
}

fn scenarios(
    main_force: bool,
    control_force: bool,
    pr_force: bool,
    manual_force: bool,
) -> Vec<Scenario> {
    let main = Scenario {
        name: "main gate on",
        event: "push",
        origin: "push",
        repository: "Mesh-LLM/mesh-llm",
        head_repository: "",
        reference: "refs/heads/main",
        date: "2026-11-03",
        main_gate: true,
        pr_gate: true,
        manual: false,
        force_hosted: main_force,
        normal_canary: "",
        sentinel_canary: "",
        expected_depot: [true, false, false, false, false],
        depot_native_cache: false,
    };
    let mut cases = vec![
        main,
        Scenario {
            name: "main control",
            force_hosted: control_force,
            expected_depot: [false; 5],
            ..main
        },
        Scenario {
            name: "main gate off",
            main_gate: false,
            expected_depot: [false; 5],
            ..main
        },
        Scenario {
            name: "push outside main",
            reference: "refs/heads/feature",
            expected_depot: [false; 5],
            ..main
        },
        Scenario {
            name: "main in foreign repository",
            repository: "fork/mesh-llm",
            expected_depot: [false; 5],
            ..main
        },
        Scenario {
            name: "manual requested",
            event: "workflow_dispatch",
            origin: "workflow_dispatch",
            main_gate: false,
            manual: true,
            force_hosted: manual_force,
            expected_depot: [true, false, false, false, false],
            ..main
        },
        Scenario {
            name: "manual request forced hosted",
            event: "workflow_dispatch",
            origin: "workflow_dispatch",
            manual: true,
            force_hosted: true,
            expected_depot: [false; 5],
            ..main
        },
        Scenario {
            name: "manual unknown origin",
            event: "workflow_dispatch",
            origin: "schedule",
            manual: true,
            expected_depot: [false; 5],
            ..main
        },
        Scenario {
            name: "dispatch from main push",
            event: "workflow_dispatch",
            origin: "push",
            expected_depot: [true, false, false, false, false],
            ..main
        },
    ];
    cases.extend(trust_scenarios(main, pr_force, manual_force));
    cases
}

fn trust_scenarios(main: Scenario, pr_force: bool, manual_force: bool) -> Vec<Scenario> {
    vec![
        Scenario {
            name: "same-repo PR before expiry",
            event: "pull_request",
            origin: "pull_request",
            head_repository: "Mesh-LLM/mesh-llm",
            reference: "refs/pull/12/merge",
            date: "2026-09-13",
            force_hosted: pr_force,
            expected_depot: [true, true, false, false, false],
            depot_native_cache: true,
            ..main
        },
        Scenario {
            name: "same-repo PR at expiry",
            event: "pull_request",
            origin: "pull_request",
            head_repository: "Mesh-LLM/mesh-llm",
            reference: "refs/pull/12/merge",
            date: "2026-09-14",
            force_hosted: pr_force,
            expected_depot: [false; 5],
            ..main
        },
        Scenario {
            name: "same-repo PR gate disabled before expiry",
            event: "pull_request",
            origin: "pull_request",
            head_repository: "Mesh-LLM/mesh-llm",
            reference: "refs/pull/12/merge",
            date: "2026-09-13",
            pr_gate: false,
            manual: true,
            force_hosted: pr_force,
            expected_depot: [false; 5],
            ..main
        },
        Scenario {
            name: "fork PR",
            event: "pull_request",
            origin: "pull_request",
            head_repository: "fork/mesh-llm",
            reference: "refs/pull/12/merge",
            date: "2026-09-13",
            force_hosted: pr_force,
            expected_depot: [false; 5],
            ..main
        },
        Scenario {
            name: "PR head ref",
            event: "pull_request",
            origin: "pull_request",
            head_repository: "Mesh-LLM/mesh-llm",
            reference: "refs/pull/12/head",
            date: "2026-09-13",
            force_hosted: pr_force,
            expected_depot: [false; 5],
            ..main
        },
        Scenario {
            name: "pull request target",
            event: "pull_request_target",
            origin: "pull_request_target",
            head_repository: "Mesh-LLM/mesh-llm",
            reference: "refs/pull/12/merge",
            date: "2026-09-13",
            force_hosted: pr_force,
            expected_depot: [false; 5],
            ..main
        },
        Scenario {
            name: "PR-origin dispatch",
            event: "workflow_dispatch",
            origin: "pull_request",
            manual: true,
            force_hosted: manual_force,
            expected_depot: [false; 5],
            ..main
        },
        Scenario {
            name: "PR-target-origin dispatch",
            event: "workflow_dispatch",
            origin: "pull_request_target",
            manual: true,
            force_hosted: manual_force,
            expected_depot: [false; 5],
            ..main
        },
        Scenario {
            name: "unsupported scheduled event",
            event: "schedule",
            origin: "schedule",
            manual: true,
            expected_depot: [false; 5],
            ..main
        },
        Scenario {
            name: "ordinary exact canary",
            event: "pull_request",
            origin: "pull_request",
            head_repository: "Mesh-LLM/mesh-llm",
            reference: "refs/pull/12/merge",
            pr_gate: false,
            normal_canary: "refs/pull/12/merge",
            force_hosted: pr_force,
            expected_depot: [true, true, false, false, false],
            ..main
        },
        Scenario {
            name: "ordinary wrong canary",
            event: "pull_request",
            origin: "pull_request",
            head_repository: "Mesh-LLM/mesh-llm",
            reference: "refs/pull/12/merge",
            pr_gate: false,
            normal_canary: "refs/pull/13/merge",
            force_hosted: pr_force,
            expected_depot: [false; 5],
            ..main
        },
        Scenario {
            name: "fork with exact canary ref",
            event: "pull_request",
            origin: "pull_request",
            head_repository: "fork/mesh-llm",
            reference: "refs/pull/12/merge",
            pr_gate: false,
            normal_canary: "refs/pull/12/merge",
            force_hosted: pr_force,
            expected_depot: [false; 5],
            ..main
        },
        Scenario {
            name: "sentinel exact canary",
            event: "pull_request",
            origin: "pull_request",
            head_repository: "Mesh-LLM/mesh-llm",
            reference: "refs/pull/12/merge",
            pr_gate: false,
            sentinel_canary: "refs/pull/12/merge",
            force_hosted: pr_force,
            expected_depot: [false, false, false, true, false],
            ..main
        },
        Scenario {
            name: "sentinel wrong canary",
            event: "pull_request",
            origin: "pull_request",
            head_repository: "Mesh-LLM/mesh-llm",
            reference: "refs/pull/12/merge",
            pr_gate: false,
            sentinel_canary: "refs/pull/13/merge",
            force_hosted: pr_force,
            expected_depot: [false; 5],
            ..main
        },
    ]
}

#[test]
fn direct_callers_project_expected_event_and_trust_cases() {
    let base = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    let main = plan("main", "push", base, "README.md");
    let control = plan("main", "push", base, ".github/workflows/main_linux.yml");
    let pr = plan(
        "pr-ready",
        "pull_request",
        base,
        "mesh/crates/mesh-llm/src/main.rs",
    );
    let manual = plan("manual-full", "workflow_dispatch", base, "README.md");
    let force = |value: &serde_json::Value| value["signals"]["force_hosted"].as_bool().unwrap();
    assert!(!force(&main) && force(&control) && !force(&pr) && !force(&manual));
    let scenarios = scenarios(force(&main), force(&control), force(&pr), force(&manual));
    let mut declared = CALLERS
        .iter()
        .map(|(file, job, step, _)| ((*file).to_owned(), (*job).to_owned(), (*step).to_owned()))
        .collect::<Vec<_>>();
    declared.sort();
    let mut actual = caller_census::direct_workflow_callers(&support::root());
    actual.sort();
    assert_eq!(
        declared, actual,
        "update the event matrix for every direct selector"
    );
    for (file, job, id, class) in CALLERS {
        let document = workflows::document(file);
        let step = workflows::steps(workflows::job(&document, job))
            .iter()
            .find(|step| workflows::text(step, "id") == Some(*id))
            .unwrap();
        let identity = format!("{file}/{job}/{id}");
        for scenario in &scenarios {
            exercise(step, *class, scenario, &identity);
        }
    }
}
