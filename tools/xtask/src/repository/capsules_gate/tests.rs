use super::no_keys::{excluded_kind, secret_kinds};
use super::*;

const PLUGIN_MUTANTS: &str =
    include_str!("../../../../../mesh/crates/mesh-llm-capsules-plugin/gates/mutants.json");
const PLUGIN_REFEREE_MUTANTS: &str = include_str!(
    "../../../../../mesh/crates/mesh-llm-capsules-plugin/vectors/parity/referee/mutants.json"
);

#[test]
fn capsules_gate_names_parse_and_unknown_names_do_not() {
    for (name, gate) in [
        ("mutants", Gate::Mutants),
        ("referee-mutants", Gate::RefereeMutants),
        ("unbounded-depth", Gate::UnboundedDepth),
        ("no-keys", Gate::NoKeys),
        ("vectors", Gate::Vectors),
        ("web-ui", Gate::WebUi),
        ("package-twice", Gate::PackageTwice),
    ] {
        assert_eq!(Gate::parse(name), Some(gate));
        assert!(GRAMMAR.usage.contains(name), "usage omits {name}");
    }
    assert_eq!(Gate::parse("all"), None);
    assert_eq!(Gate::parse("Mutants"), None);
}

#[test]
fn capsules_gate_plugin_tables_parse_and_name_mutant_features() {
    let table = MutantTable::parse(PLUGIN_MUTANTS).unwrap();
    let referee = RefereeTable::parse(PLUGIN_REFEREE_MUTANTS).unwrap();
    assert_eq!(table.mutants.len(), 9);
    assert_eq!(referee.mutants.len(), 9);
    let manifest = include_str!("../../../../../mesh/crates/mesh-llm-capsules-plugin/Cargo.toml");
    for feature in table
        .mutants
        .iter()
        .map(|m| &m.feature)
        .chain(referee.mutants.iter().map(|m| &m.feature))
    {
        assert!(feature.starts_with("mutant-"), "{feature}");
        assert!(
            manifest.contains(&format!("\n{feature} = []")),
            "{feature} is not a crate feature"
        );
    }
    assert!(referee.mutants.iter().all(|m| !m.must_fail.is_empty()));
}

#[test]
fn capsules_gate_tables_that_would_check_nothing_are_refused() {
    assert!(MutantTable::parse(r#"{"mutants":[]}"#).is_err());
    assert!(RefereeTable::parse(r#"{"mutants":[]}"#).is_err());
    let no_case = r#"{"mutants":[{"feature":"mutant-referee-x","must_fail":[]}]}"#;
    let error = RefereeTable::parse(no_case).unwrap_err().to_string();
    assert!(error.contains("mutant-referee-x"), "{error}");
}

#[test]
fn capsules_gate_mutant_table_rejects_unknown_expectations() {
    let bad = r#"{"mutants":[{"feature":"mutant-x","tests":"t","must_print":{"substring":"x"}}]}"#;
    assert!(serde_json::from_str::<MutantTable>(bad).is_err());
}

#[test]
fn capsules_gate_line_expectation_needs_the_exact_line() {
    let expect = Expect::Line("record_to_a_stranger:".into());
    assert!(printed(
        &expect,
        "golden: {}\nrecord_to_a_stranger:\n  got: {}"
    ));
    assert!(!printed(&expect, "record_to_a_stranger_unknown:"));
    assert!(!printed(&expect, "  record_to_a_stranger:"));
    assert!(!printed(&expect, "the case record_to_a_stranger: differs"));
}

#[test]
fn capsules_gate_failed_test_expectation_needs_that_test_failed() {
    let expect = Expect::FailedTest("a_failed_request_is_never_answered_as_unsigned".into());
    assert!(printed(
        &expect,
        "test evidence_answer::tests::a_failed_request_is_never_answered_as_unsigned ... FAILED"
    ));
    assert!(printed(
        &expect,
        "test a_failed_request_is_never_answered_as_unsigned ... FAILED"
    ));
    // Passing, a near-miss name, or a wildcard-shaped line never counts.
    assert!(!printed(
        &expect,
        "test evidence_answer::tests::a_failed_request_is_never_answered_as_unsigned ... ok"
    ));
    assert!(!printed(
        &expect,
        "test evidence_answer::tests::not_a_failed_request_is_never_answered_as_unsigned ... FAILED"
    ));
    assert!(!printed(
        &expect,
        "test evidence_answer::tests::a_failed_request_is_never_answered_as_unsigned xyz FAILED"
    ));
}

#[test]
fn capsules_gate_referee_run_reports_only_differs_lines() {
    let output = "running 1 test\ndiffers: select/barred_inside_d\ndiffers: hold/x\n  differs: nested/ignored\ndiffers: \ndiffers: two words\n";
    let cases: Vec<_> = differing_cases(output).into_iter().collect();
    assert_eq!(cases, ["hold/x", "select/barred_inside_d"]);
}

#[test]
fn capsules_gate_unbounded_depth_fails_only_when_the_feature_is_on() {
    assert_eq!(
        unbounded_depth_verdict("serde_json v1.0.1\n├── serde_json feature \"default\"\n").code,
        0
    );
    assert_eq!(
        unbounded_depth_verdict("serde_json v1.0.1\n└── serde_json feature \"unbounded_depth\"\n")
            .code,
        1
    );
}

#[test]
fn capsules_gate_sum_lines_need_a_digest_and_a_contained_path() {
    let digest = "a".repeat(64);
    assert_eq!(
        parse_sum_line(&format!("{digest}  parity/README.md")),
        Some((digest.as_str(), "parity/README.md"))
    );
    assert_eq!(
        parse_sum_line(&format!("{digest} *bin/x")),
        Some((digest.as_str(), "bin/x"))
    );
    for bad in [
        format!("{digest}  ../outside"),
        format!("{digest}  /etc/passwd"),
        format!("{digest} one-space"),
        format!("{}  short", "a".repeat(63)),
        format!("{}  upper", "A".repeat(64)),
        format!("{digest}  "),
    ] {
        assert_eq!(parse_sum_line(&bad), None, "{bad}");
    }
}

#[test]
fn capsules_gate_vectors_fail_on_a_changed_or_missing_file() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("a.json"), b"{}\n").unwrap();
    let good = hex::encode(Sha256::digest(b"{}\n"));
    fs::write(dir.path().join("SHA256SUMS"), format!("{good}  a.json\n")).unwrap();
    assert_eq!(vectors(dir.path()).unwrap().code, 0);

    fs::write(dir.path().join("a.json"), b"{ }\n").unwrap();
    let changed = vectors(dir.path()).unwrap();
    assert_eq!(changed.code, 1);
    assert!(changed.stderr.contains("a.json: digest differs"));

    fs::write(
        dir.path().join("SHA256SUMS"),
        format!("{good}  gone.json\n"),
    )
    .unwrap();
    assert_eq!(vectors(dir.path()).unwrap().code, 1);

    fs::write(dir.path().join("SHA256SUMS"), "\n\n").unwrap();
    let empty = vectors(dir.path()).unwrap();
    assert_eq!(empty.code, 1);
    assert!(empty.stderr.contains("lists no vector"));
}

#[test]
fn capsules_gate_live_plugin_vectors_match() {
    let vectors_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(PLUGIN_DIR)
        .join("vectors");
    let report = vectors(&vectors_dir).unwrap();
    assert_eq!(report.code, 0, "{}", report.stderr);
}

#[test]
fn capsules_gate_plugin_version_reads_plugin_toml() {
    assert_eq!(
        plugin_version("name = \"capsules\"\nversion = \"0.78.0\"\n").unwrap(),
        "0.78.0"
    );
    assert!(plugin_version("name = \"capsules\"\n").is_err());
}

#[test]
fn capsules_gate_secret_rules_match_each_kind_and_never_public_material() {
    let cases: [(String, &str); 10] = [
        (armor("PRIVATE KEY"), "private-key-block"),
        (armor("OPENSSH PRIVATE KEY"), "private-key-block"),
        (armor("PGP PRIVATE KEY BLOCK"), "private-key-block"),
        (format!("token = ghp_{}", "A1".repeat(18)), "github-token"),
        (format!("github_pat_{}", "a_".repeat(20)), "github-token"),
        (format!("cio{}", "x".repeat(32)), "crates-io-token"),
        (format!("hf_{}", "Z".repeat(34)), "hugging-face-token"),
        (format!("AKIA{}", "ABCDEFGHIJKLMNOP"), "aws-access-key"),
        (
            format!("\"signing_key\": \"{}\"", "ab".repeat(32)),
            "private-key-hex",
        ),
        (format!("seed={}", "0F".repeat(32)), "private-key-hex"),
    ];
    for (line, kind) in &cases {
        assert_eq!(secret_kinds(line), [*kind], "{line}");
    }
    for clean in [
        armor("PUBLIC KEY"),
        armor("CERTIFICATE"),
        armor("lower PRIVATE KEY"),
        format!("ghp_{}", "a".repeat(35)),
        format!("\"public_key\": \"{}\"", "ab".repeat(32)),
        format!("\"key_id\": \"{}\"", "ab".repeat(32)),
        format!("\"signing_key\": \"{}\"", "ab".repeat(31)),
        format!("signing_key -> {}", "ab".repeat(32)),
        format!("AKIA{}", "abcdefghijklmnop"),
    ] {
        assert!(secret_kinds(&clean).is_empty(), "{clean}");
    }
}

#[test]
fn capsules_gate_excluded_paths_match_each_kind() {
    for (path, kind) in [
        ("scripts/tool.py", "python-file"),
        ("x/__pycache__/m.cpython.bin", "python-file"),
        ("docs/demo.md", "demo-material"),
        ("Red-Team/notes.txt", "red-team-material"),
        ("NOTICE", "notice-file"),
        ("third/NOTICE.txt", "notice-file"),
        ("keys/node.pem", "key-file"),
        ("id_ed25519.pub", "key-file"),
        ("ci/.env.local", "token-file"),
        ("auth.token", "token-file"),
        ("received.jsonl", "node-data"),
        ("ledger/x.json", "node-data"),
    ] {
        assert_eq!(excluded_kind(path), Some(kind), "{path}");
    }
    for path in [
        "src/main.rs",
        "vectors/parity/referee/mutants.json",
        "web-ui/src/features/capsules/lib/exchange-ledger.ts",
        "docs/NOTICE-of-change.md",
        "README.md",
    ] {
        assert_eq!(excluded_kind(path), None, "{path}");
    }
}

/// A PEM armor line, assembled here so no key-shaped literal sits in source.
fn armor(label: &str) -> String {
    let dashes = "-".repeat(5);
    format!("{dashes}BEGIN {label}{dashes}")
}

#[test]
fn capsules_gate_no_keys_reports_path_line_and_kind_but_never_the_secret() {
    let root = tempfile::tempdir().unwrap();
    let plugin = root.path().join(PLUGIN_DIR);
    fs::create_dir_all(plugin.join("docs")).unwrap();
    let token = format!("ghp_{}", "Q7".repeat(18));
    fs::write(
        plugin.join("docs/notes.md"),
        format!("intro\nuse {token} here\n"),
    )
    .unwrap();
    fs::write(plugin.join("tool.py"), "").unwrap();
    fs::write(plugin.join("untracked.pem"), "").unwrap();
    fs::write(plugin.join("blob.bin"), [0u8, 1, 2]).unwrap();
    let git = |args: &[&str]| {
        let status = Command::new("git")
            .args(args)
            .current_dir(root.path())
            .status()
            .unwrap();
        assert!(status.success(), "git {args:?}");
    };
    let tracked = |name: &str| format!("{PLUGIN_DIR}/{name}");
    git(&["init", "-q"]);
    git(&[
        "add",
        "--",
        &tracked("docs/notes.md"),
        &tracked("tool.py"),
        &tracked("blob.bin"),
    ]);

    let report = no_keys::check(root.path(), &plugin).unwrap();
    assert_eq!(report.code, 1);
    for expected in [
        "no-keys: docs/notes.md:2: github-token",
        "no-keys: tool.py: excluded file (python-file)",
    ] {
        assert!(report.stderr.contains(expected), "{}", report.stderr);
    }
    assert!(
        !report.stderr.contains(&token),
        "the matched secret was printed"
    );
    assert!(
        !report.stderr.contains("untracked.pem"),
        "an untracked file was read"
    );

    fs::write(plugin.join("docs/notes.md"), "clean\n").unwrap();
    git(&["rm", "-q", "--cached", "--", &tracked("tool.py")]);
    let clean = no_keys::check(root.path(), &plugin).unwrap();
    assert_eq!(clean.code, 0, "{}", clean.stderr);
}
