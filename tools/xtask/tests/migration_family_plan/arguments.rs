use super::*;

#[test]
fn exact_options_accept_inline_values() {
    let output = run(&["--families=llama,qwen3-dense", "--shard-count=1"]);
    assert_eq!(output.status.code(), Some(0));
    assert_eq!(output.stdout, fixture("real-reversed", "stdout"));
}

#[test]
fn undocumented_abbreviations_are_rejected() {
    let output = run(&["--fam=llama"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
}

#[test]
fn exact_help_is_successful_and_malformed_help_cannot_write_outputs() {
    for help in ["-h", "--help"] {
        let output = run(&[help]);
        assert_eq!(output.status.code(), Some(0));
        assert!(!output.stdout.is_empty());
        assert!(output.stderr.is_empty());
    }
    for malformed in ["-hh", "-hgarbage", "-h=1", "-h--help", "--hel"] {
        let plan = temp_path("invalid-help-plan.json");
        let github = temp_path("invalid-help-github.txt");
        fs::write(&github, b"existing=kept\n").expect("existing output");
        let output = run(&[
            "--output",
            plan.to_str().expect("UTF-8 path"),
            "--github-output",
            github.to_str().expect("UTF-8 path"),
            malformed,
        ]);
        assert_eq!(output.status.code(), Some(2), "{malformed}");
        assert!(output.stdout.is_empty(), "{malformed}");
        assert!(!plan.exists(), "{malformed} wrote a plan");
        assert_eq!(fs::read(&github).expect("output"), b"existing=kept\n");
        fs::remove_file(github).expect("cleanup output");
    }
}

#[test]
fn shard_count_rejects_non_u64_values() {
    for count in ["-2", "+2", "1_0", "18446744073709551616", "٢"] {
        let output = run(&["--shard-count", count]);
        assert_eq!(output.status.code(), Some(2));
        assert!(output.stdout.is_empty());
    }
}
