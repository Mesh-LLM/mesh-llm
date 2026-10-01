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
fn shard_count_rejects_non_u64_values() {
    for count in ["-2", "+2", "1_0", "18446744073709551616", "٢"] {
        let output = run(&["--shard-count", count]);
        assert_eq!(output.status.code(), Some(2));
        assert!(output.stdout.is_empty());
    }
}
