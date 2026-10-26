use super::fixture::{Fixture, stderr, stdout};
use serde_json::json;
use std::fs;
#[test]
fn actual_split_prefix_cli_distinguishes_recurrent_checkpoint_reuse_partial_misses_and_all_cold() {
    for (cached, expected) in [
        ([0, 0, 512, 640, 640, 768], 1),
        ([0, 512, 512, 640, 640, 768], 0),
        ([0, 512, 0, 640, 0, 768], 0),
        ([0, 512, 512, 640, 640, 0], 1),
        ([0, 0, 0, 0, 0, 0], 75),
    ] {
        let fixture = Fixture::new();
        let responses = fixture.root.join("responses");
        fs::create_dir(&responses).unwrap();
        for (index, (prompt, cached)) in [644, 644, 788, 788, 916, 916]
            .into_iter()
            .zip(cached)
            .enumerate()
        {
            fs::write(responses.join(format!("response-{}.json", index + 1)), serde_json::to_vec(&json!({"object":"chat.completion","choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{"prompt_tokens":prompt,"prompt_tokens_details":{"cached_tokens":cached}}})).unwrap()).unwrap();
        }
        let result = fixture.run("exec \"$NATIVE_AUTOMATION\" automation split-probe prefix-verify \"$RESPONSES\" 6 kv-recurrent\n".into(), &[("RESPONSES", responses.display().to_string())]);
        assert_eq!(
            result.process.status.unwrap().code(),
            Some(expected),
            "{}",
            stderr(&result)
        );
        if expected == 0 {
            assert!(stdout(&result).contains("repeated prompts restored from cache"));
        } else {
            assert!(stdout(&result).is_empty());
            assert!(!stderr(&result).is_empty());
        }
    }
}
