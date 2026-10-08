use super::*;

#[test]
fn stable_prefix_preserves_the_fixed_repository_rules_and_numbered_invariants() {
    assert_eq!(
        stable_prefix("family-1", 2).unwrap(),
        "You are working in repository family family-1. Follow its fixed rules.\nfamily-1-context-0000: src/family-1/module_0.rs owns invariant 0; preserve it exactly.\nfamily-1-context-0001: src/family-1/module_1.rs owns invariant 1; preserve it exactly."
    );
    let wrapped = stable_prefix("family-0", 38).unwrap();
    assert!(wrapped.ends_with(
        "family-0-context-0037: src/family-0/module_0.rs owns invariant 37; preserve it exactly."
    ));
}

#[test]
fn measured_tasks_interleave_families_and_keep_their_cacheable_prefixes() {
    let prompts = interleaved(2, 3, 2).unwrap();
    assert_eq!(
        prompts
            .iter()
            .map(|p| p.family.as_str())
            .collect::<Vec<_>>(),
        [
            "family-0", "family-1", "family-0", "family-1", "family-0", "family-1"
        ]
    );
    let prefix = stable_prefix("family-0", 2).unwrap();
    for (request, prompt) in prompts.iter().step_by(2).enumerate() {
        assert_eq!(
            prompt.prompt,
            format!(
                "{prefix}\nUnique task {request}: inspect module_{request}.rs and return its invariant only."
            )
        );
    }
    let seeded = interleaved(2, 1, 2).unwrap();
    assert_eq!(seeded[0].prompt, prompts[0].prompt);
    assert_eq!(seeded[1].prompt, prompts[1].prompt);
}

#[test]
fn admission_rejects_empty_overflowing_or_unbounded_workloads_before_allocation() {
    for (families, requests, blocks) in [
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (u64::MAX, 2, 1),
        (1, 10_001, 1),
        (1, 1, u64::MAX),
    ] {
        assert!(interleaved(families, requests, blocks).is_err());
    }
    assert_eq!(admit(2, 3, 120).unwrap(), 6);
    assert_eq!(admit(8, 2, 192).unwrap(), 16);
}
