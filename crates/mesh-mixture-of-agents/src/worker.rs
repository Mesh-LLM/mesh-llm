//! Worker role assignment and text extraction helpers.

use crate::ModelEntry;

/// Worker role determines the context shape and depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerRole {
    /// Fast small model — classify, quick proposal.
    Fast,
    /// Specialist — code, domain knowledge.
    Specialist,
    /// Strong reasoner — deeper analysis.
    Strong,
    /// General-purpose worker.
    Generalist,
    /// Reducer/finalizer — only invoked for arbitration.
    Reducer,
}

impl WorkerRole {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Specialist => "specialist",
            Self::Strong => "strong",
            Self::Generalist => "generalist",
            Self::Reducer => "reducer",
        }
    }
}

/// A worker assignment: which model plays which role.
pub struct Assignment {
    pub model_name: String,
    pub backend_index: usize,
    pub role: WorkerRole,
    /// Size tier this role was assigned from, resolved once here so downstream
    /// consumers never re-derive it from the model name.
    pub small_tier: bool,
}

/// Assign roles to models.
///
/// Heuristic: more models = more specialization.
/// With 2: fast + strong.  With 3+: fast + specialist(s) + strong.
pub fn assign_roles(models: &[ModelEntry]) -> Vec<Assignment> {
    if models.is_empty() {
        return vec![];
    }
    if models.len() == 1 {
        return vec![assignment_for(&models[0], WorkerRole::Generalist)];
    }

    // Reorder by capacity tier so role assignment lines up with model
    // capability instead of arbitrary list order:
    //   - "small tier"  = under SMALL_TIER_MAX_B parameters
    //   - "big tier"    = everything else, including unknown-size models
    //
    // Prefers the host's verified parameter count and only parses the name
    // when no count was supplied, matching the main router's metadata-first
    // sizing in `pick_model_classified`.
    let mut sorted: Vec<ModelEntry> = models.to_vec();
    sort_by_capacity(&mut sorted);
    // After sort: weakest first, strongest last — small tier before big, and by
    // verified size within each tier. That way:
    //   first  = fast       (smallest model)
    //   middle = specialist
    //   last   = strong     (biggest model — also used as reducer)

    let mut assignments = vec![assignment_for(&sorted[0], WorkerRole::Fast)];
    for entry in &sorted[1..sorted.len() - 1] {
        assignments.push(assignment_for(entry, WorkerRole::Specialist));
    }
    let last = sorted.last().expect("pool is non-empty");
    assignments.push(assignment_for(last, WorkerRole::Strong));
    assignments
}

fn assignment_for(entry: &ModelEntry, role: WorkerRole) -> Assignment {
    Assignment {
        model_name: entry.name.clone(),
        backend_index: entry.backend_index,
        role,
        small_tier: entry_is_small_tier(entry),
    }
}

/// Upper bound (exclusive) of the small tier, in billions of parameters.
///
/// Matches the legacy single-digit-B name heuristic (1–9B is small) and the
/// host's own `SMALL_TIER_MAX_B` so both sides of the boundary agree.
pub const SMALL_TIER_MAX_B: f64 = 10.0;

/// Is this pool member small-tier?
///
/// Verified size wins; the name heuristic is the fallback for an entry whose
/// host supplied no count. An unknown-size model reads as big-tier here, which
/// preserves the long-standing treatment of size-less names (`MiniMax-M2.5`,
/// `Coder-Next`) as capable models.
pub fn entry_is_small_tier(entry: &ModelEntry) -> bool {
    size_is_small_tier(entry.parameter_count_b)
        .unwrap_or_else(|| is_single_digit_b_name(&entry.name))
}

/// Classify a possibly-unknown parameter count. `None` in → `None` out, so
/// callers decide what unknown means in their context.
fn size_is_small_tier(parameter_count_b: Option<f64>) -> Option<bool> {
    parameter_count_b
        .filter(|count| count.is_finite() && *count > 0.0)
        .map(|count| count < SMALL_TIER_MAX_B)
}

/// A pool member's verified size, when the host supplied a usable one.
fn usable_parameter_count(entry: &ModelEntry) -> Option<f64> {
    entry
        .parameter_count_b
        .filter(|count| count.is_finite() && *count > 0.0)
}

/// Sort a pool weakest-first: small tier before big tier, and within each tier
/// by verified size ascending.
///
/// The tier split alone is not enough to order a pool. Sorting on the tier
/// boolean is stable, so it preserves input order among same-tier models — a
/// pool arriving as `[70B, 8B, 32B]` sorts to `[8B, 70B, 32B]` and hands the
/// Strong role (and the reducer) to the 32B while the verified 70B drafts as a
/// Specialist. Ordering by size within the tier puts the largest model last.
///
/// Entries whose size is unknown keep their relative input order and sort
/// *before* sized entries of the same tier. Unknown size reads as big-tier by
/// design — a size-less name like `MiniMax-M2.5` should still count as capable
/// — but an unverified guess must not take the Strong role from a model whose
/// size the host actually confirmed. An unsized model still becomes Strong when
/// it is genuinely the most capable thing the pool has.
pub(crate) fn sort_by_capacity(models: &mut [ModelEntry]) {
    models.sort_by(capacity_cmp);
}

/// Sort a pool strongest-first: the reverse of [`sort_by_capacity`].
///
/// Uses a reversed comparator rather than sorting ascending and reversing the
/// slice. Both orders must leave equally-ranked entries in their original
/// relative order, and reversing a sorted slice would instead flip them — which
/// silently changes the primary pick for a pool of same-tier, unsized models
/// (the common shape in fan-out simulations and any mesh whose hosts advertise
/// no GGUF size).
pub(crate) fn sort_by_capacity_desc(models: &mut [ModelEntry]) {
    models.sort_by(|left, right| capacity_cmp(right, left));
}

/// Order two pool members weakest-to-strongest.
fn capacity_cmp(left: &ModelEntry, right: &ModelEntry) -> std::cmp::Ordering {
    entry_is_small_tier(left)
        .cmp(&entry_is_small_tier(right))
        .reverse()
        .then_with(|| {
            match (usable_parameter_count(left), usable_parameter_count(right)) {
                // Both sized: smaller first, so the largest lands last.
                // `total_cmp` avoids the partial-ordering unwrap.
                (Some(left), Some(right)) => left.total_cmp(&right),
                // An unsized model sorts before a sized one of the same tier, so
                // the model whose size the host actually verified is the one
                // that ends up Strong.
                (Some(_), None) => std::cmp::Ordering::Greater,
                (None, Some(_)) => std::cmp::Ordering::Less,
                // Equal rank: leave the caller's order alone. `sort_by` is
                // stable, so this preserves host-provided ordering.
                (None, None) => std::cmp::Ordering::Equal,
            }
        })
}

/// Does this worker pool have a real quality gap?
///
/// Takes `(is_small_tier, role)` pairs — the tier is resolved by the caller at
/// dispatch, from the host's verified size when one was supplied. True when the
/// Strong-role worker is big-tier AND at least one other worker is small-tier.
/// This is the "MiniMax + small Qwens" shape where mixing tiers can pull answers
/// down. When all workers are the same tier (e.g. several small models lifting
/// each other via consensus) there is no gap and tier-aware patience stays
/// disabled.
pub(crate) fn has_quality_gap(workers: impl IntoIterator<Item = (bool, WorkerRole)>) -> bool {
    let mut strong_is_big = false;
    let mut any_small_non_strong = false;
    for (small_tier, role) in workers {
        if role == WorkerRole::Strong {
            strong_is_big = !small_tier;
        } else if small_tier {
            any_small_non_strong = true;
        }
    }
    strong_is_big && any_small_non_strong
}

/// Canonical base of a model name, mirroring the host's dedup logic: lowercase,
/// drop an `@branch` segment (keeping any `:quant` tag), strip common
/// prefixes/suffixes, keep only alphanumerics. Two aliases of the same model
/// map to the same base.
pub fn canonical_base_name(name: &str) -> String {
    let lower = name.to_lowercase();
    let no_branch = match lower.find('@') {
        Some(at) => {
            let after = &lower[at + 1..];
            let rest = after.find(':').map(|c| &after[c..]).unwrap_or("");
            format!("{}{}", &lower[..at], rest)
        }
        None => lower,
    };
    no_branch
        .replace("-gguf", "")
        .replace("unsloth/", "")
        .replace("meshllm/", "")
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .collect()
}

/// A pool is homogeneous when every member shares one canonical base — i.e. it
/// is the same model, possibly as repeated instances or quant variants.
///
/// Refinement is most valuable exactly here: identical/near-identical members
/// produce correlated drafts, and the cross-peer round is what pulls them apart
/// (measured: same-model 32B ×2 wins 48/2 with refinement vs 35/10 without,
/// while a diverse strong pool is ~unchanged). See
/// `evals/moa-openrouter/RESULTS.md`.
pub fn pool_is_homogeneous(models: &[crate::backend::ModelEntry]) -> bool {
    let mut bases = models.iter().map(|m| canonical_base_name(&m.name));
    match bases.next() {
        Some(first) => bases.all(|b| b == first),
        None => false,
    }
}

/// Return true if `name` advertises a single-digit billion-parameter count,
/// e.g. "Qwen3.5-2B-Q4_K_M" or "llama-3-7b-instruct".
///
/// Accepts a standalone digit 1-9 immediately followed by `b` or `B` at a word
/// boundary (not part of a multi-digit number, decimal, or alphanumeric run
/// like "BF16" or "A3B").
///
/// This is a *fallback only*, used when a pool member carries no verified
/// parameter count. It is unreliable by nature: a quant tag such as `-4bit`
/// or `-8bit` satisfies the pattern, so `Llama-3.3-70B-Instruct-4bit` reads as
/// small. Prefer [`entry_is_small_tier`], which consults the host's verified
/// size first.
pub(crate) fn is_single_digit_b_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    for i in 0..bytes.len() {
        let c = bytes[i];
        if !c.is_ascii_digit() {
            continue;
        }
        // Must be a single digit at a word boundary: previous char must
        // not be another digit, a '.', or an ASCII letter.
        if i > 0 {
            let prev = bytes[i - 1];
            if prev.is_ascii_digit() || prev == b'.' || prev.is_ascii_alphabetic() {
                continue;
            }
        }
        // Digit must be 1-9
        if c == b'0' {
            continue;
        }
        // Next byte must be b or B
        let Some(&next) = bytes.get(i + 1) else {
            continue;
        };
        if next != b'b' && next != b'B' {
            continue;
        }
        // Byte after must not be another digit (avoid BF16-like continuations)
        if bytes.get(i + 2).is_some_and(u8::is_ascii_digit) {
            continue;
        }
        return true;
    }
    false
}

/// Truncate `text` so the returned slice is at most `max_bytes` long,
/// honouring UTF-8 char boundaries (never panics, unlike `&text[..N]`).
pub fn truncate_chars(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut idx = max_bytes;
    while idx > 0 && !text.is_char_boundary(idx) {
        idx -= 1;
    }
    &text[..idx]
}

/// Strip `<think>...</think>` tags, return the remaining content.
///
/// Single linear scan over the input: skips think blocks (matched or
/// unclosed) and removes orphan `</think>` closers. The earlier shape
/// rebuilt the whole string on every block (`format!`/`replace` in a
/// loop) which is O(n*k) on long outputs with many tags.
pub fn strip_thinking(text: &str) -> String {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < bytes.len() {
        // Match a full <think>...</think> block.
        if bytes[i..].starts_with(OPEN.as_bytes()) {
            match text[i + OPEN.len()..].find(CLOSE) {
                Some(rel_end) => {
                    i += OPEN.len() + rel_end + CLOSE.len();
                    continue;
                }
                // Unclosed <think> — drop the rest of the string.
                None => break,
            }
        }
        // Drop an orphan </think> (a closer with no matching opener).
        if bytes[i..].starts_with(CLOSE.as_bytes()) {
            i += CLOSE.len();
            continue;
        }
        // Otherwise copy one char (UTF-8 safe: walk by character).
        let ch = text[i..].chars().next().expect("char boundary");
        out.push(ch);
        i += ch.len_utf8();
    }
    out.trim().to_string()
}

/// Extract content inside `<think>` tags.
pub fn extract_thinking(text: &str) -> String {
    if let Some(start) = text.find("<think>") {
        let after = &text[start + "<think>".len()..];
        if let Some(end) = after.find("</think>") {
            return after[..end].trim().to_string();
        }
        return after.trim().to_string();
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ModelEntry;

    fn entries(names: &[&str]) -> Vec<ModelEntry> {
        names
            .iter()
            .map(|n| ModelEntry::new((*n).to_string(), 0))
            .collect()
    }

    /// Pool members with verified sizes, in the order given.
    fn sized_entries(models: &[(&str, f64)]) -> Vec<ModelEntry> {
        models
            .iter()
            .enumerate()
            .map(|(index, (name, size))| {
                ModelEntry::new((*name).to_string(), index).with_parameter_count_b(Some(*size))
            })
            .collect()
    }

    fn role_of(assignments: &[Assignment], model: &str) -> WorkerRole {
        assignments
            .iter()
            .find(|a| a.model_name == model)
            .map(|a| a.role)
            .unwrap_or_else(|| panic!("{model} missing from assignments"))
    }

    #[test]
    fn strong_role_goes_to_the_largest_verified_model_whatever_the_input_order() {
        // Tier alone cannot order a pool: sorting on the tier boolean is stable,
        // so this input used to sort to [8B, 70B, 32B] and hand Strong — and
        // the reducer — to the 32B while the verified 70B drafted as a
        // Specialist.
        let pool = sized_entries(&[("big-70b", 70.6), ("small-8b", 8.2), ("mid-32b", 32.8)]);
        let assignments = assign_roles(&pool);

        assert_eq!(role_of(&assignments, "small-8b"), WorkerRole::Fast);
        assert_eq!(role_of(&assignments, "mid-32b"), WorkerRole::Specialist);
        assert_eq!(role_of(&assignments, "big-70b"), WorkerRole::Strong);
    }

    #[test]
    fn capacity_order_is_independent_of_input_permutation() {
        // Every arrangement of the same pool must produce the same roles.
        let permutations = [
            [("a-8b", 8.0), ("b-32b", 32.0), ("c-70b", 70.0)],
            [("c-70b", 70.0), ("b-32b", 32.0), ("a-8b", 8.0)],
            [("b-32b", 32.0), ("c-70b", 70.0), ("a-8b", 8.0)],
            [("c-70b", 70.0), ("a-8b", 8.0), ("b-32b", 32.0)],
        ];
        for permutation in permutations {
            let assignments = assign_roles(&sized_entries(&permutation));
            assert_eq!(
                role_of(&assignments, "a-8b"),
                WorkerRole::Fast,
                "input order {permutation:?} changed the Fast pick"
            );
            assert_eq!(
                role_of(&assignments, "c-70b"),
                WorkerRole::Strong,
                "input order {permutation:?} changed the Strong pick"
            );
        }
    }

    #[test]
    fn a_verified_size_outranks_an_unknown_size_for_the_strong_role() {
        // Unknown size reads as big-tier, which is deliberate — a size-less
        // name like `MiniMax-M2.5` should still be treated as capable. But it
        // must not take Strong from a model the host verified to be larger.
        let pool = vec![
            ModelEntry::new("unknown-size".to_string(), 0).with_parameter_count_b(None),
            ModelEntry::new("verified-70b".to_string(), 1).with_parameter_count_b(Some(70.0)),
        ];
        let assignments = assign_roles(&pool);
        assert_eq!(role_of(&assignments, "verified-70b"), WorkerRole::Strong);
    }

    #[test]
    fn an_all_small_pool_still_orders_by_verified_size() {
        let pool = sized_entries(&[("mid-8b", 8.0), ("tiny-1b", 1.0), ("small-4b", 4.0)]);
        let assignments = assign_roles(&pool);
        assert_eq!(role_of(&assignments, "tiny-1b"), WorkerRole::Fast);
        assert_eq!(role_of(&assignments, "mid-8b"), WorkerRole::Strong);
    }

    #[test]
    fn homogeneous_pool_detects_repeated_instances() {
        assert!(pool_is_homogeneous(&entries(&["Qwen3-32B", "Qwen3-32B"])));
    }

    #[test]
    fn homogeneous_pool_matches_aliases_of_one_model() {
        // Same base once prefixes/-gguf/@branch are normalised away.
        assert!(pool_is_homogeneous(&entries(&[
            "Qwen3-8B",
            "unsloth/Qwen3-8B",
            "Qwen3-8B@main",
        ])));
    }

    #[test]
    fn different_models_are_not_homogeneous() {
        assert!(!pool_is_homogeneous(&entries(&[
            "Qwen3-8B",
            "Llama-3.1-8B"
        ])));
    }

    #[test]
    fn empty_pool_is_not_homogeneous() {
        assert!(!pool_is_homogeneous(&[]));
    }

    #[test]
    fn truncate_chars_shorter_than_limit_is_passthrough() {
        assert_eq!(truncate_chars("hello", 100), "hello");
    }

    #[test]
    fn truncate_chars_exact_limit_is_passthrough() {
        assert_eq!(truncate_chars("hello", 5), "hello");
    }

    #[test]
    fn truncate_chars_respects_utf8_boundary() {
        // "café!" is 6 bytes: c a f 0xC3 0xA9 !  (é is 2 bytes).
        let s = "café!";
        assert_eq!(s.len(), 6);
        // Byte 4 is mid-codepoint (between 0xC3 and 0xA9). Naive `&s[..4]`
        // would panic; truncate_chars must walk back to byte 3 ("caf").
        assert_eq!(truncate_chars(s, 4), "caf");
        // Byte 5 IS a valid boundary (between é and !).
        assert_eq!(truncate_chars(s, 5), "café");
        // Within limit ⇒ passthrough.
        assert_eq!(truncate_chars(s, 6), "café!");
    }

    #[test]
    fn truncate_chars_handles_multibyte_only() {
        let s = "日本語"; // each char is 3 bytes ⇒ 9 bytes total
        // Byte 4 lands mid-char ⇒ walks back to 3 (first char only).
        assert_eq!(truncate_chars(s, 4), "日");
        // Byte 0 is always safe.
        assert_eq!(truncate_chars(s, 0), "");
    }

    #[test]
    fn assign_two_models() {
        let models = vec![ModelEntry::new("small", 0), ModelEntry::new("big", 1)];
        let assignments = assign_roles(&models);
        assert_eq!(assignments.len(), 2);
        assert_eq!(assignments[0].role, WorkerRole::Fast);
        assert_eq!(assignments[1].role, WorkerRole::Strong);
    }

    #[test]
    fn assign_three_models() {
        let models = vec![
            ModelEntry::new("small", 0),
            ModelEntry::new("mid", 1),
            ModelEntry::new("big", 2),
        ];
        let assignments = assign_roles(&models);
        assert_eq!(assignments.len(), 3);
        assert_eq!(assignments[0].role, WorkerRole::Fast);
        assert_eq!(assignments[1].role, WorkerRole::Specialist);
        assert_eq!(assignments[2].role, WorkerRole::Strong);
    }

    #[test]
    fn assign_roles_sorts_by_size_tier() {
        // 3B is last in list-order, but should NOT end up as Strong —
        // MiniMax (no digit) and Qwen3-32B (multi-digit) belong in the
        // big tier; Qwen2.5-3B and Qwen3-8B belong in the small tier.
        let models = vec![
            ModelEntry::new("MiniMax-M2.5", 0),
            ModelEntry::new("unsloth/Qwen3-32B-GGUF:Q4_K_M", 1),
            ModelEntry::new("Qwen3-8B", 2),
            ModelEntry::new("Qwen2.5-3B", 3),
        ];
        let assignments = assign_roles(&models);
        assert_eq!(assignments.len(), 4);
        // Fast = a small-tier model (3B or 8B)
        assert_eq!(assignments[0].role, WorkerRole::Fast);
        assert!(
            is_single_digit_b_name(&assignments[0].model_name),
            "fast should be small-tier, got {}",
            assignments[0].model_name
        );
        // Strong = a big-tier model (MiniMax or 32B)
        assert_eq!(assignments[3].role, WorkerRole::Strong);
        assert!(
            !is_single_digit_b_name(&assignments[3].model_name),
            "strong should be big-tier, got {}",
            assignments[3].model_name
        );
    }

    /// Regression: a quantised big model must not be demoted to `Fast`.
    ///
    /// Quant tags (`-4bit`, `-8bit`, `-4bpw`) satisfy the single-digit-B name
    /// pattern, so name-only tiering ranked `Llama-3.3-70B-Instruct-4bit` as
    /// small and handed the 8B the `Strong` role. That matters beyond ordering:
    /// `Fast` is packed at 256 max tokens with no tool schemas, so the 70B was
    /// drafting truncated and tool-blind while the 8B synthesized and anchored
    /// the patience gate.
    #[test]
    fn verified_size_keeps_a_quantised_big_model_as_strong() {
        let models = vec![
            ModelEntry::new("Llama-3.3-70B-Instruct-4bit", 0).with_parameter_count_b(Some(70.6)),
            ModelEntry::new("Qwen3-8B", 1).with_parameter_count_b(Some(8.2)),
        ];
        let assignments = assign_roles(&models);

        assert_eq!(assignments[0].model_name, "Qwen3-8B");
        assert_eq!(assignments[0].role, WorkerRole::Fast);
        assert_eq!(assignments[1].model_name, "Llama-3.3-70B-Instruct-4bit");
        assert_eq!(assignments[1].role, WorkerRole::Strong);

        // Without verified sizes the name heuristic still inverts these two.
        // Pinned so the fallback's limits stay visible rather than assumed away.
        let name_only = assign_roles(&[
            ModelEntry::new("Llama-3.3-70B-Instruct-4bit", 0),
            ModelEntry::new("Qwen3-8B", 1),
        ]);
        assert_eq!(
            name_only[1].model_name, "Qwen3-8B",
            "name-only tiering is known to invert quantised names; \
             this documents the gap the verified size closes"
        );
    }

    /// The reducer ladder ranks big-tier first, and must use verified sizes for
    /// that too — otherwise a quantised 70B lands in the last-resort bucket
    /// behind an 8B.
    #[test]
    fn reducer_ladder_prefers_a_verified_big_model() {
        let config = crate::GatewayConfig {
            backends: Vec::new(),
            models: vec![
                ModelEntry::new("Qwen3-8B", 0).with_parameter_count_b(Some(8.2)),
                ModelEntry::new("Llama-3.3-70B-Instruct-4bit", 1)
                    .with_parameter_count_b(Some(70.6)),
            ],
            worker_timeout: std::time::Duration::from_secs(1),
            hedge_delay: std::time::Duration::from_secs(1),
            reducer_timeout: std::time::Duration::from_secs(1),
            first_answer_grace: std::time::Duration::ZERO,
            strong_patience: std::time::Duration::ZERO,
            enable_thinking: None,
            actor_candidates: Vec::new(),
            reference_policy: crate::ReferencePolicy::Never,
            refinement_policy: crate::RefinementPolicy::Never,
        };

        let candidates = crate::reducer::reducer_candidates(&config);
        assert_eq!(
            candidates.first().map(|(name, _)| name.as_str()),
            Some("Llama-3.3-70B-Instruct-4bit"),
            "the verified 70B must lead the reducer ladder, not the 8B"
        );
    }

    #[test]
    fn size_heuristic_classifies_known_models() {
        // Single-digit B → small tier
        assert!(is_single_digit_b_name("Qwen3-8B"));
        assert!(is_single_digit_b_name("Qwen2.5-3B"));
        assert!(is_single_digit_b_name("Qwen3.5-9B-Q4_K_M"));
        assert!(is_single_digit_b_name("llama-3-7b-instruct"));

        // Multi-digit B → big tier
        assert!(!is_single_digit_b_name("Qwen3-32B"));
        assert!(!is_single_digit_b_name("llama-3-70b"));

        // No size in name → big tier
        assert!(!is_single_digit_b_name("MiniMax-M2.5"));
        assert!(!is_single_digit_b_name("Coder-Next"));

        // Active-params subset (A3B inside larger name) → big tier
        assert!(!is_single_digit_b_name("Qwen3.6-35B-A3B"));

        // BF16-style continuation → not a single-digit-B match
        assert!(!is_single_digit_b_name("model-bf16"));
    }

    #[test]
    fn strip_thinking_tags() {
        assert_eq!(strip_thinking("<think>foo</think>bar"), "bar");
        assert_eq!(
            strip_thinking("before<think>mid</think>after"),
            "beforeafter"
        );
        assert_eq!(strip_thinking("<think>only thinking"), "");
        assert_eq!(strip_thinking("no tags here"), "no tags here");
    }

    #[test]
    fn strip_thinking_drops_orphan_close() {
        // Orphan </think> with no matching opener: drop the closer,
        // keep surrounding content.
        assert_eq!(strip_thinking("stuff</think>answer"), "stuffanswer");
    }

    #[test]
    fn strip_thinking_handles_multiple_blocks_in_linear_time() {
        // Regression for PR #566 review item #5b: the previous shape
        // rebuilt the whole string on every think block (`format!` /
        // `replace` in a loop), which is O(n*k). Verify the new linear
        // implementation produces the same output for many blocks.
        let mut input = String::new();
        for i in 0..50 {
            input.push_str(&format!("<think>think-{i}</think>seg{i} "));
        }
        let stripped = strip_thinking(&input);
        let mut expected = String::new();
        for i in 0..50 {
            expected.push_str(&format!("seg{i} "));
        }
        assert_eq!(stripped, expected.trim());
    }

    #[test]
    fn strip_thinking_preserves_utf8() {
        // Multibyte characters outside think blocks must survive intact.
        assert_eq!(strip_thinking("<think>思</think>答案"), "答案");
        assert_eq!(strip_thinking("前置</think>中间<think>隐"), "前置中间");
    }

    /// `has_quality_gap` consumes tiers resolved at dispatch, so these build
    /// the pairs the way the fan-out loop does: through `entry_is_small_tier`.
    fn tiers(models: &[(&str, Option<f64>, WorkerRole)]) -> Vec<(bool, WorkerRole)> {
        models
            .iter()
            .map(|(name, size, role)| {
                let entry = ModelEntry::new(*name, 0).with_parameter_count_b(*size);
                (entry_is_small_tier(&entry), *role)
            })
            .collect()
    }

    #[test]
    fn quality_gap_minimax_plus_small_qwens() {
        // The motivating shape: big-tier strong + small-tier workers.
        let workers = tiers(&[
            ("Qwen2.5-3B-Instruct", None, WorkerRole::Fast),
            ("Qwen3-8B", None, WorkerRole::Specialist),
            ("MiniMax-M2.5", None, WorkerRole::Strong),
        ]);
        assert!(has_quality_gap(workers));
    }

    #[test]
    fn no_quality_gap_when_all_small_tier() {
        // "Many small models lift each other" — gate must stay off so
        // same-tier consensus keeps its current latency profile.
        let workers = tiers(&[
            ("Qwen2.5-3B-Instruct", None, WorkerRole::Fast),
            ("llama-3-7b-instruct", None, WorkerRole::Specialist),
            ("Qwen3-8B", None, WorkerRole::Strong),
        ]);
        assert!(!has_quality_gap(workers));
    }

    #[test]
    fn no_quality_gap_when_all_big_tier() {
        let workers = tiers(&[
            ("Qwen3-32B", None, WorkerRole::Fast),
            ("MiniMax-M2.5", None, WorkerRole::Strong),
        ]);
        assert!(!has_quality_gap(workers));
    }

    #[test]
    fn no_quality_gap_without_strong_role() {
        let workers = tiers(&[("Qwen2.5-3B-Instruct", None, WorkerRole::Generalist)]);
        assert!(!has_quality_gap(workers));
    }

    /// A verified size must override the name. `-4bit` parses as a size, so
    /// name-only tiering called this 70B model small and let the 8B hold the
    /// Strong role — inverting the gate this function exists to arm.
    #[test]
    fn quality_gap_uses_verified_size_over_a_quant_tag() {
        let workers = tiers(&[
            ("Qwen3-8B", Some(8.0), WorkerRole::Fast),
            (
                "Llama-3.3-70B-Instruct-4bit",
                Some(70.6),
                WorkerRole::Strong,
            ),
        ]);
        assert!(
            has_quality_gap(workers),
            "verified 70B strong + 8B worker is exactly the gap the gate is for"
        );

        // Same pool, same names, no verified sizes: the name heuristic reads
        // the 70B as small, so every worker looks small-tier and the gate stays
        // off. Pinned to document what the fallback still gets wrong.
        let name_only = tiers(&[
            ("Qwen3-8B", None, WorkerRole::Fast),
            ("Llama-3.3-70B-Instruct-4bit", None, WorkerRole::Strong),
        ]);
        assert!(!has_quality_gap(name_only));
    }

    #[test]
    fn verified_size_decides_the_tier_boundary() {
        let just_under = ModelEntry::new("mystery-model", 0).with_parameter_count_b(Some(9.9));
        let exactly_at = ModelEntry::new("mystery-model", 0).with_parameter_count_b(Some(10.0));
        assert!(entry_is_small_tier(&just_under));
        assert!(!entry_is_small_tier(&exactly_at));

        // A size-less model falls back to the name; an unparseable name reads
        // as big-tier, preserving the long-standing treatment of `MiniMax-M2.5`.
        assert!(!entry_is_small_tier(&ModelEntry::new("MiniMax-M2.5", 0)));
        assert!(entry_is_small_tier(&ModelEntry::new("Qwen3-8B", 0)));

        // Nonsense counts are ignored rather than trusted.
        for bogus in [Some(0.0), Some(-1.0), Some(f64::NAN), None] {
            let entry = ModelEntry::new("Qwen3-235B-A22B", 0).with_parameter_count_b(bogus);
            assert!(
                !entry_is_small_tier(&entry),
                "bogus size {bogus:?} must fall back to the name, not be trusted"
            );
        }
    }
}
